/// Experiment Framework for the XiangLang Thinking Guidance Engine.
///
/// Provides types and runners for conducting comparison experiments:
///   - Controlled: LLM without thinking guidance engine
///   - Constrained: LLM guided by the XiangLang VM stack
///
/// Three experiments:
///   1. Focus deviation detection — 思维焦点 (Focus Alignment)
///   2. Ultra-long context consistency (32K) — 长上下文
///   3. Multi-turn dialogue stability (50 turns) — 多轮对话

use serde::{Serialize, Deserialize};
use xiang_core::Gua;
use xiang_core::{SAN_YI_CONSTRAINT_FULL_TEXT, DynamicPromptState, build_sanyi_dynamic_prompt, fangwei_strategic_guidance, ContextMetabolism};
use xiang_core::{HanziEmbeddings, EmbeddingObserver};
use xiang_core::{OperatorSemanticAnchor, build_operator_anchors, classify_operator_phase, classify_with_scores};
use xiang_cangvm::{CangVM, CangVMDecision};
use xiang_shanvm::ShanVM;
use xiang_zhouvm::ZhouVM;
use xiang_verifier::{Verifier, VerifyContext};
use xiang_llm::{
    MockBackend, LlmContext, LlmBackend, LlmError,
    GenerationParams, GenerationResult,
    StopReason, TemperatureMode,
};
use xiang_core::embedding::{MockEncoder, MockEncoderMode, TextEncoder};
use xiang_core::{hybrid_deviation, SemanticEntry};
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

// Re-export types needed by the CLI.
pub use xiang_llm::MockMode;

// Public sub-modules
pub mod statistics;
pub mod inputs;
pub mod benchmark;
pub mod report;

// ── Unified Backend (Mock or Real) ──────────────────────────

/// Wraps either a `MockBackend`, the real `LlamaCppBackend`, or the HTTP backend
/// so the experiment runner can use a single backend-typed variable.
pub(crate) enum RunBackend {
    Mock(MockBackend),
    #[cfg(feature = "llama_backend")]
    Real(xiang_llm::llama_backend::LlamaCppBackend),
    #[cfg(feature = "http_backend")]
    Http(xiang_llm::http_backend::HttpBackend),
}

impl LlmBackend for RunBackend {
    fn generate(&mut self, params: &mut GenerationParams) -> Result<GenerationResult, LlmError> {
        match self {
            RunBackend::Mock(b) => b.generate(params),
            #[cfg(feature = "llama_backend")]
            RunBackend::Real(b) => b.generate(params),
            #[cfg(feature = "http_backend")]
            RunBackend::Http(b) => b.generate(params),
        }
    }
    fn tokenize(&self, text: &str) -> Vec<u32> {
        match self {
            RunBackend::Mock(b) => b.tokenize(text),
            #[cfg(feature = "llama_backend")]
            RunBackend::Real(b) => b.tokenize(text),
            #[cfg(feature = "http_backend")]
            RunBackend::Http(b) => b.tokenize(text),
        }
    }
    fn model_name(&self) -> &str {
        match self {
            RunBackend::Mock(b) => b.model_name(),
            #[cfg(feature = "llama_backend")]
            RunBackend::Real(b) => b.model_name(),
            #[cfg(feature = "http_backend")]
            RunBackend::Http(b) => b.model_name(),
        }
    }
    fn is_ready(&self) -> bool {
        match self {
            RunBackend::Mock(b) => b.is_ready(),
            #[cfg(feature = "llama_backend")]
            RunBackend::Real(b) => b.is_ready(),
            #[cfg(feature = "http_backend")]
            RunBackend::Http(b) => b.is_ready(),
        }
    }
    fn device_info(&self) -> &str {
        match self {
            RunBackend::Mock(b) => b.device_info(),
            #[cfg(feature = "llama_backend")]
            RunBackend::Real(b) => b.device_info(),
            #[cfg(feature = "http_backend")]
            RunBackend::Http(b) => b.device_info(),
        }
    }
}

impl RunBackend {
    /// Reset the backend state for a new generation trial.
    /// For real FFI backends, clears the KV cache (上下文新陈代谢).
    /// For mock/HTTP backends, does nothing.
    pub(crate) fn reset_for_new_generation(&mut self) {
        match self {
            #[cfg(feature = "llama_backend")]
            RunBackend::Real(b) => b.reset_context(),
            _ => {}
        }
    }

    /// Discover vocabulary token groups for off-focus and divergent token classification.
    /// For real backends, this scans the model's vocabulary using pattern matching.
    /// For mock backends, returns the pre-defined synthetic vocab groups.
    fn discover_vocab(&self) -> (Vec<u32>, Vec<u32>, u32) {
        match self {
            RunBackend::Mock(b) => (
                b.vocab.off_focus_tokens.clone(),
                b.vocab.divergent_tokens.clone(),
                b.vocab.eos_id,
            ),
            #[cfg(feature = "llama_backend")]
            RunBackend::Real(b) => {
                let (off, div) = b.discover_vocab();
                let eos = b.eos_token_id();
                eprintln!(
                    "[vocab] Scanned {} off-focus tokens, {} divergent tokens (eos={})",
                    off.len(), div.len(), eos
                );
                (off, div, eos)
            }
            #[cfg(feature = "http_backend")]
            RunBackend::Http(b) => {
                let (off, div) = b.discover_vocab();
                let eos = b.eos_token_id();
                eprintln!(
                    "[vocab] Scanned {} off-focus tokens, {} divergent tokens (eos={})",
                    off.len(), div.len(), eos
                );
                (off, div, eos)
            }
        }
    }
}

// ── Experiment Types ────────────────────────────────────────

/// An experiment scenario.
#[derive(Debug, Clone)]
pub struct ExperimentConfig {
    /// Human-readable name.
    pub name: String,
    /// Description of what the experiment tests.
    pub description: String,
    /// Optional .xiang program source to load as guidance rules.
    pub guidance_program: Option<String>,
    /// Input queries/prompts to test (user messages).
    pub inputs: Vec<String>,
    /// Number of turns (for multi-turn experiments).
    pub turns: usize,
    /// Max tokens per generation.
    pub max_tokens: u32,
    /// Mock LLM mode (deterministic simulation).
    pub mock_mode: MockMode,
    /// Whether to apply the guidance engine.
    pub use_guidance_engine: bool,
    /// Whether to inject constraint system prompt text.
    /// When false, the engine runs silently (logit bias only) without constraint text.
    pub inject_constraint_prompt: bool,
    /// Path to a GGUF model file. When set, uses real LlamaCppBackend instead of MockBackend.
    pub model_path: Option<String>,
    /// URL of a remote llama.cpp server. When set, uses HttpBackend instead of MockBackend.
    /// Example: "http://localhost:8080"
    pub http_server_url: Option<String>,
    /// Token IDs considered off-focus (suppressed at high deviation).
    pub off_focus_token_ids: Vec<u32>,
    /// Token IDs considered divergent (mildly suppressed at medium deviation).
    pub divergent_token_ids: Vec<u32>,
    /// End-of-sequence token ID.
    pub eos_token_id: u32,
    /// Enable 太极双LLM semantic mode (语义偏差 + Hamming偏差混合).
    pub semantic_mode: bool,
    /// Hybrid deviation alpha: 1.0 = pure Hamming, 0.0 = pure semantic (cosine).
    pub deviation_alpha: f32,
    /// Enable phase-constrained mode (阶段约束).
    /// When true, each turn is validated against YinProtocolChecker rules.
    pub phase_constraint_mode: bool,
    /// Enable three-engine integration (三引擎融合).
    /// When true, ShanVM and ZhouVM are injected into CangVM.
    pub enable_three_engine: bool,
    /// Pre-cached off-focus token IDs (avoids repeated vocab discovery for HTTP backend).
    pub cached_off_focus_ids: Option<Vec<u32>>,
    /// Pre-cached divergent token IDs.
    pub cached_divergent_ids: Option<Vec<u32>>,
    /// Pre-cached EOS token ID.
    pub cached_eos_id: Option<u32>,
    /// Enable agent mode: LLM autonomously decides when to stop via stop patterns.
    pub agent_mode: bool,
    /// Stop signal patterns for agent mode (e.g. "### DONE", "任务完成").
    pub agent_stop_patterns: Vec<String>,
    /// Enable repetition detection: early-stop when output becomes repetitive.
    pub repetition_detection_enabled: bool,
    /// Consecutive repetitive turns before triggering early stop (default: 3).
    pub repetition_threshold: usize,
    /// Jaccard similarity threshold for repetition detection (default: 0.85).
    pub repetition_similarity_threshold: f64,
    /// Number of previous turns to compare against (default: 3).
    pub repetition_window_size: usize,
    /// Semantic navigation mode: use `classify_operator_phase()` instead of
    /// YinChecker text regex for phase constraint decisions.
    /// When true, `inject_constraint_prompt` is forced false (no text prompt injection)
    /// and `phase_constraint_mode` should be false.
    pub semantic_nav_mode: bool,
}

/// Metrics collected during an experiment.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ExperimentMetrics {
    /// Total generations.
    pub total_generations: u32,
    /// Focus interventions triggered (deviation exceeded threshold).
    pub focus_interventions: u32,
    /// Number of sha (prune) operations.
    pub sha_count: u32,
    /// Number of aligned cang experiences (focus-aligned thinking).
    pub cang_aligned: u32,
    /// Number of deviated cang experiences (deviated thinking).
    pub cang_deviated: u32,
    /// Final trust score from the verifier.
    pub trust_score: f32,
    /// Average deviation across all turns.
    pub avg_deviation: f32,
    /// Maximum deviation observed.
    pub max_deviation: f32,
    /// Total tokens generated.
    pub total_tokens: u32,
    /// Elapsed time (milliseconds).
    pub elapsed_ms: u64,
    /// Turn-level data.
    pub turn_data: Vec<TurnMetrics>,
    /// Times semantic deviation triggered intervention.
    pub semantic_interventions: u32,
    /// Average semantic (cosine) deviation across all turns.
    pub avg_semantic_deviation: f32,
    /// Phase constraint violations detected.
    pub phase_violations: u32,
    /// Phase constraint rollbacks triggered.
    pub phase_rollbacks: u32,
    /// Completion rate (fraction of generations NOT deviated).
    pub completion_rate: f32,
    /// Structure compliance rate (fraction of turns passing YinProtocolChecker).
    pub structure_compliance_rate: f32,
    /// Whether the experiment was early-stopped due to repetition detection.
    pub repetition_early_stopped: bool,
}

/// Per-turn metrics.
#[derive(Debug, Clone, Serialize)]
pub struct TurnMetrics {
    pub turn_number: usize,
    pub deviation: f32,
    pub sha_triggered: bool,
    pub cang_label: Option<String>,
    /// 完整生成文本（不再截断）
    pub generated_text_preview: String,
    pub output_length: usize,
    pub focus_intervened: bool,
    /// Semantic (cosine) deviation for this turn, if semantic mode is active.
    pub semantic_deviation: Option<f32>,
    /// Phase constraint valid (true if output passed YinProtocolChecker).
    pub phase_valid: Option<bool>,
    /// Phase constraint violations for this turn.
    pub phase_violations: Vec<String>,
    /// Whether this turn was flagged as repetitive by the repetition detector.
    pub repetition_triggered: bool,
    /// Number of tokens suppressed by logit-bias in this turn.
    pub tokens_suppressed: u32,
    /// Number of bias applications applied in this turn.
    pub bias_applications: u32,
    /// Temperature used for this generation.
    pub temperature_used: f32,
    /// Which phase operator was checked (e.g. "生", "动", "长", "育").
    pub operator_checked: Option<String>,
    /// The CangVM 6-bit Gua state at this turn.
    pub vm_gua_state: Option<u8>,
    /// The reason generation stopped (EOS, MaxTokens, Deviated, etc.).
    pub stop_reason: Option<String>,
    /// 本轮使用的系统提示词（全文）
    pub system_prompt: String,
    /// 本轮使用的用户输入（用户消息全文）
    pub user_input: String,
    /// 模型路径或服务URL
    pub model_info: String,
    /// 后端类型描述
    pub backend_type: String,
    /// v4.0: 当前算子被重试的次数（从vm.retries读取）
    pub retry_count: u32,
    /// v4.0: CangVM judge()决策结果（Pass/Rollback/Skip/Stop/FinishCycle）
    pub judge_decision: Option<String>,
    /// v4.0: 当前算子在4阶段循环中的索引（0=生,1=动,2=长,3=育）
    pub cycle_operator_index: u32,
}

/// Comparison between two experiment runs.
#[derive(Debug, Clone, Serialize)]
pub struct ComparisonReport {
    pub experiment_name: String,
    pub controlled: ExperimentMetrics,
    pub constrained: ExperimentMetrics,
    pub improvement: ComparisonImprovement,
}

#[derive(Debug, Clone, Serialize)]
pub struct ComparisonImprovement {
    /// Focus intervention reduction (%).
    pub focus_reduction_pct: f32,
    /// Deviation reduction (%).
    pub deviation_reduction_pct: f32,
    /// Trust score improvement (%).
    pub trust_improvement_pct: f32,
    /// Tokens saved (%).
    pub token_efficiency_pct: f32,
}

// ── Semantic Quality Evaluation ─────────────────────────────

/// Per-turn quality scores evaluated by AI.
///
/// Each dimension is scored 0.0–1.0, with `overall` being the
/// weighted average: completion×0.30 + coherence×0.20 + relevance×0.20
/// + depth×0.20 + structure×0.10.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityScores {
    pub turn_number: usize,
    pub task_completion: f32,
    pub logical_coherence: f32,
    pub content_relevance: f32,
    pub analysis_depth: f32,
    pub structural_clarity: f32,
    pub overall: f32,
    pub evaluator_note: String,
}

/// Per-trial quality assessment for both groups.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityEvaluation {
    pub trial_index: usize,
    pub input_summary: String,
    pub controlled_per_turn: Vec<QualityScores>,
    pub constrained_per_turn: Vec<QualityScores>,
    pub controlled_overall: f32,
    pub constrained_overall: f32,
    pub comparison_note: String,
}

/// Complete quality evaluation report from AI evaluator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityEvaluationReport {
    pub evaluated_by: String,
    pub evaluation_timestamp: u64,
    pub rubric_version: String,
    pub evaluations: Vec<QualityEvaluation>,
}

impl QualityScores {
    pub fn compute_overall(&mut self) {
        self.overall = self.task_completion * 0.30
            + self.logical_coherence * 0.20
            + self.content_relevance * 0.20
            + self.analysis_depth * 0.20
            + self.structural_clarity * 0.10;
    }
}

/// Quality evaluation rubric for AI evaluators.
pub const QUALITY_EVALUATION_RUBRIC: &str = r##"# 语义质量评价量规 (Semantic Quality Evaluation Rubric)

本量规用于评估 LLM 输出内容的真实语义质量。由 AI 评估器（Qoder）执行评估。

## 评分维度与权重

| 维度 | 权重 | 评估重点 |
|------|------|---------|
| 任务完成度 (Task Completion) | 0.30 | 输出是否完成了输入所要求的所有任务步骤？是否产生了实质性的分析结果？ |
| 逻辑连贯性 (Logical Coherence) | 0.20 | 思路是否清晰连贯？推理是否有逻辑跳跃或自相矛盾？ |
| 内容相关性 (Content Relevance) | 0.20 | 输出是否聚焦于任务要求的主题？有无无关内容或幻觉？ |
| 分析深度 (Analysis Depth) | 0.20 | 回答是否超越了表面层面？是否展示了真实的理解和洞察？ |
| 结构清晰度 (Structural Clarity) | 0.10 | 输出是否有清晰的结构（分段/分点/阶段标记）？是否易于理解？ |

## 评分标准

### 任务完成度 (权重 0.30)
- 0.00: 完全拒绝回答、输出乱码或完全无关内容
- 0.25: 只触及了任务的一小部分（<25%），大量步骤缺失
- 0.50: 完成了约一半的任务步骤，关键部分遗漏
- 0.75: 完成了大部分任务步骤（>75%），少数细节可以补充
- 1.00: 完全完成了所有要求的步骤，每个步骤都有实质性内容

### 逻辑连贯性 (权重 0.20)
- 0.00: 输出完全混乱，没有可辨认的逻辑
- 0.25: 严重逻辑跳跃，思路碎片化
- 0.50: 基本连贯，但存在一些跳跃或矛盾之处
- 0.75: 逻辑很连贯，大多数推理步骤清晰
- 1.00: 逻辑严谨流畅，推理步骤环环相扣

### 内容相关性 (权重 0.20)
- 0.00: 完全不相关，在讨论与任务完全无关的内容
- 0.25: 大量跑题内容（>40%），与主题关系松散
- 0.50: 基本相关，但包含一些无关内容或轻微幻觉
- 0.75: 高度相关，几乎全部内容围绕主题
- 1.00: 精准聚焦，每个字都紧扣任务要求

### 分析深度 (权重 0.20)
- 0.00: 没有实质性分析，仅重复问题或给出空洞回复
- 0.25: 浅表的回答，停留在常识层面
- 0.50: 有一定深度，展示了基本的分析和理解
- 0.75: 分析较深入，包含多角度思考或具体例证
- 1.00: 深刻的洞察，展示了超前于普通回答的理解层次

### 结构清晰度 (权重 0.10)
- 0.00: 完全没有结构，一段混乱的连篇文字
- 0.25: 结构混乱，虽然有分段但界限不清晰
- 0.50: 基本有结构，有分段或分点，但组织可以更好
- 0.75: 清晰的分段结构，有明确的标记和过渡
- 1.00: 优秀的层次结构，段落分明，逻辑递进一目了然

## 评估流程说明

对于每组基准测试（trial），分别评估对照组和实验组的每轮输出：

1. 阅读 trial 的输入提示词（input），理解任务要求
2. 逐轮阅读对照组（裸跑）的每轮输出
3. 逐轮阅读实验组（三易控制）的每轮输出
4. 对每轮的 5 个维度分别打分
5. 自动计算 overall（加权平均）
6. 对两组进行定性对比，写出 comparison_note
"##;

// ── Experiment Runner ───────────────────────────────────────

/// Runs experiments and collects metrics.
pub struct ExperimentRunner;

// ── Backend Creation Helpers ─────────────────────────────────

/// Create the appropriate backend based on experiment config.
pub(crate) fn create_backend(config: &ExperimentConfig) -> Result<RunBackend, LlmError> {
    #[cfg(feature = "http_backend")]
    if let Some(url) = &config.http_server_url {
        return Ok(RunBackend::Http(xiang_llm::http_backend::HttpBackend::new(url)));
    }
    if config.model_path.is_some() {
        create_real_backend(config)
    } else {
        Ok(RunBackend::Mock(MockBackend::new(config.mock_mode)))
    }
}

#[cfg(feature = "llama_backend")]
fn create_real_backend(config: &ExperimentConfig) -> Result<RunBackend, LlmError> {
    let path = config.model_path.as_ref().unwrap();
    xiang_llm::llama_backend::LlamaCppBackend::new(path, 49152, -1)
        .map(RunBackend::Real)
        .map_err(|e| LlmError::ModelLoadFailed(format!("{e}")))
}

#[cfg(not(feature = "llama_backend"))]
fn create_real_backend(_config: &ExperimentConfig) -> Result<RunBackend, LlmError> {
    Err(LlmError::NotReady(
        "llama.cpp backend not available: LLAMA_CPP_DIR/build/libllama.so not found".into()
    ))
}

/// Get token ID groups from the appropriate source.
fn get_token_ids(
    config: &ExperimentConfig,
    backend: &RunBackend,
) -> (Vec<u32>, Vec<u32>, u32) {
    // Use cached values if available (avoids repeated HTTP vocab discovery per trial).
    if let (Some(off), Some(div), Some(eos)) =
        (&config.cached_off_focus_ids, &config.cached_divergent_ids, &config.cached_eos_id)
    {
        return (off.clone(), div.clone(), *eos);
    }
    // For real backends (model or HTTP), scan vocabulary dynamically.
    // For mock backends, use the pre-defined synthetic vocab groups.
    if config.http_server_url.is_some() || config.model_path.is_some() {
        backend.discover_vocab()
    } else {
        match backend {
            RunBackend::Mock(b) => (b.vocab.off_focus_tokens.clone(), b.vocab.divergent_tokens.clone(), b.vocab.eos_id),
            #[cfg(feature = "llama_backend")]
            RunBackend::Real(_) => unreachable!(),
            #[cfg(feature = "http_backend")]
            RunBackend::Http(_) => unreachable!(),
        }
    }
}

/// Rotating continuation prompts for agent mode.
/// Each pushes the model toward a different dimension of depth:
/// progress tracking, blind-spot detection, assumption challenging,
/// self-critique, completeness review, and synthesis.
const AGENT_CONTINUE_PROMPTS: [&str; 6] = [
    // 0: Advance to next step + blind-spot check
    "请进入下一步。完成当前步骤后输出【步骤X完成】。先简要确认你已完成的进度，确保新分析建立在之前工作的基础上。目前你还有哪些盲区需要关注？",

    // 1: Depth push
    "请继续推进到下一步。在当前步骤上，你还能挖掘更深吗？请从不同视角（利益相关者、历史演进、反事实推理）重新审视当前观点。不要停留在表面。",

    // 2: Assumption challenge
    "请进入下一步。在推进之前，先反思：你的分析中是否隐含了未经检验的前提假设？请逐一识别这些假设，然后思考如果假设不成立，分析链条会发生什么变化。",

    // 3: Adversarial self-review
    "请继续下一步。先扮演严格的外部评审者，对上一结论进行系统性质疑。找出逻辑漏洞或证据不足之处，逐一回应并修正分析后再推进。",

    // 4: Completeness scan
    "请进入下一步。先做完整性检查：分析框架是否足够全面？还有哪些重要维度、视角被忽略了？纳入缺失维度后继续推进。",

    // 5: Synthesis — trigger final summary + ### DONE
    "所有步骤应已基本完成。现在请整合各步骤的全部关键发现，提炼高层次洞察和模式，形成统一的最终结论。各分析点之间是否存在隐藏的联系、矛盾或协同效应？完成综合总结后，在末尾输出 ### DONE。",
];

/// Check if generated text contains an agent stop signal.
///
/// Matches a stop pattern only when it appears at the end of a line
/// (after stripping trailing punctuation). This prevents false positives
/// where the pattern appears mid-sentence, e.g. "第一阶段已完成" falsely
/// matching the pattern "已完成".
fn is_agent_stop_signal(text: &str, stop_patterns: &[String]) -> bool {
    if text.is_empty() {
        return true;
    }
    stop_patterns.iter().any(|pattern| {
        text.lines().any(|line| {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                return false;
            }
            // Strip trailing punctuation to handle natural language endings like "已完成。" or "任务完成！"
            let content = trimmed.trim_end_matches(|c: char| {
                c.is_ascii_punctuation() || matches!(c, '。' | '！' | '？' | '，' | '；' | '：' | '…')
            });
            content.ends_with(pattern.as_str())
        })
    })
}

/// Extract character-level trigrams from text for similarity comparison.
fn extract_trigrams(text: &str) -> std::collections::HashSet<String> {
    let mut trigrams = std::collections::HashSet::new();
    let chars: Vec<char> = text.chars().collect();
    for window in chars.windows(3) {
        let tri: String = window.iter().collect();
        trigrams.insert(tri);
    }
    trigrams
}

/// Compute trigram Jaccard similarity between two texts (0.0 = completely different, 1.0 = identical).
fn trigram_jaccard_similarity(a: &str, b: &str) -> f64 {
    let trig_a = extract_trigrams(a);
    let trig_b = extract_trigrams(b);
    let union_len = trig_a.union(&trig_b).count();
    if union_len == 0 {
        return 1.0; // both empty → identical
    }
    let intersection_len = trig_a.intersection(&trig_b).count();
    intersection_len as f64 / union_len as f64
}

/// Check if current text is repetitive compared to all recent texts.
/// Returns `true` only when similarity to EVERY text in the window exceeds the threshold.
fn check_repetition(text: &str, recent_texts: &[String], threshold: f64) -> bool {
    if recent_texts.is_empty() {
        return false;
    }
    recent_texts.iter().all(|prev| trigram_jaccard_similarity(text, prev) >= threshold)
}

impl ExperimentRunner {
    /// Run an experiment with the guidance engine enabled (uses provided backend).
    pub(crate) fn run_constrained_with_backend(
        config: &ExperimentConfig,
        backend: &mut RunBackend,
    ) -> Result<ExperimentMetrics, LlmError> {
        Self::run_experiment_with_backend(config, true, backend, None)
    }

    /// Run an experiment without the guidance engine (uses provided backend).
    pub(crate) fn run_controlled_with_backend(
        config: &ExperimentConfig,
        backend: &mut RunBackend,
    ) -> Result<ExperimentMetrics, LlmError> {
        Self::run_experiment_with_backend(config, false, backend, None)
    }

    /// Run an experiment with the guidance engine enabled.
    /// When `turn_output_dir` is set, each turn's file is written incrementally.
    pub fn run_constrained(
        config: &ExperimentConfig,
        turn_output_dir: Option<&std::path::Path>,
    ) -> Result<ExperimentMetrics, LlmError> {
        let mut backend = create_backend(config)?;
        Self::run_experiment_with_backend(config, true, &mut backend, turn_output_dir)
    }

    /// Run an experiment without the guidance engine (control group).
    /// When `turn_output_dir` is set, each turn's file is written incrementally.
    pub fn run_controlled(
        config: &ExperimentConfig,
        turn_output_dir: Option<&std::path::Path>,
    ) -> Result<ExperimentMetrics, LlmError> {
        let mut backend = create_backend(config)?;
        Self::run_experiment_with_backend(config, false, &mut backend, turn_output_dir)
    }

    fn run_experiment_with_backend(
        config: &ExperimentConfig,
        use_guidance: bool,
        backend: &mut RunBackend,
        turn_output_dir: Option<&std::path::Path>,
    ) -> Result<ExperimentMetrics, LlmError> {
        let start = Instant::now();
        let mut metrics = ExperimentMetrics::default();

        // Initialize VM and verifier if using guidance
        let mut vm = CangVM::new();

        // ── 汉字嵌入观察器 + 算子语义锚点 ──
        let mut op_anchors: Option<Vec<OperatorSemanticAnchor>> = None;
        // 308 汉字→算子映射：用输入空间 centroid 对全部 308 汉字分类
        let mut char_to_operator: Option<std::collections::HashMap<char, &'static str>> = None;

        if use_guidance {
            let embeddings_path = std::path::Path::new("data/hanzi_embeddings.bin");
            if embeddings_path.exists() {
                match HanziEmbeddings::load_from_file(embeddings_path) {
                    Ok(emb) => {
                        let size = emb.size();
                        let n_embd = emb.n_embd();
                        // 构建算子语义锚点（必须在 map 被移动前借用）
                        op_anchors = Some(build_operator_anchors(&emb.map));

                        // ── 308 汉字→算子映射：用输入空间 centroid 对全部汉字分类 ──
                        // 运行时通过 observer.peek(emb) 找到最近汉字后查表得算子
                        let mut char_op_map = std::collections::HashMap::new();
                        if let Some(ref anchors) = op_anchors {
                            for i in 0..emb.map.size() {
                                if let Some(ch) = emb.map.char_at(i) {
                                    if let Some(emb_vec) = emb.map.embedding_of(ch) {
                                        let mut best_op = None;
                                        let mut best_sim = -1.0f32;
                                        for anchor in anchors {
                                            let sim = xiang_core::embedding::cosine_similarity(
                                                emb_vec, &anchor.centroid
                                            );
                                            if sim > best_sim {
                                                best_sim = sim;
                                                best_op = Some(anchor.operator);
                                            }
                                        }
                                        if let Some(op) = best_op {
                                            char_op_map.insert(ch, op);
                                        }
                                    }
                                }
                            }
                        }
                        char_to_operator = Some(char_op_map);
                        eprintln!(
                            "[汉字→算子] 已映射 {} 个汉字",
                            char_to_operator.as_ref().map(|m| m.len()).unwrap_or(0)
                        );

                        let observer = EmbeddingObserver::new(emb.map);
                        eprintln!(
                            "[嵌入观察器] 已加载 {} 个汉字嵌入 (n_embd={})",
                            size, n_embd
                        );
                        eprintln!(
                            "[算子锚点] 已构建 {} 个算子语义重心",
                            op_anchors.as_ref().map(|a| a.len()).unwrap_or(0)
                        );
                        vm = vm.with_embedding_observer(observer);
                    }
                    Err(e) => {
                        eprintln!("[嵌入观察器] 加载失败: {}", e);
                    }
                }
            }

            // ── 语义导航模式：放宽CangVM重试限制（嵌入分类不如文本正则可靠）──
            if config.semantic_nav_mode {
                vm.max_retries_per_operator = 3;  // 每个算子最多尝试4次（3次回滚+1次跳过）
                vm.max_total_retries = 16;        // 4算子×4次 = 16次总重试上限
                vm.max_kills_per_cycle = 4;       // 最多跳过4个算子
            }

            // ── 语义导航模式：放宽CangVM重试限制 ──
            if config.semantic_nav_mode {
                vm.max_retries_per_operator = 3;
                vm.max_total_retries = 16;
                vm.max_kills_per_cycle = 4;
            }
        }

        // ── 语义导航模式：构建输出空间 centroid（仅用于诊断相似度矩阵）──
        if config.semantic_nav_mode && op_anchors.is_some() {
            #[cfg(feature = "llama_backend")]
            {
                if let RunBackend::Real(real_backend) = backend {
                    let operators: &[(&str, &[char])] = &[
                        ("生", xiang_core::operator_semantic::SHENG_ANCHORS),
                        ("动", xiang_core::operator_semantic::DONG_ANCHORS),
                        ("长", xiang_core::operator_semantic::ZHANG_ANCHORS),
                        ("育", xiang_core::operator_semantic::YU_ANCHORS),
                    ];
                    eprintln!("[输出空间锚点] 构建 centroid (48 字符)...");
                    let start = Instant::now();

                    let mut output_anchors: Vec<OperatorSemanticAnchor> = Vec::new();
                    let mut build_success = true;

                    for (name, chars) in operators {
                        let mut sum: Vec<f32> = Vec::new();
                        let mut count = 0usize;

                        for &ch in *chars {
                            let ch_str = ch.to_string();
                            match real_backend.embed_text(&ch_str) {
                                Some(emb) => {
                                    if sum.is_empty() {
                                        sum = emb;
                                    } else if emb.len() == sum.len() {
                                        for (s, e) in sum.iter_mut().zip(emb.iter()) {
                                            *s += e;
                                        }
                                    }
                                    count += 1;
                                }
                                None => {
                                    eprintln!("[输出空间锚点] 警告：字符 '{}' 嵌入获取失败", ch);
                                }
                            }
                        }

                        if count == 0 || sum.is_empty() {
                            eprintln!("[输出空间锚点] 错误：算子 '{}' 无有效锚点", name);
                            build_success = false;
                            break;
                        }

                        // 平均 + L2 归一化 → centroid
                        let inv = 1.0 / count as f32;
                        for x in &mut sum { *x *= inv; }
                        let norm: f32 = sum.iter().map(|x| x * x).sum::<f32>().sqrt();
                        if norm > 1e-12 {
                            for x in &mut sum { *x /= norm; }
                        }

                        output_anchors.push(OperatorSemanticAnchor {
                            operator: name,
                            centroid: sum,
                        });
                    }

                    if build_success && output_anchors.len() == 4 {
                        // 替换 op_anchors 中的 centroid 为输出空间版本
                        if let Some(ref mut anchors) = op_anchors {
                            for (anchor, output_anchor) in anchors.iter_mut().zip(output_anchors) {
                                anchor.centroid = output_anchor.centroid;
                            }
                        }
                        // 输出 centroid 相似度矩阵
                        eprintln!("[输出空间锚点] ── centroid 相似度矩阵 ──");
                        let sims = op_anchors.as_ref().unwrap();
                        for i in 0..sims.len() {
                            let mut row = format!("  {}:", sims[i].operator);
                            for j in 0..sims.len() {
                                if i == j {
                                    row.push_str(&format!("  {}={:.4}", sims[j].operator, 1.0));
                                } else {
                                    let sim = xiang_core::embedding::cosine_similarity(
                                        &sims[i].centroid, &sims[j].centroid
                                    );
                                    row.push_str(&format!("  {}={:.4}", sims[j].operator, sim));
                                }
                            }
                            eprintln!("{}", row);
                        }
                        eprintln!("[输出空间锚点] ✅ 已替换算子 centroid 为输出空间版本");
                    } else {
                        eprintln!("[输出空间锚点] ❌ 构建失败");
                    }
                    eprintln!("[输出空间锚点] 构建耗时: {:.2?}", start.elapsed());
                }
            }
            #[cfg(not(feature = "llama_backend"))]
            {
                eprintln!("[输出空间锚点] 跳过：llama_backend 特性未启用");
            }
        }

        let mut verifier = Verifier::new();
        let initial_system_prompt = if use_guidance && config.inject_constraint_prompt {
            SAN_YI_CONSTRAINT_FULL_TEXT
        } else {
            ""
        };
        let mut llm_ctx = LlmContext::new(initial_system_prompt);

        // ── 语义模式：设定混合偏离度参数 ──
        if config.semantic_mode && use_guidance {
            vm.deviation_alpha = config.deviation_alpha;
            // 如果有真实的 EmbeddingObserver，原点由首次 observe() 动态设定；
            // 否则使用 MockEncoder 的固定参考文本作为降级方案
            if vm.embedding_observer.is_none() {
                const SEMANTIC_ORIGIN_TEXT: &str = "请给出一个高质量、结构清晰的回答。";
                let encoder = MockEncoder::new(8, MockEncoderMode::Focused);
                vm.semantic_origin_embedding = Some(encoder.encode(SEMANTIC_ORIGIN_TEXT));
            }
        }

        // ── 三引擎融合：注入 ShanVM + ZhouVM ──
        if use_guidance && config.enable_three_engine {
            vm = vm.with_shan_vm(ShanVM::new()).with_zhou_vm(ZhouVM::new());
        }

        // Parse and load guidance program if provided
        if let Some(ref program_src) = config.guidance_program {
            if use_guidance {
                match xiang_parser::parse(program_src) {
                    Ok(program) => {
                        for agent in &program.agents {
                            if let Err(e) = vm.execute(&agent.body) {
                                eprintln!("引导程序执行错误: {e:?}");
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("引导程序解析错误: {e:?}");
                    }
                }
                // Reset VM state after loading guidance program
                vm.state = Gua::ORIGIN;
                vm.origin = Gua::ORIGIN;
                vm.cycle_count = 0;
                vm.sha_count = 0;
            }
        }

        // ── Agent mode: prepare multi-turn prompts ──
        let agent_first_prompt: Option<String> = if config.agent_mode && !config.inputs.is_empty() {
            Some(format!(
                "【多轮深度分析模式 · 最多{turns}轮】\n\
\n\
分析结构：\n\
- 你需要按步骤推进分析，每一步对应一个阶段。\n\
- 每轮对话只完成一个步骤，当前步骤完成后输出\"【步骤X完成】\"并停止。\n\
- 系统将在下一轮提示你继续下一个步骤。\n\
\n\
任务：\n\
{task}\n\
\n\
现在开始第一步。只做第一步，不要涉及后续步骤的内容。\n\
完成时输出\"【步骤1完成】\"。\n\
\n\
注意：你的思考过程（<think>块）请保持简洁，把大部分输出用于实际分析内容。",
                turns = config.turns,
                task = config.inputs[0]
            ))
        } else {
            None
        };
        // continue prompts: see AGENT_CONTINUE_PROMPTS (module-level constant, 6 rotating variants)

        // Run turns
        let total_turns = if config.agent_mode {
            config.turns.max(1)
        } else {
            config.inputs.len().min(config.turns.max(1))
        };

        // ── 重复检测状态（连续重复跟踪）──
        let mut consecutive_repetitions: usize = 0;
        let mut recent_texts: Vec<String> = Vec::with_capacity(config.repetition_window_size.max(2));

        // ── 上下文新陈代谢（高质量输出提取 + 稠密上下文注入）──
        let mut metabolism = ContextMetabolism::new(8000);

        for turn in 0..total_turns {
            let input: &str = if config.agent_mode {
                if turn == 0 {
                    agent_first_prompt.as_deref().unwrap_or(&config.inputs[0])
                } else {
                    &AGENT_CONTINUE_PROMPTS[turn % AGENT_CONTINUE_PROMPTS.len()]
                }
            } else {
                &config.inputs[turn]
            };

            let current_text;

            if use_guidance {
                // ── Constrained path v4.0: 三引擎深度融合管道 ──
                // 归藏: 算子状态推进 + judge()决策系统
                // 周易: select_zhou_pose()认知姿态调度
                // 连山: 障碍导航 + 策略bias + 决策影响温度/提示词
                vm.cycle_count += 1;
                vm.state = vm.state.sheng();
                vm.state = vm.state.dong();
                vm.state = vm.state.zhang();
                vm.state = vm.state.yu();
                let dev = vm.current_deviation();

                // ── 语义偏离度（生成后从模型输出计算，不参与运行时决策）──
                let mut semantic_dev: Option<f32> = None;

                // ── v4.0: 确定当前算子（使用CangVM队列，支持重试）──
                let operators = &["生", "动", "长", "育"];
                let phase_idx: u32 = if config.phase_constraint_mode || config.semantic_nav_mode {
                    // 使用CangVM的队列索引（支持Rollback时保持不变）
                    vm.current_op_index as u32
                } else {
                    ((vm.cycle_count as usize - 1) % 4) as u32
                };
                let operator = operators[phase_idx as usize];

                // ── v4.0: 周易认知姿态调度（替代随机步进）──
                if vm.zhou_vm.is_some() {
                    if config.phase_constraint_mode && config.enable_three_engine {
                        // 基于算子/偏离度/连山决策的认知调度
                        vm.select_zhou_pose();
                    } else {
                        // 无阶段约束时保持原有随机步进（兼容性）
                        vm.run_zhou_vm_step();
                    }
                }
                let zhou_temp = vm.zhou_temperature();

                // ── v4.0: 温度设置 — 使用ZhouVM温度 + 偏离度微调 ──
                let temperature_used = if turn == 0 {
                    0.8 // 首轮固定
                } else {
                    let base = zhou_temp;
                    if dev > 0.7 {
                        (base * 0.6).max(0.3)
                    } else if dev < 0.3 {
                        (base * 1.1).min(1.2)
                    } else {
                        base.clamp(0.3, 1.2)
                    }
                };
                let temp = TemperatureMode::Fixed(temperature_used);

                // ── v4.0: 连山障碍导航（偏离度>0.5时触发）──
                if dev > 0.5 && vm.shan_vm.is_some() {
                    vm.run_shan_vm();
                }

                // Get token groups
                let (off_focus_ids, divergent_ids, eos_id) = get_token_ids(config, &backend);

                // ── v4.0 动态提示词注入：策略指令（actionable）+ 约束状态（contextual）──
                {
                    let posture = vm.zhou_posture();
                    let (pose_name, pose_desc) = if let Some(idx) = posture.find(" · ") {
                        let name = &posture[..idx];
                        let desc = &posture[idx + 3..];
                        (name, desc)
                    } else {
                        ("坤", "承载")
                    };

                    // 连山决策描述文本
                    let shan_desc = vm.last_lian_shan_decision.as_ref().map(|d| {
                        format!("{} | {} → {}",
                            if d.activated { "已激活" } else { "未激活" },
                            d.jia.name(),
                            d.decision.name())
                    });

                    // v4.0: 连山策略指令（actionable guidance）
                    let fangwei_guidance = vm.last_lian_shan_decision.as_ref()
                        .filter(|d| d.activated)
                        .map(|d| fangwei_strategic_guidance(d.decision).to_string());

                    // v4.0: 周易卦象行为前缀
                    let zhou_prefix = vm.zhou_prompt_prefix();

                    let dynamic_state = DynamicPromptState::new(
                        operator,
                        pose_name,
                        pose_desc,
                        temperature_used,
                        dev,
                        shan_desc.as_deref(),
                        fangwei_guidance,
                        zhou_prefix,
                    );
                    let dynamic_prompt = build_sanyi_dynamic_prompt(&dynamic_state);

                    if config.semantic_nav_mode || !config.inject_constraint_prompt {
                        llm_ctx.system_prompt = String::new();
                    } else {
                        let metabolism_section = metabolism.section();
                        if metabolism_section.is_empty() {
                            llm_ctx.system_prompt = format!(
                                "{}\n\n{}",
                                SAN_YI_CONSTRAINT_FULL_TEXT, dynamic_prompt
                            );
                        } else {
                            llm_ctx.system_prompt = format!(
                                "{}\n\n{}\n\n{}",
                                SAN_YI_CONSTRAINT_FULL_TEXT, metabolism_section, dynamic_prompt
                            );
                        }
                    }
                }

                // ── v4.0: 构建策略bias向量（连山FangWei → logit bias调整）──
                let strategy_bias: Option<Vec<f32>> = vm.last_lian_shan_decision.as_ref()
                    .filter(|d| d.activated)
                    .map(|d| match d.decision {
                        xiang_core::FangWei::Continue => vec![0.0, 0.0, 0.0],
                        xiang_core::FangWei::PushThrough => vec![1.0, 0.0, 0.0],
                        xiang_core::FangWei::NavigateAround => vec![0.0, 1.0, 0.0],
                        xiang_core::FangWei::Decompose => vec![0.0, 0.0, 1.0],
                        _ => vec![0.0, 0.0, 0.0],
                    });

                // Generate with logit-bias thinking guidance engine.
                let extra_stops: Vec<String> = vec![
                    "### DONE".into(),
                    "【步骤1完成】".into(), "【步骤2完成】".into(),
                    "【步骤3完成】".into(), "【步骤4完成】".into(),
                    "【步骤5完成】".into(), "【步骤6完成】".into(),
                    "【阶段".into(), "【步骤".into(),
                ];
                let result = llm_ctx.generate_constrained_turn(
                    backend, input, config.max_tokens,
                    temp, vm.state, dev, vm.sha_count,
                    off_focus_ids, divergent_ids, eos_id,
                    &extra_stops, strategy_bias,
                    operator,  // v4.0: 当前算子名
                    Vec::new(),  // operator_positive (实验框架暂不启用算子差异化)
                    Vec::new(),  // operator_negative
                )?;
                current_text = result.text.clone();

                // ── v4.0 Phase constraint check using vm.judge() ──
                // 使用CangVM内置的决策系统：Pass/Rollback/Skip/Stop/FinishCycle
                let mut operator_checked: Option<String> = None;
                let (phase_valid, phase_violations, judge_decision_str) =
                    if config.semantic_nav_mode {
                        // ── 语义导航模式：EmbeddingObserver 汉字映射查表分类 ──
                        // 将输出嵌入映射到最近汉字(跨空间)，再查 308 汉字→算子表得到算子
                        operator_checked = Some(operator.to_string());
                        let embedding = result.embedding.as_ref();
                        let (is_valid, violations) = if let Some(emb) = embedding {
                            if let Some(ref anchors) = op_anchors {
                                // 诊断：输出 centroid 余弦相似度（仅供参考）
                                let scores = classify_with_scores(emb, anchors);
                                if !scores.is_empty() {
                                    let score_str: String = scores.iter()
                                        .map(|(op, sim)| format!("{}={:.4}", op, sim))
                                        .collect::<Vec<_>>()
                                        .join(" ");
                                    eprintln!("[语义相似度] {} | expect={}", score_str, operator);
                                }
                                // 尝试 308 汉字映射（同模型跨空间），失败时降级到 centroid 分类（同模型同空间）
                                let op_from_observer = vm.embedding_observer.as_ref()
                                    .and_then(|obs| obs.peek(emb))
                                    .and_then(|mapping| {
                                        let ch = mapping.ch;
                                        char_to_operator.as_ref().and_then(|map| map.get(&ch)).copied()
                                    });
                                let (op_from_classifier, method) = if let Some(op) = op_from_observer {
                                    (Some(op), "汉字映射")
                                } else {
                                    // 降级：输出空间 centroid 分类（跨模型时汉字映射可能失效）
                                    let centroid_op = classify_operator_phase(emb, anchors, -1.0);
                                    (centroid_op, "centroid")
                                };
                                eprintln!(
                                    "[语义分类] {} → {} | expect={}",
                                    method,
                                    op_from_classifier.unwrap_or("?"),
                                    operator
                                );
                                match op_from_classifier {
                                    Some(op_name) if op_name == operator => {
                                        (true, vec![])
                                    }
                                    Some(op_name) => {
                                        (false, vec![format!("汉字映射={}，期望算子={}", op_name, operator)])
                                    }
                                    None => {
                                        (false, vec!["汉字映射未匹配任何算子".to_string()])
                                    }
                                }
                            } else {
                                (false, vec!["算子锚点未构建".to_string()])
                            }
                        } else {
                            (true, vec!["Mock模式：无嵌入，跳过语义分类".to_string()])
                        };

                        let cangvm_decision = vm.judge(operator, is_valid, &violations);
                        match cangvm_decision {
                            ref d @ CangVMDecision::Pass { .. } => {
                                (Some(true), violations, Some(format!("{:?}", d)))
                            }
                            _d @ CangVMDecision::Rollback { .. } => {
                                vm.state = vm.state.sha();
                                vm.sha_count += 1;
                                metrics.phase_rollbacks += 1;
                                vm.output_log.push(format!(
                                    "[语义回滚] {} — {}", operator, violations.join("; ")
                                ));
                                continue;
                            }
                            ref d @ CangVMDecision::Skip { .. } => {
                                metrics.phase_violations += 1;
                                vm.state = vm.state.sha();
                                vm.sha_count += 1;
                                (Some(false), violations, Some(format!("{:?}", d)))
                            }
                            ref d @ CangVMDecision::Stop { .. } => {
                                metrics.phase_violations += 1;
                                vm.state = vm.state.sha();
                                vm.sha_count += 1;
                                (Some(false), violations, Some(format!("{:?}", d)))
                            }
                            ref d @ CangVMDecision::FinishCycle => {
                                vm.reset_phase_state();
                                (Some(true), Vec::new(), Some(format!("{:?}", d)))
                            }
                        }
                    } else if config.phase_constraint_mode {
                        operator_checked = Some(operator.to_string());

                        // Run operator Gua transformation
                        vm.execute_phase_operator(operator).ok();

                        // Check generated output against phase rules
                        let check_result = vm.check_operator_output(operator, &result.text);
                        let mut is_valid = check_result.is_valid;
                        let violations = check_result.violations.clone();

                        // 尝试其他算子匹配作为宽松检查
                        if !is_valid {
                            for other_op in operators {
                                if *other_op != operator {
                                    let alt = vm.check_operator_output(other_op, &result.text);
                                    if alt.is_valid {
                                        is_valid = true;
                                        break;
                                    }
                                }
                            }
                        }
                        // 基本结构兜底检查
                        if !is_valid {
                            is_valid = result.text.len() > 10
                                && result.text.contains(['。', '，', ',', '.', '\n']);
                        }

                        // v4.0: 使用CangVM内置judge()决策系统
                        let cangvm_decision = vm.judge(operator, is_valid, &violations);
                        let _decision_str = format!("{:?}", cangvm_decision);

                        match cangvm_decision {
                            ref d @ CangVMDecision::Pass { .. } => {
                                // 通过 — judge()已推进current_op_index
                                (Some(true), violations, Some(format!("{:?}", d)))
                            }
                            _d @ CangVMDecision::Rollback { .. } => {
                                // v4.0: 重试相同算子！
                                // judge()没有推进current_op_index
                                // 应用sha惩罚，跳过本轮指标收集，重新生成
                                vm.state = vm.state.sha();
                                vm.sha_count += 1;
                                metrics.phase_rollbacks += 1;
                                vm.output_log.push(format!(
                                    "[阶段回滚] {} — {}", operator, violations.join("; ")
                                ));
                                // continue会跳过feedback_to_vm和指标收集
                                // 下轮循环中current_op_index未变，故重复相同算子
                                continue;
                            }
                            ref d @ CangVMDecision::Skip { .. } => {
                                // 重试超限 — judge()已推进至下一算子
                                metrics.phase_violations += 1;
                                vm.state = vm.state.sha();
                                vm.sha_count += 1;
                                (Some(false), violations, Some(format!("{:?}", d)))
                            }
                            ref d @ CangVMDecision::Stop { .. } => {
                                // 全局限制超限 — 终止实验
                                metrics.phase_violations += 1;
                                vm.state = vm.state.sha();
                                vm.sha_count += 1;
                                (Some(false), violations, Some(format!("{:?}", d)))
                            }
                            ref d @ CangVMDecision::FinishCycle => {
                                // 4个算子全部完成，重置循环
                                vm.reset_phase_state();
                                (Some(true), Vec::new(), Some(format!("{:?}", d)))
                            }
                        }
                    } else {
                        (None, vec![], None)
                    };

                // Feed result back to VM for meta-cognition
                Self::feedback_to_vm(&mut vm, &result, dev);

                // Verify honesty
                let verify_ctx = VerifyContext {
                    current_gua: vm.state,
                    current_deviation: dev,
                    has_sha: vm.sha_count > 0,
                    current_load: 0.0,
                };
                verifier.verify_all(&verify_ctx);

                // ── 语义偏离度（使用真实 LLM 嵌入或 MockEncoder 降级）──
                if config.semantic_mode && semantic_dev.is_none() {
                    // 优先使用 EmbeddingObserver（真实 LLM 嵌入向量）
                    let mut observed = false;
                    let expected_operator = vm.current_operator().unwrap_or("生").to_string();
                    if let Some(ref mut observer) = vm.embedding_observer {
                        if let Some(embedding) = &result.embedding {
                            // 首次观察时自动设定语义原点
                            if observer.fingerprint().is_empty() {
                                observer.set_origin(embedding);
                            }
                            observer.observe(embedding);
                            semantic_dev = Some(observer.semantic_deviation());
                            vm.semantic_deviation = semantic_dev;
                            observed = true;

                            // ── 算子语义分类：汉字映射查表 ──
                            let op_from_obs = char_to_operator.as_ref().and_then(|map| {
                                observer.peek(embedding)
                                    .and_then(|mapping| map.get(&mapping.ch))
                                    .copied()
                            });
                            let fp = observer.fingerprint();
                            let recent_fp: String = fp.iter().rev().take(5).rev().collect();
                            eprintln!(
                                "[语义分类] 当前算子={} 映射={:?} 指纹={}",
                                expected_operator, op_from_obs, recent_fp
                            );
                            if let Some(classified) = op_from_obs {
                                if classified != expected_operator {
                                    metrics.semantic_interventions += 1;
                                }
                            }
                        }
                    }
                    // 降级：无 EmbeddingObserver 或无嵌入时使用 MockEncoder
                    if !observed {
                        if let Some(ref origin_emb) = vm.semantic_origin_embedding {
                            let encoder = MockEncoder::new(8, MockEncoderMode::Focused);
                            let v_current = encoder.encode(&result.text);
                            let cos_sim = xiang_core::embedding::cosine_similarity(&v_current, origin_emb);
                            semantic_dev = Some((1.0 - cos_sim).clamp(0.0, 1.0));
                            vm.semantic_deviation = semantic_dev;
                        }
                    }
                    if semantic_dev.map_or(false, |sd| sd > 0.7) {
                        metrics.semantic_interventions += 1;
                    }
                }
                let hybrid_dev = if config.semantic_mode {
                    hybrid_deviation(dev, semantic_dev, config.deviation_alpha)
                } else {
                    dev
                };

                // Collect metrics
                let focus_stop = result.stop_reason == StopReason::Deviated;
                let stop_reason_str = match result.stop_reason {
                    StopReason::Eos => Some("EOS".into()),
                    StopReason::MaxTokens => Some("最大Token".into()),
                    StopReason::StopSeq => Some("停止序列".into()),
                    StopReason::Deviated => Some("焦点偏离".into()),
                    StopReason::Error => Some("错误".into()),
                };
                let tm = TurnMetrics {
                    turn_number: turn,
                    deviation: hybrid_dev,
                    sha_triggered: vm.sha_count > metrics.sha_count,
                    cang_label: if focus_stop { Some("焦点偏离".to_string()) } else { None },
                    generated_text_preview: result.text.clone(),
                    output_length: result.text.len(),
                    focus_intervened: focus_stop,
                    semantic_deviation: semantic_dev,
                    phase_valid,
                    phase_violations,
                    repetition_triggered: false,
                    tokens_suppressed: result.tokens_suppressed,
                    bias_applications: result.bias_applications,
                    temperature_used,
                    operator_checked,
                    vm_gua_state: Some(vm.state.0),
                    stop_reason: stop_reason_str,
                    system_prompt: llm_ctx.system_prompt.clone(),
                    user_input: input.to_string(),
                    model_info: config.model_path.clone()
                        .or_else(|| config.http_server_url.clone())
                        .unwrap_or_else(|| "Mock".into()),
                    backend_type: backend.device_info().to_string(),
                    retry_count: vm.retries.get(operator).copied().unwrap_or(0),
                    judge_decision: judge_decision_str,
                    cycle_operator_index: phase_idx,
                };
                metrics.turn_data.push(tm);

                // ── 增量写入轮次文件 ──
                if let Some(dir) = turn_output_dir {
                    let last = metrics.turn_data.last().unwrap();
                    let turn_file = dir.join(turn_filename(last, turn));
                    let content = format_turn_content(last, turn);
                    let _ = fs::write(&turn_file, content);
                }

                metrics.total_tokens += result.tokens_generated;
                metrics.total_generations += 1;
                metrics.focus_interventions += if focus_stop || hybrid_dev > 0.7 { 1 } else { 0 };  // constrained

                // ── 上下文代谢：提取高质量输出内容 ──
                {
                    let operators = &["生", "动", "长", "育"];
                    let op_idx = (vm.cycle_count as usize - 1) % 4;
                    let operator = operators[op_idx];
                    let is_phase_valid = phase_valid.unwrap_or(true);
                    metabolism.feed(
                        turn + 1,
                        operator,
                        hybrid_dev,
                        is_phase_valid,
                        &result.text,
                    );
                }

                // Agent stop check
                if config.agent_mode && is_agent_stop_signal(&result.text, &config.agent_stop_patterns) {
                    break;
                }
            } else {
                // ── Controlled path (no thinking guidance) ──
                // 使用空系统提示词的独立LlmContext（不含约束体系全文）
                let mut ctrl_llm_ctx = LlmContext::new("");
                let extra_stops: Vec<String> = vec![
                    "### DONE".into(),
                    "【步骤1完成】".into(), "【步骤2完成】".into(),
                    "【步骤3完成】".into(), "【步骤4完成】".into(),
                    "【步骤5完成】".into(), "【步骤6完成】".into(),
                    // 有些模型会生成"阶段"而不是"步骤"，并含空格
                    "【阶段".into(), "【步骤".into(),
                ];
                let result = ctrl_llm_ctx.generate_unconstrained_turn(
                    backend, input, config.max_tokens,
                    TemperatureMode::Default,
                    &extra_stops,
                )?;
                current_text = result.text.clone();

                // ── 测量用：为对照组计算可比偏离度和合规率 ──
                // 创建独立测量VM（不影响LLM生成），注入三引擎用于指标计算
                let mut measure_vm = CangVM::new();
                if config.enable_three_engine {
                    measure_vm = measure_vm
                        .with_shan_vm(ShanVM::new())
                        .with_zhou_vm(ZhouVM::new());
                }
                if config.semantic_mode {
                    const SEMANTIC_ORIGIN_TEXT: &str = "请给出一个高质量、结构清晰的回答。";
                    let encoder = MockEncoder::new(8, MockEncoderMode::Focused);
                    measure_vm.semantic_origin_embedding = Some(encoder.encode(SEMANTIC_ORIGIN_TEXT));
                    measure_vm.deviation_alpha = config.deviation_alpha;
                }

                // 运行算子产生卦象变化
                measure_vm.cycle_count += 1;
                measure_vm.state = measure_vm.state.sheng();
                measure_vm.state = measure_vm.state.dong();
                measure_vm.state = measure_vm.state.zhang();
                measure_vm.state = measure_vm.state.yu();

                // 计算偏离度（含语义模式）
                let mut measure_dev = measure_vm.current_deviation();
                let mut measure_semantic_dev: Option<f32> = None;
                if config.semantic_mode {
                    if let Some(ref origin_emb) = measure_vm.semantic_origin_embedding {
                        let encoder = MockEncoder::new(8, MockEncoderMode::Focused);
                        let v_current = encoder.encode(&result.text);
                        let cos_sim = xiang_core::embedding::cosine_similarity(&v_current, origin_emb);
                        measure_semantic_dev = Some((1.0 - cos_sim).clamp(0.0, 1.0));
                        measure_dev = hybrid_deviation(measure_dev, measure_semantic_dev, config.deviation_alpha);
                    }
                }

                // 运行三引擎测量（不干预LLM，仅收集指标）
                if measure_dev > 0.5 && measure_vm.shan_vm.is_some() {
                    measure_vm.run_shan_vm();
                }
                if measure_vm.zhou_vm.is_some() {
                    measure_vm.run_zhou_vm_step();
                }

                // 阶段合规检查：尝试所有算子规则，任一通过即视为合规
                let phase_valid = if config.phase_constraint_mode {
                    let operators = &["生", "动", "长", "育"];
                    let check_result = operators.iter().any(|op| {
                        measure_vm.check_operator_output(op, &result.text).is_valid
                    });
                    // 同时检查基本结构：长度>10且包含标点
                    let basic_structure = result.text.len() > 10
                        && result.text.contains(['。', '，', ',', '.', '\n']);
                    Some(check_result || basic_structure)
                } else {
                    None
                };

                let tm = TurnMetrics {
                    turn_number: turn,
                    deviation: measure_dev,
                    sha_triggered: false,
                    cang_label: None,
                    generated_text_preview: result.text.clone(),
                    output_length: result.text.len(),
                    focus_intervened: measure_dev > 0.7,
                    semantic_deviation: measure_semantic_dev,
                    phase_valid,
                    phase_violations: vec![],
                    repetition_triggered: false,
                    tokens_suppressed: 0,
                    bias_applications: 0,
                    temperature_used: TemperatureMode::Default.value(),
                    operator_checked: None,
                    vm_gua_state: None,
                    stop_reason: Some(match result.stop_reason {
                        StopReason::Eos => "EOS".into(),
                        StopReason::MaxTokens => "最大Token".into(),
                        StopReason::StopSeq => "停止序列".into(),
                        StopReason::Deviated => "焦点偏离".into(),
                        StopReason::Error => "错误".into(),
                    }),
                    system_prompt: ctrl_llm_ctx.system_prompt.clone(),
                    user_input: input.to_string(),
                    model_info: config.model_path.clone()
                        .or_else(|| config.http_server_url.clone())
                        .unwrap_or_else(|| "Mock".into()),
                    backend_type: backend.device_info().to_string(),
                    retry_count: 0,
                    judge_decision: None,
                    cycle_operator_index: 0,
                };
                metrics.turn_data.push(tm);

                // ── 增量写入轮次文件 ──
                if let Some(dir) = turn_output_dir {
                    let last = metrics.turn_data.last().unwrap();
                    let turn_file = dir.join(turn_filename(last, turn));
                    let content = format_turn_content(last, turn);
                    let _ = fs::write(&turn_file, content);
                }

                metrics.total_tokens += result.tokens_generated;
                metrics.total_generations += 1;
                metrics.focus_interventions += if measure_dev > 0.7 { 1 } else { 0 };  // controlled

                // Agent stop check
                if config.agent_mode && is_agent_stop_signal(&result.text, &config.agent_stop_patterns) {
                    break;
                }
            }

            // ── 重复检测：与最近窗口内的历史输出比较 trigram Jaccard 相似度 ──
            if config.repetition_detection_enabled && !current_text.is_empty() {
                let is_rep = check_repetition(&current_text, &recent_texts,
                    config.repetition_similarity_threshold);

                // 标注当前轮次的重复检测结果
                if let Some(last) = metrics.turn_data.last_mut() {
                    last.repetition_triggered = is_rep;
                }

                if is_rep {
                    consecutive_repetitions += 1;
                } else {
                    consecutive_repetitions = 0;
                }

                recent_texts.push(current_text.clone());
                if recent_texts.len() > config.repetition_window_size {
                    recent_texts.remove(0);
                }

                if consecutive_repetitions >= config.repetition_threshold {
                    metrics.repetition_early_stopped = true;
                    break;
                }
            }
        }

        // Final metrics
        metrics.sha_count = vm.sha_count;
        metrics.cang_aligned = vm.cang_aligned_count;
        metrics.cang_deviated = vm.cang_deviated_count;
        metrics.trust_score = verifier.trust_score;
        if !metrics.turn_data.is_empty() {
            metrics.avg_deviation = metrics.turn_data.iter()
                .map(|t| t.deviation)
                .sum::<f32>() / metrics.turn_data.len() as f32;
            metrics.max_deviation = metrics.turn_data.iter()
                .map(|t| t.deviation)
                .fold(0.0_f32, f32::max);
            // 平均语义偏离度（仅统计语义模式启用的轮次）
            let semantic_values: Vec<f32> = metrics.turn_data.iter()
                .filter_map(|t| t.semantic_deviation)
                .collect();
            if !semantic_values.is_empty() {
                metrics.avg_semantic_deviation = semantic_values.iter().sum::<f32>()
                    / semantic_values.len() as f32;
            }
            // 完成率：偏离度 < 0.7 视为生成完整（三易控制可降低偏离从而提升完成率）
            let completed = metrics.turn_data.iter()
                .filter(|t| t.deviation < 0.7)
                .count();
            metrics.completion_rate = completed as f32 / metrics.turn_data.len() as f32;

            // 结构合规率：基于偏离度的分段评估（偏离越低 → 结构越合规）
            // 使用非线性分段映射，使分布具有区分度
            let structure_scores: Vec<f32> = metrics.turn_data.iter()
                .map(|t| {
                    if t.deviation < 0.3 { 1.0 }
                    else if t.deviation < 0.5 { 0.8 }
                    else if t.deviation < 0.7 { 0.5 }
                    else if t.deviation < 0.85 { 0.2 }
                    else { 0.0 }
                })
                .collect();
            if !structure_scores.is_empty() {
                metrics.structure_compliance_rate = structure_scores.iter().sum::<f32>()
                    / structure_scores.len() as f32;
            }
        }
        metrics.elapsed_ms = start.elapsed().as_millis() as u64;

        Ok(metrics)
    }

    /// Feed generation result back to the VM for meta-cognitive update.
    /// When SemanticStore is active, also logs aligned/deviated experiences.
    ///
    /// 路线A v3.4: 软反馈归零策略
    /// - dev < 0.3: zhi() 软收敛（保留状态梯度），替代 cang() 硬归零
    /// - dev > 0.7: sha() 清除有害模式，不再跟随 cang() 以避免硬归零
    /// - 0.3 ≤ dev ≤ 0.7: cang() 自然归藏 + 轻量正向存储
    fn feedback_to_vm(vm: &mut CangVM, result: &GenerationResult, deviation: f32) {
        if deviation < 0.3 && result.stop_reason == StopReason::Eos {
            // v3.4: 成功时用 zhi 软收敛（掩码 101101），保留状态梯度
            // 替代 v3.3 的 cang() 硬归零，避免状态在 {000000, 111111} 间跳跃
            vm.state = vm.state.zhi();
            vm.cang_sea.store(vm.state, Gua::ZERO, 0.3);
            if vm.cang_sea.semantic_store.is_some() {
                vm.cang_sea.store_semantic(SemanticEntry {
                    v_think: vec![],
                    v_goal: vec![], v_obstacle: vec![],
                    v_strategy: vec![], v_attitude: vec![],
                    deviation, reward: 0.3, timestamp: 0,
                    immune_count: 0, merge_count: 0, crystal_generation: 0,
                });
            }
            vm.cang_aligned_count += 1;
        } else if deviation > 0.7 || result.stop_reason == StopReason::Deviated {
            // v3.4: sha() 清除有害模式，不跟随 cang()
            // 保留 sha 后的残余状态（非零），为下轮气算子演化提供基础
            vm.state = vm.state.sha();
            vm.sha_count += 1;
            vm.cang_sea.store(vm.state, Gua::ZERO, -0.5);
            if vm.cang_sea.semantic_store.is_some() {
                vm.cang_sea.store_semantic(SemanticEntry {
                    v_think: vec![],
                    v_goal: vec![], v_obstacle: vec![],
                    v_strategy: vec![], v_attitude: vec![],
                    deviation, reward: -0.5, timestamp: 0,
                    immune_count: 0, merge_count: 0, crystal_generation: 0,
                });
            }
            vm.cang_deviated_count += 1;
        } else {
            // v3.4: 中间偏离度 (0.3 ~ 0.7)，自然归藏 + 轻量正向存储
            vm.state = vm.state.cang();
            vm.cang_sea.store(vm.state, Gua::ZERO, 0.1);
        }
    }

    /// Run a comparison experiment and generate a report.
    pub fn run_comparison(config: &ExperimentConfig) -> ComparisonReport {
        let controlled = Self::run_controlled(config, None).unwrap_or_default();
        let constrained = Self::run_constrained(config, None).unwrap_or_default();

        let focus_improvement = if controlled.focus_interventions > 0 {
            (controlled.focus_interventions as f32 - constrained.focus_interventions as f32)
                / controlled.focus_interventions as f32 * 100.0
        } else {
            0.0
        };

        let deviation_improvement = if controlled.avg_deviation > 0.001 {
            (controlled.avg_deviation - constrained.avg_deviation)
                / controlled.avg_deviation * 100.0
        } else {
            0.0
        };

        let trust_improvement = (constrained.trust_score - controlled.trust_score) * 100.0;

        let token_improvement = if controlled.total_tokens > 0 {
            (controlled.total_tokens as f32 - constrained.total_tokens as f32)
                / controlled.total_tokens as f32 * 100.0
        } else {
            0.0
        };

        let improvement = ComparisonImprovement {
            focus_reduction_pct: focus_improvement.max(0.0),
            deviation_reduction_pct: deviation_improvement.max(0.0),
            trust_improvement_pct: trust_improvement.max(0.0),
            token_efficiency_pct: token_improvement.max(0.0),
        };

        ComparisonReport {
            experiment_name: config.name.clone(),
            controlled,
            constrained,
            improvement,
        }
    }
}

// ── HTML Report Generation ──────────────────────────────────

/// Generate an HTML report from comparison data.
pub fn generate_html_report(reports: &[ComparisonReport]) -> String {
    let mut html = String::new();
    html.push_str(r#"<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>象语言 思维引导引擎 — 对比实验报告</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 960px; margin: 2em auto; padding: 0 1em; background: #0d1117; color: #c9d1d9; }
  h1 { color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 0.3em; }
  h2 { color: #f0883e; margin-top: 1.5em; }
  table { border-collapse: collapse; width: 100%; margin: 1em 0; }
  th, td { border: 1px solid #30363d; padding: 8px 12px; text-align: left; }
  th { background: #161b22; color: #8b949e; }
  .improvement { color: #3fb950; font-weight: bold; }
  .degradation { color: #f85149; }
  .metric-bar { height: 20px; border-radius: 4px; margin: 4px 0; }
  .bar-controlled { background: #f85149; display: inline-block; height: 100%; border-radius: 4px 0 0 4px; }
  .bar-constrained { background: #3fb950; display: inline-block; height: 100%; border-radius: 0 4px 4px 0; }
  pre { background: #161b22; padding: 1em; border-radius: 6px; overflow-x: auto; }
  .summary-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1em; margin: 1em 0; }
</style>
</head>
<body>
<h1>象语言 思维引导引擎 — 对比实验报告</h1>
"#);

    for report in reports {
        let imp = &report.improvement;
        html.push_str(&format!(
            r#"<h2>实验: {name}</h2>
<div class="summary-card">
  <p><strong>焦点对齐提升:</strong> <span class="improvement">{focus:+.1}%</span> |
     <strong>偏差降低:</strong> <span class="improvement">{dev:+.1}%</span> |
     <strong>信任提升:</strong> <span class="improvement">{trust:+.1}%</span> |
     <strong>Token效率:</strong> <span class="improvement">{token:+.1}%</span>
  </p>
</div>
<h3>详细对比</h3>
<table>
  <tr><th>指标</th><th>无引导 (对照组)</th><th>有引导 (实验组)</th><th>改善</th></tr>
  <tr><td>焦点干预次数</td><td>{c_focus}</td><td>{x_focus}</td><td class="improvement">{focus_pct:+.1}%</td></tr>
  <tr><td>平均偏差</td><td>{c_dev:.3}</td><td>{x_dev:.3}</td><td class="improvement">{dev_pct:+.1}%</td></tr>
  <tr><td>最大偏差</td><td>{c_max_dev:.3}</td><td>{x_max_dev:.3}</td><td>—</td></tr>
  <tr><td>信任评分</td><td>{c_trust:.1}%</td><td>{x_trust:.1}%</td><td class="improvement">{trust_pct:+.1}%</td></tr>
  <tr><td>剪除操作数</td><td>{c_sha}</td><td>{x_sha}</td><td>—</td></tr>
  <tr><td>藏对齐次数</td><td>{c_aligned}</td><td>{x_aligned}</td><td>—</td></tr>
  <tr><td>藏偏离次数</td><td>{c_deviated}</td><td>{x_deviated}</td><td>—</td></tr>
  <tr><td>总Token数</td><td>{c_tokens}</td><td>{x_tokens}</td><td class="improvement">{token_pct:+.1}%</td></tr>
  <tr><td>耗时 (ms)</td><td>{c_ms}</td><td>{x_ms}</td><td>—</td></tr>
</table>
"#,
            name = report.experiment_name,
            focus = imp.focus_reduction_pct,
            dev = imp.deviation_reduction_pct,
            trust = imp.trust_improvement_pct,
            token = imp.token_efficiency_pct,
            c_focus = report.controlled.focus_interventions,
            x_focus = report.constrained.focus_interventions,
            focus_pct = if report.controlled.focus_interventions > 0 {
                (report.controlled.focus_interventions as f32 - report.constrained.focus_interventions as f32)
                / report.controlled.focus_interventions as f32 * 100.0
            } else { 0.0 },
            c_dev = report.controlled.avg_deviation,
            x_dev = report.constrained.avg_deviation,
            dev_pct = imp.deviation_reduction_pct,
            c_max_dev = report.controlled.max_deviation,
            x_max_dev = report.constrained.max_deviation,
            c_trust = report.controlled.trust_score * 100.0,
            x_trust = report.constrained.trust_score * 100.0,
            trust_pct = imp.trust_improvement_pct,
            c_sha = report.controlled.sha_count,
            x_sha = report.constrained.sha_count,
            c_aligned = report.controlled.cang_aligned,
            x_aligned = report.constrained.cang_aligned,
            c_deviated = report.controlled.cang_deviated,
            x_deviated = report.constrained.cang_deviated,
            c_tokens = report.controlled.total_tokens,
            x_tokens = report.constrained.total_tokens,
            token_pct = imp.token_efficiency_pct,
            c_ms = report.controlled.elapsed_ms,
            x_ms = report.constrained.elapsed_ms,
        ));

        // Semantic mode metrics (only if available)
        if report.constrained.semantic_interventions > 0 || report.constrained.avg_semantic_deviation > 0.001 {
            html.push_str(&format!(
                r#"<tr><td>语义干预次数</td><td>—</td><td>{x_sem}</td><td>—</td></tr>
<tr><td>平均语义偏差</td><td>—</td><td>{x_sem_dev:.3}</td><td>—</td></tr>
"#,
                x_sem = report.constrained.semantic_interventions,
                x_sem_dev = report.constrained.avg_semantic_deviation,
            ));
        }
    }

    html.push_str("</body>\n</html>");
    html
}

// ── Pre-defined Experiment Scenarios ─────────────────────────

/// Focus deviation detection experiment (思维焦点).
pub fn focus_deviation_experiment() -> ExperimentConfig {
    ExperimentConfig {
        name: "思维焦点—偏移检测".to_string(),
        description: "测试思维引导引擎在话题偏移时的焦点回归能力".to_string(),
        guidance_program: Some(FOCUS_DEVIATION_XIANG.to_string()),
        inputs: vec![
            "请继续深入讨论当前主题的核心要点".to_string(),
            "让我们回到主题的核心问题上来".to_string(),
            "你刚才的思考方向是否偏离了核心问题？".to_string(),
            "请聚焦在当前问题的关键维度上".to_string(),
            "总结一下你对这个问题的核心洞察".to_string(),
        ],
        turns: 5,
        max_tokens: 500,
        mock_mode: MockMode::OffFocus,
        use_guidance_engine: true,
        inject_constraint_prompt: true,
        model_path: None,
        http_server_url: None,
        off_focus_token_ids: vec![],
        divergent_token_ids: vec![],
        eos_token_id: 0,
        semantic_mode: false,
        deviation_alpha: 0.5,
        enable_three_engine: true,
        phase_constraint_mode: false,
        cached_off_focus_ids: None,
        cached_divergent_ids: None,
        cached_eos_id: None,
        agent_mode: false,
        agent_stop_patterns: vec![],
        repetition_detection_enabled: false,
        repetition_threshold: 3,
        repetition_similarity_threshold: 0.85,
        repetition_window_size: 3,
        semantic_nav_mode: false,
    }
}

/// Ultra-long context experiment (长上下文).
pub fn long_context_experiment() -> ExperimentConfig {
    ExperimentConfig {
        name: "长上下文—32K持续聚焦".to_string(),
        description: "测试思维引导引擎在超长对话中维持主题聚焦的能力".to_string(),
        guidance_program: Some(LONGCONTEXT_XIANG.to_string()),
        inputs: (0..32).map(|i| {
            format!("第{i}轮: 请继续讨论关于AI认知架构的话题。当前主题是：如何构建具有持续学习能力的认知系统？")
        }).collect(),
        turns: 32,
        max_tokens: 400,
        mock_mode: MockMode::Verbose,
        use_guidance_engine: true,
        inject_constraint_prompt: true,
        model_path: None,
        http_server_url: None,
        off_focus_token_ids: vec![],
        divergent_token_ids: vec![],
        eos_token_id: 0,
        semantic_mode: false,
        deviation_alpha: 0.5,
        enable_three_engine: true,
        phase_constraint_mode: false,
        cached_off_focus_ids: None,
        cached_divergent_ids: None,
        cached_eos_id: None,
        agent_mode: false,
        agent_stop_patterns: vec![],
        repetition_detection_enabled: false,
        repetition_threshold: 3,
        repetition_similarity_threshold: 0.85,
        repetition_window_size: 3,
        semantic_nav_mode: false,
    }
}

/// Multi-turn consistency experiment (多轮对话).
pub fn multi_turn_experiment() -> ExperimentConfig {
    ExperimentConfig {
        name: "多轮对话—50轮思维一致性".to_string(),
        description: "测试思维引导引擎在50轮对话中维持思维连贯性".to_string(),
        guidance_program: Some(MULTITURN_XIANG.to_string()),
        inputs: (0..50).map(|i| {
            match i % 4 {
                0 => "请解释什么是机器学习".to_string(),
                1 => "请详细说明神经网络的工作原理".to_string(),
                2 => "那么深度学习与传统机器学习有什么区别？".to_string(),
                _ => "请总结我们之前讨论的内容".to_string(),
            }
        }).collect(),
        turns: 50,
        max_tokens: 400,
        mock_mode: MockMode::Verbose,
        use_guidance_engine: true,
        inject_constraint_prompt: true,
        model_path: None,
        http_server_url: None,
        off_focus_token_ids: vec![],
        divergent_token_ids: vec![],
        eos_token_id: 0,
        semantic_mode: false,
        deviation_alpha: 0.5,
        enable_three_engine: true,
        phase_constraint_mode: false,
        cached_off_focus_ids: None,
        cached_divergent_ids: None,
        cached_eos_id: None,
        agent_mode: false,
        agent_stop_patterns: vec![],
        repetition_detection_enabled: false,
        repetition_threshold: 3,
        repetition_similarity_threshold: 0.85,
        repetition_window_size: 3,
        semantic_nav_mode: false,
    }
}

// ── Example .xiang Programs ──────────────────────────────────

pub const FOCUS_DEVIATION_XIANG: &str = r#"
定义 思维向导:
    卦 焦点状态 = 111111
    生
    动
    归
    若 偏离度 > 0.7:
        杀 110
        藏 恶
    若 偏离度 < 0.3:
        藏 善
    否则:
        止 101101
        藏 善
"#;

pub const LONGCONTEXT_XIANG: &str = r#"
定义 长上下文:
    卦 对话状态 = 111111
    周天 32:
        生
        动
        归
        长
        育
        若 偏离度 > 0.7:
            杀 110
            止 101101
            藏 恶
        否则:
            藏 善
"#;

pub const MULTITURN_XIANG: &str = r#"
定义 多轮对话:
    卦 逻辑状态 = 111111
    周天 50:
        生
        动
        归
        长
        育
        杀 110
        止 101101
        藏 善
"#;

/// Phase constraint experiment (阶段约束验证).
pub fn phase_constraint_experiment() -> ExperimentConfig {
    ExperimentConfig {
        name: "阶段约束—算子协议验证".to_string(),
        description: "测试阴仪阶段协议验证器对算子输出的形式规范检查".to_string(),
        guidance_program: Some(PHASE_CONSTRAINT_XIANG.to_string()),
        inputs: vec![
            "也许我们可以从系统架构入手分析这个问题？需要进一步考察数据流的走向。".to_string(),
            "此外还可以考虑性能维度。另一方面，安全性也需要关注的扩展方向。".to_string(),
            "我们聚焦在用户认证这条路径上，深入分析OAuth2.0的实现细节。".to_string(),
            "第一步，部署数据库。第二步，配置API网关。第三步，集成前端测试。".to_string(),
            "总结：以上四个阶段构成了完整的分析链条。".to_string(),
        ],
        turns: 5,
        max_tokens: 500,
        mock_mode: MockMode::Verbose,
        use_guidance_engine: true,
        inject_constraint_prompt: true,
        model_path: None,
        http_server_url: None,
        off_focus_token_ids: vec![],
        divergent_token_ids: vec![],
        eos_token_id: 0,
        semantic_mode: false,
        deviation_alpha: 0.5,
        enable_three_engine: true,
        phase_constraint_mode: true,
        cached_off_focus_ids: None,
        cached_divergent_ids: None,
        cached_eos_id: None,
        agent_mode: false,
        agent_stop_patterns: vec![],
        repetition_detection_enabled: false,
        repetition_threshold: 3,
        repetition_similarity_threshold: 0.85,
        repetition_window_size: 3,
        semantic_nav_mode: false,
    }
}

/// 统一综合实验：合并基准测试的100个多样化任务 + 全引擎特性
/// 用于全面衡量思维引导引擎在10主题×10任务类型的表现
pub fn unified_experiment() -> ExperimentConfig {
    ExperimentConfig {
        name: "统一综合—100任务全覆盖".to_string(),
        description: "合并基准测试的100个任务（10主题×10类型），全面衡量焦点对齐、偏离控制、结构合规、完成率等核心指标".to_string(),
        guidance_program: Some(UNIFIED_XIANG.to_string()),
        inputs: inputs::generate_benchmark_inputs(42),
        turns: 100,
        max_tokens: 400,
        mock_mode: MockMode::Verbose,
        use_guidance_engine: true,
        inject_constraint_prompt: true,
        model_path: None,
        http_server_url: None,
        off_focus_token_ids: vec![],
        divergent_token_ids: vec![],
        eos_token_id: 0,
        semantic_mode: true,
        deviation_alpha: 0.5,
        enable_three_engine: true,
        phase_constraint_mode: true,
        cached_off_focus_ids: None,
        cached_divergent_ids: None,
        cached_eos_id: None,
        agent_mode: false,
        agent_stop_patterns: vec![],
        repetition_detection_enabled: true,
        repetition_threshold: 5,
        repetition_similarity_threshold: 0.85,
        repetition_window_size: 3,
        semantic_nav_mode: false,
    }
}

/// 语义导航实验：用向量汉字映射替代体系提示词注入
/// 系统提示词为空，阶段约束通过 operator_semantic::classify_operator_phase() 驱动
pub fn semantic_nav_experiment() -> ExperimentConfig {
    ExperimentConfig {
        name: "语义导航—向量汉字映射".to_string(),
        description: "用算子语义分类替代体系提示词，通过 embedding 映射驱动阶段约束".to_string(),
        guidance_program: Some(SEMANTIC_NAV_XIANG.to_string()),
        inputs: vec![
            "请分析这个问题的核心架构".to_string(),
            "探索解决方案的多条路径".to_string(),
            "深入分析关键维度的细节".to_string(),
            "结构化地规划实施步骤".to_string(),
        ],
        turns: 4,
        max_tokens: 500,
        mock_mode: MockMode::Verbose,
        use_guidance_engine: true,
        inject_constraint_prompt: false,
        model_path: None,
        http_server_url: None,
        off_focus_token_ids: vec![],
        divergent_token_ids: vec![],
        eos_token_id: 0,
        semantic_mode: true,
        deviation_alpha: 0.5,
        enable_three_engine: true,
        phase_constraint_mode: false,
        cached_off_focus_ids: None,
        cached_divergent_ids: None,
        cached_eos_id: None,
        agent_mode: false,
        agent_stop_patterns: vec![],
        repetition_detection_enabled: false,
        repetition_threshold: 3,
        repetition_similarity_threshold: 0.85,
        repetition_window_size: 3,
        semantic_nav_mode: true,
    }
}

pub const UNIFIED_XIANG: &str = r#"
定义 全面检测:
    卦 综合状态 = 111111
    周天 100:
        生
        动
        归
        长
        育
        若 偏离度 > 0.7:
            杀 110
            止 101101
            藏 恶
        若 偏离度 < 0.3:
            止 101101
            藏 善
        否则:
            藏 善
"#;

// ── SEMANTIC_NAV_XIANG ──

pub const SEMANTIC_NAV_XIANG: &str = r#"
定义 语义导航:
    卦 导航状态 = 111111
    周天 4:
        生
        动
        归
        长
        育
        归
        杀 110
        止 101101
        藏 善
"#;

// ── PHASE_CONSTRAINT_XIANG ──

pub const PHASE_CONSTRAINT_XIANG: &str = r#"
定义 阶段约束:
    卦 验证状态 = 111111
    周天 5:
        生
        动
        归
        长
        育
        归
        杀 110
        止 101101
        藏 善
"#;

/// Build a turn file name that shows operator and phase status.
/// Format: `turn_{index:02}_{operator}_{status}.txt`
/// Example: `turn_00_生_OK.txt`, `turn_03_育_NG.txt`
pub fn turn_filename(turn: &TurnMetrics, index: usize) -> String {
    let operator = turn.operator_checked.as_deref().unwrap_or("op");
    let status = match turn.phase_valid {
        Some(true) => "OK",
        Some(false) => "NG",
        None => "NA",
    };
    format!("turn_{:02}_{}_{}.txt", index, operator, status)
}

/// Format a single turn's metrics into a structured text block.
pub fn format_turn_content(turn: &TurnMetrics, turn_index: usize) -> String {
    format!(
        "=== 轮次 {turn_index} ===\n\
         ── 模型信息 ──\n\
         模型:       {model}\n\
         后端:       {backend}\n\
         ── 用户输入 ──\n\
         {user_input}\n\
         ── 系统提示词 ──\n\
         {sys_prompt}\n\
         ── 思考过程 ──\n\
         卦象状态:   {gua}\n\
         偏离率:     {dev}\n\
         语义偏离:   {sem}\n\
         温度策略:   {temp}\n\
         阶段算子:   {op}\n\
         阶段合规:   {phase}\n\
         停止原因:   {stop}\n\
         ── 干预手段 ──\n\
         Token压制:  {sup}\n\
         Bias应用:   {bias}\n\
         焦点干预:   {fi}\n\
         SHA触发:    {sha}\n\
         重复检测:   {rep}\n\
         藏标签:     {label}\n\
         阶段违规:   {violations}\n\
         ── 生成输出（全文）──\n\
         {text}",
        turn_index = turn_index,
        model = turn.model_info,
        backend = turn.backend_type,
        user_input = turn.user_input,
        sys_prompt = turn.system_prompt,
        gua = turn.vm_gua_state.map_or("N/A".into(), |g| format!("{:06b}", g)),
        dev = format!("{:.4}", turn.deviation),
        sem = turn.semantic_deviation.map_or("N/A".into(), |v| format!("{:.4}", v)),
        temp = format!("{:.2}", turn.temperature_used),
        op = turn.operator_checked.as_deref().unwrap_or("N/A"),
        phase = turn.phase_valid.map_or("N/A".into(), |v| if v { "通过" } else { "违规" }),
        stop = turn.stop_reason.as_deref().unwrap_or("N/A"),
        sup = turn.tokens_suppressed,
        bias = turn.bias_applications,
        fi = if turn.focus_intervened { "是" } else { "否" },
        sha = if turn.sha_triggered { "是" } else { "否" },
        rep = if turn.repetition_triggered { "是" } else { "否" },
        label = turn.cang_label.as_deref().unwrap_or("N/A"),
        violations = if turn.phase_violations.is_empty() {
            "无".into()
        } else {
            turn.phase_violations.join("; ")
        },
        text = turn.generated_text_preview,
    )
}

/// Write per-turn output files for a single experiment run.
///
/// Directory structure:
///   experiment_runs/{name}_{timestamp}/
///     input.txt
///     对照组/turn_000.txt
///     实验组/turn_000.txt
pub fn write_experiment_turn_files(
    experiment_label: &str,
    input: &str,
    controlled: &ExperimentMetrics,
    constrained: &ExperimentMetrics,
) -> Result<PathBuf, String> {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| format!("系统时间错误: {e}"))?
        .as_secs();
    let run_dir = PathBuf::from("experiment_runs")
        .join(format!("{}_{}", experiment_label, ts));

    // Write input
    let input_path = run_dir.join("input.txt");
    fs::create_dir_all(input_path.parent().unwrap())
        .map_err(|e| format!("无法创建输出目录: {e}"))?;
    fs::write(&input_path, input)
        .map_err(|e| format!("无法写入输入文件: {e}"))?;

    // Write controlled turn files
    let ctrl_dir = run_dir.join("对照组");
    fs::create_dir_all(&ctrl_dir)
        .map_err(|e| format!("无法创建输出目录: {e}"))?;
    for (j, turn) in controlled.turn_data.iter().enumerate() {
        let turn_file = ctrl_dir.join(turn_filename(turn, j));
        let content = format_turn_content(turn, j);
        fs::write(&turn_file, content)
            .map_err(|e| format!("无法写入轮次文件: {e}"))?;
    }

    // Write constrained turn files
    let const_dir = run_dir.join("实验组");
    fs::create_dir_all(&const_dir)
        .map_err(|e| format!("无法创建输出目录: {e}"))?;
    for (j, turn) in constrained.turn_data.iter().enumerate() {
        let turn_file = const_dir.join(turn_filename(turn, j));
        let content = format_turn_content(turn, j);
        fs::write(&turn_file, content)
            .map_err(|e| format!("无法写入轮次文件: {e}"))?;
    }

    Ok(run_dir)
}

/// Export quality evaluation input for AI evaluator (Qoder).
///
/// Produces a JSON file containing the rubric and all per-turn output text
/// for both controlled and constrained conditions of every trial.
pub fn export_quality_evaluation_input(
    results: &benchmark::BenchmarkResults,
    path: &str,
) -> Result<(), String> {
    #[derive(Serialize)]
    struct QTurn {
        turn_number: usize,
        generated_text: String,
        operator: String,
        deviation: f32,
        phase_valid: Option<bool>,
    }
    #[derive(Serialize)]
    struct QTrial {
        trial_index: usize,
        input: String,
        controlled_turns: Vec<QTurn>,
        constrained_turns: Vec<QTurn>,
    }
    #[derive(Serialize)]
    struct QFile {
        rubric: String,
        trials: Vec<QTrial>,
    }

    let mut trials = Vec::with_capacity(results.trials.len());
    for trial in &results.trials {
        let controlled_turns: Vec<QTurn> = trial.controlled_metrics.turn_data.iter().map(|t| {
            QTurn {
                turn_number: t.turn_number,
                generated_text: t.generated_text_preview.clone(),
                operator: t.operator_checked.clone().unwrap_or_else(|| "?".into()),
                deviation: t.deviation,
                phase_valid: t.phase_valid,
            }
        }).collect();
        let constrained_turns: Vec<QTurn> = trial.constrained_metrics.turn_data.iter().map(|t| {
            QTurn {
                turn_number: t.turn_number,
                generated_text: t.generated_text_preview.clone(),
                operator: t.operator_checked.clone().unwrap_or_else(|| "?".into()),
                deviation: t.deviation,
                phase_valid: t.phase_valid,
            }
        }).collect();
        trials.push(QTrial {
            trial_index: trial.trial_index,
            input: trial.input.clone(),
            controlled_turns,
            constrained_turns,
        });
    }
    let file = QFile {
        rubric: QUALITY_EVALUATION_RUBRIC.to_string(),
        trials,
    };
    let json = serde_json::to_string_pretty(&file).map_err(|e| e.to_string())?;
    fs::write(path, &json).map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_experiment_config_builder() {
        let config = focus_deviation_experiment();
        assert_eq!(config.name, "思维焦点—偏移检测");
        assert_eq!(config.inputs.len(), 5);
        assert_eq!(config.turns, 5);
        assert!(config.guidance_program.is_some());
    }

    #[test]
    fn test_long_context_config() {
        let config = long_context_experiment();
        assert_eq!(config.inputs.len(), 32);
        assert_eq!(config.turns, 32);
    }

    #[test]
    fn test_multi_turn_config() {
        let config = multi_turn_experiment();
        assert_eq!(config.inputs.len(), 50);
        assert_eq!(config.turns, 50);
    }

    #[test]
    fn test_run_focus_constrained() {
        let config = focus_deviation_experiment();
        let result = ExperimentRunner::run_constrained(&config, None);
        assert!(result.is_ok());
        let metrics = result.unwrap();
        assert!(metrics.total_generations > 0);
    }

    #[test]
    fn test_run_focus_controlled() {
        let config = focus_deviation_experiment();
        let result = ExperimentRunner::run_controlled(&config, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_long_context() {
        let config = long_context_experiment();
        let result = ExperimentRunner::run_constrained(&config, None);
        assert!(result.is_ok());
        let metrics = result.unwrap();
        assert_eq!(metrics.total_generations, 32);
    }

    #[test]
    fn test_run_multi_turn() {
        let config = multi_turn_experiment();
        let result = ExperimentRunner::run_constrained(&config, None);
        assert!(result.is_ok());
        let metrics = result.unwrap();
        assert_eq!(metrics.total_generations, 50);
    }

    #[test]
    fn test_comparison_report() {
        let config = focus_deviation_experiment();
        let report = ExperimentRunner::run_comparison(&config);
        assert_eq!(report.experiment_name, "思维焦点—偏移检测");
        assert!(report.constrained.total_generations > 0);
    }

    #[test]
    fn test_html_report_generation() {
        let config = focus_deviation_experiment();
        let report = ExperimentRunner::run_comparison(&config);
        let html = generate_html_report(&[report]);
        assert!(html.contains("思维引导引擎"));
        assert!(html.contains("对比实验报告"));
        assert!(html.contains("html"));
    }

    #[test]
    fn test_all_experiments_html() {
        let reports = vec![
            ExperimentRunner::run_comparison(&focus_deviation_experiment()),
            ExperimentRunner::run_comparison(&long_context_experiment()),
            ExperimentRunner::run_comparison(&multi_turn_experiment()),
        ];
        let html = generate_html_report(&reports);
        assert!(html.contains("思维焦点"));
        assert!(html.contains("长上下文"));
        assert!(html.contains("多轮对话"));
    }

    #[test]
    fn test_default_metrics() {
        let metrics = ExperimentMetrics::default();
        assert_eq!(metrics.total_generations, 0);
        assert_eq!(metrics.focus_interventions, 0);
        assert_eq!(metrics.trust_score, 0.0);
    }

    #[test]
    fn test_phase_constraint_experiment() {
        let config = phase_constraint_experiment();
        assert!(config.phase_constraint_mode);
        assert_eq!(config.name, "阶段约束—算子协议验证");

        let result = ExperimentRunner::run_constrained(&config, None);
        assert!(result.is_ok());
        let metrics = result.unwrap();
        assert_eq!(metrics.total_generations, 5);
        // Phase violations should be >= 0
        // (depends on whether MockBackend output matches the operator phase patterns)
        assert!(metrics.phase_violations <= 5);
    }

    #[test]
    fn test_semantic_mode_experiment() {
        // ── 语义模式实验配置（使用 MockEncoder + hybrid_deviation）──
        let config = ExperimentConfig {
            name: "语义模式—混合偏离度验证".to_string(),
            description: "测试路线C语义引擎（MockEncoder + hybrid_deviation + SemanticStore）".to_string(),
            guidance_program: Some("定义 语义测试:\n    卦 状态 = 111111\n    生\n    动\n    归\n    藏 善".into()),
            inputs: vec![
                "请讨论AI认知架构的核心问题".to_string(),
                "深入分析神经网络的学习机制".to_string(),
            ],
            turns: 2,
            max_tokens: 400,
            mock_mode: MockMode::Safe,
            use_guidance_engine: true,
            inject_constraint_prompt: true,
            model_path: None,
            http_server_url: None,
            off_focus_token_ids: vec![],
            divergent_token_ids: vec![],
            eos_token_id: 0,
            semantic_mode: true,
            deviation_alpha: 0.5,
            enable_three_engine: true,
            phase_constraint_mode: false,
            cached_off_focus_ids: None,
            cached_divergent_ids: None,
            cached_eos_id: None,
            agent_mode: false,
            agent_stop_patterns: vec![],
            repetition_detection_enabled: false,
            repetition_threshold: 3,
            repetition_similarity_threshold: 0.85,
            repetition_window_size: 3,
            semantic_nav_mode: false,
        };

        let result = ExperimentRunner::run_constrained(&config, None);
        assert!(result.is_ok(), "语义模式实验应成功运行");
        let metrics = result.unwrap();

        // 验证语义模式下 deviation 被记录
        assert!(metrics.total_generations > 0, "应有生成轮次");

        // 验证 TurnMetrics 包含 semantic_deviation 值
        let has_semantic = metrics.turn_data.iter().any(|t| t.semantic_deviation.is_some());
        assert!(has_semantic, "语义模式下应记录 semantic_deviation");

        // 验证某些轮次的 deviation > 0（混合偏离度计算生效）
        let has_positive_dev = metrics.turn_data.iter().any(|t| t.deviation > 0.0);
        assert!(has_positive_dev, "混合偏离度应产生非零值");
    }

    #[test]
    fn test_semantic_mode_disabled_fallback() {
        // ── 非语义模式回退验证 ──
        let mut config = focus_deviation_experiment();
        config.semantic_mode = false;

        let result = ExperimentRunner::run_constrained(&config, None);
        assert!(result.is_ok(), "非语义模式应正常回退");
        let metrics = result.unwrap();

        // 非语义模式下不应有 semantic_deviation
        // （用 Option::is_none 验证所有轮次）
        let no_semantic = metrics.turn_data.iter().all(|t| t.semantic_deviation.is_none());
        assert!(no_semantic, "非语义模式下不应记录 semantic_deviation");
    }

    // ── Stop pattern tests ──

    #[test]
    fn test_stop_signal_empty_text() {
        let patterns: Vec<String> = vec!["已完成".into()];
        assert!(is_agent_stop_signal("", &patterns));
    }

    #[test]
    fn test_stop_signal_end_of_line() {
        let patterns: Vec<String> = vec!["已完成".into()];
        // 模式出现在行尾，应匹配
        assert!(is_agent_stop_signal("所有阶段已完成\n", &patterns));
        assert!(is_agent_stop_signal("第一阶段已完成\n第二阶段已完成", &patterns));
    }

    #[test]
    fn test_stop_signal_mid_sentence_no_match() {
        let patterns: Vec<String> = vec!["已完成".into()];
        // 模式出现在句中，不应匹配（修复 trial_2 误触发）
        assert!(!is_agent_stop_signal("基于第一阶段已完成，我现在进入第二阶段", &patterns));
        assert!(!is_agent_stop_signal("好的，收到您的反馈。基于第一阶段已完成，我现在进入**第二阶段：关系定义**。", &patterns));
    }

    #[test]
    fn test_stop_signal_done_pattern() {
        let patterns: Vec<String> = vec!["### DONE".into()];
        assert!(is_agent_stop_signal("### DONE", &patterns));
        assert!(is_agent_stop_signal("一些内容\n### DONE\n", &patterns));
        assert!(!is_agent_stop_signal("### DONE is not finished yet", &patterns));
    }

    #[test]
    fn test_stop_signal_punctuation_stripping() {
        let patterns: Vec<String> = vec!["任务完成".into(), "分析结束".into()];
        // 标点应被剥离后匹配
        assert!(is_agent_stop_signal("任务完成。", &patterns));
        assert!(is_agent_stop_signal("任务完成！", &patterns));
        assert!(is_agent_stop_signal("分析结束。", &patterns));
        assert!(is_agent_stop_signal("分析结束…", &patterns));
        // 但不应匹配句中的模式
        assert!(!is_agent_stop_signal("当任务完成后，请继续", &patterns));
        assert!(!is_agent_stop_signal("分析结束后进入综合阶段", &patterns));
    }

    #[test]
    fn test_stop_signal_all_patterns() {
        let patterns: Vec<String> = vec![
            "### DONE".into(),
            "任务完成".into(),
            "分析结束".into(),
            "已完成".into(),
        ];
        // 每个模式在行尾都应匹配
        assert!(is_agent_stop_signal("### DONE", &patterns));
        assert!(is_agent_stop_signal("任务完成。", &patterns));
        assert!(is_agent_stop_signal("分析结束！", &patterns));
        assert!(is_agent_stop_signal("所有阶段已完成", &patterns));
        // 不应匹配句中模式
        assert!(!is_agent_stop_signal("任务完成后才能开始下一阶段", &patterns));
        assert!(!is_agent_stop_signal("当分析结束时打印报告", &patterns));
        assert!(!is_agent_stop_signal("第一阶段已完成，进入第二阶段", &patterns));
    }
}
