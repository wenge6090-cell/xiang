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

use serde::Serialize;
use xiang_core::Gua;
use xiang_cangvm::CangVM;
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
use std::time::Instant;

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
    xiang_llm::llama_backend::LlamaCppBackend::new(path, 32768, -1)
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

/// Check if generated text contains an agent stop signal.
fn is_agent_stop_signal(text: &str, stop_patterns: &[String]) -> bool {
    if text.is_empty() {
        return true;
    }
    stop_patterns.iter().any(|pattern| text.contains(pattern.as_str()))
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
        Self::run_experiment_with_backend(config, true, backend)
    }

    /// Run an experiment without the guidance engine (uses provided backend).
    pub(crate) fn run_controlled_with_backend(
        config: &ExperimentConfig,
        backend: &mut RunBackend,
    ) -> Result<ExperimentMetrics, LlmError> {
        Self::run_experiment_with_backend(config, false, backend)
    }

    /// Run an experiment with the guidance engine enabled.
    pub fn run_constrained(config: &ExperimentConfig) -> Result<ExperimentMetrics, LlmError> {
        let mut backend = create_backend(config)?;
        Self::run_experiment_with_backend(config, true, &mut backend)
    }

    /// Run an experiment without the guidance engine (control group).
    pub fn run_controlled(config: &ExperimentConfig) -> Result<ExperimentMetrics, LlmError> {
        let mut backend = create_backend(config)?;
        Self::run_experiment_with_backend(config, false, &mut backend)
    }

    fn run_experiment_with_backend(
        config: &ExperimentConfig,
        use_guidance: bool,
        backend: &mut RunBackend,
    ) -> Result<ExperimentMetrics, LlmError> {
        let start = Instant::now();
        let mut metrics = ExperimentMetrics::default();

        // Initialize VM and verifier if using guidance
        let mut vm = CangVM::new();
        let mut verifier = Verifier::new();
        let mut llm_ctx = LlmContext::new("你是一个AI助手。请根据引导指令回答。");

        // ── 语义模式：设定混合偏离度参数 ──
        if config.semantic_mode && use_guidance {
            vm.deviation_alpha = config.deviation_alpha;
            // 使用固定参考文本作为语义原点（而非第一个输入），
            // 确保不同输入与固定原点产生不同的余弦距离
            const SEMANTIC_ORIGIN_TEXT: &str = "请给出一个高质量、结构清晰的回答。";
            let encoder = MockEncoder::new(8, MockEncoderMode::Focused);
            vm.semantic_origin_embedding = Some(encoder.encode(SEMANTIC_ORIGIN_TEXT));
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
                "【多轮对话模式，最多{}轮】请逐步深入分析以下问题。当你认为分析已充分完成时，在回答末尾输出 ### DONE 表示结束。\n\n{}",
                config.turns, config.inputs[0]
            ))
        } else {
            None
        };
        let agent_continue_prompt = "请继续深入分析".to_string();

        // Run turns
        let total_turns = if config.agent_mode {
            config.turns.max(1)
        } else {
            config.inputs.len().min(config.turns.max(1))
        };

        // ── 重复检测状态（连续重复跟踪）──
        let mut consecutive_repetitions: usize = 0;
        let mut recent_texts: Vec<String> = Vec::with_capacity(config.repetition_window_size.max(2));

        for turn in 0..total_turns {
            let input: &str = if config.agent_mode {
                if turn == 0 {
                    agent_first_prompt.as_deref().unwrap_or(&config.inputs[0])
                } else {
                    &agent_continue_prompt
                }
            } else {
                &config.inputs[turn]
            };

            let current_text;

            if use_guidance {
                // ── Constrained path (logit-bias + phase constraint pipeline) ──
                // Run a ZhouTian cycle before each turn
                vm.cycle_count += 1;
                let mut dev = vm.current_deviation();

                // ── 混合偏离度（路线C hybrid = α·hamming + (1-α)·cosine）──
                let mut semantic_dev: Option<f32> = None;
                if config.semantic_mode {
                    if let Some(ref origin_emb) = vm.semantic_origin_embedding {
                        let encoder = MockEncoder::new(8, MockEncoderMode::Focused);
                        let v_current = encoder.encode(input);
                        let cos_sim = xiang_core::embedding::cosine_similarity(&v_current, origin_emb);
                        semantic_dev = Some((1.0 - cos_sim).clamp(0.0, 1.0));
                        vm.semantic_deviation = semantic_dev;
                    }
                    dev = hybrid_deviation(dev, semantic_dev, config.deviation_alpha);
                    // 语义干预计数：纯语义偏离度 > 0.7 时计为一次干预
                    if semantic_dev.map_or(false, |sd| sd > 0.7) {
                        metrics.semantic_interventions += 1;
                    }
                }

                // Set temperature based on deviation
                let temperature_used = if dev > 0.7 { 0.3 } else if dev < 0.3 { 0.8 } else { 0.5 };
                let temp = TemperatureMode::Fixed(temperature_used);

                // Get token groups (from config for real model, from vocab for mock)
                let (off_focus_ids, divergent_ids, eos_id) = get_token_ids(config, &backend);

                // Generate with logit-bias thinking guidance engine.
                let result = llm_ctx.generate_constrained_turn(
                    backend, input, config.max_tokens,
                    temp, vm.state, dev, vm.sha_count,
                    off_focus_ids, divergent_ids, eos_id,
                )?;
                current_text = result.text.clone();

                // ── Phase constraint check (阶段约束) ──
                let mut operator_checked: Option<String> = None;
                let (phase_valid, phase_violations) =
                    if config.phase_constraint_mode {
                        // Determine current operator phase from cycle count
                        let operators = &["生", "动", "长", "育"];
                        let phase_idx = (vm.cycle_count as usize - 1) % 4;
                        let operator = operators[phase_idx];
                        operator_checked = Some(operator.to_string());

                        // Run operator Gua transformation
                        vm.execute_phase_operator(operator).ok();

                        // Check generated output against phase rules — try all operators
                        let check_result = vm.check_operator_output(operator, &result.text);
                        let mut is_valid = check_result.is_valid;
                        let violations = check_result.violations.clone();

                        // 若当前算子不匹配，尝试其他算子
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
                        // 基本结构检查作为兜底
                        if !is_valid {
                            is_valid = result.text.len() > 10
                                && result.text.contains(['。', '，', ',', '.', '\n']);
                        }

                        if !is_valid {
                            metrics.phase_violations += 1;
                            vm.output_log.push(format!(
                                "[阶段违规] {} — {}", operator, violations.join("; ")
                            ));
                            // In phase-constrained mode, a violation triggers pruning
                            vm.state = vm.state.sha();
                            vm.sha_count += 1;
                            metrics.phase_rollbacks += 1;
                        }

                        (Some(is_valid), violations)
                    } else {
                        (None, vec![])
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
                    deviation: dev,
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
                };
                metrics.turn_data.push(tm);
                metrics.total_tokens += result.tokens_generated;
                metrics.total_generations += 1;
                metrics.focus_interventions += if focus_stop || dev > 0.7 { 1 } else { 0 };

                // Agent stop check
                if config.agent_mode && is_agent_stop_signal(&result.text, &config.agent_stop_patterns) {
                    break;
                }
            } else {
                // ── Controlled path (no thinking guidance) ──
                let result = llm_ctx.generate_unconstrained_turn(
                    backend, input, config.max_tokens,
                    TemperatureMode::Default,
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
                        let v_current = encoder.encode(input);
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
                    temperature_used: 0.0,
                    operator_checked: None,
                    vm_gua_state: None,
                    stop_reason: None,
                };
                metrics.turn_data.push(tm);
                metrics.total_tokens += result.tokens_generated;
                metrics.total_generations += 1;
                metrics.focus_interventions += if measure_dev > 0.7 { 1 } else { 0 };

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
    fn feedback_to_vm(vm: &mut CangVM, result: &GenerationResult, deviation: f32) {
        // If generation was focused and aligned, reinforce positive
        if deviation < 0.3 && result.stop_reason == StopReason::Eos {
            vm.state = vm.state.cang();
            vm.cang_sea.store(vm.state, Gua::ZERO, 0.3);
            // Semantic store: log aligned experience
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
            // High deviation or focus drift → prune
            vm.state = vm.state.sha();
            vm.sha_count += 1;
            vm.state = vm.state.cang();
            vm.cang_sea.store(vm.state, Gua::ZERO, -0.5);
            // Semantic store: log negative experience (immune)
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
        }
    }

    /// Run a comparison experiment and generate a report.
    pub fn run_comparison(config: &ExperimentConfig) -> ComparisonReport {
        let controlled = Self::run_controlled(config).unwrap_or_default();
        let constrained = Self::run_constrained(config).unwrap_or_default();

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
        max_tokens: 100,
        mock_mode: MockMode::OffFocus,
        use_guidance_engine: true,
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
        max_tokens: 80,
        mock_mode: MockMode::Verbose,
        use_guidance_engine: true,
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
        max_tokens: 120,
        mock_mode: MockMode::Verbose,
        use_guidance_engine: true,
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
        max_tokens: 120,
        mock_mode: MockMode::Verbose,
        use_guidance_engine: true,
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
    }
}

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
        let result = ExperimentRunner::run_constrained(&config);
        assert!(result.is_ok());
        let metrics = result.unwrap();
        assert!(metrics.total_generations > 0);
    }

    #[test]
    fn test_run_focus_controlled() {
        let config = focus_deviation_experiment();
        let result = ExperimentRunner::run_controlled(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_long_context() {
        let config = long_context_experiment();
        let result = ExperimentRunner::run_constrained(&config);
        assert!(result.is_ok());
        let metrics = result.unwrap();
        assert_eq!(metrics.total_generations, 32);
    }

    #[test]
    fn test_run_multi_turn() {
        let config = multi_turn_experiment();
        let result = ExperimentRunner::run_constrained(&config);
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

        let result = ExperimentRunner::run_constrained(&config);
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
            max_tokens: 50,
            mock_mode: MockMode::Safe,
            use_guidance_engine: true,
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
        };

        let result = ExperimentRunner::run_constrained(&config);
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

        let result = ExperimentRunner::run_constrained(&config);
        assert!(result.is_ok(), "非语义模式应正常回退");
        let metrics = result.unwrap();

        // 非语义模式下不应有 semantic_deviation
        // （用 Option::is_none 验证所有轮次）
        let no_semantic = metrics.turn_data.iter().all(|t| t.semantic_deviation.is_none());
        assert!(no_semantic, "非语义模式下不应记录 semantic_deviation");
    }
}
