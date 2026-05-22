//! Benchmark runner for comparing bare vs 三易-guided generation.
//!
//! Runs N trials where each trial generates output under two conditions:
//!   1. Controlled (bare model, no guidance)
//!   2. Constrained (三易 three-engine stack with ShanVM + ZhouVM)
//!
//! Collects per-trial metrics and delegates statistical analysis to the statistics module.

use crate::{
    ExperimentRunner, ExperimentMetrics, ExperimentConfig, TurnMetrics,
    statistics::{BenchmarkSummary, compute_benchmark_summary},
    create_backend,
};
use serde::Serialize;
use std::path::PathBuf;
use std::fs;
use xiang_llm::LlmError;

/// Configuration for a benchmark run.
pub struct BenchmarkConfig {
    /// Number of trials (e.g., 100).
    pub num_trials: usize,
    /// The list of input prompts, one per trial.
    pub inputs: Vec<String>,
    /// Base experiment configuration (backend, token limits, etc.).
    pub base_config: ExperimentConfig,
}

/// A single benchmark trial with both controlled and constrained results.
#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkTrial {
    pub trial_index: usize,
    pub input: String,
    pub controlled_metrics: ExperimentMetrics,
    pub constrained_metrics: ExperimentMetrics,
}

/// Complete benchmark results with raw trials and statistical summary.
#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkResults {
    pub num_trials: usize,
    pub trials: Vec<BenchmarkTrial>,
    pub summary: BenchmarkSummary,
    pub model_name: String,
    pub backend_type: String,
}

/// The benchmark runner orchestrates all trials.
pub struct BenchmarkRunner;

impl BenchmarkRunner {
    /// Run the full benchmark suite.
    ///
    /// For each trial:
    /// 1. Creates a constrained config (guidance engine + three-engine stack)
    /// 2. Runs constrained with KV cache cleared beforehand
    /// 3. Clears KV cache (上下文新陈代谢)
    /// 4. Creates a controlled config (bare LLM, no guidance)
    /// 5. Runs controlled with KV cache cleared beforehand
    /// 6. Collects per-trial metrics
    /// 7. Computes statistical summary at the end
    ///
    /// The backend (model/HTTP) is created ONCE and reused across all trials,
    /// with KV cache cleared between conditions and trials to prevent overflow.
    pub fn run(config: &BenchmarkConfig) -> Result<BenchmarkResults, LlmError> {
        let n = config.num_trials.min(config.inputs.len());
        let mut trials = Vec::with_capacity(n);

        let (model_name, backend_type) = Self::detect_backend_info(&config.base_config);
        let bench_start = std::time::Instant::now();
        let output_dir = PathBuf::from("benchmark_output");

        // ── Create backend ONCE, discover vocab ONCE ──
        let mut base_config = config.base_config.clone();
        let mut backend = create_backend(&config.base_config)?;

        // Cache vocab if using real model
        if config.base_config.model_path.is_some() || config.base_config.http_server_url.is_some() {
            let (off, div, eos) = backend.discover_vocab();
            base_config.cached_off_focus_ids = Some(off);
            base_config.cached_divergent_ids = Some(div);
            base_config.cached_eos_id = Some(eos);
        }

        for i in 0..n {
            let input = &config.inputs[i];

            // ── 上下文新陈代谢: clear KV cache before each trial ──
            backend.reset_for_new_generation();

            // ── Constrained (三易 guided) ── run first
            let mut constrained_cfg = base_config.clone();
            constrained_cfg.use_guidance_engine = true;
            constrained_cfg.enable_three_engine = true;
            constrained_cfg.phase_constraint_mode = true;
            constrained_cfg.inputs = vec![input.clone()];

            let constrained = ExperimentRunner::run_constrained_with_backend(
                &constrained_cfg, &mut backend,
            ).unwrap_or_else(|e| {
                eprintln!("[WARN] 试验#{i} 实验组失败: {e}");
                ExperimentMetrics::default()
            });

            // ── 上下文新陈代谢: clear KV cache between conditions ──
            backend.reset_for_new_generation();

            // ── Controlled (bare model) ──
            let mut controlled_cfg = base_config.clone();
            controlled_cfg.use_guidance_engine = false;
            controlled_cfg.enable_three_engine = false;
            controlled_cfg.phase_constraint_mode = false;
            controlled_cfg.inputs = vec![input.clone()];

            let controlled = ExperimentRunner::run_controlled_with_backend(
                &controlled_cfg, &mut backend,
            ).unwrap_or_else(|e| {
                eprintln!("[WARN] 试验#{i} 对照组失败: {e}");
                ExperimentMetrics::default()
            });

            let ctrl_turns = controlled.total_generations;
            let const_turns = constrained.total_generations;

            // ── Export per-turn files ──
            Self::write_turn_files(&output_dir, i, input, &controlled, &constrained)?;

            trials.push(BenchmarkTrial {
                trial_index: i,
                input: input.clone(),
                controlled_metrics: controlled,
                constrained_metrics: constrained,
            });

            // Per-trial progress with timing
            let elapsed_t = bench_start.elapsed();
            let avg = elapsed_t.as_secs_f64() / (i + 1) as f64;
            let remaining = avg * (n - i - 1) as f64;
            eprintln!("  试验#{i} ({:.0}s) 剩余~{:.0}s [ctrl_t={} const_t={}]",
                      elapsed_t.as_secs_f64(), remaining,
                      ctrl_turns, const_turns);
        }

        let summary = compute_benchmark_summary(&trials);

        Ok(BenchmarkResults {
            num_trials: n,
            trials,
            summary,
            model_name,
            backend_type,
        })
    }

    /// Format a single turn's metrics into a structured text block.
    fn format_turn_content(turn: &TurnMetrics, turn_index: usize) -> String {
        format!(
            "=== 轮次 {turn_index} ===\n\
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

    /// Write per-turn output files for a single trial.
    /// Directory structure:
    ///   benchmark_output/trial_{i}/
    ///     input.txt
    ///     controlled/turn_000.txt  (each turn's generated text with full metrics)
    ///     constrained/turn_000.txt
    fn write_turn_files(
        output_dir: &PathBuf,
        trial_index: usize,
        input: &str,
        controlled: &ExperimentMetrics,
        constrained: &ExperimentMetrics,
    ) -> Result<(), LlmError> {
        let trial_dir = output_dir.join(format!("trial_{}", trial_index));

        // Write input
        let input_path = trial_dir.join("input.txt");
        fs::create_dir_all(input_path.parent().unwrap()).map_err(|e| {
            LlmError::GenerationFailed(format!("无法创建输出目录: {}", e))
        })?;
        fs::write(&input_path, input).map_err(|e| {
            LlmError::GenerationFailed(format!("无法写入输入文件: {}", e))
        })?;

        // Write controlled turn files
        let ctrl_dir = trial_dir.join("controlled");
        fs::create_dir_all(&ctrl_dir).map_err(|e| {
            LlmError::GenerationFailed(format!("无法创建输出目录: {}", e))
        })?;
        for (j, turn) in controlled.turn_data.iter().enumerate() {
            let turn_file = ctrl_dir.join(format!("turn_{:03}.txt", j));
            let content = Self::format_turn_content(turn, j);
            fs::write(&turn_file, content).map_err(|e| {
                LlmError::GenerationFailed(format!("无法写入轮次文件: {}", e))
            })?;
        }

        // Write constrained turn files
        let const_dir = trial_dir.join("constrained");
        fs::create_dir_all(&const_dir).map_err(|e| {
            LlmError::GenerationFailed(format!("无法创建输出目录: {}", e))
        })?;
        for (j, turn) in constrained.turn_data.iter().enumerate() {
            let turn_file = const_dir.join(format!("turn_{:03}.txt", j));
            let content = Self::format_turn_content(turn, j);
            fs::write(&turn_file, content).map_err(|e| {
                LlmError::GenerationFailed(format!("无法写入轮次文件: {}", e))
            })?;
        }

        Ok(())
    }

    fn detect_backend_info(config: &ExperimentConfig) -> (String, String) {
        let model = if let Some(ref url) = config.http_server_url {
            format!("远程HTTP服务 ({url})")
        } else if config.model_path.is_some() {
            config.model_path.as_ref().unwrap().clone()
        } else {
            match config.mock_mode {
                xiang_llm::MockMode::Safe => "Mock(Safe)".into(),
                xiang_llm::MockMode::Verbose => "Mock(Verbose)".into(),
                xiang_llm::MockMode::OffFocus => "Mock(OffFocus)".into(),
                xiang_llm::MockMode::Hallucinating => "Mock(Hallucinating)".into(),
                xiang_llm::MockMode::Fixed => "Mock(Fixed)".into(),
            }
        };
        let backend = if config.http_server_url.is_some() {
            "HTTP远程"
        } else if config.model_path.is_some() {
            "llama.cpp FFI"
        } else {
            "MockBackend"
        };
        (model, backend.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inputs::generate_benchmark_inputs;

    fn make_config() -> ExperimentConfig {
        ExperimentConfig {
            name: "小规模基准测试".into(),
            description: "测试基准运行器".into(),
            guidance_program: Some("定义 测试:\n    卦 状态 = 111111\n    生\n    动\n    归\n    藏 善".into()),
            inputs: vec![],
            turns: 1,
            max_tokens: 30,
            mock_mode: xiang_llm::MockMode::OffFocus,
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

    #[test]
    fn test_benchmark_small_run() {
        let inputs = generate_benchmark_inputs(42);
        let config = BenchmarkConfig {
            num_trials: 5,
            inputs: inputs[..5].to_vec(),
            base_config: make_config(),
        };

        let results = BenchmarkRunner::run(&config);
        assert!(results.is_ok());
        let r = results.unwrap();
        assert_eq!(r.num_trials, 5);
        assert_eq!(r.trials.len(), 5);
        // Summary should have valid stats
        assert!(r.summary.deviation_controlled.n > 0);
        assert!(r.summary.deviation_constrained.n > 0);
    }

    #[test]
    fn test_benchmark_result_has_valid_metrics() {
        let inputs = generate_benchmark_inputs(42);
        let config = BenchmarkConfig {
            num_trials: 3,
            inputs: inputs[..3].to_vec(),
            base_config: make_config(),
        };

        let results = BenchmarkRunner::run(&config).unwrap();
        for trial in &results.trials {
            assert!(trial.controlled_metrics.total_generations > 0,
                "对照组试验#{} 应有生成", trial.trial_index);
            assert!(trial.constrained_metrics.total_generations > 0,
                "实验组试验#{} 应有生成", trial.trial_index);
        }
    }

    #[test]
    fn test_benchmark_summary_has_interpretation() {
        let inputs = generate_benchmark_inputs(42);
        let config = BenchmarkConfig {
            num_trials: 5,
            inputs: inputs[..5].to_vec(),
            base_config: make_config(),
        };

        let results = BenchmarkRunner::run(&config).unwrap();
        assert!(!results.summary.deviation_test.interpretation.is_empty());
        assert!(!results.summary.compliance_test.interpretation.is_empty());
        assert!(!results.summary.completion_test.interpretation.is_empty());
    }
}
