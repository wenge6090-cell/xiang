//! Benchmark runner for comparing bare vs 三易-guided generation.
//!
//! Runs N trials where each trial generates output under two conditions:
//!   1. Controlled (bare model, no guidance)
//!   2. Constrained (三易 three-engine stack with ShanVM + ZhouVM)
//!
//! Collects per-trial metrics and delegates statistical analysis to the statistics module.

use crate::{
    ExperimentRunner, ExperimentMetrics, ExperimentConfig, focus_deviation_experiment,
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
    /// Optional quality evaluation report from AI evaluator.
    #[serde(skip)]
    pub quality_report: Option<crate::QualityEvaluationReport>,
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

        let summary = compute_benchmark_summary(&trials, None);

        Ok(BenchmarkResults {
            num_trials: n,
            trials,
            summary,
            model_name,
            backend_type,
            quality_report: None,
        })
    }

    /// Write per-turn output files for a single trial.
    /// Directory structure:
    ///   benchmark_output/trial_{i}/
    ///     input.txt
    ///     对照组/turn_000.txt  (each turn's generated text with full metrics)
    ///     实验组/turn_000.txt
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
        let ctrl_dir = trial_dir.join("对照组");
        fs::create_dir_all(&ctrl_dir).map_err(|e| {
            LlmError::GenerationFailed(format!("无法创建输出目录: {}", e))
        })?;
        for (j, turn) in controlled.turn_data.iter().enumerate() {
            let turn_file = ctrl_dir.join(crate::turn_filename(turn, j));
            let content = crate::format_turn_content(turn, j);
            fs::write(&turn_file, content).map_err(|e| {
                LlmError::GenerationFailed(format!("无法写入轮次文件: {}", e))
            })?;
        }

        // Write constrained turn files
        let const_dir = trial_dir.join("实验组");
        fs::create_dir_all(&const_dir).map_err(|e| {
            LlmError::GenerationFailed(format!("无法创建输出目录: {}", e))
        })?;
        for (j, turn) in constrained.turn_data.iter().enumerate() {
            let turn_file = const_dir.join(crate::turn_filename(turn, j));
            let content = crate::format_turn_content(turn, j);
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

/// Run a three-way benchmark comparing 约束组, 微调组, and 原生组.
///
/// The benchmark runs two sequential model passes:
///   1. RouteB model → obtains 约束组 (silent engine) + 微调组 (bare)
///   2. Base model  → obtains 原生组 (bare)
///
/// Returns (yue_shu_results, wei_diao_results, yuan_sheng_results) — three BenchmarkResults
/// where each carries the correct metrics in the appropriate fields.
pub fn run_threeway_bench(
    num_trials: usize,
    inputs: Vec<String>,
    routeb_model_path: &str,
    base_model_path: &str,
    max_tokens: u32,
    _seed: u64,
) -> Result<(BenchmarkResults, BenchmarkResults, BenchmarkResults), LlmError> {
    let n = num_trials.min(inputs.len());
    let output_dir = PathBuf::from("threeway_benchmark_output");
    let bench_start = std::time::Instant::now();

    // Extract model names from paths
    let model_name_routeb = std::path::Path::new(routeb_model_path)
        .file_name().map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "RouteB".into());
    let model_name_base = std::path::Path::new(base_model_path)
        .file_name().map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "Base".into());

    // ── Phase 1: RouteB model (约束组 + 微调组) ──
    eprintln!("\n=== 第一阶段: 路线B微调模型 (约束组 + 微调组) ===");
    eprintln!("模型: {}", routeb_model_path);

    let mut routeb_cfg = make_experiment_config(routeb_model_path, max_tokens);
    routeb_cfg.inputs = inputs.iter().take(n).cloned().collect();
    let mut routeb_backend = create_backend(&routeb_cfg)?;

    // Discover vocab once for RouteB
    let (off, div, eos) = routeb_backend.discover_vocab();
    routeb_cfg.cached_off_focus_ids = Some(off);
    routeb_cfg.cached_divergent_ids = Some(div);
    routeb_cfg.cached_eos_id = Some(eos);

    let mut yue_shu_metrics: Vec<ExperimentMetrics> = Vec::with_capacity(n);
    let mut wei_diao_metrics: Vec<ExperimentMetrics> = Vec::with_capacity(n);

    for i in 0..n {
        let input = &routeb_cfg.inputs[i];

        // ── 约束组: RouteB + 静默引擎 (logit bias only, no text prompts) ──
        routeb_backend.reset_for_new_generation();
        let mut yue_shu_cfg = routeb_cfg.clone();
        yue_shu_cfg.use_guidance_engine = true;
        yue_shu_cfg.inject_constraint_prompt = false;
        yue_shu_cfg.enable_three_engine = true;
        yue_shu_cfg.phase_constraint_mode = true;
        yue_shu_cfg.inputs = vec![input.clone()];

        let yue_shu = ExperimentRunner::run_constrained_with_backend(
            &yue_shu_cfg, &mut routeb_backend,
        ).unwrap_or_else(|e| {
            eprintln!("[WARN] 试验#{i} 约束组失败: {e}");
            ExperimentMetrics::default()
        });

        // ── 微调组: RouteB + 裸问题 (无引擎引导) ──
        routeb_backend.reset_for_new_generation();
        let mut wei_diao_cfg = routeb_cfg.clone();
        wei_diao_cfg.use_guidance_engine = false;
        wei_diao_cfg.enable_three_engine = false;
        wei_diao_cfg.phase_constraint_mode = false;
        wei_diao_cfg.inputs = vec![input.clone()];

        let wei_diao = ExperimentRunner::run_controlled_with_backend(
            &wei_diao_cfg, &mut routeb_backend,
        ).unwrap_or_else(|e| {
            eprintln!("[WARN] 试验#{i} 微调组失败: {e}");
            ExperimentMetrics::default()
        });

        yue_shu_metrics.push(yue_shu);
        wei_diao_metrics.push(wei_diao);

        let elapsed = bench_start.elapsed();
        eprintln!("  试验#{i} ({:.0}s) 约束组_轮={} 微调组_轮={}",
            elapsed.as_secs_f64(),
            yue_shu_metrics.last().unwrap().total_generations,
            wei_diao_metrics.last().unwrap().total_generations);
    }

    // ── Phase 2: Base model (原生组) ──
    eprintln!("\n=== 第二阶段: 基础模型 (原生组) ===");
    eprintln!("模型: {}", base_model_path);

    let mut base_cfg = make_experiment_config(base_model_path, max_tokens);
    base_cfg.inputs = inputs.iter().take(n).cloned().collect();
    let mut base_backend = create_backend(&base_cfg)?;

    let (boff, bdiv, beos) = base_backend.discover_vocab();
    base_cfg.cached_off_focus_ids = Some(boff);
    base_cfg.cached_divergent_ids = Some(bdiv);
    base_cfg.cached_eos_id = Some(beos);

    let mut yuan_sheng_metrics: Vec<ExperimentMetrics> = Vec::with_capacity(n);

    for i in 0..n {
        let input = &base_cfg.inputs[i];

        // ── 原生组: 基础模型 + 裸问题 (基线) ──
        base_backend.reset_for_new_generation();
        let mut yuan_sheng_cfg = base_cfg.clone();
        yuan_sheng_cfg.use_guidance_engine = false;
        yuan_sheng_cfg.enable_three_engine = false;
        yuan_sheng_cfg.phase_constraint_mode = false;
        yuan_sheng_cfg.inputs = vec![input.clone()];

        let yuan_sheng = ExperimentRunner::run_controlled_with_backend(
            &yuan_sheng_cfg, &mut base_backend,
        ).unwrap_or_else(|e| {
            eprintln!("[WARN] 试验#{i} 原生组失败: {e}");
            ExperimentMetrics::default()
        });

        yuan_sheng_metrics.push(yuan_sheng);

        let elapsed = bench_start.elapsed();
        eprintln!("  试验#{i} ({:.0}s) 原生组_轮={}",
            elapsed.as_secs_f64(),
            yuan_sheng_metrics.last().unwrap().total_generations);
    }

    // ── Construct BenchmarkResults for each group ──
    let yue_shu_trials: Vec<BenchmarkTrial> = (0..n).map(|i| BenchmarkTrial {
        trial_index: i,
        input: routeb_cfg.inputs[i].clone(),
        controlled_metrics: ExperimentMetrics::default(),
        constrained_metrics: yue_shu_metrics[i].clone(),
    }).collect();

    let wei_diao_trials: Vec<BenchmarkTrial> = (0..n).map(|i| BenchmarkTrial {
        trial_index: i,
        input: routeb_cfg.inputs[i].clone(),
        controlled_metrics: wei_diao_metrics[i].clone(),
        constrained_metrics: ExperimentMetrics::default(),
    }).collect();

    let yuan_sheng_trials: Vec<BenchmarkTrial> = (0..n).map(|i| BenchmarkTrial {
        trial_index: i,
        input: base_cfg.inputs[i].clone(),
        controlled_metrics: yuan_sheng_metrics[i].clone(),
        constrained_metrics: ExperimentMetrics::default(),
    }).collect();

    let yue_shu_summary = compute_benchmark_summary(&yue_shu_trials, None);
    let wei_diao_summary = compute_benchmark_summary(&wei_diao_trials, None);
    let yuan_sheng_summary = compute_benchmark_summary(&yuan_sheng_trials, None);

    let yue_shu_results = BenchmarkResults {
        num_trials: n,
        trials: yue_shu_trials,
        summary: yue_shu_summary,
        model_name: model_name_routeb.clone(),
        backend_type: "llama.cpp FFI".into(),
        quality_report: None,
    };

    let wei_diao_results = BenchmarkResults {
        num_trials: n,
        trials: wei_diao_trials,
        summary: wei_diao_summary,
        model_name: model_name_routeb.clone(),
        backend_type: "llama.cpp FFI".into(),
        quality_report: None,
    };

    let yuan_sheng_results = BenchmarkResults {
        num_trials: n,
        trials: yuan_sheng_trials,
        summary: yuan_sheng_summary,
        model_name: model_name_base,
        backend_type: "llama.cpp FFI".into(),
        quality_report: None,
    };

    // ── Write three-group turn files ──
    write_threeway_turn_files(&output_dir, &yue_shu_metrics, &wei_diao_metrics, &yuan_sheng_metrics, &inputs[..n])?;

    eprintln!("\n=== 三组基准测试完成 ({:.0}s) ===", bench_start.elapsed().as_secs_f64());
    eprintln!("  约束组: RouteB + 静默引擎 (logit bias only, 无体系提示词)");
    eprintln!("  微调组: RouteB + 裸问题 (无引擎)");
    eprintln!("  原生组: 基础模型 + 裸问题 (基线)");
    eprintln!("  输出目录: {}", output_dir.display());

    Ok((yue_shu_results, wei_diao_results, yuan_sheng_results))
}

/// Create an ExperimentConfig for a model with standard three-way benchmark settings.
fn make_experiment_config(model_path: &str, max_tokens: u32) -> ExperimentConfig {
    let mut cfg = focus_deviation_experiment();
    cfg.model_path = Some(model_path.to_string());
    cfg.semantic_mode = true;
    cfg.deviation_alpha = 0.5;
    cfg.turns = 10;
    cfg.max_tokens = max_tokens;
    cfg.agent_mode = true;
    cfg.agent_stop_patterns = vec![
        "### DONE".into(),
        "任务完成".into(),
        "分析结束".into(),
    ];
    cfg.repetition_detection_enabled = true;
    cfg.repetition_threshold = 3;
    cfg.repetition_similarity_threshold = 0.85;
    cfg.repetition_window_size = 3;
    cfg
}

/// Write three-group turn files for all trials.
///
/// Directory structure:
///   threeway_benchmark_output/trial_{i}/
///     input.txt
///     约束组/turn_000.txt  (silent engine turns)
///     微调组/turn_000.txt  (bare RouteB turns)
///     原生组/turn_000.txt  (bare base model turns)
fn write_threeway_turn_files(
    output_dir: &PathBuf,
    yue_shu_metrics: &[ExperimentMetrics],
    wei_diao_metrics: &[ExperimentMetrics],
    yuan_sheng_metrics: &[ExperimentMetrics],
    inputs: &[String],
) -> Result<(), LlmError> {
    let n = yue_shu_metrics.len()
        .min(wei_diao_metrics.len())
        .min(yuan_sheng_metrics.len())
        .min(inputs.len());

    for i in 0..n {
        let trial_dir = output_dir.join(format!("trial_{}", i));
        fs::create_dir_all(&trial_dir).map_err(|e| {
            LlmError::GenerationFailed(format!("无法创建输出目录: {}", e))
        })?;

        // Write input file
        fs::write(trial_dir.join("input.txt"), &inputs[i]).map_err(|e| {
            LlmError::GenerationFailed(format!("无法写入输入文件: {}", e))
        })?;

        // 约束组
        let yue_shu_dir = trial_dir.join("约束组");
        fs::create_dir_all(&yue_shu_dir).map_err(|e| {
            LlmError::GenerationFailed(format!("无法创建约束组目录: {}", e))
        })?;
        for (j, turn) in yue_shu_metrics[i].turn_data.iter().enumerate() {
            let turn_file = yue_shu_dir.join(crate::turn_filename(turn, j));
            let content = crate::format_turn_content(turn, j);
            fs::write(&turn_file, content).map_err(|e| {
                LlmError::GenerationFailed(format!("无法写入约束组轮次文件: {}", e))
            })?;
        }

        // 微调组
        let wei_diao_dir = trial_dir.join("微调组");
        fs::create_dir_all(&wei_diao_dir).map_err(|e| {
            LlmError::GenerationFailed(format!("无法创建微调组目录: {}", e))
        })?;
        for (j, turn) in wei_diao_metrics[i].turn_data.iter().enumerate() {
            let turn_file = wei_diao_dir.join(crate::turn_filename(turn, j));
            let content = crate::format_turn_content(turn, j);
            fs::write(&turn_file, content).map_err(|e| {
                LlmError::GenerationFailed(format!("无法写入微调组轮次文件: {}", e))
            })?;
        }

        // 原生组
        let yuan_sheng_dir = trial_dir.join("原生组");
        fs::create_dir_all(&yuan_sheng_dir).map_err(|e| {
            LlmError::GenerationFailed(format!("无法创建原生组目录: {}", e))
        })?;
        for (j, turn) in yuan_sheng_metrics[i].turn_data.iter().enumerate() {
            let turn_file = yuan_sheng_dir.join(crate::turn_filename(turn, j));
            let content = crate::format_turn_content(turn, j);
            fs::write(&turn_file, content).map_err(|e| {
                LlmError::GenerationFailed(format!("无法写入原生组轮次文件: {}", e))
            })?;
        }
    }

    Ok(())
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
            max_tokens: 400,
            mock_mode: xiang_llm::MockMode::OffFocus,
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
