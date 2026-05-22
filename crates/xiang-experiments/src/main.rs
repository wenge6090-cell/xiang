/// XiangLang Thinking Guidance Engine — Experiment CLI
///
/// Usage:
///   xiang-exp run                Run all experiments (mock backend)
///   xiang-exp run --model <GGUF> --experiment focus  Run with real model
///   xiang-exp run --http <URL>   Run using remote llama.cpp server
///   xiang-exp report             Generate HTML report
///   xiang-exp list               List available experiments

use xiang_experiments::{
    ExperimentRunner, generate_html_report,
    focus_deviation_experiment, long_context_experiment, multi_turn_experiment,
    phase_constraint_experiment,
    ExperimentConfig, MockMode,
    benchmark::{BenchmarkRunner, BenchmarkConfig},
    inputs::generate_benchmark_inputs,
    report::generate_benchmark_html_report,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        return;
    }

    let use_semantic = parse_flag(&args, "--semantic");
    let deviation_alpha = parse_flag_value(&args, "--alpha")
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.5);

    match args[1].as_str() {
        "run" => {
            let use_model = parse_flag(&args, "--model");
            let http_url = parse_flag_value(&args, "--http").map(|s| s.to_string());
            let experiment = parse_flag_value(&args, "--experiment")
                .or_else(|| args.get(2).map(|s| s.as_str()))
                .unwrap_or("all");
            run_experiments(experiment, use_model, http_url, use_semantic, deviation_alpha);
        }
        "report" => {
            let experiment = args.get(2).map(|s| s.as_str()).unwrap_or("all");
            let http_url = parse_flag_value(&args, "--http").map(|s| s.to_string());
            generate_report(experiment, http_url, use_semantic, deviation_alpha);
        }
        "list" => {
            list_experiments();
        }
        "bench" => {
            let http_url = parse_flag_value(&args, "--http").map(|s| s.to_string());
            let use_model = parse_flag(&args, "--model");
            let seed = parse_flag_value(&args, "--seed")
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(42);
            let mock_mode = parse_flag_value(&args, "--mock")
                .map(|s| match s {
                    "safe" => MockMode::Safe,
                    "verbose" => MockMode::Verbose,
                    "offfocus" => MockMode::OffFocus,
                    "hallucinating" => MockMode::Hallucinating,
                    "fixed" => MockMode::Fixed,
                    _ => MockMode::OffFocus,
                })
                .unwrap_or(MockMode::OffFocus);
            run_bench(http_url, use_model, seed, mock_mode);
        }
        _ => {
            print_usage();
        }
    }
}

/// Check if a boolean flag is present.
fn parse_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

/// Get the value of a --key <value> flag.
fn parse_flag_value<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.windows(2).find(|w| w[0] == name).map(|w| w[1].as_str())
}

fn print_usage() {
    println!("象语言 思维引导引擎 实验框架 (XiangLang Thinking Guidance Engine)");
    println!();
    println!("用法:");
    println!("  xiang-exp run                    运行所有实验 (MockBackend)");
    println!("  xiang-exp run --model <GGUF>     使用真实模型后端运行");
    println!("  xiang-exp run --http <URL>       使用远程 llama.cpp server (GPU)");
    println!("  xiang-exp run --experiment <名称> 运行指定实验");
    println!("  xiang-exp report                 生成所有实验的HTML报告");
    println!("  xiang-exp list                   列出可用实验");
    println!("  xiang-exp bench                  运行100组基准测试（裸跑 vs 三易控制）");
    println!("  xiang-exp bench --mock <模式>     指定Mock模式（offfocus推荐）");
    println!("  xiang-exp bench --seed <数字>     指定随机种子（默认42）");
    println!("  xiang-exp bench --http <URL>      使用远程llama.cpp");
    println!("  xiang-exp bench --model <GGUF>    使用真实模型");
    println!();
    println!("实验列表:");
    println!("  focus        — 思维焦点: 偏移检测测试");
    println!("  longcontext  — 长上下文: 32K持续聚焦测试");
    println!("  multiturn    — 多轮对话: 50轮思维连贯性测试");
    println!("  phase        — 阶段约束: 阴仪协议验证测试");
    println!("  all          — 运行全部四个实验");
    println!();
    println!("基准测试示例:");
    println!("  xiang-exp bench                                   100组MockBackend基准");
    println!("  xiang-exp bench --mock offfocus                   推荐：OffFocus模式");
    println!("  xiang-exp bench --seed 123 --mock hallucinating   不同种子+模式");
    println!();
    println!("实验示例:");
    println!("  xiang-exp run --model /path/to/model.gguf --experiment focus");
    println!("  xiang-exp run --http http://localhost:8080");
    println!("  xiang-exp run --semantic                         启用语义模式（路线C MockEncoder）");
}

fn list_experiments() {
    println!("可用实验:");
    println!("  focus        — 思维焦点: 偏移检测测试");
    println!("  longcontext  — 长上下文: 32K持续聚焦测试");
    println!("  multiturn    — 多轮对话: 50轮思维连贯性测试");
    println!("  phase        — 阶段约束: 阴仪协议验证测试");
    println!("  all          — 运行全部实验");
}

fn get_experiments(name: &str) -> Vec<(&'static str, ExperimentConfig)> {
    match name {
        "focus" => vec![("思维焦点—偏移检测", focus_deviation_experiment())],
        "longcontext" => vec![("长上下文—32K", long_context_experiment())],
        "multiturn" => vec![("多轮对话—50轮", multi_turn_experiment())],
        "phase" => vec![("阶段约束—协议验证", phase_constraint_experiment())],
        _ => vec![
            ("思维焦点—偏移检测", focus_deviation_experiment()),
            ("长上下文—32K", long_context_experiment()),
            ("多轮对话—50轮", multi_turn_experiment()),
            ("阶段约束—协议验证", phase_constraint_experiment()),
        ],
    }
}

/// Apply real model settings to a config.
fn apply_real_model(mut config: ExperimentConfig, model_path: &str) -> ExperimentConfig {
    config.model_path = Some(model_path.to_string());
    // EOS and token groups will be auto-detected from the model vocabulary
    // at runtime via backend.discover_vocab().
    config.eos_token_id = 0;
    config.mock_mode = xiang_experiments::MockMode::Safe; // unused but valid
    config
}

/// Apply semantic mode settings to a config.
fn apply_semantic_config(mut config: ExperimentConfig, use_semantic: bool, alpha: f32) -> ExperimentConfig {
    config.semantic_mode = use_semantic;
    config.deviation_alpha = alpha;
    config
}

fn run_experiments(name: &str, use_real_model: bool, http_url: Option<String>, use_semantic: bool, deviation_alpha: f32) {
    let experiments: Vec<(String, ExperimentConfig)> = if let Some(url) = http_url {
        // HTTP backend mode
        get_experiments(name).into_iter()
            .map(|(label, mut cfg)| {
                cfg = apply_semantic_config(cfg, use_semantic, deviation_alpha);
                cfg.http_server_url = Some(url.clone());
                // EOS and token groups will be auto-detected via HttpBackend::discover_vocab()
                cfg.eos_token_id = 0;
                (format!("{} [HTTP: {}]", label, url), cfg)
            })
            .collect()
    } else if use_real_model {
        // Direct llama.cpp FFI backend mode
        let all_args: Vec<String> = std::env::args().collect();
        let model_path = all_args
            .windows(2)
            .find(|w| w[0] == "--model")
            .map(|w| w[1].clone())
            .expect("缺少 --model <GGUF路径> 参数");
        get_experiments(name).into_iter()
            .map(|(label, cfg)| {
                let cfg = apply_semantic_config(cfg, use_semantic, deviation_alpha);
                (format!("{} [真实模型]", label), apply_real_model(cfg, &model_path))
            })
            .collect()
    } else {
        // Mock backend mode
        get_experiments(name).into_iter()
            .map(|(label, cfg)| {
                let cfg = apply_semantic_config(cfg, use_semantic, deviation_alpha);
                (label.to_string(), cfg)
            })
            .collect()
    };

    let count = experiments.len();
    for (i, (label, config)) in experiments.iter().enumerate() {
        println!("[{}/{}] 实验: {}", i + 1, count, label);
        println!("  描述: {}", config.description);
        println!("  输入数: {}  轮数: {}", config.inputs.len(), config.turns);
        if config.http_server_url.is_some() {
            println!("  后端: HTTP远程服务器 ({})", config.http_server_url.as_ref().unwrap());
            println!("  EOS: {}", config.eos_token_id);
        } else if config.model_path.is_some() {
            println!("  后端: 真实模型 ({}), EOS: {}", 
                config.model_path.as_ref().unwrap(), config.eos_token_id);
        } else {
            println!("  后端: Mock ({:?})", config.mock_mode);
        }

        println!();
        println!("  [对照组] 无引导引擎...");
        let controlled = match ExperimentRunner::run_controlled(config) {
            Ok(m) => {
                println!("    焦点干预: {} | 总Token: {} | 耗时: {}ms",
                    m.focus_interventions, m.total_tokens, m.elapsed_ms);
                m
            }
            Err(e) => {
                eprintln!("    ✗ 失败: {e}");
                continue;
            }
        };

        println!();
        println!("  [实验组] 有引导引擎...");
        let constrained = match ExperimentRunner::run_constrained(config) {
            Ok(m) => {
                println!("    焦点干预: {} | 总Token: {} | 耗时: {}ms",
                    m.focus_interventions, m.total_tokens, m.elapsed_ms);
                m
            }
            Err(e) => {
                eprintln!("    ✗ 失败: {e}");
                continue;
            }
        };

        // Calculate improvements
        let focus_pct = if controlled.focus_interventions > 0 {
            (controlled.focus_interventions as f32 - constrained.focus_interventions as f32)
                / controlled.focus_interventions as f32 * 100.0
        } else { 0.0 };

        let dev_pct = if controlled.avg_deviation > 0.001 {
            (controlled.avg_deviation - constrained.avg_deviation)
                / controlled.avg_deviation * 100.0
        } else { 0.0 };

        let trust_pct = (constrained.trust_score - controlled.trust_score) * 100.0;

        println!();
        println!("  ── 对比结果 ──");
        println!("  焦点对齐改善: {:+.1}%", focus_pct);
        println!("  偏差降低: {:+.1}%", dev_pct);
        println!("  信任提升: {:+.1}%", trust_pct);
        if config.semantic_mode {
            println!("  语义干预(实验组): {} | 平均语义偏差: {:.3}",
                constrained.semantic_interventions, constrained.avg_semantic_deviation);
        }
        println!();
    }

    println!("所有实验完成。使用 'xiang-exp report' 生成HTML报告。");
}

fn generate_report(name: &str, http_url: Option<String>, use_semantic: bool, deviation_alpha: f32) {
    let experiments: Vec<(String, ExperimentConfig)> = if let Some(url) = http_url {
        get_experiments(name).into_iter()
            .map(|(label, mut cfg)| {
                cfg = apply_semantic_config(cfg, use_semantic, deviation_alpha);
                cfg.http_server_url = Some(url.clone());
                cfg.eos_token_id = 0; // auto-detected via HttpBackend::discover_vocab()
                (label.to_string(), cfg)
            })
            .collect()
    } else {
        get_experiments(name).into_iter()
            .map(|(label, cfg)| {
                let cfg = apply_semantic_config(cfg, use_semantic, deviation_alpha);
                (label.to_string(), cfg)
            })
            .collect()
    };
    let mut reports = Vec::new();

    println!("生成报告...");
    for (_label, config) in &experiments {
        println!("  运行: {} ...", config.name);
        let report = ExperimentRunner::run_comparison(config);
        reports.push(report);
    }

    let html = generate_html_report(&reports);
    let path = "experiment_report.html";
    std::fs::write(path, &html).expect("无法写入报告文件");
    println!("报告已生成: {path}");
    println!("在浏览器中打开以查看详细对比结果。");
}

fn run_bench(http_url: Option<String>, use_model: bool, seed: u64, mock_mode: MockMode) {
    println!("=== 归藏项目 三易认知约束引擎 基准测试 ===");
    println!();

    let inputs = generate_benchmark_inputs(seed);
    println!("输入: 10主题 x 10变体 = {}组 (种子: {})", inputs.len(), seed);

    let base_config = if let Some(ref url) = http_url {
        println!("后端: HTTP远程服务器 ({url})");
        let mut cfg = focus_deviation_experiment();
        cfg.http_server_url = Some(url.clone());
        cfg.phase_constraint_mode = true;
        cfg.enable_three_engine = true;
        cfg.semantic_mode = true;
        cfg.deviation_alpha = 0.5;
        cfg.turns = 50;
        cfg.max_tokens = 80;
        cfg.agent_mode = true;
        cfg.agent_stop_patterns = vec![]; // benchmark 需要跑满 50 轮, 不清空轮次
        cfg.repetition_detection_enabled = true;
        cfg.repetition_threshold = 3;
        cfg.repetition_similarity_threshold = 0.85;
        cfg.repetition_window_size = 3;
        cfg
    } else if use_model {
        let all_args: Vec<String> = std::env::args().collect();
        let model_path = all_args
            .windows(2)
            .find(|w| w[0] == "--model")
            .map(|w| w[1].clone())
            .expect("缺少 --model <GGUF路径> 参数");
        println!("后端: 真实模型 ({})", model_path);
        let mut cfg = focus_deviation_experiment();
        cfg.model_path = Some(model_path);
        cfg.phase_constraint_mode = true;
        cfg.enable_three_engine = true;
        cfg.semantic_mode = true;
        cfg.deviation_alpha = 0.5;
        cfg.turns = 50;
        cfg.max_tokens = 80;
        cfg.agent_mode = true;
        cfg.agent_stop_patterns = vec![]; // benchmark 需要跑满 50 轮, 不清空轮次
        cfg.repetition_detection_enabled = true;
        cfg.repetition_threshold = 3;
        cfg.repetition_similarity_threshold = 0.85;
        cfg.repetition_window_size = 3;
        cfg
    } else {
        println!("后端: MockBackend ({:?})", mock_mode);
        let mut cfg = focus_deviation_experiment();
        cfg.mock_mode = mock_mode;
        cfg.http_server_url = None;
        cfg.model_path = None;
        cfg.phase_constraint_mode = true;
        cfg.enable_three_engine = true;
        cfg.semantic_mode = true;
        cfg.deviation_alpha = 0.5;
        cfg.turns = 50;
        cfg.max_tokens = 80;
        cfg.agent_mode = true;
        cfg.agent_stop_patterns = vec!["### DONE".into(), "任务完成".into(), "分析结束".into()];
        cfg.repetition_detection_enabled = true;
        cfg.repetition_threshold = 3;
        cfg.repetition_similarity_threshold = 0.85;
        cfg.repetition_window_size = 3;
        cfg
    };

    let num_trials = inputs.len().min(100);
    let inputs_used: Vec<String> = inputs.iter().take(num_trials).cloned().collect();
    eprintln!("试验次数: {}/{}组 (KV cache 新陈代谢, 50轮/试验, 每轮前清除)", num_trials, inputs.len());

    let bench_config = BenchmarkConfig {
        num_trials,
        inputs: inputs_used.clone(),
        base_config,
    };

    println!();
    println!("开始 {num_trials} 组对比基准测试...");
    println!("  条件A: 裸跑 (无三易控制)");
    println!("  条件B: 三易控制 (归藏+连山+周易)");
    println!();

    let start = std::time::Instant::now();

    match BenchmarkRunner::run(&bench_config) {
        Ok(results) => {
            let elapsed = start.elapsed();
            println!();
            println!("=== 测试完成 ({:.1}s) ===", elapsed.as_secs_f64());
            println!();

            let s = &results.summary;
            println!("┌──────────────┬─────────────────────┬─────────────────────┬──────────┬─────────┬──────────┬────────────┐");
            println!("│ 指标          │ 对照组 (裸跑)        │ 实验组 (三易控制)    │ t        │ p-value │ Cohen's d│ 显著       │");
            println!("├──────────────┼─────────────────────┼─────────────────────┼──────────┼─────────┼──────────┼────────────┤");

            print_bench_row("偏离率", &s.deviation_controlled, &s.deviation_constrained, &s.deviation_test);
            print_bench_row("结构合规率", &s.compliance_controlled, &s.compliance_constrained, &s.compliance_test);
            print_bench_row("完成率", &s.completion_controlled, &s.completion_constrained, &s.completion_test);

            println!("└──────────────┴─────────────────────┴─────────────────────┴──────────┴─────────┴──────────┴────────────┘");
            println!();

            if s.deviation_test.significant && s.compliance_test.significant && s.completion_test.significant {
                println!("结论: 三易认知约束引擎在所有核心指标上均表现出统计显著提升!");
            }
            println!("  {} {}", if s.deviation_test.significant { "[V]" } else { "[X]" }, s.deviation_test.interpretation);
            println!("  {} {}", if s.compliance_test.significant { "[V]" } else { "[X]" }, s.compliance_test.interpretation);
            println!("  {} {}", if s.completion_test.significant { "[V]" } else { "[X]" }, s.completion_test.interpretation);

            let html = generate_benchmark_html_report(&results);
            let path = "benchmark_report.html";
            std::fs::write(path, &html).expect("无法写入报告文件");
            println!();
            println!("详细报告已生成: {path}");

            // Export raw data as JSON
            let json = serde_json::to_string_pretty(&results).expect("JSON序列化失败");
            let json_path = "benchmark_raw_data.json";
            std::fs::write(json_path, &json).expect("无法写入JSON数据文件");
            println!("原始数据已导出: {json_path}");
            println!("在浏览器中打开以查看完整基准测试结果。");
        }
        Err(e) => {
            eprintln!("基准测试失败: {e}");
        }
    }
}

fn print_bench_row(
    name: &str,
    c_stats: &xiang_experiments::statistics::MetricStats,
    x_stats: &xiang_experiments::statistics::MetricStats,
    test: &xiang_experiments::statistics::TTestResult,
) {
    let sig = if test.significant { "V 极显著" } else { "-" };
    println!(
        "│ {:<12} │ {:.4}+-{:.4} (n={}) │ {:.4}+-{:.4} (n={}) │ {:>8.2} │ {:>7.4} │ {:>8.2} │ {:<10} │",
        name,
        c_stats.mean, c_stats.std_dev, c_stats.n,
        x_stats.mean, x_stats.std_dev, x_stats.n,
        test.t_statistic,
        test.p_value,
        test.cohens_d,
        sig,
    );
}
