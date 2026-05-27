/// XiangLang Thinking Guidance Engine — Experiment CLI
///
/// 路线A v3.1：全实验强制 Vulkan GPU 直接推理，不使用 HTTP 远程模式。
///
/// Usage:
///   xiang-exp run                            Run all experiments (mock backend)
///   xiang-exp run --model <GGUF>             Run with Vulkan GPU (llama.cpp FFI)
///   xiang-exp run --model <GGUF> --experiment <名称>  指定实验
///   xiang-exp report                         Generate HTML report (mock)
///   xiang-exp bench                          Run 100-trial benchmark (mock)
///   xiang-exp bench --model <GGUF>           100-trial benchmark (Vulkan GPU)
///   xiang-exp list                           List available experiments
///   xiang-exp routeb-bench --model <RouteB> --base-model <Base>  三组对照基准测试

use xiang_experiments::{
    ExperimentRunner, generate_html_report,
    focus_deviation_experiment, long_context_experiment, multi_turn_experiment,
    phase_constraint_experiment, unified_experiment, semantic_nav_experiment,
    ExperimentConfig, MockMode,
    benchmark::{BenchmarkRunner, BenchmarkConfig, run_threeway_bench},
    inputs::generate_benchmark_inputs,
    report::generate_benchmark_html_report,
    report::generate_threeway_benchmark_html_report,
    export_quality_evaluation_input,
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
    let no_prompt = parse_flag(&args, "--no-prompt");

    match args[1].as_str() {
        "run" => {
            let use_model = parse_flag(&args, "--model");
            let experiment = parse_flag_value(&args, "--experiment")
                .or_else(|| args.get(2).map(|s| s.as_str()))
                .unwrap_or("all");
            run_experiments(experiment, use_model, use_semantic, deviation_alpha, no_prompt);
        }
        "report" => {
            let experiment = args.get(2).map(|s| s.as_str()).unwrap_or("all");
            generate_report(experiment, use_semantic, deviation_alpha, no_prompt);
        }
        "list" => {
            list_experiments();
        }
        "bench" => {
            let use_model = parse_flag(&args, "--model");
            let seed = parse_flag_value(&args, "--seed")
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(42);
            let max_tokens_override = parse_flag_value(&args, "--max-tokens")
                .and_then(|s| s.parse::<u32>().ok());
            let quality_path = parse_flag_value(&args, "--load-quality")
                .map(|s| s.to_string());
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
            let no_prompt = parse_flag(&args, "--no-prompt");
            run_bench(use_model, seed, mock_mode, max_tokens_override, quality_path, no_prompt);
        }
        "rubric" => {
            print_rubric();
        }
        "routeb-bench" => {
            let routeb_model = parse_flag_value(&args, "--model")
                .expect("缺少 --model <路线B模型路径> 参数");
            let base_model = parse_flag_value(&args, "--base-model")
                .expect("缺少 --base-model <基础模型路径> 参数");
            let num_trials = parse_flag_value(&args, "--trials")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(3);
            run_routeb_bench(routeb_model, base_model, num_trials);
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
    println!("路线A v3.1：全实验 Vulkan GPU 直接推理（llama.cpp FFI, n_gpu_layers=-1）");
    println!();
    println!("用法:");
    println!("  xiang-exp run                    运行所有实验 (MockBackend 快速验证)");
    println!("  xiang-exp run --model <GGUF>     使用 Vulkan GPU 真实模型运行全部实验");
    println!("  xiang-exp run --model <GGUF> --experiment <名称>  运行指定实验");
    println!("  xiang-exp run --semantic          启用语义偏离度检测 (EmbeddingObserver)");
    println!("  xiang-exp run --no-prompt         无约束提示模式 (不注入约束全文)");
    println!("  xiang-exp run --semanticnav       语义导航实验 (向量汉字映射替代体系提示词)");
    println!("  xiang-exp report                 生成所有实验的HTML报告 (MockBackend)");
    println!("  xiang-exp list                   列出可用实验");
    println!("  xiang-exp bench                  运行100组基准测试 (MockBackend)");
    println!("  xiang-exp bench --model <GGUF>   使用 Vulkan GPU 真实模型基准测试");
    println!("  xiang-exp bench --mock <模式>    指定Mock模式（offfocus推荐）");
    println!("  xiang-exp bench --seed <数字>    指定随机种子（默认42）");
    println!("  xiang-exp bench --max-tokens <N> 覆盖每回合最大token数（默认1024）");
    println!("  xiang-exp bench --load-quality <JSON>  加载AI评估的质量评分并生成报告");
    println!("  xiang-exp bench --no-prompt       无约束提示模式 (不注入约束全文，仅语义导航)");
    println!("  xiang-exp rubric                 打印语义质量评估量规");
    println!();
    println!("实验列表:");
    println!("  focus        — 思维焦点: 偏移检测测试 (5轮, max_tokens=500)");
    println!("  longcontext  — 长上下文: 32K持续聚焦测试 (32轮, max_tokens=400)");
    println!("  multiturn    — 多轮对话: 50轮思维一致性测试 (50轮, max_tokens=400)");
    println!("  phase        — 阶段约束: 阴仪协议验证测试 (5轮, max_tokens=500)");
    println!("  unified      — 统一综合: 100任务全覆盖测试 (100轮, max_tokens=400)");
    println!("  semanticnav  — 语义导航: 向量汉字映射替代体系提示词测试 (4轮)");
    println!("  all          — 运行全部六个实验");
    println!();
    println!("GPU 加速说明:");
    println!("  使用 --model 时，自动调用 llama.cpp FFI, n_gpu_layers=-1 (全部卸载到 GPU)");
    println!("  模型通过 Vulkan 后端在 AMD RX 6650 XT (gfx1032) 上运行");
    println!("  要求 llama.cpp 编译时启用 -DGGML_VULKAN=ON");
    println!();
    println!("实验示例:");
    println!("  xiang-exp run --model ../models/qwen3.5-4b-q4_k_m.gguf");
    println!("  xiang-exp run --model ../models/qwen3.5-4b-q4_k_m.gguf --experiment focus --semantic");
    println!("  xiang-exp bench --model ../models/qwen3.5-4b-q4_k_m.gguf");
    println!("  xiang-exp bench --no-prompt  无约束提示基准测试 (纯语义导航)");
    println!("  xiang-exp run --experiment semanticnav  运行语义导航实验");
}

fn list_experiments() {
    println!("可用实验:");
    println!("  focus        — 思维焦点: 偏移检测测试");
    println!("  longcontext  — 长上下文: 32K持续聚焦测试");
    println!("  multiturn    — 多轮对话: 50轮思维连贯性测试");
    println!("  phase        — 阶段约束: 阴仪协议验证测试");
    println!("  unified      — 统一综合: 100任务全覆盖测试");
    println!("  semanticnav  — 语义导航: 向量汉字映射替代体系提示词测试");
    println!("  all          — 运行全部实验");
}

fn get_experiments(name: &str) -> Vec<(&'static str, ExperimentConfig)> {
    match name {
        "focus" => vec![("思维焦点—偏移检测", focus_deviation_experiment())],
        "longcontext" => vec![("长上下文—32K", long_context_experiment())],
        "multiturn" => vec![("多轮对话—50轮", multi_turn_experiment())],
        "phase" => vec![("阶段约束—协议验证", phase_constraint_experiment())],
        "unified" => vec![("统一综合—100任务", unified_experiment())],
        "semanticnav" => vec![("语义导航—向量汉字映射", semantic_nav_experiment())],
        _ => vec![
            ("思维焦点—偏移检测", focus_deviation_experiment()),
            ("长上下文—32K", long_context_experiment()),
            ("多轮对话—50轮", multi_turn_experiment()),
            ("阶段约束—协议验证", phase_constraint_experiment()),
            ("统一综合—100任务", unified_experiment()),
            ("语义导航—向量汉字映射", semantic_nav_experiment()),
        ],
    }
}

/// Apply Vulkan GPU model settings to a config.
fn apply_real_model(mut config: ExperimentConfig, model_path: &str) -> ExperimentConfig {
    config.model_path = Some(model_path.to_string());
    // EOS and token groups auto-detected from model vocabulary
    config.eos_token_id = 0;
    config.mock_mode = xiang_experiments::MockMode::Safe; // unused but valid
    config
}

/// Apply semantic mode settings to a config.
fn apply_semantic_config(mut config: ExperimentConfig, use_semantic: bool, alpha: f32, no_prompt: bool) -> ExperimentConfig {
    config.semantic_mode = use_semantic;
    config.deviation_alpha = alpha;
    if no_prompt {
        config.inject_constraint_prompt = false;
        config.semantic_nav_mode = true;
        config.semantic_mode = true;
    }
    config
}

fn run_experiments(name: &str, use_real_model: bool, use_semantic: bool, deviation_alpha: f32, no_prompt: bool) {
    let experiments: Vec<(String, ExperimentConfig)> = if use_real_model {
        // Vulkan GPU direct mode (llama.cpp FFI, n_gpu_layers=-1)
        let all_args: Vec<String> = std::env::args().collect();
        let model_path = all_args
            .windows(2)
            .find(|w| w[0] == "--model")
            .map(|w| w[1].clone())
            .expect("缺少 --model <GGUF路径> 参数");
        get_experiments(name).into_iter()
            .map(|(label, cfg)| {
                let cfg = apply_semantic_config(cfg, use_semantic, deviation_alpha, no_prompt);
                (format!("{} [Vulkan GPU]", label), apply_real_model(cfg, &model_path))
            })
            .collect()
    } else {
        // Mock backend mode (快速验证)
        get_experiments(name).into_iter()
            .map(|(label, cfg)| {
                let cfg = apply_semantic_config(cfg, use_semantic, deviation_alpha, no_prompt);
                (label.to_string(), cfg)
            })
            .collect()
    };

    let count = experiments.len();
    for (i, (label, config)) in experiments.iter().enumerate() {
        println!("[{}/{}] 实验: {}", i + 1, count, label);
        println!("  描述: {}", config.description);
        println!("  输入数: {}  轮数: {}  max_tokens: {}", config.inputs.len(), config.turns, config.max_tokens);
        if config.model_path.is_some() {
            println!("  后端: Vulkan GPU (llama.cpp FFI, 全部层卸载)");
            println!("  模型: {}", config.model_path.as_ref().unwrap());
        } else {
            println!("  后端: Mock ({:?})", config.mock_mode);
        }

        println!();
        println!("  [对照组] 仅约束体系全文...");

        // ── 预创建输出目录（启动后立即可见）──
        let input_text = config.inputs.first().map(|s| s.as_str()).unwrap_or("");
        let safe_label = label.replace(['/', '\\', ':', ' '], "_");
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let run_dir = std::path::PathBuf::from("experiment_runs")
            .join(format!("{}_{}", safe_label, ts));
        let ctrl_dir = run_dir.join("对照组");
        let const_dir = run_dir.join("实验组");

        // Write input.txt immediately
        std::fs::create_dir_all(&ctrl_dir).ok();
        std::fs::create_dir_all(&const_dir).ok();
        std::fs::write(run_dir.join("input.txt"), input_text).ok();
        println!("  输出目录: {} (增量写入中...)", run_dir.display());

        let controlled = match ExperimentRunner::run_controlled(config, Some(&ctrl_dir)) {
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
        println!("  [实验组] 体系全文 + 动态提示词 + 引擎引导...");
        let constrained = match ExperimentRunner::run_constrained(config, Some(&const_dir)) {
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

        println!("  轮次文件已保存: {}", run_dir.display());

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

fn generate_report(name: &str, use_semantic: bool, deviation_alpha: f32, no_prompt: bool) {
    let experiments: Vec<(String, ExperimentConfig)> = get_experiments(name).into_iter()
        .map(|(label, cfg)| {
            let cfg = apply_semantic_config(cfg, use_semantic, deviation_alpha, no_prompt);
            (label.to_string(), cfg)
        })
        .collect();

    let mut reports = Vec::new();

    println!("生成报告 (MockBackend)...");
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

fn run_bench(use_model: bool, seed: u64, mock_mode: MockMode, max_tokens_override: Option<u32>, quality_path: Option<String>, no_prompt: bool) {
    println!("=== 归藏项目 三易认知约束引擎 基准测试 ===");
    println!();

    // If --load-quality is provided, load it for later attachment
    let quality_report = match &quality_path {
        Some(path) => {
            println!("加载质量评估: {path}");
            match std::fs::read_to_string(path) {
                Ok(json) => match serde_json::from_str(&json) {
                    Ok(qr) => Some(qr),
                    Err(e) => { eprintln!("质量评估文件解析失败: {e}"); None }
                },
                Err(e) => { eprintln!("无法读取质量评估文件: {e}"); None }
            }
        }
        None => None,
    };

    let inputs = generate_benchmark_inputs(seed);
    println!("输入: 10主题 x 10变体 = {}组 (种子: {})", inputs.len(), seed);

    let mut base_config = if use_model {
        // Vulkan GPU direct mode
        let all_args: Vec<String> = std::env::args().collect();
        let model_path = all_args
            .windows(2)
            .find(|w| w[0] == "--model")
            .map(|w| w[1].clone())
            .expect("缺少 --model <GGUF路径> 参数");
        println!("后端: Vulkan GPU (llama.cpp FFI, n_gpu_layers=-1)");
        println!("模型: {model_path}");
        let mut cfg = focus_deviation_experiment();
        cfg.model_path = Some(model_path);
        cfg.phase_constraint_mode = true;
        cfg.enable_three_engine = true;
        cfg.semantic_mode = true;
        cfg.deviation_alpha = 0.5;
        cfg.turns = 10;
        cfg.max_tokens = 4096;
        cfg.agent_mode = true;
        cfg.agent_stop_patterns = vec!["### DONE".into(), "任务完成".into(), "分析结束".into()];
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
        cfg.turns = 10;
        cfg.max_tokens = 4096;
        cfg.agent_mode = true;
        cfg.agent_stop_patterns = vec!["### DONE".into(), "任务完成".into(), "分析结束".into()];
        cfg.repetition_detection_enabled = true;
        cfg.repetition_threshold = 3;
        cfg.repetition_similarity_threshold = 0.85;
        cfg.repetition_window_size = 3;
        cfg
    };

    // Apply --max-tokens override if provided
    if let Some(mt) = max_tokens_override {
        base_config.max_tokens = mt;
        eprintln!("max_tokens 覆盖: {}", mt);
    }

    // Apply --no-prompt: disable constraint prompt injection
    if no_prompt {
        base_config.inject_constraint_prompt = false;
        eprintln!("no-prompt 模式: 不注入约束提示词");
    }

    let num_trials = inputs.len().min(3); // FIXME: restore to 10 after verification
    let inputs_used: Vec<String> = inputs.iter().take(num_trials).cloned().collect();
    eprintln!("试验次数: {}/{}组 (KV cache 新陈代谢, 10轮/试验, 每轮前清除)", num_trials, inputs.len());

    let bench_config = BenchmarkConfig {
        num_trials,
        inputs: inputs_used.clone(),
        base_config,
    };

    println!();
    println!("开始 {num_trials} 组对比基准测试...");
    if no_prompt {
        println!("  [无约束提示模式] 仅依赖语义导航，不注入SAN_YI_CONSTRAINT_FULL_TEXT");
    }
    println!("  条件A: 裸跑 (无引擎引导)");
    println!("  条件B: 三易控制 (动态提示词 + 归藏/连山/周易)");
    println!();

    let start = std::time::Instant::now();

    match BenchmarkRunner::run(&bench_config) {
        Ok(mut results) => {
            // Attach quality report if loaded
            results.quality_report = quality_report;

            // Recompute summary with quality data if available
            if results.quality_report.is_some() {
                use xiang_experiments::statistics::compute_benchmark_summary;
                results.summary = compute_benchmark_summary(
                    &results.trials,
                    results.quality_report.as_ref(),
                );
            }

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

            // Export quality evaluation input for AI evaluator
            match export_quality_evaluation_input(&results, "benchmark_quality_input.json") {
                Ok(()) => println!("质量评估输入已导出: benchmark_quality_input.json"),
                Err(e) => eprintln!("质量评估输入导出失败: {e}"),
            }

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

fn run_routeb_bench(routeb_model: &str, base_model: &str, num_trials: usize) {
    println!("=== 路线B 三组对照基准测试 ===");
    println!();
    println!("路线B模型: {}", routeb_model);
    println!("基础模型:  {}", base_model);
    println!("试验次数:  {}", num_trials);

    let seed = 42u64;
    let inputs = generate_benchmark_inputs(seed);
    let max_tokens = 4096u32;

    let start = std::time::Instant::now();

    match run_threeway_bench(num_trials, inputs, routeb_model, base_model, max_tokens, seed) {
        Ok((yue_shu_results, wei_diao_results, yuan_sheng_results)) => {
            let elapsed = start.elapsed();

            // Generate three-way report
            let html = generate_threeway_benchmark_html_report(
                &yue_shu_results, &wei_diao_results, &yuan_sheng_results,
            );
            let report_path = "threeway_benchmark_report.html";
            std::fs::write(report_path, &html).expect("无法写入三组对照报告");
            println!("三组对照报告: {}", report_path);

            // Save raw data
            let data = serde_json::json!({
                "yue_shu": {
                    "model": yue_shu_results.model_name,
                    "num_trials": yue_shu_results.num_trials,
                    "mean_deviation": yue_shu_results.summary.deviation_constrained.mean,
                    "mean_compliance": yue_shu_results.summary.compliance_constrained.mean,
                    "mean_completion": yue_shu_results.summary.completion_constrained.mean,
                    "mean_trust": yue_shu_results.summary.trust_constrained.mean,
                },
                "wei_diao": {
                    "model": wei_diao_results.model_name,
                    "num_trials": wei_diao_results.num_trials,
                    "mean_deviation": wei_diao_results.summary.deviation_controlled.mean,
                    "mean_compliance": wei_diao_results.summary.compliance_controlled.mean,
                    "mean_completion": wei_diao_results.summary.completion_controlled.mean,
                    "mean_trust": wei_diao_results.summary.trust_controlled.mean,
                },
                "yuan_sheng": {
                    "model": yuan_sheng_results.model_name,
                    "num_trials": yuan_sheng_results.num_trials,
                    "mean_deviation": yuan_sheng_results.summary.deviation_controlled.mean,
                    "mean_compliance": yuan_sheng_results.summary.compliance_controlled.mean,
                    "mean_completion": yuan_sheng_results.summary.completion_controlled.mean,
                    "mean_trust": yuan_sheng_results.summary.trust_controlled.mean,
                }
            });
            let json_path = "threeway_benchmark_data.json";
            std::fs::write(json_path, serde_json::to_string_pretty(&data).unwrap())
                .expect("无法写入JSON数据");
            println!("原始数据:  {}", json_path);

            println!();
            println!("=== 测试完成 ({:.1}s) ===", elapsed.as_secs_f64());
            println!();

            // Print summary
            let ys = &yue_shu_results.summary;    // 约束组 → constrained fields
            let ws = &wei_diao_results.summary;    // 微调组 → controlled fields
            let ys2 = &yuan_sheng_results.summary;  // 原生组 → controlled fields
            println!("┌──────────────┬─────────────────────┬─────────────────────┬─────────────────────┐");
            println!("│ 指标          │ 约束组               │ 微调组               │ 原生组               │");
            println!("├──────────────┼─────────────────────┼─────────────────────┼─────────────────────┤");
            println!("│ 偏离率        │ {:.4} ± {:.4}    │ {:.4} ± {:.4}    │ {:.4} ± {:.4}    │",
                ys.deviation_constrained.mean, ys.deviation_constrained.std_dev,
                ws.deviation_controlled.mean, ws.deviation_controlled.std_dev,
                ys2.deviation_controlled.mean, ys2.deviation_controlled.std_dev);
            println!("│ 结构合规率     │ {:.4} ± {:.4}    │ {:.4} ± {:.4}    │ {:.4} ± {:.4}    │",
                ys.compliance_constrained.mean, ys.compliance_constrained.std_dev,
                ws.compliance_controlled.mean, ws.compliance_controlled.std_dev,
                ys2.compliance_controlled.mean, ys2.compliance_controlled.std_dev);
            println!("│ 完成率        │ {:.4} ± {:.4}    │ {:.4} ± {:.4}    │ {:.4} ± {:.4}    │",
                ys.completion_constrained.mean, ys.completion_constrained.std_dev,
                ws.completion_controlled.mean, ws.completion_controlled.std_dev,
                ys2.completion_controlled.mean, ys2.completion_controlled.std_dev);
            println!("└──────────────┴─────────────────────┴─────────────────────┴─────────────────────┘");
            println!();
            println!("详细报告: {report_path}");
        }
        Err(e) => {
            eprintln!("三组基准测试失败: {e}");
        }
    }
}

fn print_rubric() {
    println!("{}", xiang_experiments::QUALITY_EVALUATION_RUBRIC);
}
