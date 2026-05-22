/// Full pipeline integration test: llama.cpp + XiangLogitBias thinking guidance engine.
///
/// This example:
///   1. Loads Qwen3.5-4B via LlamaCppBackend
///   2. Creates a XiangLogitBias with configurable token groups
///   3. Runs the guided generation pipeline
///   4. Prints bias application log for verification
///
/// Usage:
///   MODEL=/root/xing/models/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf \
///   LD_LIBRARY_PATH=/root/xing/llama.cpp/build/bin:/root/xing/llama.cpp/build \
///   cargo run --example constrained -p xiang-llm

use std::env;

use xiang_llm::{
    LlmBackend, LlmContext, TemperatureMode,
};
use xiang_core::Gua;

fn main() {
    let model_path = env::var("MODEL").unwrap_or_else(|_| {
        "/root/xing/models/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf".to_string()
    });

    println!("╔══════════════════════════════════════════════╗");
    println!("║  XiangLang 引导引擎 — 全链路集成测试         ║");
    println!("╚══════════════════════════════════════════════╝");
    println!();

    // ── 1. Load model ──
    println!("[1/4] 加载模型...");
    let mut backend = xiang_llm::llama_backend::LlamaCppBackend::new(&model_path, 2048, 0)
        .expect("模型加载失败");
    println!("      模型: {}", backend.model_name());
    println!("      设备: {}", backend.device_info());
    println!("      就绪: {}", backend.is_ready());
    println!();

    // ── 2. Create LLM context ──
    let mut ctx = LlmContext::new("你是一个聚焦的AI助手。保持回答紧扣用户的核心问题，避免偏离主题。");

    // ── 3. Run guided generation with bias engine ──
    println!("[2/4] 运行引导生成...");

    // Simulate an off-focus state to trigger bias rules
    // D > 0.7 -> suppress off_focus tokens (-8), promote EOS (+4)
    // D > 0.95 -> force stop (emit EOS immediately)
    let deviation = 0.85;  // High deviation → EOS promotion
    let state = Gua::ORIGIN;  // Full state (0b111111 = 63)
    let sha_count = 0;

    // Token groups (empty for this test — bias engine will promote EOS)
    // For real deployment, scan vocabulary to populate these.
    let off_focus_tokens: Vec<u32> = vec![];
    let deviating_tokens: Vec<u32> = vec![];
    let eos_id = 248046u32;  // Qwen3.5 <|im_end|>

    let prompt = "请解释如何用Rust编写一个简单的HTTP服务器。";

    println!("      输入: {}", prompt);
    println!("      偏差: {:.2}", deviation);
    println!("      卦象: {}", state);
    println!("      脱焦Token: {}", off_focus_tokens.len());
    println!("      发散Token: {}", deviating_tokens.len());
    println!("      EOS Token: {}", eos_id);
    println!();

    let result = ctx.generate_constrained_turn(
        &mut backend, prompt, 50,
        TemperatureMode::Fixed(0.7),
        state, deviation, sha_count,
        off_focus_tokens, deviating_tokens, eos_id,
    ).expect("引导生成失败");

    println!("      输出: {}", result.text);
    println!("      Token数: {}", result.tokens_generated);
    println!("      停止原因: {:?}", result.stop_reason);
    println!("      Bias应用: {}", result.bias_applications);
    println!("      Token压制: {}", result.tokens_suppressed);
    println!();

    // ── 4. Now run with low deviation (no bias should trigger) ──
    println!("[3/4] 运行低偏差对照生成...");
    let deviation_low = 0.1;

    let result2 = ctx.generate_constrained_turn(
        &mut backend, "请解释什么是机器学习。", 50,
        TemperatureMode::Fixed(0.7),
        state, deviation_low, sha_count,
        vec![], vec![], 248046,
    ).expect("对照生成失败");

    println!("      输入: 请解释什么是机器学习。");
    println!("      偏差: {:.2}", deviation_low);
    println!("      输出: {}", result2.text);
    println!("      Token数: {}", result2.tokens_generated);
    println!("      停止原因: {:?}", result2.stop_reason);
    println!("      Bias应用: {}", result2.bias_applications);
    println!();

    // ── 5. Context stats ──
    println!("[4/4] 上下文统计:");
    println!("      总Token: {}", ctx.total_tokens);
    println!("      总调用: {}", ctx.total_calls);
    println!("      焦点干预: {}", ctx.focus_interventions);
    println!("      总压制: {}", ctx.total_suppressions);
    println!();

    println!("╔══════════════════════════════════════════════╗");
    println!("║  全链路集成测试完成!                          ║");
    println!("╚══════════════════════════════════════════════╝");
}
