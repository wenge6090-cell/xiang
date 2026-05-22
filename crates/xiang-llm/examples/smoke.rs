/// Quick smoke test for the llama.cpp backend.
///
/// Usage:
///   MODEL=/root/xing/models/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf \
///   LD_LIBRARY_PATH=/root/xing/llama.cpp/build/bin:/root/xing/llama.cpp/build \
///   cargo run --example smoke
use std::env;

use xiang_llm::{LlmBackend, GenerationParams, TemperatureMode};
use xiang_core::Gua;

fn main() {
    let model_path = env::var("MODEL").unwrap_or_else(|_| {
        "/root/xing/models/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf".to_string()
    });

    println!("Loading model: {}", model_path);

    let backend = xiang_llm::llama_backend::LlamaCppBackend::new(&model_path, 2048, 0);
    match backend {
        Ok(mut ll) => {
            println!("[OK] Model loaded!");
            println!("     Name: {}", ll.model_name());
            println!("     Device: {}", ll.device_info());
            println!("     Ready: {}", ll.is_ready());

            // Test tokenization
            let tokens = ll.tokenize("Hello, world!");
            println!("     Tokenize: 'Hello, world!' -> {} tokens", tokens.len());

            // Test generation
            let mut params = GenerationParams {
                system_prompt: "You are a helpful AI assistant.".into(),
                user_input: "Introduce yourself in one sentence.".into(),
                history: vec![],
                max_tokens: 50,
                temperature: TemperatureMode::Fixed(0.7),
                stop_sequences: vec!["</s>".into()],
                apply_focus_constraint: false,
                vm_state: Gua::ZERO,
                deviation: 0.0,
                logit_bias: None,
            };

            println!("     Generating...");
            match ll.generate(&mut params) {
                Ok(result) => {
                    println!("[OK] Generation OK! ({} tokens)", result.tokens_generated);
                    println!("     Output: {}", result.text);
                    println!("     Stop: {:?}", result.stop_reason);
                }
                Err(e) => {
                    eprintln!("[FAIL] Generation failed: {e}");
                }
            }
        }
        Err(e) => {
            eprintln!("[FAIL] Model load failed: {e}");
        }
    }
}
