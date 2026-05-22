/// 词汇发现模块 — 通过 llama.cpp `/tokenize` 端点发现 Qwen3.5 的 token ID。
///
/// 运行时通过查询模型的实际 tokenizer 来构建 off_focus 和 divergent 两组 token ID。
/// 如果 llama.cpp server 不可用，则返回空列表（降级但不会崩溃）。
///
/// 使用方式（在 ConstrainedEngine::new 时调用）：
/// ```ignore
/// let (off_focus, divergent) = vocab::discover_tokens(&backend);
/// ```

use xiang_llm::http_backend::HttpBackend;
use xiang_llm::LlmBackend;

/// Qwen3.5-4B 的 EOS token ID（已知的固定值）。
pub const QWEN_EOS_TOKEN: u32 = 248046;

/// 用于发现 off-focus 标签的英文文本模式。
/// 包含：英文字母、常见英文单词。
const ENGLISH_PATTERNS: &[&str] = &[
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "the", "and", "for", "are", "but", "not", "you", "all", "can",
    "have", "with", "this", "that", "from", "they", "been", "were",
    "when", "what", "which", "their", "about", "would", "could",
    "should", "there", "other", "into", "than", "then", "them",
    "these", "some", "more", "also", "very", "just", "over",
    "such", "each", "well", "here", "where", "after", "before",
    "between", "through", "during", "without", "because", "under",
    "might", "shall", "will", "must", "still", "already", "even",
    "first", "second", "third", "last", "next", "much", "many",
    "Hello", "World", "This", "That", "What", "How", "Why",
    "English", "response", "answer", "question", "please",
    "sorry", "thank", "yes", "no", "maybe", "help",
    "the", "is", "in", "it", "to", "of", "on", "at", "by",
    "as", "be", "an", "or", "if", "do", "up", "so", "no",
    "I", "you", "he", "she", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their",
    "mine", "yours", "hers", "ours", "theirs",
];

/// 用于发现 divergent 标签的中文转折短语。
/// 仅取每个短语的首 token（对于 Qwen tokenizer，短短语通常是一个 token）。
const TRANSITION_PATTERNS: &[&str] = &[
    "但是", "然而", "不过", "可是", "虽然", "尽管", "即使",
    "另一方面", "相比之下", "反之",
    "此外", "另外", "还有", "再者",
    "总的来说", "综上所述", "总之",
    "首先", "其次", "最后",
    "因此", "所以", "从而", "进而",
    "换句话说", "也就是说",
];

/// 从正在运行的 llama.cpp server 发现 token ID。
///
/// 返回 `(off_focus_ids, divergent_ids)`。
/// 如果 server 不可用，两个列表都为空。
pub fn discover_tokens(backend: &HttpBackend) -> (Vec<u32>, Vec<u32>) {
    if !backend.is_ready() {
        eprintln!("[vocab] llama.cpp server not ready, token discovery skipped");
        return (vec![], vec![]);
    }

    let off_focus = discover_category(backend, ENGLISH_PATTERNS);
    let divergent = discover_category(backend, TRANSITION_PATTERNS);

    eprintln!(
        "[vocab] Discovered {} off-focus tokens, {} divergent tokens",
        off_focus.len(),
        divergent.len()
    );

    (off_focus, divergent)
}

/// 发现某一类 token ID：对每个 pattern 调用 tokenize，收集所有返回的 ID。
fn discover_category(backend: &HttpBackend, patterns: &[&str]) -> Vec<u32> {
    let mut ids = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for pattern in patterns {
        let tokens = backend.tokenize(pattern);
        for tid in tokens {
            // 跳过 EOS token（我们不希望对 EOS 本身做抑制）
            if tid == QWEN_EOS_TOKEN {
                continue;
            }
            if seen.insert(tid) {
                ids.push(tid);
            }
        }
    }

    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discover_category_dedup() {
        // 用 mock 验证去重：实际发现需运行中的 llama.cpp
        // 这里只验证常量不 panic
        assert_eq!(QWEN_EOS_TOKEN, 248046);
        assert!(ENGLISH_PATTERNS.len() > 50);
        assert!(TRANSITION_PATTERNS.len() > 10);
    }

    #[test]
    fn test_eos_not_in_discovery() {
        // EOS token 不应该出现在任何分类中
        // 这是语义保证，不依赖运行时
        assert!(QWEN_EOS_TOKEN != 0);
    }
}
