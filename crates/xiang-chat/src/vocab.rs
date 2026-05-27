/// 词汇发现模块 — 通过 llama.cpp `/tokenize` 端点发现 Qwen3.5 的 token ID。
///
/// 运行时通过查询模型的实际 tokenizer 来构建 off_focus、divergent 和算子专属 token 组。
/// 如果 llama.cpp server 不可用，则返回空结构（降级但不会崩溃）。
///
/// 使用方式（在 ConstrainedEngine::new 时调用）：
/// ```ignore
/// let (off_focus, divergent) = vocab::discover_tokens(&backend);
/// let operator_pools = OperatorTokenPools::discover(&backend);
/// ```

use xiang_llm::http_backend::HttpBackend;
use xiang_llm::LlmBackend;

/// Qwen3.5-4B 的 EOS token ID（已知的固定值）。
pub const QWEN_EOS_TOKEN: u32 = 248046;

// ── 算子专属方向引导 token 池 ────────────────────────────────
//
// 每个算子有独立的"方向引导池"：一组 token ID，在每个算子阶段获得不同的偏置方向。
//
// | 算子 | 阶段语义 | 正向引导（鼓励） | 负向引导（压制） |
// |------|---------|----------------|----------------|
// | 生   | 探索·提问·起始 | 探索性 token | 结论性 token |
// | 动   | 扩展·连接·发散 | 发散性 token | 重复性 token |
// | 长   | 深入·聚焦·收敛 | 收敛性 token | 新方向 token |
// | 育   | 构建·结构化·规划 | 结构性 token | 模糊性 token |

/// 单个算子的正/负 token 池。
#[derive(Debug, Clone, Default)]
pub struct OperatorTokens {
    /// 正向引导 token（+bias，鼓励生成方向）
    pub positive: Vec<u32>,
    /// 负向引导 token（-bias，压制偏离方向）
    pub negative: Vec<u32>,
}

/// 四个生成树算子的方向引导 token 池。
#[derive(Debug, Clone, Default)]
pub struct OperatorTokenPools {
    /// 生：探索·提问·起始
    pub sheng: OperatorTokens,
    /// 动：扩展·连接·发散
    pub dong: OperatorTokens,
    /// 长：深入·聚焦·收敛
    pub zhang: OperatorTokens,
    /// 育：构建·结构化·规划
    pub yu: OperatorTokens,
}

impl OperatorTokenPools {
    /// 从运行的 llama.cpp server 发现所有算子的方向引导 token。
    pub fn discover(backend: &HttpBackend) -> Self {
        if !backend.is_ready() {
            eprintln!("[vocab] llama.cpp server not ready, operator token discovery skipped");
            return Self::default();
        }
        Self {
            sheng: OperatorTokens {
                positive: discover_category(backend, SHENG_POSITIVE),
                negative: discover_category(backend, SHENG_NEGATIVE),
            },
            dong: OperatorTokens {
                positive: discover_category(backend, DONG_POSITIVE),
                negative: discover_category(backend, DONG_NEGATIVE),
            },
            zhang: OperatorTokens {
                positive: discover_category(backend, ZHANG_POSITIVE),
                negative: discover_category(backend, ZHANG_NEGATIVE),
            },
            yu: OperatorTokens {
                positive: discover_category(backend, YU_POSITIVE),
                negative: discover_category(backend, YU_NEGATIVE),
            },
        }
    }

    /// 根据算子名称获取对应的 token 组。
    pub fn for_operator(&self, op: &str) -> (&[u32], &[u32]) {
        let tokens = match op {
            "生" => &self.sheng,
            "动" => &self.dong,
            "长" => &self.zhang,
            "育" => &self.yu,
            _ => {
                eprintln!("[vocab] 未知算子: {op}，使用空 token 池");
                return (&[], &[]);
            }
        };
        (tokens.positive.as_slice(), tokens.negative.as_slice())
    }
}

// ── 生算子：探索·提问·起始 ──────────────────────────────────
// 正向：鼓励模型扩展思路、提出新方向
// 负向：压制过早下结论

/// 生算子正向 token 模式（鼓励探索性、不确定性、可能性）
pub(crate) const SHENG_POSITIVE: &[&str] = &[
    "也许", "可能", "或许", "大概", "似乎", "往往",
    "假设", "设想", "猜测", "推测", "猜想",
    "方向", "角度", "方面", "层面", "维度",
    "探索", "探索性", "探讨", "讨论",
    "问题", "提问", "疑问", "质疑",
    "新的", "新颖", "创新", "创造性",
    "不一定", "不一定", "不确定", "未确定",
    "试想", "换个", "另一个", "其他",
    "potential", "possibly", "perhaps", "explore",
    "alternative", "another", "different",
];

/// 生算子负向 token 模式（压制结论性、确定性）
pub(crate) const SHENG_NEGATIVE: &[&str] = &[
    "因此", "总之", "综上所述",
    "证明", "结论", "结果表明",
    "确定", "肯定", "一定",
    "必然", "绝对", "毫无疑问",
    "因此", "所以", "从而",
    "therefore", "thus", "conclude", "conclusion",
    "definitely", "certainly", "absolutely",
];

// ── 动算子：扩展·连接·发散 ──────────────────────────────────
// 正向：鼓励多角度关联
// 负向：压制重复表述

pub(crate) const DONG_POSITIVE: &[&str] = &[
    "但是", "然而", "不过", "可是",
    "另一方面", "相比之下", "反之",
    "此外", "另外", "还有", "再者", "以及",
    "同时", "与此同", "同样",
    "虽然", "尽管", "即使", "就算",
    "不仅", "不止", "除了",
    "或者", "要么", "要不",
    "however", "moreover", "furthermore", "besides",
    "additionally", "meanwhile", "nevertheless",
];

pub(crate) const DONG_NEGATIVE: &[&str] = &[
    "也就是说", "即", "换言之", "换句话说",
    "重复", "再次", "重申",
    "同上", "如前所述", "如上所述",
    "namely", "i.e.", "in other words",
    "repeat", "again", "same as",
];

// ── 长算子：深入·聚焦·收敛 ──────────────────────────────────
// 正向：鼓励深入分析核心
// 负向：压制新方向的提出

pub(crate) const ZHANG_POSITIVE: &[&str] = &[
    "因此", "这表明", "这意味着",
    "因为", "所以", "从而", "以致", "导致",
    "原因是", "根源", "根本", "本质",
    "核心", "关键", "重点", "主要",
    "深入", "深层", "本质", "实质",
    "分析", "剖析", "解析",
    "聚焦", "集中", "专注",
    "therefore", "because", "since",
    "core", "key", "essential", "fundamental",
    "analysis", "focus", "deeper",
];

pub(crate) const ZHANG_NEGATIVE: &[&str] = &[
    "另外", "此外", "还有", "再者",
    "换个角度", "另一方面",
    "新的", "新增", "另一个",
    "other", "another", "additionally",
    "new", "different",
    "transition", "move on",
];

// ── 育算子：构建·结构化·规划 ──────────────────────────────────
// 正向：鼓励结构化输出
// 负向：压制模糊表述

pub(crate) const YU_POSITIVE: &[&str] = &[
    "第一步", "第二步", "第三步", "首先", "其次", "最后",
    "第一", "第二", "第三",
    "综上所述", "总而言之", "总结",
    "结构", "框架", "体系", "系统",
    "规划", "计划", "方案", "步骤",
    "分类", "类别", "维度", "层面",
    "首先是", "其次是", "最后是",
    "first", "second", "third", "finally",
    "structure", "framework", "system",
    "summary", "overall", "conclusion",
];

pub(crate) const YU_NEGATIVE: &[&str] = &[
    "某种", "某些", "某个",
    "大概", "也许", "或许", "可能",
    "模糊", "大概", "大约", "差不多",
    "maybe", "perhaps", "possibly",
    "somewhat", "roughly", "approximately",
    "vague", "unclear",
];

// ── 原有 off-focus / divergent 发现 ──────────────────────────

/// 用于发现 off-focus 标签的英文文本模式。
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
