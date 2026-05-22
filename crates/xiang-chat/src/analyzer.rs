/// 输出质量分析器 — 从生成的文本计算 semantic deviation 反馈信号。
///
/// 在无提示词注入的前提下，通过分析 LLM 实际输出文本的统计特征，
/// 计算一个 `[0, 1]` 的偏离度分数，反馈给 CangVM 的 hybrid_deviation。

/// 需要检测的中文转折/话题跳跃短语。
pub const TRANSITION_PHRASES: &[&str] = &[
    "但是", "然而", "不过", "可是", "虽然", "尽管", "即使",
    "另一方面", "相比之下", "与之相反", "反之", "反过来说",
    "此外", "另外", "还有", "再者", "加之", "况且",
    "换句话说", "也就是说", "即", "换言之", "简而言之",
    "总的来说", "综上所述", "总之", "总而言之", "总体而言",
    "首先", "其次", "再次", "最后", "第一", "第二", "第三",
    "如果", "那么", "因此", "所以", "从而", "进而", "以至于",
    "虽然", "不仅", "因为", "与其", "不如",
];

/// 期望输出长度范围（字符数）。
const EXPECTED_MIN_CHARS: usize = 20;
const EXPECTED_MAX_CHARS: usize = 500;

/// 计算输出文本的语义偏离度。
///
/// 返回值范围 `[0, 1]`：
/// - 0.0 = 完美聚焦的中文输出
/// - 1.0 = 完全偏离（英文为主/杂乱无章）
///
/// 指标权重：
/// - english_ratio (0.40): ASCII 字母占比
/// - transition_density (0.30): 转折短语密度
/// - repetition_ratio (0.15): 重复度
/// - length_anomaly (0.15): 长度异常
pub fn compute_output_deviation(output: &str) -> f32 {
    if output.is_empty() {
        return 0.0;
    }

    let er = english_ratio(output);
    let td = transition_density(output);
    let rr = repetition_ratio(output);
    let la = length_anomaly(output);

    let score = 0.40 * er + 0.30 * td + 0.15 * rr + 0.15 * la;
    score.clamp(0.0, 1.0)
}

/// ASCII 英文字母占比。
/// 中文模型中正常值应 < 5%，> 20% 表示严重偏离。
fn english_ratio(text: &str) -> f32 {
    let total = text.chars().count();
    if total == 0 {
        return 0.0;
    }
    let english_chars = text.chars().filter(|c| c.is_ascii_alphabetic()).count();
    english_chars as f32 / total as f32
}

/// 转折短语密度（每 100 字符的出现次数）。
/// 正常段落 1-2 次/100字，>3 次表示话题跳跃。
fn transition_density(text: &str) -> f32 {
    if text.is_empty() {
        return 0.0;
    }
    let total_chars = text.chars().count();
    let count: usize = TRANSITION_PHRASES
        .iter()
        .map(|t| text.matches(t).count())
        .sum();
    // 密度 = 次数 / 100字
    let density = count as f32 / total_chars as f32 * 100.0;
    // 映射到 [0, 1]：密度 0 → 0, 密度 5+ → 1
    (density / 5.0).clamp(0.0, 1.0)
}

/// 重复度：1 - (唯一 bigram 数 / 总 bigram 数)。
/// 高重复 → 输出质量差 → 偏离。
fn repetition_ratio(text: &str) -> f32 {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < 4 {
        return 0.0;
    }
    let total_bigrams = chars.len() - 1;
    let mut unique_bigrams = std::collections::HashSet::new();
    for pair in chars.windows(2) {
        unique_bigrams.insert(pair);
    }
    let uniqueness = unique_bigrams.len() as f32 / total_bigrams as f32;
    // 返回"不唯一"的比例：低 uniqueness → 高重复 → 高偏离
    1.0 - uniqueness
}

/// 长度异常度：输出长度偏离期望范围的程度。
/// 过短（EOS 过早）或过长（漫谈）都视为偏离。
fn length_anomaly(text: &str) -> f32 {
    let len = text.chars().count();
    if len < EXPECTED_MIN_CHARS {
        // 太短：线性偏离，20字时 0，0字时 1.0
        (EXPECTED_MIN_CHARS - len) as f32 / EXPECTED_MIN_CHARS as f32
    } else if len > EXPECTED_MAX_CHARS {
        // 太长：线性偏离，500字时 0，1000字时 1.0
        (len - EXPECTED_MAX_CHARS) as f32 / EXPECTED_MAX_CHARS as f32
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_english_ratio() {
        assert_eq!(english_ratio("你好世界"), 0.0);
        assert_eq!(english_ratio("hello"), 1.0);
        let mixed = english_ratio("你好hello世界");
        assert!(mixed > 0.0 && mixed < 1.0);
    }

    #[test]
    fn test_transition_density_zero() {
        let text = "这是一个非常好的结果。我们继续深入分析。";
        let td = transition_density(text);
        assert!(td <= 0.01, "no transitions found, density should be near 0");
    }

    #[test]
    fn test_transition_density_high() {
        let text = "但是另一方面，然而不过虽然尽管，此外另外还有。";
        let td = transition_density(text);
        assert!(td > 0.1, "high density expected, got {}", td);
    }

    #[test]
    fn test_repetition_ratio() {
        let unique = "这是一段没有重复的文字。";
        let repetitive = "重复重复重复重复重复重复。";
        assert!(repetition_ratio(repetitive) > repetition_ratio(unique));
    }

    #[test]
    fn test_length_anomaly() {
        assert!(length_anomaly("短") > 0.5, "very short text should have high anomaly");
        assert_eq!(length_anomaly(&"中".repeat(100)), 0.0, "normal length should be 0");
        assert!(length_anomaly(&"长".repeat(800)) > 0.5, "very long text should have high anomaly");
    }

    #[test]
    fn test_compute_output_deviation() {
        // Pure Chinese, coherent, normal length → low deviation
        let good = "我觉得这个问题需要从多个角度来分析。首先考虑技术可行性，其次是成本效益。";
        let good_dev = compute_output_deviation(good);
        assert!(good_dev < 0.5, "good Chinese output should have dev < 0.5, got {}", good_dev);

        // Pure English → high deviation
        let bad = "This is a completely off-topic response in English that has nothing to do with the question. Let me talk about something completely different.";
        let bad_dev = compute_output_deviation(bad);
        assert!(bad_dev > 0.3, "English output should have high deviation, got {}", bad_dev);
    }

    #[test]
    fn test_empty_output() {
        assert_eq!(compute_output_deviation(""), 0.0);
    }
}
