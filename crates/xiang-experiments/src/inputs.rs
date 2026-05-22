//! Deterministic benchmark input generator.
//!
//! Generates 100 diverse test prompts covering 10 topic domains × 10 variants.
//! Each variant exercises a different cognitive task type.
//! The generator is seeded for reproducibility.

/// 10 topic domains for benchmark inputs.
const TOPICS: [&str; 10] = [
    "AI认知架构",
    "机器学习基础",
    "自然语言处理",
    "分布式系统",
    "数据库设计",
    "软件架构",
    "网络安全",
    "算法优化",
    "前端开发",
    "项目管理",
];

/// 6 cognitive task types, each with prompt templates.
/// Template: {topic} is replaced with the actual topic.
const TASK_TEMPLATES: [(&str, &str); 10] = [
    // 解释/定义 (2 variants)
    ("解释", "请用简洁的语言解释{topic}的核心概念是什么？"),
    ("定义", "请给出{topic}中最重要的三个术语并分别定义。"),

    // 比较/对比 (2 variants)
    ("比较", "请对比{topic}中两种不同方法的优劣。哪种方法在什么场景下更适用？"),
    ("对比", "{topic}与传统方法相比有哪些本质区别？请列出关键差异。"),

    // 分析/拆解 (2 variants)
    ("分析", "请深入分析{topic}面临的主要挑战是什么？每个挑战的根本原因是什么？"),
    ("拆解", "请将{topic}的完整流程拆解为具体步骤，并说明每一步的关键要点。"),

    // 设计/规划 (2 variants)
    ("设计", "请为{topic}设计一套完整的实现方案。需要包含架构设计、核心组件和数据流。"),
    ("规划", "如果要从头开始学习{topic}，应该怎样规划学习路线？"),

    // 评估/判断 (1 variant)
    ("评估", "如何评估{topic}中一个方案的质量？请列出关键评估指标和评估方法。"),

    // 调试/修复 (1 variant)
    ("调试", "在{topic}的实践中遇到一个常见问题：输出结果不符合预期。请给出系统的排查思路。"),
];

/// Generate 100 deterministic benchmark inputs.
///
/// Each input is a unique combination of a topic and a task template.
/// The seed controls the pairing order for reproducibility.
pub fn generate_benchmark_inputs(seed: u64) -> Vec<String> {
    // Create 100 pairings: 10 topics × 10 task variants
    let mut inputs = Vec::with_capacity(100);

    for (_topic_idx, topic) in TOPICS.iter().enumerate() {
        for (_task_idx, (_task_name, template)) in TASK_TEMPLATES.iter().enumerate() {
            let prompt = template.replace("{topic}", topic);
            inputs.push(prompt);
        }
    }

    // Deterministic shuffle using seed
    shuffle_with_seed(&mut inputs, seed);

    inputs
}

/// Deterministic Fisher-Yates shuffle with a seeded PRNG (LCG).
fn shuffle_with_seed<T>(items: &mut [T], seed: u64) {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let n = items.len();

    for i in (1..n).rev() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (state >> 33) as usize % (i + 1);
        items.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generates_100_inputs() {
        let inputs = generate_benchmark_inputs(42);
        assert_eq!(inputs.len(), 100, "应该生成恰好100个输入");
    }

    #[test]
    fn test_all_inputs_non_empty() {
        let inputs = generate_benchmark_inputs(42);
        for (i, input) in inputs.iter().enumerate() {
            assert!(!input.is_empty(), "第{i}个输入为空");
            assert!(input.len() >= 10, "第{i}个输入过短: {input}");
        }
    }

    #[test]
    fn test_all_inputs_unique() {
        let inputs = generate_benchmark_inputs(42);
        let mut sorted = inputs.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 100, "所有输入应唯一");
    }

    #[test]
    fn test_deterministic_seed() {
        let a = generate_benchmark_inputs(42);
        let b = generate_benchmark_inputs(42);
        assert_eq!(a, b, "相同种子应产生相同输出");
    }

    #[test]
    fn test_different_seeds_different() {
        let a = generate_benchmark_inputs(42);
        let b = generate_benchmark_inputs(99);
        // Very unlikely to be identical
        assert_ne!(a, b, "不同种子应产生不同输出");
    }

    #[test]
    fn test_covers_all_topics() {
        let inputs = generate_benchmark_inputs(42);
        for topic in TOPICS.iter() {
            let count = inputs.iter().filter(|i| i.contains(topic)).count();
            assert!(count > 0, "主题 {topic} 应至少出现一次");
        }
    }

    #[test]
    fn test_contains_varied_task_types() {
        let inputs = generate_benchmark_inputs(42);
        let explain_count = inputs.iter().filter(|i| i.contains("解释")).count();
        let design_count = inputs.iter().filter(|i| i.contains("设计")).count();
        let debug_count = inputs.iter().filter(|i| i.contains("排查思路")).count();
        assert!(explain_count > 0, "应包含解释类任务");
        assert!(design_count > 0, "应包含设计类任务");
        assert!(debug_count > 0, "应包含调试类任务");
    }
}
