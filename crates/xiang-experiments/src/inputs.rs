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

/// 10 multi-phase task templates, each demanding sustained multi-turn reasoning.
/// Each template explicitly structures the task into phases/steps/layers
/// that the model must complete sequentially across multiple turns.
/// Template: {topic} is replaced with the actual topic.
const TASK_TEMPLATES: [(&str, &str); 10] = [
    // 1. 迭代设计方案 — 5阶段
    ("迭代设计", "你是一位资深{topic}架构师。请完成以下迭代设计任务，每轮聚焦一个阶段：\n\
    阶段1：分析{topic}领域中一个常见系统的现状与2-3个核心痛点\n\
    阶段2：提出至少3种不同的解决方案，分别描述核心思路\n\
    阶段3：对每个方案从可行性、性能、维护性三维度打分并批判性分析\n\
    阶段4：基于批判结果优化最佳方案，形成最终设计\n\
    阶段5：评估最终方案的局限性和未来改进方向\n\n\
    请逐步完成。每轮完成一个阶段后再进入下一阶段。不要在单个阶段中预先跳到后续阶段。"),

    // 2. 多层深度分析 — 5层次
    ("多层分析", "请对{topic}进行一次从表到里的五层深度分析：\n\
    层次1-表面层：描述{topic}领域中广为人知的现象和常见做法\n\
    层次2-机制层：分析这些现象背后的运作机制和因果关系\n\
    层次3-原理层：挖掘更深层的第一性原理或理论基础\n\
    层次4-范式层：反思这些原理所依赖的假设和范式是否可能被颠覆\n\
    层次5-整合层：将各层分析整合，形成一个完整的认知框架\n\n\
    每轮深入一个层次。在进入下一层之前，请确保当前层分析已充分展开。"),

    // 3. 辩证对话 — 5轮交替
    ("辩证对话", "请以自我辩证的方式，从正方和反方两个角度交替深入分析{topic}：\n\
    第1轮-正方立论：从{topic}的主流观点出发，系统论证其合理性和有效性\n\
    第2轮-反方质疑：严格审视正方论点，提出反例、边界条件和潜在缺陷\n\
    第3轮-正方修正：吸收反方的合理质疑，修正或强化原有论点\n\
    第4轮-反方追问：更深层质疑修正后的观点，挖掘未触及的前提假设\n\
    第5轮-综合升华：整合正反双方的洞见，形成超越二元对立的辩证综合\n\n\
    每轮扮演一个角色，充分展开论证后再切换角色。"),

    // 4. 渐进式问题解决 — 6步骤
    ("问题解决", "在{topic}领域中，请采用系统化方法解决此核心挑战：如何在资源受限的情况下实现高性能和高可靠性？\n\
    步骤1-问题界定：精确定义问题的边界、约束条件和成功标准\n\
    步骤2-根因分析：使用5-Why等方法追溯问题的根本原因链\n\
    步骤3-方案发散：头脑风暴至少5种可能的解决路径\n\
    步骤4-方案收敛：构建评估矩阵，筛选出最优的2-3个候选方案\n\
    步骤5-详细设计：为最佳方案设计具体的实施架构和关键细节\n\
    步骤6-风险分析：识别实施过程中的关键风险和缓解策略\n\n\
    每轮完成一个步骤。确保当前步骤的输出质量达标后再推进。"),

    // 5. 多维对比框架 — 6维度
    ("对比框架", "请为{topic}构建一个系统的六维对比分析框架：\n\
    维度1-历史演进：梳理{topic}领域的关键里程碑和技术代际变迁\n\
    维度2-技术哲学：剖析不同流派的底层假设、价值观和范式分歧\n\
    维度3-工程实践：对比不同方案在真实场景中的表现差异和适用条件\n\
    维度4-经济学视角：分析不同方案的投入产出比和组织适配性\n\
    维度5-前瞻判断：评估各流派的未来发展方向和融合趋势\n\
    维度6-综合矩阵：整合前五个维度，构建完整的决策对比矩阵\n\n\
    每轮聚焦一个维度，进行充分论述后再切换维度。"),

    // 6. 自反批判循环 — 3轮×(提出→批判→改进) = 9步
    ("自反批判", "你将在{topic}领域中经历三轮\"提出→批判→改进\"循环：\n\
    第1轮：a)提出你对{topic}的一个核心论点或设计方案 b)从理论完备性、实践可行性、边界条件三个角度进行自我批判 c)基于批判给出改进版本\n\
    第2轮：a)对第1轮改进版本再次进行更深层的自我批判 b)重点关注前提假设是否合理、是否存在更优框架 c)给出二次改进版本\n\
    第3轮：a)对二次改进进行终极批判 b)审视是否还有未被考虑的维度、该方案的哲学局限 c)形成最终版本并标注其适用范围和固有局限\n\n\
    每一轮必须完整完成\"提出→批判→改进\"闭环后再进入下一轮。"),

    // 7. 系统思维建模 — 5阶段
    ("系统建模", "请用系统思维方法对{topic}进行五阶段建模分析：\n\
    阶段1-元素识别：识别{topic}系统中的所有关键元素（目标至少8个），描述每个元素的功能\n\
    阶段2-关系映射：分析元素之间的因果关系、正负反馈回路和延迟效应\n\
    阶段3-杠杆点分析：基于系统动力学，找出系统中影响力最高的干预点（至少3个）\n\
    阶段4-动态推演：描述系统在关键杠杆点被干预后至少2条不同的演化路径\n\
    阶段5-策略设计：基于系统分析，设计一个稳健的干预策略，说明其预期的系统行为变化\n\n\
    每轮完成一个阶段的分析后再推进。"),

    // 8. 知识体系构建 — 5步骤
    ("知识体系", "请为{topic}构建一个结构化的五层知识体系：\n\
    第1步-概念枚举：列出{topic}中10-15个核心概念或术语，给出简明定义\n\
    第2步-关系定义：为每对相关概念定义关系类型（因果、包含、对立、依赖、协同等）\n\
    第3步-层次结构：将概念组织为基础层→方法层→应用层→前沿层四个层次\n\
    第4步-交叉链接：识别跨层次的关键连接点，解释它们为何重要\n\
    第5步-知识地图：标注当前体系中的知识空白和活跃争议区域\n\n\
    每轮完成一步。在构建出足够稠密的概念网络后再进行总结。"),

    // 9. 未来场景推演 — 5场景
    ("场景推演", "假设当前是2026年，请为{topic}进行五场景推演（展望至2030年）：\n\
    场景1-乐观基线：技术突破顺利，描述{topic}的最佳发展路径，包含具体的标志性里程碑\n\
    场景2-悲观基线：遭遇重大瓶颈或负面事件，描述可能的危机链和发展受阻路径\n\
    场景3-黑天鹅：一个完全意外但影响深远的事件改变了{topic}的发展轨迹\n\
    场景4-线性外推：最可能的中庸发展路径，基于当前趋势的合理延伸\n\
    场景5-综合对策：基于前四个场景，设计一个稳健的适应性策略和早期预警信号\n\n\
    每轮完整推演一个场景，充分描述关键事件、因果链和最终状态后再切换。"),

    // 10. 第一性原理重构 — 5步骤
    ("原理重构", "请从第一性原理出发，彻底重构你对{topic}的理解：\n\
    步骤A-假设审查：列出{topic}中所有被视为\"理所当然\"的假设，逐一质疑其有效性和适用范围\n\
    步骤B-公理化：将{topic}的核心问题分解到不可再分的基本公理层面\n\
    步骤C-从零构建：仅基于步骤B得出的基本公理，重新推导{topic}的核心原理和架构\n\
    步骤D-差异分析：将重构结果与现有主流方法系统对比，标注每个差异点及其深层含义\n\
    步骤E-创新机会：基于重构中发现的\"未被利用的可能性\"，提出至少3个具体创新方向\n\n\
    每轮完成一个步骤。确保底层逻辑自洽且推演严格后再前进。"),
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
        let phase_count = inputs.iter().filter(|i| i.contains("阶段")).count();
        let layer_count = inputs.iter().filter(|i| i.contains("层次")).count();
        let step_count = inputs.iter().filter(|i| i.contains("步骤")).count();
        assert!(phase_count > 0, "应包含阶段类任务");
        assert!(layer_count > 0, "应包含层次类任务");
        assert!(step_count > 0, "应包含步骤类任务");
    }
}
