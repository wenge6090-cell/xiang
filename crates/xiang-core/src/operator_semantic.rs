/// 算子语义锚点 —— 语义导航的第二层核心组件。
///
/// 每个算子（生/动/长/育）在 embedding 空间预定义一个语义重心（centroid），
/// 由 12 个锚定汉字的 L2 归一化嵌入向量的均值构成。
///
/// `classify_operator_phase()` 将 LLM 输出 embedding 分类到最接近的算子语义阶段，
/// 用于判断模型当前输出更接近哪个算子的语义域，从而驱动算子推进/回滚决策。
///
/// ## 与体系提示词的关系
///
/// 体系提示词告诉模型"你现在应该做生"，但模型可能因容量不足无法执行。
/// 语义分类器改为观察模型实际做了什么，然后决定算子是否推进：
/// - 模型输出匹配"生"语义 → 推进到"动"
/// - 模型输出匹配"动"语义但当前算子是"生" → 说明模型跳过了，rollback 到"生"
use crate::hanzi_map::HanziMap;
use crate::embedding::cosine_similarity;

/// 算子语义锚点 —— 每个算子在 embedding 空间的语义重心。
#[derive(Debug, Clone)]
pub struct OperatorSemanticAnchor {
    /// 算子名称（"生"/"动"/"长"/"育"）
    pub operator: &'static str,
    /// 预计算的语义重心向量（L2 归一化）
    pub centroid: Vec<f32>,
}

// ── 算子锚定汉字定义 ──────────────────────────────────────────
//
// 基于 YinProtocol 的语义域定义每个算子的 12 个锚定汉字。
// 所有汉字均来自 HANZI_TABLE（308 核心字表）。

/// 生算子锚定汉字：探索·提问·起始
pub const SHENG_ANCHORS: &[char] = &[
    '\u{5F00}', // 开
    '\u{751F}', // 生
    '\u{95EE}', // 问
    '\u{51FA}', // 出
    '\u{5165}', // 入
    '\u{65E5}', // 日
    '\u{5BDF}', // 察
    '\u{89C2}', // 观
    '\u{601D}', // 思
    '\u{60F3}', // 想
    '\u{5B66}', // 学
    '\u{521D}', // 初
];

/// 动算子锚定汉字：扩展·连接·发散
pub const DONG_ANCHORS: &[char] = &[
    '\u{52A8}', // 动
    '\u{884C}', // 行
    '\u{901A}', // 通
    '\u{8FBE}', // 达
    '\u{8FDE}', // 连
    '\u{4EA4}', // 交
    '\u{5408}', // 合
    '\u{53D8}', // 变
    '\u{5316}', // 化
    '\u{8D70}', // 走
    '\u{5347}', // 升
    '\u{5206}', // 分
];

/// 长算子锚定汉字：深入·聚焦·收敛
pub const ZHANG_ANCHORS: &[char] = &[
    '\u{957F}', // 长
    '\u{6DF1}', // 深
    '\u{9AD8}', // 高
    '\u{5927}', // 大
    '\u{5F3A}', // 强
    '\u{539A}', // 厚
    '\u{5185}', // 内
    '\u{4E2D}', // 中
    '\u{5BDF}', // 察
    '\u{77E5}', // 知
    '\u{609F}', // 悟
    '\u{6B62}', // 止
];

/// 育算子锚定汉字：构建·结构化·规划
pub const YU_ANCHORS: &[char] = &[
    '\u{80B2}', // 育
    '\u{6210}', // 成
    '\u{5B89}', // 安
    '\u{5408}', // 合
    '\u{8BA1}', // 计
    '\u{7B97}', // 算
    '\u{5B58}', // 存
    '\u{5B88}', // 守
    '\u{6539}', // 改
    '\u{6559}', // 教
    '\u{5B66}', // 学
    '\u{4E60}', // 习
];

/// 从 HanziMap 构建四个算子的语义锚点。
///
/// 对每个算子的锚定汉字：
/// 1. 从 HanziMap 查找每个汉字的 L2 归一化嵌入向量
/// 2. 对所有锚定汉字的嵌入取均值
/// 3. L2 归一化均值得到该算子的 centroid
///
/// 如果某汉字不在 HanziMap 中，跳过（不影响 centroid 计算）。
pub fn build_operator_anchors(map: &HanziMap) -> Vec<OperatorSemanticAnchor> {
    let operators: &[(&str, &[char])] = &[
        ("\u{751F}", SHENG_ANCHORS),   // 生
        ("\u{52A8}", DONG_ANCHORS),    // 动
        ("\u{957F}", ZHANG_ANCHORS),   // 长
        ("\u{80B2}", YU_ANCHORS),      // 育
    ];

    operators
        .iter()
        .map(|(name, chars)| {
            let centroid = compute_centroid(map, chars);
            OperatorSemanticAnchor {
                operator: name,
                centroid,
            }
        })
        .collect()
}

/// 计算一组汉字嵌入的 L2 归一化均值（centroid）。
fn compute_centroid(map: &HanziMap, chars: &[char]) -> Vec<f32> {
    let mut sum: Vec<f32> = Vec::new();
    let mut count = 0usize;

    for &ch in chars {
        if let Some(emb) = map.embedding_of(ch) {
            if sum.is_empty() {
                sum = emb.to_vec();
            } else {
                for (s, &e) in sum.iter_mut().zip(emb.iter()) {
                    *s += e;
                }
            }
            count += 1;
        }
    }

    if count == 0 || sum.is_empty() {
        return Vec::new();
    }

    // 均值
    let inv = 1.0 / count as f32;
    for x in &mut sum {
        *x *= inv;
    }

    // L2 归一化
    l2_normalize(&mut sum);
    sum
}

fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// 将 LLM 输出 embedding 分类到最接近的算子语义阶段。
///
/// 对每个算子的 centroid 计算余弦相似度，返回相似度最高的算子名称。
/// 如果所有相似度都低于 `min_similarity`，返回 `None`（无意义匹配）。
pub fn classify_operator_phase(
    embedding: &[f32],
    anchors: &[OperatorSemanticAnchor],
    min_similarity: f32,
) -> Option<&'static str> {
    if anchors.is_empty() || embedding.is_empty() {
        return None;
    }

    let mut emb_norm = embedding.to_vec();
    l2_normalize(&mut emb_norm);

    let mut best_op: Option<&str> = None;
    let mut best_sim = min_similarity;

    for anchor in anchors {
        if anchor.centroid.len() != emb_norm.len() {
            continue;
        }
        let sim = cosine_similarity(&emb_norm, &anchor.centroid);
        if sim > best_sim {
            best_sim = sim;
            best_op = Some(anchor.operator);
        }
    }

    best_op
}

/// 对每个算子计算余弦相似度，返回完整结果列表（按相似度降序）。
pub fn classify_with_scores(
    embedding: &[f32],
    anchors: &[OperatorSemanticAnchor],
) -> Vec<(&'static str, f32)> {
    if anchors.is_empty() || embedding.is_empty() {
        return Vec::new();
    }

    let mut emb_norm = embedding.to_vec();
    l2_normalize(&mut emb_norm);

    let mut results: Vec<(&str, f32)> = anchors
        .iter()
        .filter(|a| a.centroid.len() == emb_norm.len() && !a.centroid.is_empty())
        .map(|a| {
            let sim = cosine_similarity(&emb_norm, &a.centroid);
            (a.operator, sim)
        })
        .collect();

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hanzi_map::HanziMap;

    /// Build a test HanziMap using anchor characters that exist in HANZI_TABLE.
    fn test_map() -> HanziMap {
        let chars: Vec<char> = vec![
            '\u{5F00}', '\u{751F}', '\u{95EE}', '\u{51FA}', '\u{5165}', '\u{65E5}',
            '\u{5BDF}', '\u{89C2}', '\u{601D}', '\u{60F3}', '\u{5B66}', '\u{521D}',
            '\u{52A8}', '\u{884C}', '\u{901A}', '\u{8FBE}', '\u{8FDE}', '\u{4EA4}',
            '\u{5408}', '\u{53D8}', '\u{5316}', '\u{8D70}', '\u{5347}', '\u{5206}',
            '\u{957F}', '\u{6DF1}', '\u{9AD8}', '\u{5927}', '\u{5185}', '\u{4E2D}',
            '\u{80B2}', '\u{6210}', '\u{5B89}', '\u{8BA1}', '\u{7B97}',
        ];
        let n = chars.len();
        let embeddings: Vec<Vec<f32>> = chars
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let mut v = vec![0.0f32; n];
                v[i] = 1.0;
                v
            })
            .collect();
        HanziMap::from_embeddings(chars, embeddings)
    }

    #[test]
    fn test_build_anchors() {
        let map = test_map();
        let anchors = build_operator_anchors(&map);
        assert_eq!(anchors.len(), 4);
        for a in &anchors {
            assert!(!a.centroid.is_empty(), "{:?} centroid should not be empty", a.operator);
            let norm: f32 = a.centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-4, "{:?} centroid L2 norm={}", a.operator, norm);
        }
    }

    #[test]
    fn test_classify_sheng() {
        let map = test_map();
        let anchors = build_operator_anchors(&map);
        let emb = map.embedding_of('\u{5F00}').unwrap().to_vec(); // 开
        let result = classify_operator_phase(&emb, &anchors, 0.1);
        assert_eq!(result, Some("\u{751F}")); // 生
    }

    #[test]
    fn test_classify_below_threshold() {
        let map = test_map();
        let anchors = build_operator_anchors(&map);
        let mut emb = vec![1.0f32; anchors[0].centroid.len()];
        l2_normalize(&mut emb);
        let result = classify_operator_phase(&emb, &anchors, 0.7);
        assert!(result.is_none(), "uniform vector should not match any operator, got {:?}", result);
    }

    #[test]
    fn test_classify_with_scores() {
        let map = test_map();
        let anchors = build_operator_anchors(&map);
        let emb = map.embedding_of('\u{751F}').unwrap().to_vec(); // 生
        let scores = classify_with_scores(&emb, &anchors);
        assert!(!scores.is_empty());
        assert_eq!(scores[0].0, "\u{751F}"); // 生
    }
}
