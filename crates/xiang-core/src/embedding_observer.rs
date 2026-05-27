/// EmbeddingObserver — LLM 输出嵌入到汉字的语义观察器。
///
/// 这是**归**算子的核心：在连续向量空间（LLM 的嵌入输出）与离散符号空间
/// （汉字）的交界处，持续观察每个 token 的语义指纹变化，计算语义偏离度。
///
/// ## 架构位置
///
/// ```text
/// LLM decode → embedding → [EmbeddingObserver.map_single()] → 汉字 → 语义指纹
///                                      ↑
///                               HanziMap (余弦相似度)
/// ```
///
/// ## 与归的关系
///
/// 归是八气中唯一不产生修改的算子——纯粹的观察。EmbeddingObserver 实现了这种
/// 观察：将连续的嵌入向量映射为离散汉字，形成可追踪的"语义指纹"序列。
/// 偏离度 = 当前嵌入与初始锚点嵌入的余弦距离。
use crate::hanzi_map::{HanziMap, HanziMapping};

/// 语义观察记录。
#[derive(Debug, Clone)]
pub struct Observation {
    /// 映射到的汉字
    pub ch: char,
    /// 余弦相似度
    pub similarity: f32,
    /// 相对于初始锚点的余弦偏离度
    pub deviation_from_origin: Option<f32>,
    /// 序列位置
    pub position: usize,
}

/// 嵌入观察器。
///
/// 维护一个语义指纹序列（映射后的汉字序列），持续追踪 LLM 输出的语义漂移。
pub struct EmbeddingObserver {
    /// 汉字映射器
    map: HanziMap,
    /// 初始锚点嵌入（设置后用于计算偏离度）
    origin_embedding: Option<Vec<f32>>,
    /// 语义指纹：映射后的汉字序列
    fingerprint: Vec<char>,
    /// 观察历史
    observations: Vec<Observation>,
    /// 当前语义偏离度 [-1, 1] → 归一化到 [0, 1]
    /// 0 = 完全一致, 1 = 完全偏离
    semantic_deviation: f32,
    /// 相似度阈值：低于此值的映射被视为"弱锚定"
    similarity_threshold: f32,
}

impl EmbeddingObserver {
    /// 创建新的观察器。
    pub fn new(map: HanziMap) -> Self {
        EmbeddingObserver {
            map,
            origin_embedding: None,
            fingerprint: Vec::new(),
            observations: Vec::new(),
            semantic_deviation: 0.0,
            similarity_threshold: 0.3,
        }
    }

    /// 设置相似度阈值（默认 0.3）。
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.similarity_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// 设置初始锚点嵌入。
    ///
    /// 之后每次 `observe()` 都会计算与锚点的余弦偏离度。
    /// 通常用第一轮生成的第一个 token 嵌入作为锚点。
    pub fn set_origin(&mut self, embedding: &[f32]) {
        self.origin_embedding = Some(embedding.to_vec());
    }

    /// 观察一个嵌入向量，返回映射结果。
    ///
    /// - 将嵌入映射为最近的汉字
    /// - 记录到语义指纹中
    /// - 如果已设置锚点，计算偏离度
    pub fn observe(&mut self, embedding: &[f32]) -> Option<Observation> {
        let mapping = self.map.map_single(embedding)?;
        let position = self.fingerprint.len();

        // 计算与锚点的偏离度
        let deviation_from_origin = self.origin_embedding.as_ref().map(|origin| {
            let sim = crate::embedding::cosine_similarity(embedding, origin);
            // 余弦相似度 [-1, 1] → 偏离度 [0, 1]
            // sim=1.0 → dev=0.0 (完全一致)
            // sim=-1.0 → dev=1.0 (完全偏离)
            (1.0 - sim) / 2.0
        });

        let obs = Observation {
            ch: mapping.ch,
            similarity: mapping.similarity,
            deviation_from_origin,
            position,
        };

        self.fingerprint.push(mapping.ch);
        self.observations.push(obs.clone());

        // 更新当前偏离度
        if let Some(dev) = deviation_from_origin {
            self.semantic_deviation = dev;
        }

        Some(obs)
    }

    /// 不记录地查一次映射（用于预览，不影响指纹序列）。
    pub fn peek(&self, embedding: &[f32]) -> Option<HanziMapping> {
        self.map.map_single(embedding)
    }

    /// 当前语义偏离度 [0, 1]。
    pub fn semantic_deviation(&self) -> f32 {
        self.semantic_deviation
    }

    /// 语义指纹：映射后的汉字序列。
    pub fn fingerprint(&self) -> &[char] {
        &self.fingerprint
    }

    /// 最近的 N 个汉字指纹。
    pub fn recent_fingerprint(&self, n: usize) -> &[char] {
        let start = self.fingerprint.len().saturating_sub(n);
        &self.fingerprint[start..]
    }

    /// 观察历史。
    pub fn observations(&self) -> &[Observation] {
        &self.observations
    }

    /// 指纹中弱锚定的比例（相似度 < 阈值的映射）。
    pub fn weak_anchor_ratio(&self) -> f32 {
        if self.observations.is_empty() {
            return 0.0;
        }
        let weak_count = self.observations.iter()
            .filter(|o| o.similarity < self.similarity_threshold)
            .count();
        weak_count as f32 / self.observations.len() as f32
    }

    /// 重置观察器（清空指纹和历史，保留锚点）。
    pub fn reset(&mut self) {
        self.fingerprint.clear();
        self.observations.clear();
        self.semantic_deviation = 0.0;
    }

    /// 完全重置（包括锚点）。
    pub fn reset_all(&mut self) {
        self.reset();
        self.origin_embedding = None;
    }

    /// 观察器配置状态。
    pub fn is_ready(&self) -> bool {
        self.map.is_loaded()
    }

    /// 是否有锚点。
    pub fn has_origin(&self) -> bool {
        self.origin_embedding.is_some()
    }

    /// 指纹长度。
    pub fn fingerprint_len(&self) -> usize {
        self.fingerprint.len()
    }

    /// 统计指纹中最常见的 N 个汉字及其频率。
    pub fn top_chars(&self, n: usize) -> Vec<(char, usize)> {
        use std::collections::HashMap;
        let mut counts: HashMap<char, usize> = HashMap::new();
        for &ch in &self.fingerprint {
            *counts.entry(ch).or_insert(0) += 1;
        }
        let mut sorted: Vec<_> = counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted.truncate(n);
        sorted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hanzi_map::HanziMap;

    fn test_observer() -> EmbeddingObserver {
        let chars: Vec<char> = vec!['水', '火', '木', '金', '土', '日', '月', '人'];
        let embeddings = vec![
            vec![1.0, 0.0, 0.0, 0.0],  // 水
            vec![0.0, 1.0, 0.0, 0.0],  // 火
            vec![0.0, 0.0, 1.0, 0.0],  // 木
            vec![0.0, 0.0, 0.0, 1.0],  // 金
            vec![-1.0, 0.0, 0.0, 0.0], // 土
            vec![0.707, 0.707, 0.0, 0.0],  // 日
            vec![0.0, 0.707, 0.0, 0.707],  // 月
            vec![0.5, 0.5, 0.5, 0.5],      // 人
        ];
        let map = HanziMap::from_embeddings(chars, embeddings);
        EmbeddingObserver::new(map)
    }

    #[test]
    fn test_observe_basic() {
        let mut obs = test_observer();
        // 水方向嵌入
        let emb = vec![0.9, 0.1, 0.0, 0.0];
        let result = obs.observe(&emb).unwrap();
        assert_eq!(result.ch, '水');
        assert!(result.similarity > 0.8);
        assert_eq!(obs.fingerprint_len(), 1);
    }

    #[test]
    fn test_deviation_from_origin() {
        let mut obs = test_observer();
        // 设置锚点 = 水
        obs.set_origin(&[1.0, 0.0, 0.0, 0.0]);

        // 观察水（应低偏离）
        obs.observe(&[0.95, 0.05, 0.0, 0.0]);
        assert!(obs.semantic_deviation() < 0.1);

        // 观察火（应高偏离）
        obs.observe(&[0.0, 0.95, 0.0, 0.0]);
        assert!(obs.semantic_deviation() > 0.4);
    }

    #[test]
    fn test_fingerprint_accumulation() {
        let mut obs = test_observer();
        obs.observe(&[1.0, 0.0, 0.0, 0.0]); // 水
        obs.observe(&[0.0, 1.0, 0.0, 0.0]); // 火
        obs.observe(&[0.0, 0.0, 1.0, 0.0]); // 木
        assert_eq!(obs.fingerprint(), &['水', '火', '木']);
        assert_eq!(obs.recent_fingerprint(2), &['火', '木']);
    }

    #[test]
    fn test_weak_anchor_ratio() {
        let mut obs = test_observer();
        // 人的嵌入是均匀的 [0.5,0.5,0.5,0.5]，与水/火距离都较大
        obs.observe(&[1.0, 0.0, 0.0, 0.0]); // 水 → 高相似度
        obs.observe(&[0.5, 0.5, 0.5, 0.5]); // 人 → 低相似度
        // 人的均匀嵌入与任何单方向距离都适中，相似度 ≈ 0.5
        // 阈值 0.3，应该都不算弱锚定
        let ratio = obs.weak_anchor_ratio();
        assert!(ratio < 0.6, "weak ratio should be moderate, got {ratio}");
    }

    #[test]
    fn test_top_chars() {
        let mut obs = test_observer();
        obs.observe(&[1.0, 0.0, 0.0, 0.0]); // 水
        obs.observe(&[0.95, 0.05, 0.0, 0.0]); // 水
        obs.observe(&[0.0, 1.0, 0.0, 0.0]); // 火
        let top = obs.top_chars(2);
        assert_eq!(top[0].0, '水');
        assert_eq!(top[0].1, 2);
        assert_eq!(top[1].0, '火');
        assert_eq!(top[1].1, 1);
    }

    #[test]
    fn test_reset() {
        let mut obs = test_observer();
        obs.set_origin(&[1.0, 0.0, 0.0, 0.0]);
        obs.observe(&[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(obs.fingerprint_len(), 1);

        obs.reset();
        assert_eq!(obs.fingerprint_len(), 0);
        assert!(obs.has_origin()); // 锚点保留

        obs.reset_all();
        assert!(!obs.has_origin());
    }

    #[test]
    fn test_peek_does_not_record() {
        let obs = test_observer();
        let result = obs.peek(&[1.0, 0.0, 0.0, 0.0]).unwrap();
        assert_eq!(result.ch, '水');
        assert_eq!(obs.fingerprint_len(), 0);
    }
}
