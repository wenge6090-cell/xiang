/// HanziMap — 嵌入向量到汉字的映射器。
///
/// 利用 LLM 自身的 token embedding 表，将每个输出嵌入向量通过余弦相似度
/// 映射到核心字表中最接近的汉字。这是连续向量空间到离散符号空间的桥梁。

use crate::hanzi_table::{HanziEntry, lookup};

/// 一次汉字映射的结果。
#[derive(Debug, Clone)]
pub struct HanziMapping {
    /// Top-1 匹配的汉字
    pub ch: char,
    /// 余弦相似度 [0, 1]
    pub similarity: f32,
    /// 字表条目
    pub entry: &'static HanziEntry,
}

/// 汉字嵌入映射器。
///
/// 维护一个预归一化的字表嵌入向量数组，支持多种构建方式：
/// - `from_embeddings()`: 从外部提供的嵌入数据构建（用于测试和 mock）
/// - `from_embedded_bytes()`: 从 include_bytes! 嵌入的二进制数据构建（编译时）
/// - `empty()`: 空映射器（嵌入数据未加载时降级使用）
pub struct HanziMap {
    /// 字表字符列表（与 embeddings 同序）
    chars: Vec<char>,
    /// 预 L2 归一化的字表嵌入向量，shape [n_chars, n_embd]
    embeddings: Vec<Vec<f32>>,
    /// 对应的字表条目
    entries: Vec<&'static HanziEntry>,
    /// 嵌入维度（全零表示未加载）
    n_embd: usize,
}

impl HanziMap {
    /// 创建空映射器（嵌入数据未就绪时使用）。
    pub fn empty() -> Self {
        HanziMap {
            chars: Vec::new(),
            embeddings: Vec::new(),
            entries: Vec::new(),
            n_embd: 0,
        }
    }

    /// 从外部嵌入数据构建映射器。
    ///
    /// # Arguments
    /// * `chars` - 字表字符列表
    /// * `embeddings` - 对应嵌入向量，每个向量已 L2 归一化，shape [n_chars, n_embd]
    pub fn from_embeddings(
        chars: Vec<char>,
        embeddings: Vec<Vec<f32>>,
    ) -> Self {
        let n_embd = embeddings.first().map_or(0, |v| v.len());
        // 验证：每个字符必须在字表中有对应条目
        let entries: Vec<_> = chars.iter().enumerate().map(|(i, &ch)| {
            lookup(ch).unwrap_or_else(|| {
                // 回退：构造一个临时条目（仅在测试场景中出现）
                panic!("字表嵌入中的字符 '{}' (索引 {}) 不在 HANZI_TABLE 中", ch, i)
            })
        }).collect();

        HanziMap { chars, embeddings, entries, n_embd }
    }

    /// 从编译时嵌入的二进制数据构建映射器。
    ///
    /// 二进制格式:
    ///   [u32 n_chars] [u32 n_embd]
    ///   重复 n_chars 次: [u32 char_utf32] [u32 token_id] [f32×n_embd]
    pub fn from_embedded_bytes(data: &[u8]) -> Result<Self, &'static str> {
        if data.len() < 8 {
            return Err("嵌入数据太短");
        }

        let n_chars = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let n_embd = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;

        let header_size = 8;
        let entry_size = 8 + n_embd * 4; // 4(char_utf32) + 4(token_id) + n_embd*4(f32)
        let expected_size = header_size + n_chars * entry_size;

        if data.len() < expected_size {
            return Err("嵌入数据长度不匹配");
        }

        let mut chars = Vec::with_capacity(n_chars);
        let mut embeddings = Vec::with_capacity(n_chars);

        for i in 0..n_chars {
            let offset = header_size + i * entry_size;
            let char_utf32 = u32::from_le_bytes([
                data[offset], data[offset+1], data[offset+2], data[offset+3]
            ]);
            // 跳过 token_id (offset+4..offset+8)
            let ch = char::from_u32(char_utf32).ok_or("非法 Unicode 码点")?;

            let emb_start = offset + 8;
            let emb_end = emb_start + n_embd * 4;
            let emb_bytes = &data[emb_start..emb_end];

            let emb: Vec<f32> = emb_bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();

            chars.push(ch);
            embeddings.push(emb);
        }

        Ok(HanziMap::from_embeddings(chars, embeddings))
    }

    /// 是否已加载嵌入数据。
    pub fn is_loaded(&self) -> bool {
        self.n_embd > 0 && !self.embeddings.is_empty()
    }

    /// 字表大小。
    pub fn size(&self) -> usize {
        self.chars.len()
    }

    /// 嵌入维度。
    pub fn n_embd(&self) -> usize {
        self.n_embd
    }

    /// 对输出嵌入做 L2 归一化。
    fn l2_normalize(v: &mut [f32]) {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-12 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
    }

    /// 计算两个等长向量的点积。
    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// K=1 映射——返回最匹配的汉字。
    ///
    /// 如果映射器未加载数据，返回 None。
    pub fn map_single(&self, embedding: &[f32]) -> Option<HanziMapping> {
        self.map_top_k(embedding, 1).map(|v| v.into_iter().next().unwrap())
    }

    /// Top-K 映射——返回最匹配的 K 个汉字（按相似度降序）。
    ///
    /// 如果映射器未加载数据或嵌入维度不匹配，返回 None。
    pub fn map_top_k(&self, embedding: &[f32], k: usize) -> Option<Vec<HanziMapping>> {
        if !self.is_loaded() || embedding.len() != self.n_embd {
            return None;
        }

        // L2 归一化输入嵌入
        let mut emb_norm = embedding.to_vec();
        Self::l2_normalize(&mut emb_norm);

        // 计算与每个字表嵌入的余弦相似度（字表嵌入已预归一化，点积=余弦）
        let mut sims: Vec<(usize, f32)> = self.embeddings.iter()
            .enumerate()
            .map(|(i, ref_emb)| (i, Self::dot(&emb_norm, ref_emb)))
            .collect();

        // 部分排序选出 top-K
        let actual_k = k.min(sims.len());
        sims.select_nth_unstable_by(actual_k - 1, |a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sims.truncate(actual_k);
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Some(sims.into_iter().map(|(idx, sim)| HanziMapping {
            ch: self.chars[idx],
            similarity: sim,
            entry: self.entries[idx],
        }).collect())
    }

    /// 返回相似度 ≥ threshold 的所有汉字。
    pub fn map_with_threshold(&self, embedding: &[f32], threshold: f32) -> Option<Vec<HanziMapping>> {
        if !self.is_loaded() || embedding.len() != self.n_embd {
            return None;
        }

        let mut emb_norm = embedding.to_vec();
        Self::l2_normalize(&mut emb_norm);

        let mut results: Vec<HanziMapping> = self.embeddings.iter()
            .enumerate()
            .map(|(i, ref_emb)| {
                let sim = Self::dot(&emb_norm, ref_emb);
                (i, sim)
            })
            .filter(|(_, sim)| *sim >= threshold)
            .map(|(idx, sim)| HanziMapping {
                ch: self.chars[idx],
                similarity: sim,
                entry: self.entries[idx],
            })
            .collect();

        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        Some(results)
    }

    /// 获取字符在字表中的索引（用于构建语义指纹）。
    pub fn char_index(&self, ch: char) -> Option<usize> {
        self.chars.iter().position(|&c| c == ch)
    }

    /// 获取指定索引的字符。
    pub fn char_at(&self, idx: usize) -> Option<char> {
        self.chars.get(idx).copied()
    }

    /// 获取指定字符的 L2 归一化嵌入向量。
    ///
    /// 返回的向量已 L2 归一化，可直接用于余弦相似度计算（点积 = 余弦）。
    pub fn embedding_of(&self, ch: char) -> Option<&[f32]> {
        let idx = self.char_index(ch)?;
        self.embeddings.get(idx).map(|v| v.as_slice())
    }
}

impl std::fmt::Debug for HanziMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HanziMap")
            .field("n_chars", &self.chars.len())
            .field("n_embd", &self.n_embd)
            .field("loaded", &self.is_loaded())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 构建一个测试用的小型 HanziMap。
    fn test_map() -> HanziMap {
        let chars: Vec<char> = vec!['水', '火', '木', '金', '土', '日', '月', '人'];
        // 模拟 4 维嵌入（已 L2 归一化）
        let embeddings = vec![
            vec![1.0, 0.0, 0.0, 0.0],  // 水
            vec![0.0, 1.0, 0.0, 0.0],  // 火
            vec![0.0, 0.0, 1.0, 0.0],  // 木
            vec![0.0, 0.0, 0.0, 1.0],  // 金
            vec![-1.0, 0.0, 0.0, 0.0], // 土
            vec![0.707, 0.707, 0.0, 0.0],  // 日 (水+火方向)
            vec![0.0, 0.707, 0.0, 0.707],  // 月 (火+金方向)
            vec![0.5, 0.5, 0.5, 0.5],      // 人 (均匀)
        ];
        HanziMap::from_embeddings(chars, embeddings)
    }

    #[test]
    fn test_self_mapping() {
        let map = test_map();
        // 每个字表嵌入应该映射回自己
        let emb = vec![1.0, 0.0, 0.0, 0.0]; // 水
        let result = map.map_single(&emb).unwrap();
        assert_eq!(result.ch, '水');
        assert!(result.similarity > 0.99, "自相似度应接近 1.0, got {}", result.similarity);
    }

    #[test]
    fn test_mapping_near() {
        let map = test_map();
        // 偏水方向的向量应映射到水
        let emb = vec![0.9, 0.1, 0.0, 0.0];
        let result = map.map_single(&emb).unwrap();
        assert_eq!(result.ch, '水');
    }

    #[test]
    fn test_top_k() {
        let map = test_map();
        // 水+火方向（日）应该 top-2 为日和日（水/火接近）
        let emb = vec![0.6, 0.6, 0.0, 0.0];
        let results = map.map_top_k(&emb, 3).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].ch, '日');
        // 前两个应该是日和水/火
        assert!(['水', '火'].contains(&results[1].ch) || ['水', '火'].contains(&results[2].ch));
    }

    #[test]
    fn test_map_with_threshold() {
        let map = test_map();
        let emb = vec![0.6, 0.6, 0.0, 0.0];
        let results = map.map_with_threshold(&emb, 0.5).unwrap();
        assert!(results.len() >= 2, "至少2个字相似度>=0.5");
        for r in &results {
            assert!(r.similarity >= 0.5);
        }
    }

    #[test]
    fn test_empty_map_returns_none() {
        let map = HanziMap::empty();
        assert!(!map.is_loaded());
        assert!(map.map_single(&[1.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn test_wrong_dimension_returns_none() {
        let map = test_map();
        // 输入维度 3，但字表嵌入维度为 4
        assert!(map.map_single(&[1.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn test_l2_normalize() {
        let mut v = vec![3.0, 4.0];
        HanziMap::l2_normalize(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "L2 norm 应为 1.0, got {}", norm);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_similarity_symmetry() {
        let map = test_map();
        // 水嵌入（索引 0）与人嵌入（索引 7）的相似度应该是对称的
        let water_emb = &map.embeddings[0];
        let human_emb = &map.embeddings[7];
        let sim = HanziMap::dot(water_emb, human_emb);
        let sim_rev = HanziMap::dot(human_emb, water_emb);
        assert!((sim - sim_rev).abs() < 1e-6);
    }
}
