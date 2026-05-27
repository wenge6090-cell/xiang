/// 汉字嵌入数据加载器。
///
/// 负责加载预先从 LLM token embedding 表中提取的汉字嵌入向量，
/// 并构建 [`HanziMap`] 映射器。
///
/// 支持两种加载方式：
/// - **编译时嵌入**: 通过 `include_bytes!` 将嵌入数据直接编译进二进制
/// - **运行时加载**: 从文件路径读取（用于开发调试和动态切换模型）
///
/// 二进制格式参见 [`HanziMap::from_embedded_bytes`]。
use crate::hanzi_map::HanziMap;
use std::fs;
use std::path::Path;

/// 汉字嵌入数据容器。
///
/// 封装了嵌入数据的加载与 `HanziMap` 的构建。
pub struct HanziEmbeddings {
    /// 构建好的汉字映射器
    pub map: HanziMap,
    /// 数据来源标识（文件名或 "embedded"）
    pub source: String,
}

impl HanziEmbeddings {
    /// 从文件路径加载嵌入数据。
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let data = fs::read(path)
            .map_err(|e| format!("读取嵌入数据文件失败: {}: {}", path.display(), e))?;

        let map = HanziMap::from_embedded_bytes(&data)
            .map_err(|e| format!("解析嵌入数据失败: {}", e))?;

        Ok(HanziEmbeddings {
            map,
            source: path.display().to_string(),
        })
    }

    /// 从编译时嵌入的字节数据构建。
    ///
    /// 使用方式：
    /// ```ignore
    /// const EMBEDDED_DATA: &[u8] = include_bytes!("../data/hanzi_embeddings.bin");
    /// let emb = HanziEmbeddings::from_embedded_bytes(EMBEDDED_DATA);
    /// ```
    pub fn from_embedded_bytes(data: &[u8]) -> Result<Self, String> {
        let map = HanziMap::from_embedded_bytes(data)
            .map_err(|e| format!("解析嵌入数据失败: {}", e))?;

        Ok(HanziEmbeddings {
            map,
            source: "embedded".to_string(),
        })
    }

    /// 尝试加载嵌入数据，按优先级搜索：
    /// 1. 嵌入数据字节（编译时）
    /// 2. 文件路径
    ///
    /// 如果提供了嵌入数据则优先使用，否则尝试从文件中加载。
    ///
    /// 如果两者都失败，返回 `Ok(None)` 表示嵌入未就绪（此时 `HanziMap::empty()` 可作为降级方案）。
    pub fn try_load(
        embedded_data: Option<&[u8]>,
        file_path: Option<&Path>,
    ) -> Result<Option<Self>, String> {
        if let Some(data) = embedded_data {
            if !data.is_empty() {
                return Self::from_embedded_bytes(data).map(Some);
            }
        }

        if let Some(path) = file_path {
            if path.exists() {
                return Self::load_from_file(path).map(Some);
            }
        }

        Ok(None)
    }

    /// 从尝试加载失败时返回的空映射器。
    ///
    /// 当嵌入数据未就绪时，可以用这个创建一个不加载任何嵌入数据的 `HanziEmbeddings`。
    pub fn empty() -> Self {
        HanziEmbeddings {
            map: HanziMap::empty(),
            source: "empty".to_string(),
        }
    }

    /// 是否有有效的嵌入数据。
    pub fn is_loaded(&self) -> bool {
        self.map.is_loaded()
    }

    /// 嵌入维度。
    pub fn n_embd(&self) -> usize {
        self.map.n_embd()
    }

    /// 字表大小。
    pub fn size(&self) -> usize {
        self.map.size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 构建一个测试用的二进制嵌入数据。
    fn build_test_embedding_bytes() -> Vec<u8> {
        // 3 个字符: 水, 火, 木，嵌入维度 4
        let n_chars: u32 = 3;
        let n_embd: u32 = 4;

        let mut buf = Vec::new();
        buf.extend_from_slice(&n_chars.to_le_bytes());
        buf.extend_from_slice(&n_embd.to_le_bytes());

        // 水 U+6C34
        buf.extend_from_slice(&0x6C34u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // token_id
        buf.extend_from_slice(&1.0f32.to_le_bytes());
        buf.extend_from_slice(&0.0f32.to_le_bytes());
        buf.extend_from_slice(&0.0f32.to_le_bytes());
        buf.extend_from_slice(&0.0f32.to_le_bytes());

        // 火 U+706B
        buf.extend_from_slice(&0x706Bu32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0.0f32.to_le_bytes());
        buf.extend_from_slice(&1.0f32.to_le_bytes());
        buf.extend_from_slice(&0.0f32.to_le_bytes());
        buf.extend_from_slice(&0.0f32.to_le_bytes());

        // 木 U+6728
        buf.extend_from_slice(&0x6728u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0.0f32.to_le_bytes());
        buf.extend_from_slice(&0.0f32.to_le_bytes());
        buf.extend_from_slice(&1.0f32.to_le_bytes());
        buf.extend_from_slice(&0.0f32.to_le_bytes());

        buf
    }

    #[test]
    fn test_from_embedded_bytes() {
        let data = build_test_embedding_bytes();
        let emb = HanziEmbeddings::from_embedded_bytes(&data).unwrap();
        assert!(emb.is_loaded());
        assert_eq!(emb.n_embd(), 4);
        assert_eq!(emb.size(), 3);

        // 验证映射功能
        let water = vec![1.0, 0.0, 0.0, 0.0];
        let result = emb.map.map_single(&water).unwrap();
        assert_eq!(result.ch, '水');
    }

    #[test]
    fn test_empty() {
        let emb = HanziEmbeddings::empty();
        assert!(!emb.is_loaded());
        assert_eq!(emb.size(), 0);
    }

    #[test]
    fn test_try_load_with_embedded() {
        let data = build_test_embedding_bytes();
        let emb = HanziEmbeddings::try_load(Some(&data), None).unwrap().unwrap();
        assert!(emb.is_loaded());
    }

    #[test]
    fn test_try_load_embedded_priority() {
        // 即使两者都提供，嵌入数据优先
        let data = build_test_embedding_bytes();
        let emb = HanziEmbeddings::try_load(Some(&data), Some(Path::new("nonexistent.bin"))).unwrap().unwrap();
        assert!(emb.is_loaded());
        assert_eq!(emb.source, "embedded");
    }

    #[test]
    fn test_try_load_none() {
        let emb = HanziEmbeddings::try_load(None, None).unwrap();
        assert!(emb.is_none());
    }
}
