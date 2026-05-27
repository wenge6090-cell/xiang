/// 核心汉字词表 —— 按照六书分类的语义锚定字表
///
/// 精选约 300+ 个有强语义锚定性的汉字，分为三类:
/// - 象形字 (Pictogram): 世界实体锚点
/// - 指事字 (Ideogram): 变化形式化语法
/// - 会意字 (CompoundIdeogram): 复合概念
///
/// 数据由 `gen_hanzi_table.py` 生成。

// 用 include! 嵌入生成的字表数据
include!("hanzi_table_data.rs");

/// 六书分类。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HanziCategory {
    /// 象形 — 世界实体锚点
    Pictogram,
    /// 指事 — 变化形式化语法
    Ideogram,
    /// 会意 — 复合概念
    CompoundIdeogram,
}

/// 五行元素（用于汉字语义域分类）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WuXing {
    金,
    木,
    水,
    火,
    土,
}

/// 一个字表条目。
#[derive(Debug, Clone)]
pub struct HanziEntry {
    /// 汉字字符
    pub ch: char,
    /// 六书分类
    pub category: HanziCategory,
    /// 语义标签
    pub tag: &'static str,
    /// 五行归属（如适用）
    pub wuxing: Option<WuXing>,
}

impl HanziEntry {
    /// 创建象形字条目。
    pub const fn p(ch: char, tag: &'static str) -> Self {
        HanziEntry { ch, category: HanziCategory::Pictogram, tag, wuxing: None }
    }

    /// 创建指事字条目。
    pub const fn i(ch: char, tag: &'static str) -> Self {
        HanziEntry { ch, category: HanziCategory::Ideogram, tag, wuxing: None }
    }

    /// 创建会意字条目。
    pub const fn c(ch: char, tag: &'static str) -> Self {
        HanziEntry { ch, category: HanziCategory::CompoundIdeogram, tag, wuxing: None }
    }
}

/// 根据字符查找词表条目。O(N) 线性搜索，词表仅 ~300 字。
pub fn lookup(ch: char) -> Option<&'static HanziEntry> {
    HANZI_TABLE.iter().find(|e| e.ch == ch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hanzi_table_size() {
        let pictogram_count = HANZI_TABLE.iter().filter(|e| e.category == HanziCategory::Pictogram).count();
        let ideogram_count = HANZI_TABLE.iter().filter(|e| e.category == HanziCategory::Ideogram).count();
        let compound_count = HANZI_TABLE.iter().filter(|e| e.category == HanziCategory::CompoundIdeogram).count();
        assert!(pictogram_count >= 80, "Pictograms: {}", pictogram_count);
        assert!(ideogram_count >= 80, "Ideograms: {}", ideogram_count);
        assert!(compound_count >= 80, "Compounds: {}", compound_count);
        assert!(HANZI_TABLE.len() >= 240, "Total: {}", HANZI_TABLE.len());
    }

    #[test]
    fn test_eight_operators_present() {
        for op in &["生", "动", "长", "育", "杀", "止", "归", "藏"] {
            let ch = op.chars().next().unwrap();
            let entry = lookup(ch);
            assert!(entry.is_some(), "operator '{}' missing", op);
            assert_eq!(entry.unwrap().category, HanziCategory::Ideogram);
        }
    }

    #[test]
    fn test_eight_trigrams_present() {
        for bg in &["乾", "兑", "离", "震", "巽", "坎", "艮", "坤"] {
            let ch = bg.chars().next().unwrap();
            let entry = lookup(ch);
            assert!(entry.is_some(), "trigram '{}' missing", bg);
            assert_eq!(entry.unwrap().category, HanziCategory::Ideogram);
        }
    }

    #[test]
    fn test_lookup_roundtrip() {
        let entry = lookup('日').unwrap();
        assert_eq!(entry.ch, '日');
        assert_eq!(entry.category, HanziCategory::Pictogram);
        assert_eq!(entry.tag, "太阳");
    }

    #[test]
    fn test_no_duplicates() {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        for entry in HANZI_TABLE {
            assert!(seen.insert(entry.ch), "duplicate: {}", entry.ch);
        }
    }
}
