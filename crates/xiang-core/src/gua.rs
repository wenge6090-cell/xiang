/// 卦 (Gua) — 6-bit state vector S ∈ {0..63}
///
/// Maps to the 64 hexagrams of 归藏易.
/// Invariant: self.0 ∈ [0, 63].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Gua(pub u8);

impl Gua {
    pub const ZERO: Gua = Gua(0);
    pub const MAX: Gua = Gua(63);
    /// 问题焦点向量 (question focus origin) — default 全阳 111111
    /// 此值应由当前问题的语义焦点动态设定，而非固定为全阳。
    pub const ORIGIN: Gua = Gua(0b111111);

    /// Construct from a u8, returning None if value >= 64 (invalid 6-bit).
    pub fn try_from_u8(v: u8) -> Option<Gua> {
        if v < 64 { Some(Gua(v)) } else { None }
    }

    /// Unsafe: panics if v >= 64.
    pub fn from_u8(v: u8) -> Gua {
        assert!(v < 64, "Gua value must be < 64, got {v}");
        Gua(v)
    }

    /// Construct from 6 individual bits (b5..b0, MSB..LSB).
    pub fn from_bits(bits: [u8; 6]) -> Gua {
        let v = (bits[0] << 5) | (bits[1] << 4) | (bits[2] << 3)
              | (bits[3] << 2) | (bits[4] << 1) |  bits[5];
        Gua(v)
    }

    /// Upper trigram (bits 5..3) as 3-bit u8.
    pub fn upper(self) -> u8 {
        (self.0 >> 3) & 0b111
    }

    /// Lower trigram (bits 2..0) as 3-bit u8.
    pub fn lower(self) -> u8 {
        self.0 & 0b111
    }

    /// Population count (number of 1 bits) in the 6-bit value.
    pub fn popcount(self) -> u32 {
        self.0.count_ones()
    }

    /// Hamming distance to another Gua (number of differing bits).
    pub fn hamming(self, other: Gua) -> u32 {
        (self.0 ^ other.0).count_ones()
    }

    /// Read a single bit (0 = LSB, 5 = MSB).
    pub fn bit(self, n: u8) -> bool {
        debug_assert!(n < 6);
        (self.0 >> n) & 1 == 1
    }
}

impl std::fmt::Display for Gua {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:06b}", self.0)
    }
}

// ── 64 卦名常量表 ────────────────────────────────────────────
// 归藏易 64 卦命名：上卦(天气) + 下卦(地气) + 归藏卦名

impl Gua {
    /// 64 卦名（中文），索引 0..63
    pub const NAMES: [&'static str; 64] = [
        /* 00 */ "藏藏始基",  /* 01 */ "木气萌芽",  /* 02 */ "风气初始",  /* 03 */ "火气初生",
        /* 04 */ "水气始动",  /* 05 */ "山气凝基",  /* 06 */ "金气初锋",  /* 07 */ "天气初归",
        /* 08 */ "木藏生根",  /* 09 */ "木木并生",  /* 10 */ "风木相激",  /* 11 */ "火木同燃",
        /* 12 */ "水木相滋",  /* 13 */ "山木定根",  /* 14 */ "金木相克",  /* 15 */ "天木舒展",
        /* 16 */ "风藏入微",  /* 17 */ "木风入理",  /* 18 */ "风风流转",  /* 19 */ "火风相煽",
        /* 20 */ "水风化雨",  /* 21 */ "山风止散",  /* 22 */ "金风断流",  /* 23 */ "天风广布",
        /* 24 */ "火藏蓄势",  /* 25 */ "木火通明",  /* 26 */ "风火相值",  /* 27 */ "火火同辉",
        /* 28 */ "水火既济",  /* 29 */ "山火贲饰",  /* 30 */ "金火锻炼",  /* 31 */ "天火同人",
        /* 32 */ "水藏潜渊",  /* 33 */ "木水养根",  /* 34 */ "风水涣散",  /* 35 */ "火水未济",
        /* 36 */ "水水重险",  /* 37 */ "山水蒙昧",  /* 38 */ "金水相生",  /* 39 */ "天水需待",
        /* 40 */ "山藏止定",  /* 41 */ "木山损抑",  /* 42 */ "风山渐进",  /* 43 */ "火山旅寄",
        /* 44 */ "水山蹇难",  /* 45 */ "山山艮止",  /* 46 */ "金山剥落",  /* 47 */ "天山遁退",
        /* 48 */ "金藏收敛",  /* 49 */ "木金革变",  /* 50 */ "风金鼎定",  /* 51 */ "火金睽违",
        /* 52 */ "水金节限",  /* 53 */ "山金损减",  /* 54 */ "金金夬决",  /* 55 */ "天金履践",
        /* 56 */ "归藏定位",  /* 57 */ "天木益增",  /* 58 */ "天风姤遇",  /* 59 */ "天火大有",
        /* 60 */ "天水讼争",  /* 61 */ "天山谦退",  /* 62 */ "天金小畜",  /* 63 */ "天天归元",
    ];

    /// Get the Chinese name for this Gua.
    pub fn name(self) -> &'static str {
        Gua::NAMES[self.0 as usize]
    }

    /// Look up a Gua by name.
    pub fn by_name(name: &str) -> Option<Gua> {
        Gua::NAMES.iter().position(|&n| n == name).map(|i| Gua(i as u8))
    }
}

// ── 八气算子 (Eight Qi Operators) ───────────────────────────

/// The eight core qi operators of the 归藏 layer.
/// All operators are pure functions on the 6-bit state, except `cang` which
/// requires a CangSea for Hebbian learning.
impl Gua {
    /// 生 (Sheng) — Sprout an idea: `S |= 0b000001`
    pub fn sheng(self) -> Gua {
        Gua(self.0 | 0b000001)
    }

    /// 动 (Dong) — Divergent association: `S = ROL(S,1) XOR 0b000010`
    pub fn dong(self) -> Gua {
        // ROL within 6 bits: shift left 1, wrap bit5 to bit0
        let rol = ((self.0 << 1) | (self.0 >> 5)) & 0x3F;
        Gua(rol ^ 0b000010)
    }

    /// 长 (Zhang) — Amplify intent: copy lower trigram (bits 2..0) to upper (bits 5..3)
    pub fn zhang(self) -> Gua {
        let lower = self.lower();
        Gua((lower << 3) | lower)
    }

    /// 育 (Yu) — Trigger plan decomposition: `S = S XOR 0b100000` (toggle bit 5)
    pub fn yu(self) -> Gua {
        Gua(self.0 ^ 0b100000)
    }

    /// 杀 (Sha) — Prune interference patterns: clear all 110 patterns in both trigrams.
    /// If lower trigram == 110 → clear lower. If upper trigram == 110 → clear upper.
    pub fn sha(self) -> Gua {
        let mut s = self.0;
        if (s & 0b111) == 0b110 {
            s &= !0b111; // clear lower 3 bits
        }
        if ((s >> 3) & 0b111) == 0b110 {
            s &= !0b111000; // clear upper 3 bits
        }
        Gua(s)
    }

    /// 止 (Zhi) — Freeze boundary: `S = S & 0b101101`
    pub fn zhi(self) -> Gua {
        Gua(self.0 & 0b101101)
    }

    /// 归 (Gui) — Deviation measurement (pure, no state mutation).
    /// Returns deviation ∈ [0, 1], where 0 = perfect alignment with origin,
    /// 1 = complete opposition.
    pub fn gui(self, origin: Gua) -> f32 {
        // deviation = 1 - popcount(~(S ^ origin) & 0x3F) / 6
        let xor = self.0 ^ origin.0;
        let matching = (!xor) & 0x3F;
        1.0 - (matching.count_ones() as f32 / 6.0)
    }

    /// 藏 (Cang) — Reset state to zero.
    /// Note: Hebbian learning (write to CangSea) is handled separately by the VM.
    pub fn cang(self) -> Gua {
        Gua(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(Gua::ZERO.0, 0);
        assert_eq!(Gua::ORIGIN.0, 0b111111);
        assert_eq!(Gua::MAX.0, 63);
    }

    #[test]
    fn test_try_from_u8() {
        assert!(Gua::try_from_u8(0).is_some());
        assert!(Gua::try_from_u8(63).is_some());
        assert!(Gua::try_from_u8(64).is_none());
        assert!(Gua::try_from_u8(255).is_none());
    }

    #[test]
    fn test_upper_lower() {
        let g = Gua::from_bits([1,1,1, 0,0,0]); // 111000 = 56
        assert_eq!(g.upper(), 0b111);
        assert_eq!(g.lower(), 0b000);
    }

    #[test]
    fn test_popcount() {
        assert_eq!(Gua(0b000000).popcount(), 0);
        assert_eq!(Gua(0b111111).popcount(), 6);
        assert_eq!(Gua(0b101010).popcount(), 3);
    }

    #[test]
    fn test_hamming() {
        assert_eq!(Gua(0b000000).hamming(Gua(0b111111)), 6);
        assert_eq!(Gua(0b101010).hamming(Gua(0b101010)), 0);
    }

    #[test]
    fn test_sheng() {
        assert_eq!(Gua(0).sheng(), Gua(0b000001));
        assert_eq!(Gua(0b000001).sheng(), Gua(0b000001)); // idempotent (LSB already 1)
        assert_eq!(Gua(63).sheng(), Gua(63));
    }

    #[test]
    fn test_dong() {
        // ROL(000001, 1) = 000010, XOR 000010 = 000000
        assert_eq!(Gua(0b000001).dong(), Gua(0b000000));
        // ROL(000000, 1) = 000000, XOR 000010 = 000010
        assert_eq!(Gua(0).dong(), Gua(0b000010));
        // ROL(100000, 1) = 000001, XOR 000010 = 000011
        assert_eq!(Gua(0b100000).dong(), Gua(0b000011));
    }

    #[test]
    fn test_zhang() {
        // lower=000 → 000000
        assert_eq!(Gua(0b000000).zhang(), Gua(0b000000));
        // lower=111 → 111111
        assert_eq!(Gua(0b000111).zhang(), Gua(0b111111));
        // lower=010 → 010010
        assert_eq!(Gua(0b000010).zhang(), Gua(0b010010));
        // S=101010 → lower=010 → 010010 (upper overwritten)
        assert_eq!(Gua(0b101010).zhang(), Gua(0b010010));
    }

    #[test]
    fn test_yu() {
        // XOR with 100000 (toggle bit 5)
        assert_eq!(Gua(0).yu(), Gua(0b100000));
        assert_eq!(Gua(0b100000).yu(), Gua(0));
        assert_eq!(Gua(0b111000).yu(), Gua(0b011000));
    }

    #[test]
    fn test_sha() {
        // lower=110 → clear lower
        assert_eq!(Gua(0b000110).sha(), Gua(0b000000));
        // upper=110 → clear upper: 110xxx → 000xxx
        assert_eq!(Gua(0b110000).sha(), Gua(0b000000));
        // both trigrams = 110 → clear all
        assert_eq!(Gua(0b110110).sha(), Gua(0b000000));
        // no 110 pattern → unchanged
        assert_eq!(Gua(0b111111).sha(), Gua(0b111111));
        assert_eq!(Gua(0b101101).sha(), Gua(0b101101));
    }

    #[test]
    fn test_zhi() {
        // AND with 0b101101 = mask bits [5,3,2,0]
        assert_eq!(Gua(0b111111).zhi(), Gua(0b101101));
        assert_eq!(Gua(0b000000).zhi(), Gua(0));
        assert_eq!(Gua(0b101101).zhi(), Gua(0b101101));
    }

    #[test]
    fn test_gui_perfect_match() {
        assert_eq!(Gua(0b111111).gui(Gua::ORIGIN), 0.0);
    }

    #[test]
    fn test_gui_complete_opposition() {
        assert_eq!(Gua(0b000000).gui(Gua::ORIGIN), 1.0);
    }

    #[test]
    fn test_gui_partial() {
        // 3 bits match, 3 differ → deviation = 1 - 3/6 = 0.5
        assert_eq!(Gua(0b111000).gui(Gua::ORIGIN), 0.5);
        assert_eq!(Gua(0b000111).gui(Gua::ORIGIN), 0.5);
    }

    #[test]
    fn test_cang() {
        assert_eq!(Gua(0b111111).cang(), Gua(0));
        assert_eq!(Gua(0b101010).cang(), Gua(0));
    }

    #[test]
    fn test_names() {
        assert_eq!(Gua(0).name(), "藏藏始基");
        assert_eq!(Gua(56).name(), "归藏定位");
        assert_eq!(Gua(63).name(), "天天归元");
    }

    #[test]
    fn test_by_name() {
        assert_eq!(Gua::by_name("归藏定位"), Some(Gua(56)));
        assert_eq!(Gua::by_name("不存在"), None);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", Gua(0)), "000000");
        assert_eq!(format!("{}", Gua(63)), "111111");
        assert_eq!(format!("{}", Gua(56)), "111000");
    }

    #[test]
    fn test_all_ops_stay_in_range() {
        // Verify that all operators produce valid 6-bit Gua values for all 64 states.
        for i in 0..64 {
            let g = Gua(i);
            assert!(g.sheng().0 < 64, "sheng({i}) overflow");
            assert!(g.dong().0 < 64, "dong({i}) overflow");
            assert!(g.zhang().0 < 64, "zhang({i}) overflow");
            assert!(g.yu().0 < 64, "yu({i}) overflow");
            assert!(g.sha().0 < 64, "sha({i}) overflow");
            assert!(g.zhi().0 < 64, "zhi({i}) overflow");
            assert!(g.cang().0 < 64, "cang({i}) overflow");
            let dev = g.gui(Gua::ORIGIN);
            assert!(dev >= 0.0 && dev <= 1.0, "gui({i}) deviation {dev} out of range");
        }
    }

    #[test]
    fn test_sheng_idempotent() {
        for i in 0..64 {
            let g = Gua(i);
            assert_eq!(g.sheng().sheng(), g.sheng(), "sheng not idempotent for {i}");
        }
    }
}
