/// 八卦 (Bagua) — 8-trigram system for the Zhouyi cognitive grid.
///
/// Each Bagua maps to a cognitive posture with an associated LLM temperature.
/// The Wu Xing (五行) generation/restraint rules govern state transitions.

use std::fmt;

/// The eight trigrams (八卦).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Bagua {
    乾, // ☰ 111 — 金 — 创造
    兑, // ☱ 110 — 金 — 表达
    离, // ☲ 101 — 火 — 明照
    震, // ☳ 100 — 木 — 启动
    巽, // ☴ 011 — 风 — 渗透
    坎, // ☵ 010 — 水 — 破局
    艮, // ☶ 001 — 山 — 止定
    坤, // ☷ 000 — 土 — 承载
}

impl Bagua {
    /// All eight trigrams in reverse-binary order (归藏易序).
    pub const ALL: [Bagua; 8] = [
        Bagua::乾, Bagua::兑, Bagua::离, Bagua::震,
        Bagua::巽, Bagua::坎, Bagua::艮, Bagua::坤,
    ];

    /// 3-bit binary encoding (0b000..0b111).
    pub fn to_bits(self) -> u8 {
        match self {
            Bagua::乾 => 0b111,
            Bagua::兑 => 0b110,
            Bagua::离 => 0b101,
            Bagua::震 => 0b100,
            Bagua::巽 => 0b011,
            Bagua::坎 => 0b010,
            Bagua::艮 => 0b001,
            Bagua::坤 => 0b000,
        }
    }

    /// Construct from 3-bit value.
    pub fn from_bits(bits: u8) -> Option<Bagua> {
        match bits & 0b111 {
            0b111 => Some(Bagua::乾),
            0b110 => Some(Bagua::兑),
            0b101 => Some(Bagua::离),
            0b100 => Some(Bagua::震),
            0b011 => Some(Bagua::巽),
            0b010 => Some(Bagua::坎),
            0b001 => Some(Bagua::艮),
            0b000 => Some(Bagua::坤),
            _ => unreachable!("3-bit values only have 8 cases"),
        }
    }

    /// LLM temperature for this cognitive posture.
    pub fn temperature(self) -> f32 {
        match self {
            Bagua::乾 => 1.2,
            Bagua::兑 => 0.9,
            Bagua::离 => 0.5,
            Bagua::震 => 1.0,
            Bagua::巽 => 0.7,
            Bagua::坎 => 1.1,
            Bagua::艮 => 0.3,
            Bagua::坤 => 0.6,
        }
    }

    /// Chinese name.
    pub fn name(self) -> &'static str {
        match self {
            Bagua::乾 => "乾",
            Bagua::兑 => "兑",
            Bagua::离 => "离",
            Bagua::震 => "震",
            Bagua::巽 => "巽",
            Bagua::坎 => "坎",
            Bagua::艮 => "艮",
            Bagua::坤 => "坤",
        }
    }

    /// Cognitive posture description.
    pub fn posture(self) -> &'static str {
        match self {
            Bagua::乾 => "创造",
            Bagua::兑 => "表达",
            Bagua::离 => "明照",
            Bagua::震 => "启动",
            Bagua::巽 => "渗透",
            Bagua::坎 => "破局",
            Bagua::艮 => "止定",
            Bagua::坤 => "承载",
        }
    }

    /// Wu Xing element.
    pub fn wuxing(self) -> &'static str {
        match self {
            Bagua::乾 => "金",
            Bagua::兑 => "金",
            Bagua::离 => "火",
            Bagua::震 => "木",
            Bagua::巽 => "风",
            Bagua::坎 => "水",
            Bagua::艮 => "山",
            Bagua::坤 => "土",
        }
    }

    /// Check if `self` generates (生) `other` per Wu Xing rules.
    pub fn generates(self, other: Bagua) -> bool {
        matches!(
            (self, other),
            (Bagua::震, Bagua::离)  // 木生火
            | (Bagua::离, Bagua::坤) // 火生土
            | (Bagua::坤, Bagua::兑) // 土生金
            | (Bagua::兑, Bagua::坎) // 金生水
            | (Bagua::坎, Bagua::震) // 水生木
            | (Bagua::巽, Bagua::离) // 风生火
            | (Bagua::艮, Bagua::坤) // 山生土
        )
    }

    /// Check if `self` restrains (克) `other` per Wu Xing rules.
    pub fn restrains(self, other: Bagua) -> bool {
        matches!(
            (self, other),
            (Bagua::兑, Bagua::震)  // 金克木
            | (Bagua::离, Bagua::兑) // 火克金
            | (Bagua::坎, Bagua::离) // 水克火
            | (Bagua::震, Bagua::艮) // 木克山
            | (Bagua::坤, Bagua::坎) // 土克水
            | (Bagua::乾, Bagua::坤) // 金(乾)克土
        )
    }
}

impl fmt::Display for Bagua {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// 八卦认知格栅 (Zhouyi Cognitive Grid).
///
/// Manages an 8-state Markov chain where transitions are governed by
/// Wu Xing 生克 (generation/restraint) rules. Each state maps to a
/// cognitive posture with an associated LLM temperature.
pub struct ZhouGrid {
    /// Current cognitive posture.
    pub current: Bagua,
    /// 生克 weight matrix: weights[from][to] — learned transition preferences.
    pub weights: [[f32; 8]; 8],
}

impl ZhouGrid {
    pub fn new(initial: Bagua) -> Self {
        let mut weights = [[0.0; 8]; 8];
        // Initialize with Wu Xing generation weights
        for from in &Bagua::ALL {
            let fi = from.to_bits() as usize;
            for to in &Bagua::ALL {
                let ti = to.to_bits() as usize;
                if from.generates(*to) {
                    weights[fi][ti] = 1.0; // base generation weight
                } else if from.restrains(*to) {
                    weights[fi][ti] = -0.5; // restraint inhibits transition
                }
            }
        }
        ZhouGrid { current: initial, weights }
    }

    /// Set the current posture (姿 operation).
    pub fn set_pose(&mut self, bagua: Bagua) {
        self.current = bagua;
    }

    /// Apply a 生克 declaration — adjust weights for (from → to) transition.
    /// `sheng`: true = generation (生), false = restraint (克).
    pub fn apply_shengke(&mut self, from: Bagua, to: Bagua, sheng: bool) {
        let fi = from.to_bits() as usize;
        let ti = to.to_bits() as usize;
        if sheng {
            self.weights[fi][ti] += 0.1;
        } else {
            self.weights[fi][ti] -= 0.1;
        }
    }

    /// Propose the next posture based on learned weights.
    /// Uses softmax over weights from the current posture.
    pub fn next_pose<R: rand::Rng>(&self, rng: &mut R) -> Bagua {
        let fi = self.current.to_bits() as usize;
        let row = &self.weights[fi];

        // Softmax: exp(weight) for positive weights, small epsilon for negative
        let exp_weights: Vec<f32> = row.iter()
            .map(|&w| if w > 0.0 { w.exp() } else { 0.001 })
            .collect();
        let total: f32 = exp_weights.iter().sum();

        let threshold: f32 = rng.random::<f32>() * total;
        let mut cumulative = 0.0;
        for (i, &ew) in exp_weights.iter().enumerate() {
            cumulative += ew;
            if cumulative >= threshold {
                return Bagua::from_bits(i as u8).unwrap();
            }
        }
        // Fallback
        Bagua::坤
    }

    /// Current LLM temperature based on posture.
    pub fn current_temperature(&self) -> f32 {
        self.current.temperature()
    }
}

impl Default for ZhouGrid {
    fn default() -> Self {
        ZhouGrid::new(Bagua::坤) // Start in stable execution posture
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temperature_values() {
        assert!((Bagua::乾.temperature() - 1.2).abs() < f32::EPSILON);
        assert!((Bagua::兑.temperature() - 0.9).abs() < f32::EPSILON);
        assert!((Bagua::离.temperature() - 0.5).abs() < f32::EPSILON);
        assert!((Bagua::震.temperature() - 1.0).abs() < f32::EPSILON);
        assert!((Bagua::巽.temperature() - 0.7).abs() < f32::EPSILON);
        assert!((Bagua::坎.temperature() - 1.1).abs() < f32::EPSILON);
        assert!((Bagua::艮.temperature() - 0.3).abs() < f32::EPSILON);
        assert!((Bagua::坤.temperature() - 0.6).abs() < f32::EPSILON);
    }

    #[test]
    fn test_generates() {
        assert!(Bagua::震.generates(Bagua::离)); // 木生火
        assert!(!Bagua::离.generates(Bagua::震)); // 火不生木
        assert!(Bagua::兑.generates(Bagua::坎));  // 金生水
    }

    #[test]
    fn test_restrains() {
        assert!(Bagua::兑.restrains(Bagua::震)); // 金克木
        assert!(!Bagua::震.restrains(Bagua::兑)); // 木不克金
        assert!(Bagua::坎.restrains(Bagua::离));  // 水克火
    }

    #[test]
    fn test_bits_roundtrip() {
        for &bagua in &Bagua::ALL {
            let bits = bagua.to_bits();
            let recovered = Bagua::from_bits(bits).unwrap();
            assert_eq!(bagua, recovered);
        }
    }

    #[test]
    fn test_grid_new() {
        let grid = ZhouGrid::new(Bagua::坤);
        assert_eq!(grid.current, Bagua::坤);
        assert_eq!(grid.current_temperature(), 0.6);
    }

    #[test]
    fn test_set_pose() {
        let mut grid = ZhouGrid::new(Bagua::坤);
        grid.set_pose(Bagua::乾);
        assert_eq!(grid.current, Bagua::乾);
        assert_eq!(grid.current_temperature(), 1.2);
    }

    #[test]
    fn test_next_pose() {
        let grid = ZhouGrid::new(Bagua::震);
        let mut rng = rand::rng();
        // Test that next_pose always returns a valid Bagua
        for _ in 0..10 {
            let next = grid.next_pose(&mut rng);
            assert!(Bagua::ALL.contains(&next));
        }
    }

    #[test]
    fn test_apply_shengke() {
        let mut grid = ZhouGrid::default();
        let old_w = grid.weights[Bagua::震.to_bits() as usize][Bagua::离.to_bits() as usize];
        grid.apply_shengke(Bagua::震, Bagua::离, true);
        assert!(grid.weights[Bagua::震.to_bits() as usize][Bagua::离.to_bits() as usize] > old_w);
    }
}
