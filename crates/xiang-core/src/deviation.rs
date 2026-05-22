/// 偏离度 (Deviation) — standalone calculation.
///
/// `deviation(state, origin)` computes how far the current state vector
/// has drifted from the origin (question focus vector).
///
/// Returns a value in [0, 1], where 0 = perfect focus alignment and 1 = complete deviation.

use crate::gua::Gua;

/// Compute the deviation of `state` from `origin`.
///
/// Formula: `1.0 - popcount(~(state ^ origin) & 0x3F) / 6.0`
///
/// This is equivalent to the Hamming distance normalized to [0, 1]:
/// `hamming(state, origin) / 6.0`
pub fn deviation(state: Gua, origin: Gua) -> f32 {
    state.hamming(origin) as f32 / 6.0
}

// ─── Hybrid Deviation (Haming + Semantic) ────────────────────────

/// Source of the deviation component.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeviationSource {
    /// Pure Hamming distance (6-bit Gua).
    Hamming,
    /// Hybrid: Hamming + Cosine semantic deviation.
    Hybrid { alpha: f32, hamming: f32, semantic: f32 },
}

/// Compute a hybrid deviation score combining symbolic (Hamming) and
/// semantic (cosine) deviation.
///
/// `hamming_dev` — current Hamming deviation in [0.0, 1.0]
/// `semantic_dev_opt` — semantic deviation in [0.0, 1.0]; None = disabled
/// `alpha` — weight of Hamming vs semantic: alpha · hamming + (1-alpha) · semantic
///
/// Returns a value in [0.0, 1.0]. If `semantic_dev_opt` is None, returns
/// pure Hamming deviation (identical to `deviation()`).
pub fn hybrid_deviation(
    hamming_dev: f32,
    semantic_dev_opt: Option<f32>,
    alpha: f32,
) -> f32 {
    match semantic_dev_opt {
        None => hamming_dev,
        Some(semantic_dev) => {
            let alpha_clamped = alpha.clamp(0.0, 1.0);
            alpha_clamped * hamming_dev + (1.0 - alpha_clamped) * semantic_dev
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gua::Gua;

    #[test]
    fn test_deviation_zero() {
        assert_eq!(deviation(Gua::ORIGIN, Gua::ORIGIN), 0.0);
        assert_eq!(deviation(Gua(0), Gua(0)), 0.0);
    }

    #[test]
    fn test_deviation_full() {
        assert_eq!(deviation(Gua(0), Gua::ORIGIN), 1.0);
        assert_eq!(deviation(Gua::ORIGIN, Gua(0)), 1.0);
    }

    #[test]
    fn test_deviation_half() {
        assert_eq!(deviation(Gua(0b111000), Gua::ORIGIN), 0.5);
        assert_eq!(deviation(Gua(0b000111), Gua::ORIGIN), 0.5);
    }

    #[test]
    fn test_deviation_consistent_with_gua_gui() {
        for i in 0..64 {
            let g = Gua(i);
            let expected = g.gui(Gua::ORIGIN);
            let actual = deviation(g, Gua::ORIGIN);
            assert!((expected - actual).abs() < f32::EPSILON,
                "deviation vs gui mismatch for {i}: {expected} vs {actual}");
        }
    }

    #[test]
    fn test_hybrid_fallback_to_pure_hamming() {
        assert_eq!(hybrid_deviation(0.3, None, 0.5), 0.3);
        assert_eq!(hybrid_deviation(1.0, None, 0.5), 1.0);
    }

    #[test]
    fn test_hybrid_alpha_one_full_hamming() {
        let result = hybrid_deviation(0.3, Some(0.9), 1.0);
        assert!((result - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_hybrid_alpha_zero_full_semantic() {
        let result = hybrid_deviation(0.3, Some(0.9), 0.0);
        assert!((result - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_hybrid_alpha_half() {
        let result = hybrid_deviation(0.2, Some(0.8), 0.5);
        let expected = 0.5 * 0.2 + 0.5 * 0.8;
        assert!((result - expected).abs() < 0.001);
    }
}
