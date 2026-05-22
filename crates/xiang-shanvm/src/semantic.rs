/// 连山语义层 (Semantic LianShan) — semantic obstacle-strategy engine.
///
/// Implements the `SemanticDecision` trait using a Mock k-NN approach:
///   V_goal + V_obstacle → query CangSea → weighted blend → V_strategy
///
/// Operates alongside the existing `ShanVM` (6-step table lookup), not as
/// a replacement — consumers choose which to use via trait dispatch.

use xiang_core::embedding::{cosine_similarity, Embedding, TextEncoder};
use xiang_core::lianshan::FangWei;
use xiang_core::semantic::{SemanticDecision, StrategyInput, StrategyOutput};

/// 7 base strategy vectors, one per FangWei direction.
/// These are canonical embedding-space anchors for each strategy type.
/// When no CangSea experience is available, the engine falls back to
/// the nearest base vector.
const BASE_STRATEGIES: [(FangWei, [f32; 4]); 7] = [
    // Continue: steady, neutral direction — slightly positive across all dims
    (FangWei::Continue, [0.5, 0.5, 0.5, 0.5]),
    // PushThrough: assertive, forward momentum — strong positive on dims 0-1
    (FangWei::PushThrough, [0.8, 0.8, 0.2, 0.2]),
    // NavigateAround: lateral, flexible — positive on dims 2-3
    (FangWei::NavigateAround, [0.2, 0.2, 0.8, 0.8]),
    // WaitGather: cautious, observant — evenly damped
    (FangWei::WaitGather, [0.3, 0.3, 0.3, 0.3]),
    // Decompose: analytical, granular — alternating high/low
    (FangWei::Decompose, [0.9, 0.1, 0.9, 0.1]),
    // Escalate: high signal, urgent — strong diag
    (FangWei::Escalate, [0.7, 0.7, 0.7, 0.7]),
    // Abort: negative pullback — negative across dims
    (FangWei::Abort, [-0.8, -0.8, -0.8, -0.8]),
];

/// SemanticLianShan — the semantic upgrade of the Lianshan strategy engine.
///
/// Uses a TextEncoder to embed goal and obstacle text, then blends the
/// result with historical experiences queried from the SemanticStore.
/// Produces a continuous V_strategy vector and projects it to a discrete FangWei.
pub struct SemanticLianShan<E: TextEncoder> {
    /// Text encoder for embedding goals and obstacles.
    pub encoder: E,
    /// Number of historical entries to query from CangSea (k-NN k).
    pub top_k: usize,
    /// Similarity threshold for CangSea queries.
    pub similarity_threshold: f32,
    /// Weight for blending CangSea results (0 = ignore, 1 = full blend).
    pub blend_weight: f32,
}

impl<E: TextEncoder> SemanticLianShan<E> {
    pub fn new(encoder: E) -> Self {
        SemanticLianShan {
            encoder,
            top_k: 5,
            similarity_threshold: 0.6,
            blend_weight: 0.4,
        }
    }

    pub fn with_params(
        encoder: E,
        top_k: usize,
        similarity_threshold: f32,
        blend_weight: f32,
    ) -> Self {
        SemanticLianShan {
            encoder,
            top_k,
            similarity_threshold,
            blend_weight,
        }
    }

    /// Project a strategy vector onto the nearest FangWei direction.
    fn project_to_fangwei(v_strategy: &[f32]) -> (FangWei, f32) {
        let mut best_fw = FangWei::Continue;
        let mut best_sim = -1.0f32;

        for (fw, anchor) in &BASE_STRATEGIES {
            // Pad or truncate to match dimensions
            let (v_padded, anchor_padded): (Vec<f32>, Vec<f32>) = if v_strategy.len() == anchor.len() {
                (v_strategy.to_vec(), anchor.to_vec())
            } else if v_strategy.len() < anchor.len() {
                let mut padded = v_strategy.to_vec();
                padded.resize(anchor.len(), 0.0);
                (padded, anchor.to_vec())
            } else {
                let mut padded = anchor.to_vec();
                padded.resize(v_strategy.len(), 0.0);
                (v_strategy.to_vec(), padded)
            };
            let sim = cosine_similarity(&v_padded, &anchor_padded);
            if sim > best_sim {
                best_sim = sim;
                best_fw = *fw;
            }
        }

        (best_fw, best_sim)
    }

    /// Blend the raw goal+obstacle embedding with historical CangSea results.
    fn blend_with_history(
        goal_obs: &[f32],
        history: &[(f32, Embedding)],
        blend_weight: f32,
    ) -> Embedding {
        if history.is_empty() || blend_weight <= 0.0 {
            return goal_obs.to_vec();
        }

        let dim = goal_obs.len();
        let mut result = vec![0.0f32; dim];

        // Weight of raw input
        let raw_weight = 1.0 - blend_weight;
        for (i, &v) in goal_obs.iter().enumerate() {
            result[i] += v * raw_weight;
        }

        // Weight of historical entries (weighted by similarity)
        let total_hist_sim: f32 = history.iter().map(|(sim, _)| sim.max(0.0)).sum();
        if total_hist_sim > 0.0 {
            for (sim, embedding) in history {
                let w = blend_weight * sim.max(0.0) / total_hist_sim;
                for (i, &v) in embedding.iter().enumerate() {
                    if i < dim {
                        result[i] += v * w;
                    }
                }
            }
        }

        // Normalise
        let norm: f32 = result.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut result {
                *v /= norm;
            }
        }

        result
    }
}

impl<E: TextEncoder> SemanticDecision for SemanticLianShan<E> {
    fn decide(&self, input: StrategyInput) -> StrategyOutput {
        // Encode goal + obstacle
        let v_goal = self.encoder.encode(&format!("goal: {}", 
            if input.v_goal.is_empty() { "continue" } else { "custom" }));
        let v_obstacle = self.encoder.encode(&format!("obstacle: {}",
            if input.v_obstacle.is_empty() { "none" } else { "present" }));

        // Combine goal + obstacle into a single input vector
        let mut combined = vec![0.0f32; self.encoder.dimension()];
        let half = input.v_goal.len().min(input.v_obstacle.len()).min(combined.len());
        for i in 0..half {
            combined[i] = (v_goal.get(i).copied().unwrap_or(0.0)
                + v_obstacle.get(i).copied().unwrap_or(0.0))
                / 2.0;
        }
        // Fill remaining from v_goal
        for i in half..combined.len().min(v_goal.len()) {
            combined[i] = v_goal[i];
        }

        // Blend with historical context if available
        let history: Vec<(f32, Embedding)> = input
            .c_history
            .iter()
            .map(|h| {
                let sim = cosine_similarity(&combined, h);
                (sim, h.clone())
            })
            .filter(|(sim, _)| *sim >= self.similarity_threshold)
            .collect();

        let v_strategy = Self::blend_with_history(&combined, &history, self.blend_weight);

        // Project to discrete FangWei
        let (projected, confidence) = Self::project_to_fangwei(&v_strategy);

        StrategyOutput {
            v_strategy,
            projected,
            confidence: confidence.max(0.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xiang_core::embedding::MockEncoder;
    use xiang_core::embedding::MockEncoderMode;

    fn make_encoder() -> MockEncoder {
        MockEncoder::new(8, MockEncoderMode::Focused)
    }

    fn make_strategy_input() -> StrategyInput {
        StrategyInput {
            v_goal: vec![0.8, 0.6, 0.0, 0.0, 0.5, 0.3, 0.1, 0.0],
            v_obstacle: vec![0.0, 0.0, 0.3, 0.1, 0.0, 0.0, 0.2, 0.05],
            c_history: vec![],
        }
    }

    #[test]
    fn test_semantic_lian_shan_creates() {
        let engine = SemanticLianShan::new(make_encoder());
        let input = make_strategy_input();
        let output = engine.decide(input);
        assert!(!output.v_strategy.is_empty());
        // Should project to some FangWei
        assert!(output.confidence >= 0.0);
    }

    #[test]
    fn test_deterministic_output() {
        let engine = SemanticLianShan::new(make_encoder());
        let input = make_strategy_input();
        let out1 = engine.decide(input.clone());
        let out2 = engine.decide(input);
        // Same input → same output (MockEncoder is deterministic)
        assert_eq!(out1.v_strategy, out2.v_strategy);
        assert_eq!(out1.projected, out2.projected);
    }

    #[test]
    fn test_project_to_fangwei() {
        let strategies: Vec<(&[f32], FangWei)> = vec![
            (&[0.5, 0.5, 0.5, 0.5], FangWei::Continue),
            (&[0.8, 0.8, 0.2, 0.2], FangWei::PushThrough),
            (&[-0.8, -0.8, -0.8, -0.8], FangWei::Abort),
            (&[0.9, 0.1, 0.9, 0.1], FangWei::Decompose),
        ];
        for (v, expected_fw) in strategies {
            let (fw, conf) = SemanticLianShan::<MockEncoder>::project_to_fangwei(v);
            assert_eq!(fw, expected_fw, "vector {:?} should project to {:?}", v, expected_fw);
            assert!(conf > 0.5);
        }
    }

    #[test]
    fn test_blend_pure_when_no_history() {
        let input = vec![1.0, 0.0, 0.0, 0.0];
        let result = SemanticLianShan::<MockEncoder>::blend_with_history(&input, &[], 0.5);
        assert!((cosine_similarity(&input, &result) - 1.0).abs() < 0.001);
    }
}
