/// 周易语义层 (Semantic ZhouYi) — semantic cognitive-attitude encoder.
///
/// Implements the `AttitudeEncoder` trait using vector interpolation:
///   V_origin + V_strategy → interpolate → V_attitude + prompt_prefix
///
/// Operates alongside the existing `ZhouVM` (8-state Markov grid), not as
/// a replacement — consumers choose which to use via trait dispatch.

use xiang_core::bagua::Bagua;
use xiang_core::embedding::{cosine_similarity, Embedding, TextEncoder};
use xiang_core::semantic::{AttitudeEncoder, AttitudeInput, AttitudeOutput};

/// 8 base attitude vectors, one per Bagua posture.
/// Anchored in embedding space as canonical posture vectors.
const BASE_ATTITUDES: [(Bagua, [f32; 4]); 8] = [
    // 乾 (Qian) — Creation: high energy, expansive
    (Bagua::乾, [0.9, 0.7, 0.3, 0.1]),
    // 兑 (Dui) — Expression: communicative, outward
    (Bagua::兑, [0.6, 0.8, 0.4, 0.2]),
    // 离 (Li) — Illumination: clarity, analytical precision
    (Bagua::离, [0.7, 0.3, 0.8, 0.2]),
    // 震 (Zhen) — Initiation: motion, breakthrough
    (Bagua::震, [0.8, 0.4, 0.1, 0.7]),
    // 巽 (Xun) — Penetration: deep, persistent
    (Bagua::巽, [0.3, 0.7, 0.1, 0.8]),
    // 坎 (Kan) — Break: adaptive, problem-solving
    (Bagua::坎, [0.5, 0.2, 0.7, 0.6]),
    // 艮 (Gen) — Stillness: conservative, cautious
    (Bagua::艮, [0.1, 0.1, 0.3, 0.3]),
    // 坤 (Kun) — Bearing: supportive, stable, receptive
    (Bagua::坤, [0.4, 0.4, 0.5, 0.5]),
];

/// 8 prompt templates, one per Bagua posture.
/// Each template is a meta-instruction to be prepended to the user input.
const PROMPT_TEMPLATES: [(Bagua, &str); 8] = [
    (Bagua::乾, "以创造性思维展开回答。大胆假设，积极构建。"),
    (Bagua::兑, "以清晰流畅的表达回应。注重沟通的完整性与准确度。"),
    (Bagua::离, "以明辨洞察的方式分析。逐层拆解，追根溯源。"),
    (Bagua::震, "果断启动思维。直面核心问题，不绕弯子。"),
    (Bagua::巽, "深入渗透问题本质。持续深挖，不浮于表面。"),
    (Bagua::坎, "识别危险信号并绕行。寻找问题的突破口与替代路径。"),
    (Bagua::艮, "保持审慎。当前环境可能存在风险，需要稳扎稳打。"),
    (Bagua::坤, "以包容承载的立场回应。提供全面稳健的分析。"),
];

/// SemanticZhouYi — the semantic upgrade of the ZhouYi cognitive-attitude engine.
///
/// Takes V_origin (question focus) and V_strategy (from 连山) embeddings,
/// interpolates between the 8 base attitude vectors, and produces a
/// continuous V_attitude with associated temperature, Bagua pose, and
/// meta-prompt prefix.
pub struct SemanticZhouYi<E: TextEncoder> {
    /// Text encoder for embedding origin and strategy descriptions.
    pub encoder: E,
    /// Number of top base attitudes to blend.
    pub top_k: usize,
    /// Conservative bias: when danger is detected, shift toward 艮/坤.
    pub conservative_bias: f32,
    /// Whether danger was detected in the most recent encode.
    pub danger_detected: bool,
}

impl<E: TextEncoder> SemanticZhouYi<E> {
    pub fn new(encoder: E) -> Self {
        SemanticZhouYi {
            encoder,
            top_k: 3,
            conservative_bias: 0.3,
            danger_detected: false,
        }
    }

    pub fn with_params(
        encoder: E,
        top_k: usize,
        conservative_bias: f32,
    ) -> Self {
        SemanticZhouYi {
            encoder,
            top_k,
            conservative_bias,
            danger_detected: false,
        }
    }

    /// Set the danger flag — when true, bias toward conservative (低温度) postures.
    pub fn set_danger(&mut self, danger: bool) {
        self.danger_detected = danger;
    }

    /// Interpolate between the top-k base attitude vectors weighted by cosine
    /// similarity to the input blend.
    fn interpolate_attitude(
        blended: &[f32],
        top_k: usize,
        conservative_bias: f32,
        danger: bool,
    ) -> (Embedding, f32, Bagua) {
        // Score each base attitude
        let mut scored: Vec<(f32, &[f32], Bagua)> = BASE_ATTITUDES
            .iter()
            .map(|(bagua, anchor)| {
                // Pad or truncate to match dimensions
                let (blended_padded, anchor_padded): (Vec<f32>, Vec<f32>) = if blended.len() == anchor.len() {
                    (blended.to_vec(), anchor.to_vec())
                } else if blended.len() < anchor.len() {
                    let mut padded = blended.to_vec();
                    padded.resize(anchor.len(), 0.0);
                    (padded, anchor.to_vec())
                } else {
                    let mut padded = anchor.to_vec();
                    padded.resize(blended.len(), 0.0);
                    (blended.to_vec(), padded)
                };
                let mut sim = cosine_similarity(&blended_padded, &anchor_padded);
                // Apply conservative bias for 艮 and 坤 when danger detected
                if danger && (*bagua == Bagua::艮 || *bagua == Bagua::坤) {
                    sim += conservative_bias;
                }
                // Penalise 乾 (high temperature) when danger detected
                if danger && *bagua == Bagua::乾 {
                    sim -= conservative_bias * 0.5;
                }
                (sim, anchor.as_slice(), *bagua)
            })
            .collect();

        // Sort by similarity descending
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        // Blend the top-k vectors weighted by similarity
        let dim = blended.len();
        let mut v_attitude = vec![0.0f32; dim];
        let total_sim: f32 = scored.iter().map(|(s, _, _)| s.max(0.0)).sum();

        let mut weighted_temp = 0.0f32;
        let mut best_bagua = Bagua::坤;
        let mut best_sim = -1.0f32;

        for (sim, anchor, bagua) in &scored {
            let w = sim.max(0.0) / total_sim.max(0.001);
            for (i, &v) in anchor.iter().enumerate() {
                if i < dim {
                    v_attitude[i] += v * w;
                }
            }
            weighted_temp += bagua.temperature() * w;
            if *sim > best_sim {
                best_sim = *sim;
                best_bagua = *bagua;
            }
        }

        // Normalise
        let norm: f32 = v_attitude.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut v_attitude {
                *v /= norm;
            }
        }

        (v_attitude, weighted_temp, best_bagua)
    }

    /// Get the prompt template for a given Bagua posture.
    fn prompt_for(bagua: Bagua) -> String {
        for (b, template) in &PROMPT_TEMPLATES {
            if *b == bagua {
                return template.to_string();
            }
        }
        // Fallback to 坤
        "请给出客观、全面的回应。".to_string()
    }
}

impl<E: TextEncoder> AttitudeEncoder for SemanticZhouYi<E> {
    fn encode_attitude(&self, input: AttitudeInput) -> AttitudeOutput {
        // Use actual input embeddings from AttitudeInput
        let v_origin = &input.v_origin;
        let v_strategy = &input.v_strategy;

        let dim = v_origin.len().max(v_strategy.len());
        let mut blended = vec![0.0f32; dim];
        let half = v_origin.len().min(v_strategy.len());
        for i in 0..half {
            blended[i] = (v_origin[i] + v_strategy[i]) / 2.0;
        }

        // Interpolate attitude from base vectors
        let (v_attitude, temperature, pose) = Self::interpolate_attitude(
            &blended,
            self.top_k,
            self.conservative_bias,
            self.danger_detected,
        );

        let prompt_prefix = Self::prompt_for(pose);

        AttitudeOutput {
            v_attitude,
            temperature,
            pose,
            prompt_prefix,
            attention_bias: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xiang_core::embedding::MockEncoder;
    use xiang_core::embedding::MockEncoderMode;
    use xiang_core::semantic::AttitudeInput;

    fn make_encoder() -> MockEncoder {
        MockEncoder::new(8, MockEncoderMode::Focused)
    }

    fn make_attitude_input() -> AttitudeInput {
        AttitudeInput {
            v_origin: vec![0.5, 0.5, 0.5, 0.5],
            v_strategy: vec![0.8, 0.6, 0.2, 0.1],
        }
    }

    #[test]
    fn test_semantic_zhou_yi_creates() {
        let engine = SemanticZhouYi::new(make_encoder());
        let input = make_attitude_input();
        let output = engine.encode_attitude(input);
        assert!(!output.v_attitude.is_empty());
        assert!(output.temperature > 0.0);
        assert!(output.temperature <= 1.2);
        assert!(!output.prompt_prefix.is_empty());
    }

    #[test]
    fn test_deterministic_output() {
        let engine = SemanticZhouYi::new(make_encoder());
        let input = make_attitude_input();
        let out1 = engine.encode_attitude(input.clone());
        let out2 = engine.encode_attitude(input);
        assert_eq!(out1.v_attitude, out2.v_attitude);
        assert_eq!(out1.pose, out2.pose);
        assert!((out1.temperature - out2.temperature).abs() < 0.001);
    }

    #[test]
    fn test_danger_bias_to_low_temperature() {
        let mut engine = SemanticZhouYi::new(make_encoder());
        let input = make_attitude_input();

        engine.set_danger(false);
        let safe_out = engine.encode_attitude(input.clone());

        engine.set_danger(true);
        let danger_out = engine.encode_attitude(input);

        // When danger is detected, temperature should be lower or equal
        // (Conservative bias toward 艮(0.3) and 坤(0.6))
        assert!(danger_out.temperature <= safe_out.temperature + 0.01,
            "Danger temperature ({}) should be <= safe temperature ({})",
            danger_out.temperature, safe_out.temperature);
    }

    #[test]
    fn test_prompt_for_each_bagua() {
        for bagua in Bagua::ALL {
            let prompt = SemanticZhouYi::<MockEncoder>::prompt_for(bagua);
            assert!(!prompt.is_empty(), "Prompt should exist for {:?}", bagua);
        }
    }

    #[test]
    fn test_attention_bias_empty() {
        let engine = SemanticZhouYi::new(make_encoder());
        let output = engine.encode_attitude(make_attitude_input());
        // Attention bias is currently unimplemented (placeholder)
        assert!(output.attention_bias.is_empty());
    }
}
