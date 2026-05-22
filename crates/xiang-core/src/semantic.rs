/// 语义决策与姿态编码 (Semantic Decision & Attitude Encoding) — core traits.
///
/// These traits define the interface for the semantic upgrade of
/// 连山 (LianShan obstacle-strategy) and 周易 (ZhouYi cognitive-attitude).

use crate::bagua::Bagua;
use crate::embedding::Embedding;
use crate::lianshan::FangWei;

// ─── 连山 (LianShan) Semantic Types ──────────────────────────────

/// Input to the semantic LianShan strategy engine.
#[derive(Debug, Clone)]
pub struct StrategyInput {
    /// Embedding of the current goal text.
    pub v_goal: Embedding,
    /// Embedding of the current obstacle text.
    pub v_obstacle: Embedding,
    /// Historical context vectors (previous strategies or state summaries).
    /// Empty for the first turn.
    pub c_history: Vec<Embedding>,
}

/// Output from the semantic LianShan strategy engine.
#[derive(Debug, Clone)]
pub struct StrategyOutput {
    /// The strategy vector in embedding space.
    pub v_strategy: Embedding,
    /// Projected discrete direction (best-match FangWei).
    pub projected: FangWei,
    /// Confidence score in [0.0, 1.0].
    pub confidence: f32,
}

/// SemanticDecision trait — semantic obstacle-strategy engine.
///
/// Takes goal + obstacle embeddings and produces a strategy vector
/// with a projected discrete direction and confidence score.
pub trait SemanticDecision: Send + Sync {
    /// Decide on a strategy given goal, obstacle, and historical context.
    fn decide(&self, input: StrategyInput) -> StrategyOutput;
}

// ─── 周易 (ZhouYi) Semantic Types ────────────────────────────────

/// Input to the semantic ZhouYi cognitive-attitude encoder.
#[derive(Debug, Clone)]
pub struct AttitudeInput {
    /// Embedding of the origin (question focus).
    pub v_origin: Embedding,
    /// Embedding of the current strategy (from LianShan).
    pub v_strategy: Embedding,
}

/// Output from the semantic ZhouYi cognitive-attitude encoder.
#[derive(Debug, Clone)]
pub struct AttitudeOutput {
    /// The attitude vector in embedding space.
    pub v_attitude: Embedding,
    /// LLM temperature for this attitude posture.
    pub temperature: f32,
    /// Projected discrete Bagua pose.
    pub pose: Bagua,
    /// Meta-instruction text to prepend to the user prompt.
    pub prompt_prefix: String,
    /// Attention bias hints (token IDs to attend to / suppress).
    /// Can be empty.
    pub attention_bias: Vec<(u32, f32)>,
}

/// AttitudeEncoder trait — semantic cognitive-attitude engine.
///
/// Takes origin + strategy embeddings and produces an attitude vector
/// with associated temperature, pose, and prompt prefix.
pub trait AttitudeEncoder: Send + Sync {
    /// Encode origin + strategy into a cognitive attitude.
    fn encode_attitude(&self, input: AttitudeInput) -> AttitudeOutput;
}
