pub mod gua;
pub mod cang_sea;
pub mod project_context;
pub mod bagua;
pub mod lianshan;
pub mod deviation;
pub mod embedding;
pub mod semantic;
pub mod yin_checker;

pub use gua::Gua;
pub use cang_sea::CangSea;
pub use cang_sea::{CangSeaEntry, SemanticStore, SemanticEntry, MergeStrategy, project_embedding_to_bits};
pub use project_context::{ProjectContext, DecisionEntry};
pub use bagua::{Bagua, ZhouGrid};
pub use lianshan::{FangWei, SixQi, SixJia, SanYuan, LianShanInput, LianShanDecision, ZhiForces};
pub use deviation::{deviation, hybrid_deviation, DeviationSource};
pub use embedding::{Embedding, MockEncoder, MockEncoderMode, TextEncoder, cosine_similarity, advance_drift};
pub use semantic::{StrategyInput, StrategyOutput, SemanticDecision,
                   AttitudeInput, AttitudeOutput, AttitudeEncoder};
pub use yin_checker::{YinProtocolChecker, OperatorRule, RuleResult, OutputStructure};
