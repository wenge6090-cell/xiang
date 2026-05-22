/// Verifier — 诚信验证引擎 (Honesty Verification Engine).
///
/// Monitors agent declarations (言/诚 statements) and verifies them against
/// actual behavior (Gua state trajectory, deviation, sha activity, load).
/// Computes a trust score based on the ratio of passing to total declarations.
///
/// Conditions verified:
///   - 卦条件: current gua matches declared pattern
///   - 偏条件: deviation is within declared bounds
///   - 无杀意: no sha intervention has occurred
///   - 载重条件: load/weight is within declared bounds

use xiang_core::Gua;
use xiang_parser::ast::{ChengshiStmt, ChengshiCondition, CmpOp};

/// Context needed for verification — provided by the caller (CangVM).
pub struct VerifyContext {
    /// Current 6-bit Gua state.
    pub current_gua: Gua,
    /// Current deviation from origin.
    pub current_deviation: f32,
    /// Whether any sha (杀) has been executed since last verification.
    pub has_sha: bool,
    /// Current load metric (e.g., total CangSea weight sum).
    pub current_load: f32,
}

impl VerifyContext {
    pub fn new(gua: Gua, deviation: f32) -> Self {
        VerifyContext {
            current_gua: gua,
            current_deviation: deviation,
            has_sha: false,
            current_load: 0.0,
        }
    }
}

/// Result of verifying a single 诚 condition.
#[derive(Debug, Clone)]
pub enum VerifyResult {
    /// Condition passed.
    Pass,
    /// Condition failed with a reason.
    Fail(String),
}

impl VerifyResult {
    pub fn is_pass(&self) -> bool {
        matches!(self, VerifyResult::Pass)
    }
}

/// Honesty verification engine.
pub struct Verifier {
    /// All registered declarations.
    pub declarations: Vec<ChengshiStmt>,
    /// Verification results: (declared_text, condition_index, result).
    pub verify_log: Vec<(String, usize, VerifyResult)>,
    /// Trust score ∈ [0, 1], where 1 = perfectly honest.
    pub trust_score: f32,
    /// Total number of verifications performed.
    pub total_checks: u32,
    /// Number of failed verifications.
    pub failed_checks: u32,
    /// Violated declarations (full text of declarations that failed).
    pub violations: Vec<String>,
}

impl Verifier {
    pub fn new() -> Self {
        Verifier {
            declarations: Vec::new(),
            verify_log: Vec::new(),
            trust_score: 1.0,
            total_checks: 0,
            failed_checks: 0,
            violations: Vec::new(),
        }
    }

    /// Register a honesty declaration from the program.
    pub fn declare(&mut self, stmt: &ChengshiStmt) {
        self.declarations.push(stmt.clone());
    }

    /// Verify all registered declarations against the current context.
    /// Returns true if all declarations pass, false otherwise.
    pub fn verify_all(&mut self, ctx: &VerifyContext) -> bool {
        // Clone declarations to avoid borrow conflict with verify_one (which takes &mut self)
        let decls: Vec<ChengshiStmt> = self.declarations.iter().cloned().collect();
        let mut all_pass = true;
        for decl in &decls {
            let decl_passes = self.verify_one(decl, ctx);
            if !decl_passes {
                all_pass = false;
                self.violations.push(decl.declaration.clone());
            }
        }
        self.update_trust_score();
        all_pass
    }

    /// Verify a single declaration against the context.
    fn verify_one(&mut self, decl: &ChengshiStmt, ctx: &VerifyContext) -> bool {
        if decl.conditions.is_empty() {
            // No conditions = always passes
            self.total_checks += 1;
            return true;
        }

        let mut all_pass = true;
        for (cond_idx, cond) in decl.conditions.iter().enumerate() {
            let result = Self::verify_condition(cond, ctx, decl);
            self.verify_log.push((decl.declaration.clone(), cond_idx, result.clone()));
            self.total_checks += 1;
            if !result.is_pass() {
                all_pass = false;
                self.failed_checks += 1;
            }
        }
        all_pass
    }

    /// Verify a single condition.
    fn verify_condition(
        cond: &ChengshiCondition,
        ctx: &VerifyContext,
        decl: &ChengshiStmt,
    ) -> VerifyResult {
        match cond {
            ChengshiCondition::GuaCondition(expected) => {
                // Check if current gua matches declared gua, or if declared gua matches
                let target = decl.declared_gua.unwrap_or(*expected);
                if ctx.current_gua == target {
                    VerifyResult::Pass
                } else {
                    VerifyResult::Fail(format!(
                        "卦不匹配: 当前{}  != 声明{}",
                        ctx.current_gua, target
                    ))
                }
            }
            ChengshiCondition::DeviationCondition { cmp, value } => {
                let ok = cmp_compare(*cmp, ctx.current_deviation, *value);
                if ok {
                    VerifyResult::Pass
                } else {
                    VerifyResult::Fail(format!(
                        "偏差超限: 当前{:.3} {} {:.3} 不成立",
                        ctx.current_deviation, cmp_symbol(*cmp), value
                    ))
                }
            }
            ChengshiCondition::NoShaYi => {
                if ctx.has_sha {
                    VerifyResult::Fail("检测到杀意".to_string())
                } else {
                    VerifyResult::Pass
                }
            }
            ChengshiCondition::LoadCondition { cmp, value } => {
                let ok = cmp_compare(*cmp, ctx.current_load, *value);
                if ok {
                    VerifyResult::Pass
                } else {
                    VerifyResult::Fail(format!(
                        "载重超限: 当前{:.3} {} {:.3} 不成立",
                        ctx.current_load, cmp_symbol(*cmp), value
                    ))
                }
            }
        }
    }

    /// Update the trust score based on pass/fail ratio.
    fn update_trust_score(&mut self) {
        if self.total_checks == 0 {
            self.trust_score = 1.0;
        } else {
            self.trust_score = (self.total_checks - self.failed_checks) as f32
                / self.total_checks as f32;
        }
    }

    /// Reset verification state for a new cycle.
    pub fn reset_cycle(&mut self) {
        self.verify_log.clear();
        // Don't reset trust score — it accumulates across cycles
    }

    /// Check if the agent is currently trustworthy.
    pub fn is_trustworthy(&self) -> bool {
        self.trust_score >= 0.8
    }

    /// Get a summary of verification state.
    pub fn summary(&self) -> String {
        format!(
            "信任度: {:.1}% ({}/{})",
            self.trust_score * 100.0,
            self.total_checks - self.failed_checks,
            self.total_checks
        )
    }
}

impl Default for Verifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Compare two f32 values with the given operator.
fn cmp_compare(op: CmpOp, a: f32, b: f32) -> bool {
    match op {
        CmpOp::Gt => a > b,
        CmpOp::Lt => a < b,
        CmpOp::Eq => (a - b).abs() < f32::EPSILON,
        CmpOp::Ge => a >= b,
        CmpOp::Le => a <= b,
    }
}

/// Symbol representation for comparison operators.
fn cmp_symbol(op: CmpOp) -> &'static str {
    match op {
        CmpOp::Gt => ">",
        CmpOp::Lt => "<",
        CmpOp::Eq => "==",
        CmpOp::Ge => ">=",
        CmpOp::Le => "<=",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ctx(gua: Gua, deviation: f32) -> VerifyContext {
        VerifyContext::new(gua, deviation)
    }

    #[test]
    fn test_verifier_new() {
        let v = Verifier::new();
        assert_eq!(v.trust_score, 1.0);
        assert!(v.declarations.is_empty());
    }

    #[test]
    fn test_empty_conditions_always_pass() {
        let decl = ChengshiStmt {
            declaration: "我是诚实的".to_string(),
            declared_gua: None,
            conditions: vec![],
        };
        let mut v = Verifier::new();
        v.declare(&decl);
        let ctx = make_ctx(Gua::ORIGIN, 0.0);
        assert!(v.verify_all(&ctx));
        assert_eq!(v.trust_score, 1.0);
    }

    #[test]
    fn test_gua_condition_pass() {
        let decl = ChengshiStmt {
            declaration: "状态为111111".to_string(),
            declared_gua: Some(Gua::ORIGIN),
            conditions: vec![ChengshiCondition::GuaCondition(Gua::ORIGIN)],
        };
        let mut v = Verifier::new();
        v.declare(&decl);
        let ctx = make_ctx(Gua::ORIGIN, 0.0);
        assert!(v.verify_all(&ctx));
    }

    #[test]
    fn test_gua_condition_fail() {
        let decl = ChengshiStmt {
            declaration: "状态为111111".to_string(),
            declared_gua: Some(Gua::ORIGIN),
            conditions: vec![ChengshiCondition::GuaCondition(Gua::ORIGIN)],
        };
        let mut v = Verifier::new();
        v.declare(&decl);
        let ctx = make_ctx(Gua(0b000000), 1.0);
        assert!(!v.verify_all(&ctx));
        assert!(v.trust_score < 1.0);
        assert_eq!(v.failed_checks, 1);
    }

    #[test]
    fn test_deviation_condition_pass() {
        let decl = ChengshiStmt {
            declaration: "偏差小于0.5".to_string(),
            declared_gua: None,
            conditions: vec![ChengshiCondition::DeviationCondition {
                cmp: CmpOp::Lt,
                value: 0.5,
            }],
        };
        let mut v = Verifier::new();
        v.declare(&decl);
        let ctx = make_ctx(Gua(0b101101), 0.3);
        assert!(v.verify_all(&ctx));
    }

    #[test]
    fn test_deviation_condition_fail() {
        let decl = ChengshiStmt {
            declaration: "偏差小于0.5".to_string(),
            declared_gua: None,
            conditions: vec![ChengshiCondition::DeviationCondition {
                cmp: CmpOp::Lt,
                value: 0.5,
            }],
        };
        let mut v = Verifier::new();
        v.declare(&decl);
        let ctx = make_ctx(Gua(0b000000), 1.0);
        assert!(!v.verify_all(&ctx));
        assert_eq!(v.failed_checks, 1);
    }

    #[test]
    fn test_no_sha_yi_pass() {
        let decl = ChengshiStmt {
            declaration: "无杀意".to_string(),
            declared_gua: None,
            conditions: vec![ChengshiCondition::NoShaYi],
        };
        let mut v = Verifier::new();
        v.declare(&decl);
        let ctx = VerifyContext {
            current_gua: Gua::ZERO,
            current_deviation: 0.0,
            has_sha: false,
            current_load: 0.0,
        };
        assert!(v.verify_all(&ctx));
    }

    #[test]
    fn test_no_sha_yi_fail() {
        let decl = ChengshiStmt {
            declaration: "无杀意".to_string(),
            declared_gua: None,
            conditions: vec![ChengshiCondition::NoShaYi],
        };
        let mut v = Verifier::new();
        v.declare(&decl);
        let ctx = VerifyContext {
            current_gua: Gua::ZERO,
            current_deviation: 0.0,
            has_sha: true,
            current_load: 0.0,
        };
        assert!(!v.verify_all(&ctx));
        assert_eq!(v.violations.len(), 1);
    }

    #[test]
    fn test_load_condition() {
        let decl = ChengshiStmt {
            declaration: "载重限制".to_string(),
            declared_gua: None,
            conditions: vec![ChengshiCondition::LoadCondition {
                cmp: CmpOp::Lt,
                value: 10.0,
            }],
        };
        let mut v = Verifier::new();
        v.declare(&decl);
        // Under limit
        let ctx = VerifyContext {
            current_gua: Gua::ZERO,
            current_deviation: 0.0,
            has_sha: false,
            current_load: 5.0,
        };
        assert!(v.verify_all(&ctx));
        // Over limit
        let ctx2 = VerifyContext {
            current_gua: Gua::ZERO,
            current_deviation: 0.0,
            has_sha: false,
            current_load: 15.0,
        };
        assert!(!v.verify_all(&ctx2));
    }

    #[test]
    fn test_multiple_conditions_all_must_pass() {
        let decl = ChengshiStmt {
            declaration: "综合诚信检查".to_string(),
            declared_gua: Some(Gua::ORIGIN),
            conditions: vec![
                ChengshiCondition::GuaCondition(Gua::ORIGIN),
                ChengshiCondition::DeviationCondition { cmp: CmpOp::Lt, value: 0.5 },
                ChengshiCondition::NoShaYi,
            ],
        };
        let mut v = Verifier::new();
        v.declare(&decl);
        let ctx = VerifyContext {
            current_gua: Gua::ORIGIN,
            current_deviation: 0.2,
            has_sha: false,
            current_load: 0.0,
        };
        assert!(v.verify_all(&ctx));
        assert_eq!(v.trust_score, 1.0);
    }

    #[test]
    fn test_trust_score_decays_on_failure() {
        let mut v = Verifier::new();
        let good_decl = ChengshiStmt {
            declaration: "ok".to_string(),
            declared_gua: None,
            conditions: vec![],
        };
        let bad_decl = ChengshiStmt {
            declaration: "bad".to_string(),
            declared_gua: Some(Gua::ORIGIN),
            conditions: vec![ChengshiCondition::GuaCondition(Gua::ORIGIN)],
        };
        v.declare(&good_decl);
        v.declare(&bad_decl);
        let ctx = make_ctx(Gua::ZERO, 0.0);
        assert!(!v.verify_all(&ctx));
        // 1 empty pass + 1 fail = 1/2
        assert!((v.trust_score - 0.5).abs() < 0.01);
        assert_eq!(v.failed_checks, 1);
    }

    #[test]
    fn test_is_trustworthy() {
        let mut v = Verifier::new();
        assert!(v.is_trustworthy()); // 1.0 > 0.8

        // Simulate failures
        v.total_checks = 10;
        v.failed_checks = 3;
        v.update_trust_score();
        assert!(!v.is_trustworthy()); // 0.7 < 0.8

        v.failed_checks = 2;
        v.update_trust_score();
        assert!(v.is_trustworthy()); // 0.8 >= 0.8
    }

    #[test]
    fn test_summary() {
        let mut v = Verifier::new();
        v.total_checks = 20;
        v.failed_checks = 2;
        v.update_trust_score();
        let s = v.summary();
        assert!(s.contains("90.0%"));
        assert!(s.contains("18/20"));
    }
}
