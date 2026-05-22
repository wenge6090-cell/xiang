/// ShanVM — 连山 (Lianshan) strategy engine.
///
/// Implements the 6-step decision chain for navigating obstacles during task execution:
///   艮 (gate) → 气 (phase) → 甲 (obstruction level) → 元 (freshness) → 峙 (forces) → 决 (decision)
///
/// Produces one of 7 directional decisions that guide the agent's next action.

pub mod semantic;

use xiang_core::{Gua, FangWei, SixQi, SixJia, SanYuan, LianShanInput, LianShanDecision, ZhiForces};

/// ShanVM — the Lianshan strategy execution engine.
///
/// Consumes the current Gua state, deviation, and task metadata to produce
/// a directional decision via the 6-step chain.
pub struct ShanVM {
    /// Current Gua state (shared with CangVM).
    pub state: Gua,
    /// Current deviation from origin.
    pub deviation: f32,
    /// Number of active tasks.
    pub task_count: usize,
    /// How many times the current goal has been blocked.
    pub obstruction_count: usize,
    /// Rounds since last perception (感).
    pub rounds_since_perception: usize,
    /// Whether the current strategy has been verified effective.
    pub strategy_verified: bool,
    /// Decision log: history of all decisions.
    pub decision_log: Vec<LianShanDecision>,
    /// Learned forces from CangSea for the current obstacle pattern.
    /// Populated externally before decide() is called.
    pub forces: ZhiForces,
}

impl ShanVM {
    pub fn new() -> Self {
        ShanVM {
            state: Gua::ZERO,
            deviation: 0.0,
            task_count: 0,
            obstruction_count: 0,
            rounds_since_perception: 0,
            strategy_verified: false,
            decision_log: Vec::new(),
            forces: ZhiForces::empty(),
        }
    }

    /// Build the input structure for the Lianshan decision engine.
    pub fn build_input(&self) -> LianShanInput {
        LianShanInput {
            task_count: self.task_count,
            obstruction_count: self.obstruction_count,
            rounds_since_perception: self.rounds_since_perception,
            strategy_verified: self.strategy_verified,
            forces: self.forces.clone(),
        }
    }

    /// Run the full 6-step decision chain.
    pub fn decide(&mut self) -> LianShanDecision {
        let input = self.build_input();
        let decision = Self::run_chain(&input);
        self.decision_log.push(decision.clone());
        decision
    }

    /// Execute the 6-step Lianshan chain from an input.
    pub fn run_chain(input: &LianShanInput) -> LianShanDecision {
        // Step 1: 艮 — activation gate
        let activated = input.task_count >= 2;

        // Step 2: 气 — goal progress phase based on deviation proxy
        // deviation is passed from outside (CangVM), use obstruction_count and verification
        let qi = Self::classify_qi(input);

        // Step 3: 甲 — obstruction count level
        let jia = Self::classify_jia(input.obstruction_count);

        // Step 4: 元 — context freshness
        let yuan = Self::classify_yuan(input.rounds_since_perception);

        // Step 5: 峙 — forces (use provided forces)
        let forces = input.forces.clone();

        // Step 6: 决 — final directional decision
        let decision = Self::make_decision(activated, qi, jia, yuan, &forces);

        LianShanDecision {
            activated,
            qi,
            jia,
            yuan,
            forces,
            decision,
        }
    }

    /// Step 2: Classify the goal progress phase.
    fn classify_qi(input: &LianShanInput) -> SixQi {
        if input.obstruction_count == 0 && input.strategy_verified {
            SixQi::夏 // Strategy working, full push
        } else if input.obstruction_count == 0 {
            SixQi::春 // Just starting
        } else if input.obstruction_count >= 3 {
            SixQi::冬 // Severely blocked
        } else {
            SixQi::秋 // Out of summer, need adjustment
        }
    }

    /// Step 3: Classify obstruction level.
    fn classify_jia(count: usize) -> SixJia {
        match count {
            0 => SixJia::初,
            1 => SixJia::再,
            _ => SixJia::三,
        }
    }

    /// Step 4: Classify context freshness.
    fn classify_yuan(rounds: usize) -> SanYuan {
        if rounds <= 1 { SanYuan::新 } else { SanYuan::陈 }
    }

    /// Step 6: Make the final directional decision.
    fn make_decision(
        activated: bool,
        qi: SixQi,
        jia: SixJia,
        yuan: SanYuan,
        forces: &ZhiForces,
    ) -> FangWei {
        if !activated {
            return FangWei::Continue;
        }

        // If severely blocked (冬) and stale context (陈), abort
        if qi == SixQi::冬 && yuan == SanYuan::陈 {
            return FangWei::Abort;
        }

        // No CangSea experience and early in the task → wait and gather
        if forces.push_forces.is_empty()
            && forces.resist_forces.is_empty()
            && qi == SixQi::春
            && jia == SixJia::初
        {
            return FangWei::WaitGather;
        }

        // If we have learned push forces, try pushing through
        if !forces.push_forces.is_empty() && qi != SixQi::冬 {
            return FangWei::PushThrough;
        }

        // If we have resist forces, try navigating around
        if !forces.resist_forces.is_empty() {
            return FangWei::NavigateAround;
        }

        // Use jia-based suggestion
        match jia {
            SixJia::初 => {
                if qi == SixQi::春 {
                    FangWei::Continue // Trial phase, keep going
                } else {
                    FangWei::PushThrough
                }
            }
            SixJia::再 => {
                match qi {
                    SixQi::秋 => FangWei::NavigateAround,
                    SixQi::冬 => FangWei::Decompose,
                    _ => FangWei::Continue,
                }
            }
            SixJia::三 => {
                match qi {
                    SixQi::冬 => FangWei::Escalate,
                    _ => FangWei::Decompose,
                }
            }
        }
    }
}

impl Default for ShanVM {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shanvm_new() {
        let vm = ShanVM::new();
        assert_eq!(vm.task_count, 0);
        assert_eq!(vm.obstruction_count, 0);
        assert!(vm.decision_log.is_empty());
    }

    #[test]
    fn test_not_activated_single_task() {
        let input = LianShanInput {
            task_count: 1,
            obstruction_count: 2,
            rounds_since_perception: 0,
            strategy_verified: false,
            forces: ZhiForces::empty(),
        };
        let decision = ShanVM::run_chain(&input);
        assert!(!decision.activated);
        assert_eq!(decision.decision, FangWei::Continue);
    }

    #[test]
    fn test_activated_two_tasks() {
        let input = LianShanInput {
            task_count: 2,
            obstruction_count: 0,
            rounds_since_perception: 0,
            strategy_verified: true,
            forces: ZhiForces::empty(),
        };
        let decision = ShanVM::run_chain(&input);
        assert!(decision.activated);
        assert_eq!(decision.qi, SixQi::夏);
    }

    #[test]
    fn test_qi_classification() {
        // Spring: no obstruction, not verified
        let input = LianShanInput {
            task_count: 0, obstruction_count: 0,
            rounds_since_perception: 0, strategy_verified: false,
            forces: ZhiForces::empty(),
        };
        assert_eq!(ShanVM::run_chain(&input).qi, SixQi::春);

        // Summer: no obstruction, verified
        let input = LianShanInput {
            task_count: 0, obstruction_count: 0,
            rounds_since_perception: 0, strategy_verified: true,
            forces: ZhiForces::empty(),
        };
        assert_eq!(ShanVM::run_chain(&input).qi, SixQi::夏);

        // Autumn: 1 obstruction
        let input = LianShanInput {
            task_count: 0, obstruction_count: 1,
            rounds_since_perception: 0, strategy_verified: false,
            forces: ZhiForces::empty(),
        };
        assert_eq!(ShanVM::run_chain(&input).qi, SixQi::秋);

        // Winter: 3+ obstructions
        let input = LianShanInput {
            task_count: 0, obstruction_count: 5,
            rounds_since_perception: 0, strategy_verified: false,
            forces: ZhiForces::empty(),
        };
        assert_eq!(ShanVM::run_chain(&input).qi, SixQi::冬);
    }

    #[test]
    fn test_jia_classification() {
        assert_eq!(ShanVM::classify_jia(0), SixJia::初);
        assert_eq!(ShanVM::classify_jia(1), SixJia::再);
        assert_eq!(ShanVM::classify_jia(2), SixJia::三);
        assert_eq!(ShanVM::classify_jia(100), SixJia::三);
    }

    #[test]
    fn test_yuan_classification() {
        assert_eq!(ShanVM::classify_yuan(0), SanYuan::新);
        assert_eq!(ShanVM::classify_yuan(1), SanYuan::新);
        assert_eq!(ShanVM::classify_yuan(2), SanYuan::陈);
        assert_eq!(ShanVM::classify_yuan(10), SanYuan::陈);
    }

    #[test]
    fn test_decision_winter_stale_aborts() {
        let input = LianShanInput {
            task_count: 2, // activated
            obstruction_count: 5,
            rounds_since_perception: 5, // stale
            strategy_verified: false,
            forces: ZhiForces::empty(),
        };
        let decision = ShanVM::run_chain(&input);
        assert!(decision.activated);
        assert_eq!(decision.qi, SixQi::冬);
        assert_eq!(decision.yuan, SanYuan::陈);
        assert_eq!(decision.decision, FangWei::Abort);
    }

    #[test]
    fn test_decision_with_push_forces() {
        let mut forces = ZhiForces::empty();
        forces.push_forces.push("已知策略A".to_string());
        let input = LianShanInput {
            task_count: 2,
            obstruction_count: 1,
            rounds_since_perception: 0,
            strategy_verified: false,
            forces,
        };
        let decision = ShanVM::run_chain(&input);
        assert_eq!(decision.decision, FangWei::PushThrough);
    }

    #[test]
    fn test_decision_with_resist_forces() {
        let mut forces = ZhiForces::empty();
        forces.resist_forces.push("避开区域B".to_string());
        let input = LianShanInput {
            task_count: 2,
            obstruction_count: 1,
            rounds_since_perception: 0,
            strategy_verified: false,
            forces,
        };
        let decision = ShanVM::run_chain(&input);
        assert_eq!(decision.decision, FangWei::NavigateAround);
    }

    #[test]
    fn test_shanvm_decide_logs() {
        let mut vm = ShanVM::new();
        vm.task_count = 2;
        vm.obstruction_count = 0;
        vm.strategy_verified = true;
        let decision = vm.decide();
        assert!(decision.activated);
        assert_eq!(vm.decision_log.len(), 1);
    }

    #[test]
    fn test_decompose_on_third_obstruction_autumn() {
        let input = LianShanInput {
            task_count: 2,
            obstruction_count: 2, // 三 level
            rounds_since_perception: 0,
            strategy_verified: false,
            forces: ZhiForces::empty(),
        };
        let decision = ShanVM::run_chain(&input);
        assert_eq!(decision.jia, SixJia::三);
        assert_eq!(decision.qi, SixQi::秋);
        assert_eq!(decision.decision, FangWei::Decompose);
    }

    #[test]
    fn test_escalate_on_third_obstruction_winter() {
        let input = LianShanInput {
            task_count: 2,
            obstruction_count: 3,
            rounds_since_perception: 0,
            strategy_verified: false,
            forces: ZhiForces::empty(),
        };
        let decision = ShanVM::run_chain(&input);
        assert_eq!(decision.qi, SixQi::冬);
        assert_eq!(decision.jia, SixJia::三);
        assert_eq!(decision.decision, FangWei::Escalate);
    }

    #[test]
    fn test_wait_gather_when_no_forces_early_spring() {
        let input = LianShanInput {
            task_count: 2,
            obstruction_count: 0,
            rounds_since_perception: 0,
            strategy_verified: false,
            forces: ZhiForces::empty(),
        };
        let decision = ShanVM::run_chain(&input);
        assert_eq!(decision.qi, SixQi::春);
        assert_eq!(decision.jia, SixJia::初);
        assert_eq!(decision.decision, FangWei::WaitGather);
    }
}
