/// 连山 (Lianshan) — Strategy engine types.
///
/// The Lianshan layer provides a 6-step decision chain for navigating
/// obstacles during task execution. It produces 7 possible directional decisions.

use std::fmt;

/// Goal progress phase (四气).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SixQi {
    春, // Goal just established, strategy trial
    夏, // Strategy verified, full push
    秋, // Hitting boundaries, need adjustment
    冬, // Severely blocked or resource exhausted
}

impl SixQi {
    pub fn name(self) -> &'static str {
        match self {
            SixQi::春 => "春",
            SixQi::夏 => "夏",
            SixQi::秋 => "秋",
            SixQi::冬 => "冬",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            SixQi::春 => "策略试水",
            SixQi::夏 => "全力推进",
            SixQi::秋 => "边界调整",
            SixQi::冬 => "严重阻塞",
        }
    }
}

/// Obstruction count level (六甲).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SixJia {
    初, // First obstruction
    再, // Second obstruction
    三, // Third+ obstruction
}

impl SixJia {
    pub fn name(self) -> &'static str {
        match self {
            SixJia::初 => "初甲",
            SixJia::再 => "再甲",
            SixJia::三 => "三甲",
        }
    }

    /// Default suggestion based on obstruction level.
    pub fn suggested_action(self) -> FangWei {
        match self {
            SixJia::初 => FangWei::PushThrough,
            SixJia::再 => FangWei::NavigateAround,
            SixJia::三 => FangWei::Decompose,
        }
    }
}

/// Context freshness (三元).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SanYuan {
    新, // Fresh: perceived within last 1 round
    陈, // Stale: >3 rounds without perception
}

impl SanYuan {
    pub fn name(self) -> &'static str {
        match self {
            SanYuan::新 => "新",
            SanYuan::陈 => "陈",
        }
    }
}

/// 7-direction decision (七种方位决策).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FangWei {
    Continue,         // Maintain status, keep current strategy
    PushThrough,      // Push hard — force through known obstacle
    NavigateAround,   // Detour — different path, same goal
    WaitGather,       // Wait — collect more info first
    Decompose,        // Decompose — break into smaller sub-goals
    Escalate,         // Escalate — mark for external intervention
    Abort,            // Abort — terminate this goal line
}

impl FangWei {
    pub fn name(self) -> &'static str {
        match self {
            FangWei::Continue => "继续",
            FangWei::PushThrough => "强力突破",
            FangWei::NavigateAround => "迂回绕行",
            FangWei::WaitGather => "等待收集",
            FangWei::Decompose => "拆解分解",
            FangWei::Escalate => "升级上报",
            FangWei::Abort => "放弃终止",
        }
    }

    /// All 7 directions.
    pub const ALL: [FangWei; 7] = [
        FangWei::Continue,
        FangWei::PushThrough,
        FangWei::NavigateAround,
        FangWei::WaitGather,
        FangWei::Decompose,
        FangWei::Escalate,
        FangWei::Abort,
    ];
}

impl fmt::Display for FangWei {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Confrontation forces: learned push/resist patterns from CangSea.
#[derive(Debug, Clone)]
pub struct ZhiForces {
    /// Push forces — patterns that succeeded in similar situations.
    pub push_forces: Vec<String>,
    /// Drag forces — patterns that failed or caused resistance.
    pub resist_forces: Vec<String>,
}

impl ZhiForces {
    pub fn empty() -> Self {
        ZhiForces {
            push_forces: Vec::new(),
            resist_forces: Vec::new(),
        }
    }
}

/// Input to the Lianshan decision engine.
#[derive(Debug, Clone)]
pub struct LianShanInput {
    /// Number of active tasks (for 艮 check: ≥2 triggers activation)
    pub task_count: usize,
    /// How many times this goal has been blocked
    pub obstruction_count: usize,
    /// Rounds since last perception
    pub rounds_since_perception: usize,
    /// Has the current strategy been verified effective?
    pub strategy_verified: bool,
    /// Learned forces from CangSea for the current obstacle pattern
    pub forces: ZhiForces,
}

/// Output from the Lianshan decision engine.
#[derive(Debug, Clone)]
pub struct LianShanDecision {
    /// Whether sibian is activated (艮 result)
    pub activated: bool,
    /// Goal progress phase (气)
    pub qi: SixQi,
    /// Obstruction count level (甲)
    pub jia: SixJia,
    /// Context freshness (元)
    pub yuan: SanYuan,
    /// Confrontation forces (峙)
    pub forces: ZhiForces,
    /// Decision (决)
    pub decision: FangWei,
}

impl LianShanDecision {
    pub fn deactivated() -> Self {
        LianShanDecision {
            activated: false,
            qi: SixQi::春,
            jia: SixJia::初,
            yuan: SanYuan::新,
            forces: ZhiForces::empty(),
            decision: FangWei::Continue,
        }
    }
}
