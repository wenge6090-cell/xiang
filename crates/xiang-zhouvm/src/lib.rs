/// ZhouVM — 周易 (Zhouyi) cognitive grid engine.
///
/// Manages an 8-state Markov chain (ZhouGrid) where each state maps to
/// a Bagua cognitive posture with an associated LLM temperature.
/// Transitions are governed by Wu Xing 生克 (generation/restraint) rules.
///
/// Operations:
///   姿 卦名   — set the current cognitive posture
///   生 from 生 to   — declare a generation rule from→to
///   生 from 克 to   — declare a restraint rule from→to

pub mod semantic;

use xiang_core::{Bagua, ZhouGrid};

/// 8 prompt templates, one per Bagua posture.
/// Each template is a meta-instruction prepended to the user input.
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

/// ZhouVM — the Zhouyi cognitive grid execution engine.
pub struct ZhouVM {
    /// Underlying 8-state cognitive grid.
    pub grid: ZhouGrid,
    /// Current temperature output (for LLM integration).
    pub temperature: f32,
    /// Pose change log.
    pub pose_log: Vec<Bagua>,
    /// Shengke application log.
    pub shengke_log: Vec<(Bagua, Bagua, bool)>, // (from, to, is_sheng)
}

impl ZhouVM {
    pub fn new() -> Self {
        let grid = ZhouGrid::default();
        let temperature = grid.current_temperature();
        ZhouVM {
            grid,
            temperature,
            pose_log: vec![Bagua::坤],
            shengke_log: Vec::new(),
        }
    }

    pub fn with_initial(initial: Bagua) -> Self {
        let grid = ZhouGrid::new(initial);
        let temperature = grid.current_temperature();
        ZhouVM {
            grid,
            temperature,
            pose_log: vec![initial],
            shengke_log: Vec::new(),
        }
    }

    /// Execute 姿 (pose): set the current cognitive posture.
    /// This directly changes the LLM temperature and state.
    pub fn execute_pose(&mut self, bagua: Bagua) {
        self.grid.set_pose(bagua);
        self.temperature = self.grid.current_temperature();
        self.pose_log.push(bagua);
    }

    /// Execute 生 (generation): declare from generates to.
    /// Strengthens the transition weight from→to.
    pub fn execute_sheng(&mut self, from: Bagua, to: Bagua) {
        self.grid.apply_shengke(from, to, true);
        self.shengke_log.push((from, to, true));
    }

    /// Execute 克 (restraint): declare from restrains to.
    /// Weakens the transition weight from→to.
    pub fn execute_ke(&mut self, from: Bagua, to: Bagua) {
        self.grid.apply_shengke(from, to, false);
        self.shengke_log.push((from, to, false));
    }

    /// Transition to the next posture based on learned weights.
    /// This is the Markov chain step that simulates the AI's cognitive shift.
    pub fn step<R: rand::Rng>(&mut self, rng: &mut R) -> Bagua {
        let next = self.grid.next_pose(rng);
        self.execute_pose(next);
        next
    }

    /// Get the current bagua posture name.
    pub fn current_pose_name(&self) -> &'static str {
        self.grid.current.name()
    }

    /// Get the current cognitive posture description.
    pub fn current_posture(&self) -> &'static str {
        self.grid.current.posture()
    }

    /// Get the prompt prefix meta-instruction for the current posture.
    pub fn prompt_prefix(&self) -> &'static str {
        for (bagua, template) in &PROMPT_TEMPLATES {
            if *bagua == self.grid.current {
                return template;
            }
        }
        "请给出客观、全面的回应。"
    }
}

impl Default for ZhouVM {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zhouvm_new() {
        let vm = ZhouVM::new();
        assert_eq!(vm.grid.current, Bagua::坤);
        assert!((vm.temperature - 0.6).abs() < f32::EPSILON);
        assert_eq!(vm.pose_log.len(), 1);
    }

    #[test]
    fn test_execute_pose() {
        let mut vm = ZhouVM::new();
        vm.execute_pose(Bagua::乾);
        assert_eq!(vm.grid.current, Bagua::乾);
        assert!((vm.temperature - 1.2).abs() < f32::EPSILON);
        assert_eq!(vm.pose_log.len(), 2);
        assert_eq!(vm.pose_log[1], Bagua::乾);
    }

    #[test]
    fn test_execute_sheng() {
        let mut vm = ZhouVM::new();
        let fi = Bagua::震.to_bits() as usize;
        let ti = Bagua::离.to_bits() as usize;
        let old_weight = vm.grid.weights[fi][ti];
        vm.execute_sheng(Bagua::震, Bagua::离);
        assert!(vm.grid.weights[fi][ti] > old_weight);
        assert_eq!(vm.shengke_log.len(), 1);
    }

    #[test]
    fn test_execute_ke() {
        let mut vm = ZhouVM::new();
        let fi = Bagua::兑.to_bits() as usize;
        let ti = Bagua::震.to_bits() as usize;
        let old_weight = vm.grid.weights[fi][ti];
        vm.execute_ke(Bagua::兑, Bagua::震);
        assert!(vm.grid.weights[fi][ti] < old_weight);
        assert_eq!(vm.shengke_log.len(), 1);
    }

    #[test]
    fn test_step_transition() {
        let mut vm = ZhouVM::new();
        let mut rng = rand::rng();
        let _old = vm.grid.current;
        let next = vm.step(&mut rng);
        // Step should always produce a valid Bagua
        assert!(Bagua::ALL.contains(&next));
        // Should have been logged
        assert_eq!(vm.pose_log.len(), 2);
    }

    #[test]
    fn test_multiple_poses() {
        let mut vm = ZhouVM::new();
        vm.execute_pose(Bagua::乾);
        vm.execute_pose(Bagua::离);
        vm.execute_pose(Bagua::坎);
        assert_eq!(vm.grid.current, Bagua::坎);
        assert_eq!(vm.pose_log.len(), 4);
    }

    #[test]
    fn test_temperature_changes_with_pose() {
        let mut vm = ZhouVM::new();
        assert!((vm.temperature - 0.6).abs() < f32::EPSILON);
        vm.execute_pose(Bagua::艮);
        assert!((vm.temperature - 0.3).abs() < f32::EPSILON);
        vm.execute_pose(Bagua::乾);
        assert!((vm.temperature - 1.2).abs() < f32::EPSILON);
    }

    #[test]
    fn test_with_initial() {
        let vm = ZhouVM::with_initial(Bagua::震);
        assert_eq!(vm.grid.current, Bagua::震);
        assert!((vm.temperature - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_sheng_ke_weights_persistence() {
        let mut vm = ZhouVM::new();
        // Initially, 木(震) → 火(离) has base generation weight 1.0
        let fi = Bagua::震.to_bits() as usize;
        let ti = Bagua::离.to_bits() as usize;
        let base = vm.grid.weights[fi][ti];
        assert!(base > 0.0);
        // Strengthen
        vm.execute_sheng(Bagua::震, Bagua::离);
        vm.execute_sheng(Bagua::震, Bagua::离);
        assert!(vm.grid.weights[fi][ti] > base + 0.15);
    }
}
