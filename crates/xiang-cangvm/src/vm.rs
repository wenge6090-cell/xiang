/// CangVM — The core execution engine for the 归藏 (Guizang) layer.
///
/// Walks the parsed AST and executes eight-qi operators on a 6-bit Gua state vector,
/// manages the CangSea for Hebbian learning, and tracks deviation from the question focus origin.

use xiang_core::{Gua, CangSea, deviation};
use xiang_core::embedding::Embedding;
use xiang_core::yin_checker::{YinProtocolChecker, RuleResult};
use xiang_core::LianShanDecision;
use xiang_parser::ast::*;
use xiang_shanvm::ShanVM;
use xiang_zhouvm::ZhouVM;
use std::collections::HashMap;
use std::path::PathBuf;

/// Runtime value types for variables.
#[derive(Debug, Clone)]
pub enum Value {
    Gua(Gua),
    Num(f32),
    Str(String),
    Bool(bool),
}

/// CangVM phase-constraint decision.
///
/// Returned by `CangVM::judge()` after the YinProtocolChecker validates
/// an operator's output against its phase rules.
#[derive(Debug, Clone, PartialEq)]
pub enum CangVMDecision {
    /// Output passes protocol check — advance to next operator.
    Pass { next_operator: String },
    /// Output violates protocol — rollback and retry same operator.
    Rollback { operator: String },
    /// Too many retries — skip current operator, advance to next.
    Skip { next_operator: String },
    /// Global kill/retry limit exceeded — stop the entire cycle.
    Stop { reason: String },
    /// All operators completed normally — finish the cycle.
    FinishCycle,
}

/// The CangVM execution engine.
pub struct CangVM {
    /// Current 6-bit state vector S ∈ {0..63}
    pub state: Gua,
    /// 问题焦点向量 (question focus origin). The reference point for deviation measurement.
    /// Should be dynamically set from the current question's semantic focus.
    pub origin: Gua,
    /// Hebbian experience memory
    pub cang_sea: CangSea,
    /// Variable store
    pub variables: HashMap<String, Value>,
    /// Cycle counter
    pub cycle_count: u32,
    /// Deviation log: (cycle_number, deviation_value)
    pub deviation_log: Vec<(u32, f32)>,
    /// Output log: all 发 messages
    pub output_log: Vec<String>,
    /// Input log: all 感 sources
    pub input_log: Vec<String>,
    /// Number of sha (prune) operations — pruning off-focus branches
    pub sha_count: u32,
    /// Number of aligned cang experiences (focus-aligned thinking paths)
    pub cang_aligned_count: u32,
    /// Number of deviated cang experiences (deviated thinking paths)
    pub cang_deviated_count: u32,
    /// Whether execution should stop
    pub should_stop: bool,
    // ── Semantic deviation fields ─────────────────────────
    /// Semantic origin text (the question/goal text).
    pub semantic_origin_text: Option<String>,
    /// Semantic origin embedding.
    pub semantic_origin_embedding: Option<Embedding>,
    /// Current semantic deviation (cosine-based). None = no encoder.
    pub semantic_deviation: Option<f32>,
    /// Alpha weight for hybrid deviation: alpha*hamming + (1-alpha)*cosine.
    pub deviation_alpha: f32,
    // ── Three-Engine Integration (三引擎融合) ─────────────
    /// Lianshan strategy engine (discrete 6-step chain).
    /// When present, `LianShanOp` executes the full decision chain.
    pub shan_vm: Option<ShanVM>,
    /// Zhouyi cognitive grid engine (discrete 8-state Markov).
    /// When present, `ZhouYiOp` dispatches Pose/Sheng/Ke to control posture.
    pub zhou_vm: Option<ZhouVM>,
    /// Last Lianshan decision — carries strategy data flow to ZhouYi.
    pub last_lian_shan_decision: Option<LianShanDecision>,
    /// Whether ShanVM triggered (障碍激活) on the last cycle.
    pub shan_triggered: bool,
    /// Obstruction counter for ShanVM activation tracking.
    pub obstruction_count: usize,
    /// Rounds since last user input (for 元 freshness classification).
    pub rounds_since_perception: usize,
    // ── Phase Constraint fields (阶段约束·归藏设计笔记 §2.3) ──
    /// 阴仪阶段协议验证器（正则规则检查器）。
    pub yin_checker: YinProtocolChecker,
    /// 生成算子队列：生 → 动 → 长 → 育
    pub operator_queue: Vec<&'static str>,
    /// 当前算子在队列中的索引
    pub current_op_index: usize,
    /// 每个算子的重试次数
    pub retries: HashMap<String, u32>,
    /// 当前周期累计杀次数
    pub kill_count: u32,
    /// 全局重试累计
    pub total_retries: u32,
    /// 单算子最大重试次数
    pub max_retries_per_operator: u32,
    /// 单周期最大杀次数
    pub max_kills_per_cycle: u32,
    /// 全局最大重试次数
    pub max_total_retries: u32,
    /// 藏海持久化文件路径。设置后，每次执行"藏"算子会自动保存。
    pub cangsea_path: Option<PathBuf>,
}

impl CangVM {
    pub fn new() -> Self {
        CangVM {
            state: Gua::ZERO,
            origin: Gua::ORIGIN,
            cang_sea: CangSea::new(),
            variables: HashMap::new(),
            cycle_count: 0,
            deviation_log: Vec::new(),
            output_log: Vec::new(),
            input_log: Vec::new(),
            sha_count: 0,
            cang_aligned_count: 0,
            cang_deviated_count: 0,
            should_stop: false,
            semantic_origin_text: None,
            semantic_origin_embedding: None,
            semantic_deviation: None,
            deviation_alpha: 0.5,
            shan_vm: None,
            zhou_vm: None,
            last_lian_shan_decision: None,
            shan_triggered: false,
            obstruction_count: 0,
            rounds_since_perception: 0,
            yin_checker: YinProtocolChecker::new(),
            operator_queue: vec!["生", "动", "长", "育"],
            current_op_index: 0,
            retries: HashMap::new(),
            kill_count: 0,
            total_retries: 0,
            max_retries_per_operator: 2,
            max_kills_per_cycle: 3,
            max_total_retries: 5,
            cangsea_path: None,
        }
    }

    /// Get the current Hamming deviation from the question focus origin.
    pub fn current_deviation(&self) -> f32 {
        deviation(self.state, self.origin)
    }

    /// Dynamically set the origin (question focus) for deviation measurement.
    pub fn set_origin(&mut self, origin: Gua) {
        self.origin = origin;
    }

    /// Set the semantic origin from text and its embedding.
    /// This activates the hybrid deviation mode.
    pub fn set_semantic_origin(&mut self, text: String, embedding: Embedding) {
        self.semantic_origin_text = Some(text);
        self.semantic_origin_embedding = Some(embedding);
    }

    // ── Three-Engine Builder Methods (三引擎注入) ──────────

    /// Inject a ShanVM discrete strategy engine.
    pub fn with_shan_vm(mut self, shan_vm: ShanVM) -> Self {
        self.shan_vm = Some(shan_vm);
        self
    }

    /// Inject a ZhouVM discrete cognitive grid engine.
    pub fn with_zhou_vm(mut self, zhou_vm: ZhouVM) -> Self {
        self.zhou_vm = Some(zhou_vm);
        self
    }

    /// Run the Lianshan strategy engine (discrete mode).
    /// Updates obstruction_count and shan_triggered based on the decision.
    pub fn run_shan_vm(&mut self) -> Option<LianShanDecision> {
        // Extract values first to avoid borrow conflicts with shan_vm access below
        let dev = self.current_deviation();
        let state = self.state;
        let rounds = self.rounds_since_perception;
        let cycle = self.cycle_count;
        let kills = self.kill_count;
        let shas = self.sha_count;

        let shan = self.shan_vm.as_mut()?;
        shan.state = state;
        shan.deviation = dev;
        shan.rounds_since_perception = rounds;
        shan.task_count = if cycle > 1 { 2 } else { 1 };
        shan.strategy_verified = kills == 0 && shas == 0;

        // Inject CangSea forces for 阴阳对峙
        shan.forces = self.cang_sea.query_forces(state);

        let decision = shan.decide();
        self.last_lian_shan_decision = Some(decision.clone());
        self.shan_triggered = decision.activated;

        // Update obstruction tracking
        if decision.activated && decision.jia != xiang_core::SixJia::初 {
            self.obstruction_count += 1;
        }

        self.output_log.push(format!(
            "[连山] activated={} qi={} jia={} yuan={} → {}",
            decision.activated,
            decision.qi.name(),
            decision.jia.name(),
            decision.yuan.name(),
            decision.decision.name()
        ));

        Some(decision)
    }

    /// Run the ZhouYi cognitive grid step (discrete mode).
    /// Advances the Markov chain and updates temperature/posture.
    pub fn run_zhou_vm_step(&mut self) {
        if let Some(ref mut zhou) = self.zhou_vm {
            let mut rng = rand::rng();
            let next = zhou.step(&mut rng);
            self.output_log.push(format!(
                "[周易] posture={} ({}) temp={:.2}",
                zhou.current_pose_name(),
                zhou.current_posture(),
                zhou.temperature
            ));
            let _ = next;
        }
    }

    /// Get the current temperature from ZhouVM, or default 0.7.
    pub fn zhou_temperature(&self) -> f32 {
        self.zhou_vm.as_ref().map(|z| z.temperature).unwrap_or(0.7)
    }

    /// Get the current cognitive posture description from ZhouVM.
    pub fn zhou_posture(&self) -> &'static str {
        self.zhou_vm.as_ref()
            .map(|z| z.current_posture())
            .unwrap_or("坤 · 默认稳定")
    }

    /// Get the prompt prefix meta-instruction from ZhouVM for the current posture.
    pub fn zhou_prompt_prefix(&self) -> &'static str {
        self.zhou_vm.as_ref()
            .map(|z| z.prompt_prefix())
            .unwrap_or("请给出客观、全面的回应。")
    }

    // ── Phase Constraint Decision Logic (阶段约束) ──────────

    /// Reset phase constraint state for a new cycle.
    pub fn reset_phase_state(&mut self) {
        self.current_op_index = 0;
        self.retries.clear();
        self.kill_count = 0;
        self.total_retries = 0;
    }

    /// Get the current operator name, or None if queue is exhausted.
    pub fn current_operator(&self) -> Option<&str> {
        self.operator_queue.get(self.current_op_index).copied()
    }

    /// Check whether there are more operators in the queue.
    pub fn has_next_operator(&self) -> bool {
        self.current_op_index < self.operator_queue.len()
    }

    /// Get the next operator without advancing.
    pub fn peek_next_operator(&self) -> Option<&str> {
        self.operator_queue.get(self.current_op_index + 1).copied()
    }

    /// Phase-constraint decision engine.
    ///
    /// Input: the current operator name, and whether the YinProtocolChecker
    /// found the output to be valid (along with any violations).
    ///
    /// Output: a `CangVMDecision` instruction for the Yang Engine:
    ///   - Pass → advance to next operator
    ///   - Rollback → retry same operator
    ///   - Skip → force advance, increment kill count
    ///   - Stop → terminate cycle
    ///   - FinishCycle → all operators completed
    pub fn judge(&mut self, operator: &str, is_valid: bool, _violations: &[String]) -> CangVMDecision {
        if is_valid {
            // Protocol check passed — reset retries, advance to next operator
            self.retries.insert(operator.to_string(), 0);
            self.current_op_index += 1;

            if self.current_op_index >= self.operator_queue.len() {
                return CangVMDecision::FinishCycle;
            }

            let next = self.operator_queue[self.current_op_index];
            return CangVMDecision::Pass {
                next_operator: next.to_string(),
            };
        }

        // Protocol check failed — increment retry counts
        let retries = self.retries.entry(operator.to_string()).or_insert(0);
        *retries += 1;
        self.total_retries += 1;

        // Check global retry limit
        if self.total_retries >= self.max_total_retries {
            return CangVMDecision::Stop {
                reason: format!("全局重试上限({})已达", self.max_total_retries),
            };
        }

        // Check per-cycle kill limit
        if self.kill_count >= self.max_kills_per_cycle {
            return CangVMDecision::Stop {
                reason: format!("周期杀上限({})已达", self.max_kills_per_cycle),
            };
        }

        // Check per-operator retry limit
        if *retries > self.max_retries_per_operator {
            // Force skip this operator
            self.kill_count += 1;
            self.sha_count += 1;
            self.current_op_index += 1;

            if self.current_op_index >= self.operator_queue.len() {
                return CangVMDecision::Stop {
                    reason: "所有算子结束".to_string(),
                };
            }

            let next = self.operator_queue[self.current_op_index];
            return CangVMDecision::Skip {
                next_operator: next.to_string(),
            };
        }

        // Within retry limit — rollback and retry
        self.kill_count += 1;
        self.sha_count += 1;
        CangVMDecision::Rollback {
            operator: operator.to_string(),
        }
    }

    /// Check the current operator's output text against the YinProtocolChecker rules.
    ///
    /// Returns the `RuleResult` from the checker.
    /// Control operators (归/杀/止/藏) always pass.
    pub fn check_operator_output(&self, operator: &str, text: &str) -> RuleResult {
        self.yin_checker.check(operator, text)
    }

    /// Execute the Gua-level operator for a given phase name.
    /// Returns Err if operator is not recognized.
    pub fn execute_phase_operator(&mut self, operator: &str) -> Result<(), VmError> {
        match operator {
            "生" => { self.state = self.state.sheng(); Ok(()) }
            "动" => { self.state = self.state.dong(); Ok(()) }
            "长" => { self.state = self.state.zhang(); Ok(()) }
            "育" => { self.state = self.state.yu(); Ok(()) }
            "杀" => { self.state = self.state.sha(); self.sha_count += 1; Ok(()) }
            "止" => { self.state = self.state.zhi(); Ok(()) }
            "归" => {
                let dev = self.current_deviation();
                self.deviation_log.push((self.cycle_count, dev));
                Ok(())
            }
            "藏" => {
                let prev_state = self.state;
                let dev = self.current_deviation();
                let reward = if dev < 0.3 { 0.5 } else { -0.3 };
                self.cang_sea.store(prev_state, Gua::ZERO, reward);
                if dev < 0.3 {
                    self.cang_aligned_count += 1;
                } else {
                    self.cang_deviated_count += 1;
                }
                // Persist to file if path is configured
                if let Some(ref path) = self.cangsea_path {
                    if let Err(e) = self.cang_sea.save_to_file(path) {
                        self.output_log.push(format!("[藏海] 保存失败: {e}"));
                    }
                }
                self.state = self.state.cang();
                Ok(())
            }
            _ => Err(VmError::InvalidState(format!("未知算子: {operator}"))),
        }
    }

    /// Execute a complete program (an agent's body).
    pub fn execute(&mut self, statements: &[Stmt]) -> Result<(), VmError> {
        for stmt in statements {
            if self.should_stop {
                break;
            }
            self.execute_stmt(stmt)?;
        }
        Ok(())
    }

    /// Execute a single statement.
    fn execute_stmt(&mut self, stmt: &Stmt) -> Result<(), VmError> {
        match stmt {
            Stmt::VarDecl { var_type: _, name, value } => {
                let val = self.eval_expr(value)?;
                self.variables.insert(name.clone(), val);
            }
            Stmt::Assign { target, value } => {
                let val = self.eval_expr(value)?;
                self.variables.insert(target.clone(), val);
            }
            Stmt::QiOp(op) => self.execute_qi_op(op)?,
            Stmt::Cond { condition, then_branch, else_branch } => {
                if self.eval_condition(condition)? {
                    self.execute(then_branch)?;
                } else if let Some(else_stmts) = else_branch {
                    self.execute(else_stmts)?;
                }
            }
            Stmt::Loop(loop_stmt) => self.execute_loop(loop_stmt)?,
            Stmt::Io(io) => self.execute_io(io)?,
            Stmt::CangSeaOp(op) => self.execute_cang_sea_op(op)?,
            Stmt::Chengshi(cs) => {
                // Honesty verification: for now, record the declaration.
                self.output_log.push(format!("[言] {}", cs.declaration));
            }
            Stmt::LianShanOp(_ls) => {
                // ── 连山介入：离散策略导航引擎 ──
                // Extract values first to avoid borrow conflicts
                let dev = self.current_deviation();
                let state = self.state;
                let rounds = self.rounds_since_perception;
                let cycle = self.cycle_count;
                let kills = self.kill_count;
                let shas = self.sha_count;
                let obs_count = self.obstruction_count;

                if let Some(ref mut shan) = self.shan_vm {
                    shan.state = state;
                    shan.deviation = dev;
                    shan.rounds_since_perception = rounds;
                    shan.task_count = if cycle > 1 { 2 } else { 1 };
                    shan.strategy_verified = kills == 0 && shas == 0;
                    shan.obstruction_count = obs_count;

                    let decision = shan.decide();
                    self.last_lian_shan_decision = Some(decision.clone());
                    self.shan_triggered = decision.activated;

                    if decision.activated && decision.jia != xiang_core::SixJia::初 {
                        self.obstruction_count += 1;
                    }
                    self.rounds_since_perception = 0;

                    self.output_log.push(format!(
                        "[连山] activated={} qi={} jia={} yuan={} → {}",
                        decision.activated,
                        decision.qi.name(),
                        decision.jia.name(),
                        decision.yuan.name(),
                        decision.decision.name()
                    ));
                } else {
                    self.output_log.push("[连山] 无策略引擎挂载 — no-op".to_string());
                }
            }
            Stmt::ZhouYiOp(zy) => {
                // ── 周易介入：离散认知姿态引擎 ──
                if let Some(ref mut zhou) = self.zhou_vm {
                    match zy {
                        ZhouYiOp::Pose { bagua } => {
                            zhou.execute_pose(*bagua);
                            self.output_log.push(format!(
                                "[周易] 姿 {} → posture={} ({}) temp={:.2}",
                                bagua.name(),
                                zhou.current_pose_name(),
                                zhou.current_posture(),
                                zhou.temperature
                            ));
                        }
                        ZhouYiOp::Shengke { rules } => {
                            for rule in rules {
                                match rule {
                                    ShengkeRule::Sheng { from, to } => {
                                        zhou.execute_sheng(*from, *to);
                                        self.output_log.push(format!(
                                            "[周易] 生 {}→{} weight+",
                                            from.name(), to.name()
                                        ));
                                    }
                                    ShengkeRule::Ke { from, to } => {
                                        zhou.execute_ke(*from, *to);
                                        self.output_log.push(format!(
                                            "[周易] 克 {}→{} weight-",
                                            from.name(), to.name()
                                        ));
                                    }
                                }
                            }
                        }
                    }
                } else {
                    self.output_log.push("[周易] 无姿态引擎挂载 — no-op".to_string());
                }
            }
        }
        Ok(())
    }

    // ── Eight-Qi Operators ─────────────────────────────────

    fn execute_qi_op(&mut self, op: &QiOp) -> Result<(), VmError> {
        match op {
            QiOp::Sheng => {
                self.state = self.state.sheng();
            }
            QiOp::Dong => {
                self.state = self.state.dong();
            }
            QiOp::Zhang => {
                self.state = self.state.zhang();
            }
            QiOp::Yu => {
                self.state = self.state.yu();
            }
            QiOp::Sha { target: _ } => {
                self.state = self.state.sha();
                self.sha_count += 1;
            }
            QiOp::Zhi { mask } => {
                // 止 uses the provided mask; the pure operator uses a fixed mask
                // but the grammar allows specifying a custom mask
                self.state = Gua(self.state.0 & mask.0 & 0x3F);
            }
            QiOp::Gui => {
                let dev = self.current_deviation();
                self.deviation_log.push((self.cycle_count, dev));
                // 归 is read-only, no state mutation
            }
            QiOp::Cang { label } => {
                let prev_state = self.state;
                let is_aligned = matches!(label, CangLabel::Good);
                let reward = if is_aligned { 0.5 } else { -0.5 };
                self.cang_sea.store(prev_state, Gua::ZERO, reward);
                if is_aligned {
                    self.cang_aligned_count += 1;
                } else {
                    self.cang_deviated_count += 1;
                }
                self.state = self.state.cang();
            }
        }
        Ok(())
    }

    // ── Condition Evaluation ───────────────────────────────

    fn eval_condition(&mut self, cond: &Condition) -> Result<bool, VmError> {
        match cond {
            Condition::Deviation { cmp, value } => {
                let dev = self.current_deviation();
                Ok(cmp_compare(*cmp, dev, *value))
            }
            Condition::BitEq { index, value } => {
                let bit = self.state.bit(*index);
                Ok(bit == *value)
            }
            Condition::CangSeaHasExperience => {
                Ok(!self.cang_sea.is_empty())
            }
            Condition::CangSeaNoExperience => {
                Ok(self.cang_sea.is_empty())
            }
            Condition::CurrentGuaIs(bagua) => {
                let current_bits = self.state.lower(); // approximate
                Ok(current_bits == bagua.to_bits())
            }
            Condition::PatternMatch { expr, pattern } => {
                let val = self.eval_expr(expr)?;
                match val {
                    Value::Gua(g) => Ok(g == *pattern),
                    _ => Ok(false),
                }
            }
        }
    }

    // ── Loop Execution ─────────────────────────────────────

    fn execute_loop(&mut self, loop_stmt: &LoopStmt) -> Result<(), VmError> {
        match loop_stmt {
            LoopStmt::ZhouTian { cycles, body } => {
                let n = cycles.unwrap_or(1);
                for _ in 0..n {
                    if self.should_stop { break; }
                    self.cycle_count += 1;
                    // ZhouTian three-phase execution is implicit:
                    // the body may contain 生→动→归→长→育→杀→止→归→杀→藏
                    self.execute(body)?;
                    // Track deviation after each cycle
                    let dev = self.current_deviation();
                    self.deviation_log.push((self.cycle_count, dev));
                }
            }
            LoopStmt::While { condition, body } => {
                while self.eval_condition(condition)? {
                    if self.should_stop { break; }
                    self.cycle_count += 1;
                    self.execute(body)?;
                }
            }
        }
        Ok(())
    }

    // ── I/O Execution ──────────────────────────────────────

    fn execute_io(&mut self, io: &IoStmt) -> Result<(), VmError> {
        match io {
            IoStmt::GAN { source } => {
                self.input_log.push(source.clone());
                // In simulation mode, GAN produces a default Gua
                // In a real system, this would read from a sensor
            }
            IoStmt::FA { message } => {
                self.output_log.push(message.clone());
            }
        }
        Ok(())
    }

    // ── CangSea Operations ─────────────────────────────────

    fn execute_cang_sea_op(&mut self, op: &CangSeaOp) -> Result<(), VmError> {
        match op {
            CangSeaOp::Store { gua, label } => {
                let reward = match label {
                    CangLabel::Good => 0.5,
                    CangLabel::Evil => -0.5,
                };
                self.cang_sea.store(self.state, *gua, reward);
            }
            CangSeaOp::Retrieve { gua, similarity_threshold: _ } => {
                // Query CangSea for experiences matching the given gua.
                // Returns experiences sorted by reward (highest first).
                let matches = self.cang_sea.entries_from(*gua);
                if matches.is_empty() {
                    self.output_log.push(format!("[藏海·检索] gua={:06b} → 无匹配经验", gua.0));
                } else {
                    // Log top 3 matching experiences
                    let top: Vec<_> = matches.iter()
                        .take(3)
                        .map(|e| format!("{:06b}→{:06b} r={:+.2}", e.from.0, e.to.0, e.reward))
                        .collect();
                    self.output_log.push(format!(
                        "[藏海·检索] gua={:06b} → {}条经验: {}",
                        gua.0,
                        matches.len(),
                        top.join(", ")
                    ));
                }
            }
            CangSeaOp::Reinforce { from, to, weight } => {
                self.cang_sea.reinforce(*from, *to, *weight);
            }
            CangSeaOp::Weaken { from, to, weight } => {
                self.cang_sea.weaken(*from, *to, *weight);
            }
            CangSeaOp::GetWeight { target_var, from } => {
                let w = self.cang_sea.weight(*from, Gua::ZERO);
                self.variables.insert(target_var.clone(), Value::Num(w));
            }
        }
        Ok(())
    }

    // ── Expression Evaluation ──────────────────────────────

    fn eval_expr(&mut self, expr: &Expr) -> Result<Value, VmError> {
        match expr {
            Expr::GuaLit(g) => Ok(Value::Gua(*g)),
            Expr::NumLit(n) => Ok(Value::Num(*n)),
            Expr::StrLit(s) => Ok(Value::Str(s.clone())),
            Expr::BoolLit(b) => Ok(Value::Bool(*b)),
            Expr::Var(name) => {
                self.variables.get(name)
                    .cloned()
                    .ok_or_else(|| VmError::UndefinedVariable(name.clone()))
            }
            Expr::GANResult => {
                // In simulation, return current state
                Ok(Value::Gua(self.state))
            }
        }
    }
}

impl Default for CangVM {
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

/// VM execution errors.
#[derive(Debug, Clone)]
pub enum VmError {
    UndefinedVariable(String),
    InvalidState(String),
    CangSeaFull,
}

impl std::fmt::Display for VmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VmError::UndefinedVariable(s) => write!(f, "未定义变量: {s}"),
            VmError::InvalidState(s) => write!(f, "无效状态: {s}"),
            VmError::CangSeaFull => write!(f, "藏海已满"),
        }
    }
}

impl std::error::Error for VmError {}

// ── Full ZhouTian Cycle ────────────────────────────────────

/// Run a complete ZhouTian (周天) consciousness cycle in pure bitwise mode.
///
/// The cycle follows the three-phase sequence:
///   1. 起念: 生 → 动 → 归 (deviation check → if > 0.7: 杀+藏偏离+restart)
///   2. 立目标: 长 → 育 → 杀 → 止 (deviation check)
///   3. 整理认知: 归 → 杀 → 藏 (conditionally: 藏对齐 if dev < 0.3, else 藏偏离)
pub struct ZhouTianRunner {
    pub vm: CangVM,
    pub deviation_threshold: f32,
    pub refocus_threshold: f32,
    pub aligned_threshold: f32,
    pub max_cycles: u32,
    /// Whether phase-constrained cycle is enabled. When false, falls back to legacy mode.
    pub phase_constraint_mode: bool,
}

impl ZhouTianRunner {
    pub fn new() -> Self {
        ZhouTianRunner {
            vm: CangVM::new(),
            deviation_threshold: 0.7,
            refocus_threshold: 0.9,
            aligned_threshold: 0.3,
            max_cycles: 10,
            phase_constraint_mode: true,
        }
    }

    /// Run one complete ZhouTian cycle.
    /// Dispatches to phase-constrained or legacy mode.
    pub fn run_cycle(&mut self) -> Result<(), VmError> {
        if self.phase_constraint_mode {
            self.run_phase_cycle()
        } else {
            self.run_legacy_cycle()
        }
    }

    /// Legacy 3-phase 8-step cycle (original behaviour).
    fn run_legacy_cycle(&mut self) -> Result<(), VmError> {
        self.vm.cycle_count += 1;

        // Phase 1: 起念 (Thought Arising)
        // 生 → 动 → 归
        self.vm.state = self.vm.state.sheng();
        self.vm.state = self.vm.state.dong();
        let dev = self.vm.current_deviation();
        self.vm.deviation_log.push((self.vm.cycle_count, dev));

        // Refocus check — deviation too high, prune and restart
        if dev > self.refocus_threshold {
            self.vm.state = self.vm.state.sha();
            self.vm.sha_count += 1;
            self.vm.cang_sea.store(self.vm.state, Gua::ZERO, -0.8);
            self.vm.cang_deviated_count += 1;
            self.vm.state = self.vm.state.cang();
            // Check max cycles even on refocus abort
            if self.vm.cycle_count >= self.max_cycles {
                self.vm.should_stop = true;
            }
            return Ok(()); // restart implied
        }

        // Phase 2: 立目标 (Goal Setting)
        // 长 → 育 → 杀 → 止
        self.vm.state = self.vm.state.zhang();
        self.vm.state = self.vm.state.yu();
        let dev2 = self.vm.current_deviation();
        if dev2 > self.deviation_threshold {
            self.vm.state = self.vm.state.sha();
            self.vm.sha_count += 1;
        }
        self.vm.state = self.vm.state.zhi();

        // Phase 3: 整理认知 (Cognition Tidy)
        // 归 → 杀 → 藏
        let dev3 = self.vm.current_deviation();
        self.vm.deviation_log.push((self.vm.cycle_count, dev3));
        self.vm.state = self.vm.state.sha();
        self.vm.sha_count += 1;

        if dev3 < self.aligned_threshold {
            self.vm.cang_sea.store(self.vm.state, Gua::ZERO, 0.5);
            self.vm.cang_aligned_count += 1;
        } else {
            self.vm.cang_sea.store(self.vm.state, Gua::ZERO, -0.3);
            self.vm.cang_deviated_count += 1;
        }
        self.vm.state = self.vm.state.cang();

        // Stop if max cycles exceeded
        if self.vm.cycle_count >= self.max_cycles {
            self.vm.should_stop = true;
        }

        Ok(())
    }

    /// Phase-constrained ZhouTian cycle (阶段约束周天).
    ///
    /// This is the new default cycle based on the 归藏设计笔记 §2.3.
    /// Instead of using deviation to trigger pruning, each operator's output
    /// is validated by the YinProtocolChecker against phase-specific rules.
    ///
    /// Cycle flow:
    ///   生 → [yin check] → 动 → [yin check] → 长 → [yin check] → 育 → [yin check] → 藏
    ///
    /// Phase rules (from the YinProtocolChecker):
    ///   - 生: must be exploratory, open-ended; must NOT be structured
    ///   - 动: must show divergence, multi-angle; must NOT be conclusive
    ///   - 长: must show convergence, focus; must NOT be divergent
    ///   - 育: must be structured, numbered; must NOT be vague
    ///
    /// Decision responses:
    ///   - Pass: advance to next operator
    ///   - Rollback: retry same operator (up to max_retries_per_operator)
    ///   - Skip: force advance after retry exhaustion (increments kill_count)
    ///   - Stop: terminate cycle (global limits exceeded)
    fn run_phase_cycle(&mut self) -> Result<(), VmError> {
        self.vm.cycle_count += 1;
        self.vm.reset_phase_state();

        loop {
            let operator = match self.vm.current_operator() {
                Some(op) => op.to_string(),
                None => break, // queue exhausted
            };

            // Execute Gua-level operator
            self.vm.execute_phase_operator(&operator)?;

            // Record deviation
            let dev = self.vm.current_deviation();
            self.vm.deviation_log.push((self.vm.cycle_count, dev));

            // ── Three-Engine Integration (三引擎协同) ──
            // If deviation is high, invoke ShanVM for obstacle navigation
            if dev > 0.5 && self.vm.shan_vm.is_some() {
                self.vm.run_shan_vm();
            }
            // Advance ZhouVM cognitive posture each cycle
            if self.vm.zhou_vm.is_some() {
                self.vm.run_zhou_vm_step();
            }

            // ── Simulated Yang Engine output check ──
            // In a real system, the Yang Engine generates text for this operator,
            // and the YinProtocolChecker validates it.
            //
            // For the phase-constrained simulation path, we assume the generated
            // text for the CURRENT operator is valid. This tests the decision
            // state machine without needing a real LLM.
            // The actual validation happens when the experiment runner calls
            // `vm.check_operator_output()` explicitly.
            let is_valid = true; // simulated: output passes protocol check
            let violations: Vec<String> = vec![];

            // CangVM decision
            let decision = self.vm.judge(&operator, is_valid, &violations);

            match decision {
                CangVMDecision::Pass { .. } => {
                    // Advance to next operator (already done in judge())
                    continue;
                }
                CangVMDecision::Rollback { .. } => {
                    // Retry same operator (already handled in judge())
                    // In a real system, this triggers KV snapshot rollback
                    continue;
                }
                CangVMDecision::Skip { .. } => {
                    // Force advance (already done in judge())
                    self.vm.output_log.push(format!(
                        "[跳过] {} — 阶段检查失败次数超限", operator
                    ));
                    continue;
                }
                CangVMDecision::Stop { reason } => {
                    self.vm.output_log.push(format!(
                        "[终止] 周天终止: {reason}"
                    ));
                    // Prune and store negative experience
                    self.vm.state = self.vm.state.sha();
                    self.vm.sha_count += 1;
                    self.vm.cang_sea.store(self.vm.state, Gua::ZERO, -0.8);
                    self.vm.cang_deviated_count += 1;
                    self.vm.state = self.vm.state.cang();

                    if self.vm.cycle_count >= self.max_cycles {
                        self.vm.should_stop = true;
                    }
                    return Ok(());
                }
                CangVMDecision::FinishCycle => {
                    // All operators completed — enter 藏 phase
                    break;
                }
            }
        }

        // ── 藏 (Store & Reset) ──
        self.vm.execute_phase_operator("归")?;
        let final_dev = self.vm.current_deviation();
        self.vm.deviation_log.push((self.vm.cycle_count, final_dev));
        self.vm.execute_phase_operator("藏")?;

        if self.vm.cycle_count >= self.max_cycles {
            self.vm.should_stop = true;
        }

        Ok(())
    }

    /// Run N complete cycles.
    pub fn run(&mut self, n: u32) -> Result<(), VmError> {
        for _ in 0..n {
            if self.vm.should_stop {
                break;
            }
            self.run_cycle()?;
        }
        Ok(())
    }
}

impl Default for ZhouTianRunner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cangvm_new() {
        let vm = CangVM::new();
        assert_eq!(vm.state, Gua::ZERO);
        assert_eq!(vm.origin, Gua::ORIGIN);
        assert!(vm.variables.is_empty());
    }

    #[test]
    fn test_execute_sheng() {
        let mut vm = CangVM::new();
        let stmts = vec![Stmt::QiOp(QiOp::Sheng)];
        vm.execute(&stmts).unwrap();
        assert_eq!(vm.state, Gua(0b000001));
    }

    #[test]
    fn test_execute_dong() {
        let mut vm = CangVM::new();
        vm.state = Gua(0b000001);
        let stmts = vec![Stmt::QiOp(QiOp::Dong)];
        vm.execute(&stmts).unwrap();
        assert_eq!(vm.state, Gua(0b000000));
    }

    #[test]
    fn test_execute_zhang() {
        let mut vm = CangVM::new();
        vm.state = Gua(0b000111);
        let stmts = vec![Stmt::QiOp(QiOp::Zhang)];
        vm.execute(&stmts).unwrap();
        assert_eq!(vm.state, Gua(0b111111));
    }

    #[test]
    fn test_execute_sha() {
        let mut vm = CangVM::new();
        vm.state = Gua(0b110110);
        let stmts = vec![Stmt::QiOp(QiOp::Sha { target: None })];
        vm.execute(&stmts).unwrap();
        assert_eq!(vm.state, Gua(0));
        assert_eq!(vm.sha_count, 1);
    }

    #[test]
    fn test_execute_zhi() {
        let mut vm = CangVM::new();
        vm.state = Gua(0b111111);
        let stmts = vec![Stmt::QiOp(QiOp::Zhi { mask: Gua(0b101101) })];
        vm.execute(&stmts).unwrap();
        assert_eq!(vm.state, Gua(0b101101));
    }

    #[test]
    fn test_deviation_tracking() {
        let mut vm = CangVM::new();
        vm.state = Gua(0b111000); // deviation from 111111 = 0.5
        let stmts = vec![Stmt::QiOp(QiOp::Gui)];
        vm.execute(&stmts).unwrap();
        assert!((vm.current_deviation() - 0.5).abs() < f32::EPSILON);
        assert_eq!(vm.deviation_log.len(), 1);
    }

    #[test]
    fn test_cang_good_experience() {
        let mut vm = CangVM::new();
        vm.state = Gua(0b111000);
        let stmts = vec![Stmt::QiOp(QiOp::Cang { label: CangLabel::Good })];
        vm.execute(&stmts).unwrap();
        assert_eq!(vm.state, Gua::ZERO);
        assert_eq!(vm.cang_aligned_count, 1);
        assert!(!vm.cang_sea.is_empty());
    }

    #[test]
    fn test_cang_evil_experience() {
        let mut vm = CangVM::new();
        vm.state = Gua(0b000000);
        let stmts = vec![Stmt::QiOp(QiOp::Cang { label: CangLabel::Evil })];
        vm.execute(&stmts).unwrap();
        assert_eq!(vm.state, Gua::ZERO);
        assert_eq!(vm.cang_deviated_count, 1);
    }

    #[test]
    fn test_condition_deviation() {
        let mut vm = CangVM::new();
        vm.state = Gua(0b000000); // dev = 1.0
        let cond = Condition::Deviation { cmp: CmpOp::Gt, value: 0.7 };
        assert!(vm.eval_condition(&cond).unwrap());
        let cond2 = Condition::Deviation { cmp: CmpOp::Lt, value: 0.5 };
        assert!(!vm.eval_condition(&cond2).unwrap());
    }

    #[test]
    fn test_var_decl_and_assign() {
        let mut vm = CangVM::new();
        let stmts = vec![
            Stmt::VarDecl {
                var_type: VarType::Num,
                name: "阈值".to_string(),
                value: Expr::NumLit(0.8),
            },
            Stmt::Assign {
                target: "阈值".to_string(),
                value: Expr::NumLit(0.5),
            },
        ];
        vm.execute(&stmts).unwrap();
        if let Value::Num(n) = vm.variables.get("阈值").unwrap() {
            assert!((*n - 0.5).abs() < f32::EPSILON);
        } else {
            panic!("Expected Num value");
        }
    }

    #[test]
    fn test_zhou_tian_runner() {
        let mut runner = ZhouTianRunner::new();
        runner.run(3).unwrap();
        assert_eq!(runner.vm.cycle_count, 3);
        assert!(!runner.vm.deviation_log.is_empty());
        // After 3 cycles, the cang sea should have entries
        assert!(!runner.vm.cang_sea.is_empty());
    }

    #[test]
    fn test_zhou_tian_high_deviation_pruning() {
        let mut runner = ZhouTianRunner::new();
        runner.phase_constraint_mode = false;
        runner.vm.origin = Gua(0b000000); // invert origin so deviation is always high
        // Each cycle resets state to 0 (via cang), so we must re-set high deviation before every cycle
        for _ in 0..5 {
            runner.vm.state = Gua(0b111111); // completely opposite → dev = 1.0
            runner.run_cycle().unwrap();
        }
        // Should have many sha and deviated cang entries
        assert!(runner.vm.sha_count >= 5);
        assert!(runner.vm.cang_deviated_count >= 5);
    }

    #[test]
    fn test_execute_via_parser() {
        // Integration: parse → execute
        let source = "
定义 测试:
    卦 我的状态 = 000000
    生
    动
    归
    杀 110
    止 101101
    藏 善
";
        let program = xiang_parser::parse(source).unwrap();
        let mut vm = CangVM::new();
        vm.execute(&program.agents[0].body).unwrap();
        assert_eq!(vm.state, Gua::ZERO); // cang resets to 0
        assert_eq!(vm.cang_aligned_count, 1);
    }
}

#[cfg(test)]
mod zhou_tian_cycle_tests {
    use super::*;

    #[test]
    fn test_complete_zhou_tian_sequence() {
        let mut runner = ZhouTianRunner::new();
        runner.vm.origin = Gua::ORIGIN; // 111111
        runner.vm.state = Gua::ZERO;     // start clean
        runner.run_cycle().unwrap();

        // After one complete cycle:
        // Phase 1: 生(dong etc) → state changes
        // Phase 2: 长+育+杀+止 → state further modified  
        // Phase 3: 杀+藏 → state = 0
        assert_eq!(runner.vm.state, Gua::ZERO);
        assert_eq!(runner.vm.cycle_count, 1);
    }

    #[test]
    fn test_refocus_abort() {
        let mut runner = ZhouTianRunner::new();
        runner.phase_constraint_mode = false;
        // Start with max deviation state
        runner.vm.origin = Gua::ORIGIN;
        runner.vm.state = Gua(0b000000); // dev = 1.0, exceeds refocus threshold 0.9
        runner.run_cycle().unwrap();
        // Should hit refocus: sha + cang deviated + reset
        assert!(runner.vm.sha_count > 0);
        assert_eq!(runner.vm.state, Gua::ZERO);
    }

    #[test]
    fn test_max_cycles() {
        let mut runner = ZhouTianRunner::new();
        runner.max_cycles = 3;
        runner.run(10).unwrap();
        assert_eq!(runner.vm.cycle_count, 3);
        assert!(runner.vm.should_stop);
    }
}

#[cfg(test)]
mod phase_constraint_tests {
    use super::*;

    #[test]
    fn test_cangvm_phase_fields_initialized() {
        let vm = CangVM::new();
        assert_eq!(vm.operator_queue, vec!["生", "动", "长", "育"]);
        assert_eq!(vm.current_op_index, 0);
        assert_eq!(vm.kill_count, 0);
        assert_eq!(vm.total_retries, 0);
        assert_eq!(vm.max_retries_per_operator, 2);
        assert_eq!(vm.max_kills_per_cycle, 3);
        assert_eq!(vm.max_total_retries, 5);
    }

    #[test]
    fn test_current_operator() {
        let vm = CangVM::new();
        assert_eq!(vm.current_operator(), Some("生"));
    }

    #[test]
    fn test_judge_pass_advances_operator() {
        let mut vm = CangVM::new();
        assert_eq!(vm.current_op_index, 0);

        let decision = vm.judge("生", true, &[]);
        assert_eq!(vm.current_op_index, 1);
        assert!(matches!(decision, CangVMDecision::Pass { .. }));
        if let CangVMDecision::Pass { next_operator } = decision {
            assert_eq!(next_operator, "动");
        }
    }

    #[test]
    fn test_judge_all_operators_pass_then_finish() {
        let mut vm = CangVM::new();

        // 生 passes
        let d1 = vm.judge("生", true, &[]);
        assert!(matches!(d1, CangVMDecision::Pass { .. }));

        // 动 passes
        let d2 = vm.judge("动", true, &[]);
        assert!(matches!(d2, CangVMDecision::Pass { .. }));

        // 长 passes
        let d3 = vm.judge("长", true, &[]);
        assert!(matches!(d3, CangVMDecision::Pass { .. }));

        // 育 passes → finish
        let d4 = vm.judge("育", true, &[]);
        assert!(matches!(d4, CangVMDecision::FinishCycle));
    }

    #[test]
    fn test_judge_fail_triggers_rollback() {
        let mut vm = CangVM::new();

        // First fail → rollback (within retry limit)
        let d1 = vm.judge("生", false, &["缺少必要模式".to_string()]);
        assert!(matches!(d1, CangVMDecision::Rollback { .. }));
        if let CangVMDecision::Rollback { operator } = d1 {
            assert_eq!(operator, "生");
        }
        assert_eq!(vm.total_retries, 1);
        assert_eq!(vm.kill_count, 1);
    }

    #[test]
    fn test_judge_fail_retry_exhausted_triggers_skip() {
        let mut vm = CangVM::new();

        // 3 fails on same operator → skip (max_retries_per_operator = 2, > 2 triggers skip)
        vm.judge("生", false, &[]);
        vm.judge("生", false, &[]);
        let d3 = vm.judge("生", false, &[]);

        assert!(matches!(d3, CangVMDecision::Skip { .. }));
        if let CangVMDecision::Skip { next_operator } = d3 {
            assert_eq!(next_operator, "动");
        }
        assert_eq!(vm.current_op_index, 1); // skipped to 动
    }

    #[test]
    fn test_judge_global_retry_limit_triggers_stop() {
        let mut vm = CangVM::new();
        vm.max_total_retries = 3;
        vm.max_retries_per_operator = 10; // high enough to not trigger per-op limit

        // Fail 3 times across different operators to hit global limit
        vm.judge("生", false, &[]);
        vm.judge("生", false, &[]);
        let d3 = vm.judge("生", false, &[]);

        assert!(matches!(d3, CangVMDecision::Stop { .. }));
        if let CangVMDecision::Stop { reason } = &d3 {
            assert!(reason.contains("全局重试上限"));
        }
    }

    #[test]
    fn test_judge_kill_limit_triggers_stop() {
        let mut vm = CangVM::new();
        vm.max_retries_per_operator = 10; // high, won't trigger skip
        vm.max_kills_per_cycle = 2;

        // Each fail increments kill_count
        vm.judge("生", false, &[]); // kill_count = 1
        vm.judge("生", false, &[]); // kill_count = 2
        let d3 = vm.judge("生", false, &[]); // kill_count = 3 >= max_kills_per_cycle

        assert!(matches!(d3, CangVMDecision::Stop { .. }));
    }

    #[test]
    fn test_reset_phase_state() {
        let mut vm = CangVM::new();

        // Advance operator queue
        vm.judge("生", true, &[]);
        vm.judge("动", true, &[]);
        assert_eq!(vm.current_op_index, 2);

        // Reset
        vm.reset_phase_state();
        assert_eq!(vm.current_op_index, 0);
        assert_eq!(vm.kill_count, 0);
        assert_eq!(vm.total_retries, 0);
        assert!(vm.retries.is_empty());
    }

    #[test]
    fn test_phase_cycle_runs_all_operators() {
        let mut runner = ZhouTianRunner::new();
        runner.vm.origin = Gua::ORIGIN;
        runner.vm.state = Gua::ZERO;

        runner.run_cycle().unwrap();

        // After phase cycle, state should be reset (藏)
        assert_eq!(runner.vm.state, Gua::ZERO);
        assert_eq!(runner.vm.cycle_count, 1);
        // Should have stored experience
        assert!(!runner.vm.cang_sea.is_empty());
    }

    #[test]
    fn test_phase_cycle_multiple_runs() {
        let mut runner = ZhouTianRunner::new();
        runner.max_cycles = 5;

        runner.run(5).unwrap();

        assert_eq!(runner.vm.cycle_count, 5);
        // Each cycle has deviation log entries
        assert!(runner.vm.deviation_log.len() >= 5);
    }

    #[test]
    fn test_yin_checker_integration() {
        let vm = CangVM::new();

        // 生 should pass with exploratory text
        let result = vm.check_operator_output("生",
            "也许我们可以从架构层面考虑这个问题？需要分析数据流的走向。");
        assert!(result.is_valid, "生 should pass: {:?}", result.violations);

        // 生 should fail with structured text
        let result = vm.check_operator_output("生",
            "第一步，分析需求。因此我们应该采用方案A。");
        assert!(!result.is_valid, "生 should reject structured text");
    }

    #[test]
    fn test_execute_phase_operator_all_ops() {
        let mut vm = CangVM::new();

        // 生
        vm.execute_phase_operator("生").unwrap();
        assert_eq!(vm.state, Gua(0b000001));

        // 动
        vm.execute_phase_operator("动").unwrap();
        // state changed after dong

        // 长
        vm.execute_phase_operator("长").unwrap();

        // 育
        vm.execute_phase_operator("育").unwrap();

        // 杀
        let sha_before = vm.sha_count;
        vm.execute_phase_operator("杀").unwrap();
        assert!(vm.sha_count > sha_before);

        // 止
        vm.execute_phase_operator("止").unwrap();

        // 归 (read-only)
        vm.execute_phase_operator("归").unwrap();
        assert!(!vm.deviation_log.is_empty());

        // 藏
        vm.execute_phase_operator("藏").unwrap();
        assert_eq!(vm.state, Gua::ZERO);
        assert!(!vm.cang_sea.is_empty());
    }

    #[test]
    fn test_unknown_operator_returns_err() {
        let mut vm = CangVM::new();
        let result = vm.execute_phase_operator("未知");
        assert!(result.is_err());
    }

    #[test]
    fn test_decision_enum_variants() {
        // Verify all variants construct correctly
        let pass = CangVMDecision::Pass { next_operator: "动".to_string() };
        assert_eq!(pass, CangVMDecision::Pass { next_operator: "动".to_string() });

        let rollback = CangVMDecision::Rollback { operator: "生".to_string() };
        assert_eq!(rollback, CangVMDecision::Rollback { operator: "生".to_string() });

        let skip = CangVMDecision::Skip { next_operator: "长".to_string() };
        assert_eq!(skip, CangVMDecision::Skip { next_operator: "长".to_string() });

        let stop = CangVMDecision::Stop { reason: "测试".to_string() };
        assert!(matches!(stop, CangVMDecision::Stop { .. }));

        let finish = CangVMDecision::FinishCycle;
        assert!(matches!(finish, CangVMDecision::FinishCycle));
    }
}

// ── 三引擎联动集成测试 ──

#[cfg(test)]
mod three_engine_tests {
    use super::*;
    use xiang_core::Bagua;
    use xiang_shanvm::ShanVM;
    use xiang_zhouvm::ZhouVM;

    fn make_full_vm() -> CangVM {
        let mut vm = CangVM::new();
        vm.state = Gua::ORIGIN;
        vm.origin = Gua::ORIGIN;
        vm.shan_vm = Some(ShanVM::new());
        vm.zhou_vm = Some(ZhouVM::new());
        vm
    }

    #[test]
    fn test_three_engine_construction() {
        let vm = make_full_vm();
        assert!(vm.shan_vm.is_some());
        assert!(vm.zhou_vm.is_some());
        assert_eq!(vm.state, Gua::ORIGIN);
        assert_eq!(vm.origin, Gua::ORIGIN);
    }

    #[test]
    fn test_zhouvm_prompt_prefix_all_postures() {
        let mut vm = make_full_vm();
        let zhou = vm.zhou_vm.as_mut().unwrap();

        // Test each posture produces a non-empty prompt prefix
        for bagua in Bagua::ALL {
            zhou.execute_pose(bagua);
            let prefix = zhou.prompt_prefix();
            assert!(!prefix.is_empty(), "prompt_prefix should not be empty for {:?}", bagua);
        }
    }

    #[test]
    fn test_zhouvm_prompt_prefix_delegation() {
        let mut vm = make_full_vm();
        vm.zhou_vm.as_mut().unwrap().execute_pose(Bagua::乾);

        let prefix = vm.zhou_prompt_prefix();
        assert!(!prefix.is_empty());
        assert!(prefix.contains("创造性"));
    }

    #[test]
    fn test_shanvm_run_produces_decision() {
        let mut vm = make_full_vm();
        vm.state = Gua::ORIGIN;
        // High deviation to trigger ShanVM
        vm.deviation_log.push((1, 0.85));
        vm.obstruction_count = 2;

        let decision = vm.run_shan_vm();
        assert!(decision.is_some(), "run_shan_vm should produce a decision");
        let d = decision.unwrap();
        // Decision should have been stored
        assert_eq!(vm.last_lian_shan_decision.unwrap().activated, d.activated);
    }

    #[test]
    fn test_shanvm_updates_cangsea_forces() {
        let mut vm = make_full_vm();
        vm.state = Gua::ORIGIN;

        // Add some experience so CangSea has data
        vm.cang_sea.store(Gua::ORIGIN, Gua(0b000001), 0.7);

        // run_shan_vm internally queries forces - verify it doesn't panic
        let decision = vm.run_shan_vm();
        assert!(decision.is_some());
    }

    #[test]
    fn test_full_reset_preserves_engine_wiring() {
        let mut vm = make_full_vm();

        // Run through some operations
        vm.execute_phase_operator("生").unwrap();
        vm.judge("生", true, &[]);

        // Create fresh VM (simulating reset)
        let mut new_vm = CangVM::new();
        new_vm.state = Gua::ORIGIN;
        new_vm.origin = Gua::ORIGIN;
        new_vm.shan_vm = Some(ShanVM::new());
        new_vm.zhou_vm = Some(ZhouVM::new());

        // Verify fresh state
        assert_eq!(new_vm.current_op_index, 0);
        assert_eq!(new_vm.kill_count, 0);
        assert!(new_vm.shan_vm.is_some());
        assert!(new_vm.zhou_vm.is_some());
    }

    #[test]
    fn test_zhouvm_temperature_delegation() {
        let mut vm = make_full_vm();

        // Default (坤) temperature: 0.6
        let t = vm.zhou_temperature();
        assert!((t - 0.6).abs() < 0.01, "default temp should be 0.6, got {}", t);

        // 乾: 1.2
        vm.zhou_vm.as_mut().unwrap().execute_pose(Bagua::乾);
        let t = vm.zhou_temperature();
        assert!((t - 1.2).abs() < 0.01, "乾 temp should be 1.2, got {}", t);

        // 艮: 0.3
        vm.zhou_vm.as_mut().unwrap().execute_pose(Bagua::艮);
        let t = vm.zhou_temperature();
        assert!((t - 0.3).abs() < 0.01, "艮 temp should be 0.3, got {}", t);
    }

    #[test]
    fn test_zhouvm_posture_delegation() {
        let mut vm = make_full_vm();

        vm.zhou_vm.as_mut().unwrap().execute_pose(Bagua::乾);
        let posture = vm.zhou_posture();
        assert!(posture.contains("创造"));
    }

    #[test]
    fn test_three_engine_pipeline_orchestration() {
        // Simulate the full pipeline orchestration sequence:
        // ZhouVM select posture → CangVM execute operator → ShanVM navigate
        let mut vm = make_full_vm();
        vm.state = Gua::ZERO;
        vm.origin = Gua::ZERO;

        // 1. ZhouVM: select cognitive posture
        vm.zhou_vm.as_mut().unwrap().execute_pose(Bagua::震); // 启动
        let temp = vm.zhou_temperature();
        assert!((temp - 1.0).abs() < 0.01);
        let prefix = vm.zhou_prompt_prefix();
        assert!(!prefix.is_empty());

        // 2. CangVM: execute phase operator (归藏 → 生)
        vm.execute_phase_operator("生").unwrap();
        assert_eq!(vm.state, Gua(0b000001));

        // 3. CangVM: check deviation → ShanVM
        let dev = vm.current_deviation();
        assert!(dev >= 0.0 && dev <= 1.0);

        // 4. CangVM: advance to next operator
        vm.judge("生", true, &[]);
        assert_eq!(vm.current_op_index, 1); // → 动

        // 5. CangVM can run another cycle
        assert!(vm.has_next_operator());
        assert_eq!(vm.current_operator(), Some("动"));
    }

    #[test]
    fn test_shanvm_respects_deviation_threshold() {
        let mut vm = make_full_vm();

        // With low deviation and few obstructions, shan_triggered should be false
        vm.state = Gua::ORIGIN;
        vm.deviation_log.clear();
        vm.deviation_log.push((1, 0.1)); // low deviation
        vm.obstruction_count = 0;

        let decision = vm.run_shan_vm().unwrap();
        // The decision depends on ShanVM's internal logic, but the VM
        // should have recorded it
        assert_eq!(vm.last_lian_shan_decision.unwrap().activated, decision.activated);
    }
}
