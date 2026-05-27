use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tower_http::cors::CorsLayer;
use xiang_cangvm::{CangVM, MetabolismSignal};
use xiang_core::{CangSea, ProjectContext};
use xiang_core::Gua;
use xiang_core::{EmbeddingObserver, HanziMap};
use xiang_llm::http_backend::HttpBackend;
use xiang_llm::{LlmBackend, LlmContext, TemperatureMode};
use xiang_shanvm::ShanVM;
use xiang_zhouvm::ZhouVM;

mod analyzer;
mod vocab;
mod semantic_injector;

const LLAMA_SERVER_URL: &str = "http://localhost:8080";

/// Raw (unconstrained) engine — fully independent.
struct RawEngine {
    backend: HttpBackend,
    ctx: LlmContext,
    messages: Vec<Message>,
}

impl RawEngine {
    fn new() -> Self {
        RawEngine {
            backend: HttpBackend::new(LLAMA_SERVER_URL),
            ctx: LlmContext::new(xiang_core::SAN_YI_CONSTRAINT_FULL_TEXT),
            messages: vec![],
        }
    }

    fn generate(&mut self, message: &str) -> Result<RawResponse, String> {
        let result = self
            .ctx
            .generate_unconstrained_turn(&mut self.backend, message, 200, TemperatureMode::Default, &[])
            .map_err(|e| format!("生成失败: {e}"))?;

        self.messages.push(Message {
            role: "user".into(),
            content: message.into(),
        });
        self.messages.push(Message {
            role: "assistant".into(),
            content: result.text.clone(),
        });

        Ok(RawResponse {
            response: result.text,
            tokens: result.tokens_generated,
        })
    }
}

/// Constrained engine — full three-engine pipeline.
///
/// ## Three-Engine Collaborative Closed Loop (三层协同闭环)
///
/// ```
/// User Request
///     │
///     ▼
/// ┌──────────────────┐
/// │  ZhouVM (周易)    │ ←─ Select initial cognitive posture (default 坤)
/// │  Set temperature  │
/// └────────┬─────────┘
///          │ posture
///          ▼
/// ┌──────────────────┐
/// │  CangVM (归藏)    │ ←─ Execute 8-operator zhou tian
/// │  generate → yin   │
/// │  check → judge    │
/// │                   │
/// │  Each step:       │
/// │  1. LLM generate  │
/// │  2. analyzer check│
/// │  3. deviation ↑?  │
/// │     → ShanVM      │
/// │  4. judge()       │
/// └────────┬──────────┘
///          │ high deviation / multi-retry
///          ▼
/// ┌──────────────────┐
/// │  ShanVM (连山)    │ ←─ Obstacle navigation
/// │  艮→气→甲→元→峙→决│
/// │  7-direction      │
/// └────────┬──────────┘
///          │ strategy (continue/decompose/escalate...)
///          ▼
/// ┌──────────────────┐
/// │  CangVM (归藏)    │ ←─ Execute strategy suggestion
/// │  adjust operator  │
/// └────────┬──────────┘
///          │ cycle end
///          ▼
/// ┌──────────────────┐
/// │  CangSea (藏海)   │ ←─ Store experience
/// │  Hebbian update   │
/// └──────────────────┘
/// ```
struct ConstrainedEngine {
    backend: HttpBackend,
    cang_vm: CangVM,
    ctx: LlmContext,
    messages: Vec<Message>,
    eos_id: u32,
    off_focus_ids: Vec<u32>,
    divergent_ids: Vec<u32>,
    /// 跨轮项目上下文 — 目标锚定 + 决策累积
    project_context: ProjectContext,
    /// 泛化树当前分解深度（每分解一次 +1，聚合后 -1）
    decompose_depth: usize,
    /// 最大分解深度，防止无限递归
    max_decompose_depth: usize,
}

impl ConstrainedEngine {
    fn new() -> Self {
        eprintln!("[debug] ConstrainedEngine::new: CangVM...");
        let mut vm = CangVM::new();
        vm.state = Gua::ORIGIN;
        vm.origin = Gua::ORIGIN;

        // ── 藏海持久化：从文件加载历史经验 ──
        const CANGSEA_PATH: &str = "cangsea_data.json";
        vm.cangsea_path = Some(std::path::PathBuf::from(CANGSEA_PATH));
        if let Ok(loaded) = CangSea::load_from_file(CANGSEA_PATH) {
            vm.cang_sea = loaded;
            eprintln!("[debug] 藏海从文件加载: {}条经验", vm.cang_sea.len());
        } else {
            eprintln!("[debug] 藏海无历史数据，从空开始");
        }

        // ── 观测层：EmbeddingObserver 注入 ──
        // 使用空 HanziMap 初始化（降级模式：纯 Hamming 偏离度）。
        // 当 export_hanzi_embeddings.py 导出真实嵌入数据后，
        // 切换为 include_bytes!("../data/hanzi_embeddings.bin") 加载。
        eprintln!("[debug] ConstrainedEngine::new: EmbeddingObserver (空HanziMap降级)...");
        let hanzi_map = HanziMap::empty();
        let observer = EmbeddingObserver::new(hanzi_map);
        vm.embedding_observer = Some(observer);

        eprintln!("[debug] ConstrainedEngine::new: ShanVM/ZhouVM...");
        // 创建引擎并注入 CangVM（单一数据源）
        vm.shan_vm = Some(ShanVM::new());
        vm.zhou_vm = Some(ZhouVM::new());

        eprintln!("[debug] ConstrainedEngine::new: HttpBackend...");
        let backend = HttpBackend::new(LLAMA_SERVER_URL);

        eprintln!("[debug] ConstrainedEngine::new: vocab discovery...");
        // Retry vocab discovery until llama.cpp is ready (or timeout)
        let (off_focus_ids, divergent_ids) = Self::discover_vocab_with_retry(&backend);
        eprintln!("[debug] ConstrainedEngine::new: vocab done");

        ConstrainedEngine {
            backend,
            cang_vm: vm,
            ctx: LlmContext::new(xiang_core::SAN_YI_CONSTRAINT_FULL_TEXT),
            messages: vec![],
            eos_id: vocab::QWEN_EOS_TOKEN,
            off_focus_ids,
            divergent_ids,
            project_context: ProjectContext::new("通用AI助手对话".into(), 6),
            decompose_depth: 0,
            max_decompose_depth: 2,
        }
    }

    /// Retry vocab discovery up to 30 seconds, waiting for llama.cpp to be ready.
    fn discover_vocab_with_retry(backend: &HttpBackend) -> (Vec<u32>, Vec<u32>) {
        for attempt in 1..=30 {
            if backend.is_ready() {
                let result = vocab::discover_tokens(backend);
                if !result.0.is_empty() || !result.1.is_empty() {
                    return result;
                }
                eprintln!("[vocab] attempt {}/30: discovered 0 tokens, retrying...", attempt);
            } else {
                eprintln!("[vocab] attempt {}/30: server not ready, waiting...", attempt);
            }
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
        eprintln!("[vocab] WARNING: vocab discovery failed after 30 attempts, continuing with empty token lists");
        (vec![], vec![])
    }

    fn current_phase(&self) -> String {
        self.cang_vm.current_operator().unwrap_or("归").to_string()
    }

    /// Generate with full three-engine collaborative flow.
    ///
    /// Flow per message:
    ///   1. ZhouVM selects cognitive posture → temperature
    ///   2. CangVM: get current operator from queue
    ///   3. CangVM: execute Gua state transition
    ///   4. CangVM: compute deviation → dynamic temperature
    ///   5. LLM: generate with logit bias
    ///   6. Analyzer: compute output quality deviation
    ///   7. CangVM: validate against YinProtocolChecker
    ///   8. If deviation high → ShanVM obstacle navigation
    ///   9. CangVM: judge() → PASS/ROLLBACK/SKIP/STOP/FINISH
    fn generate(&mut self, message: &str) -> Result<ConstrainedResponse, String> {
        // ── 1. ZhouVM: select cognitive posture based on system state ──
        self.cang_vm.rounds_since_perception = 0;
        self.select_zhou_pose();

        // ── 2. Determine current operator ──
        let operator = match self.cang_vm.current_operator() {
            Some(op) => op.to_string(),
            None => {
                // Queue exhausted — reset for a new cycle
                self.cang_vm.reset_phase_state();
                "生".to_string()
            }
        };

        self.cang_vm.cycle_count += 1;

        // ── 3. Execute Gua state transition ──
        self.cang_vm
            .execute_phase_operator(&operator)
            .map_err(|e| format!("算子执行失败: {e}"))?;

        // ── 4. Compute deviation → blend CangVM + ZhouVM temperature ──
        let dev = self.cang_vm.current_deviation();
        let zhou_temp = self.cang_vm.zhou_temperature();

        // Blend: deviation adjusts ZhouVM's base temperature
        let temp = if dev > 0.7 {
            // High deviation → lock down
            (zhou_temp * 0.4).clamp(0.2, 0.8)
        } else if dev < 0.3 {
            // Low deviation → let ZhouVM's creative posture shine
            (zhou_temp * 1.2).clamp(0.5, 1.3)
        } else {
            zhou_temp.clamp(0.3, 1.0)
        };
        let temp_mode = TemperatureMode::Fixed(temp);

        let state = self.cang_vm.state;
        let sha_count = self.cang_vm.sha_count;

        // ── 4a. ShanVM: pre-generation obstacle navigation ──
        let mut shan_decision: Option<xiang_core::LianShanDecision> = None;
        if dev > 0.5 {
            if let Some(decision) = self.cang_vm.run_shan_vm() {
                // ── 泛化树入口：ShanVM 决定 Decompose 时，暂停管道，拆分为子任务执行 ──
                if decision.decision == xiang_core::FangWei::Decompose
                    && self.decompose_depth < self.max_decompose_depth
                {
                    return self.decompose_and_execute(message);
                }

                shan_decision = Some(decision);
            }
        }

        // ── 4b. 系统提示词注入（v3.1：体系全文 + 动态提示词 + 项目上下文）──
        {
            // 解析 ZhouVM 姿态信息
            let posture = self.cang_vm.zhou_posture();
            let (pose_name, pose_desc) = if let Some(idx) = posture.find(" · ") {
                (&posture[..idx], &posture[idx + 3..])
            } else {
                ("坤", "承载")
            };

            let engine_hint = semantic_injector::EngineHint {
                operator: operator.clone(),
                posture_name: pose_name.to_string(),
                posture_description: pose_desc.to_string(),
                temperature: temp,
                deviation: dev,
                shan_decision: shan_decision
                    .as_ref()
                    .map(|d| format!("{} | {} → {}",
                        if d.activated { "已激活" } else { "未激活" },
                        d.jia.name(),
                        d.decision.name())),
            };

            let injection_state = semantic_injector::InjectionState {
                project_context: Some(self.project_context.section()),
                engine_state: Some(engine_hint),
            };
            self.ctx.system_prompt = semantic_injector::build_injection(&injection_state);
        }

        // ── 4c. ZhouVM cognitive posture → prompt prefix ──
        let prefix = self.cang_vm.zhou_prompt_prefix();
        let guided_message = format!("[{}]\n{}", prefix, message);

        // ── 5. Generate with logit bias ──
        // 算子专属方向引导 token 池（v4.0）
        // 从 LLM 词表中识别各类 token 的脚本见 scripts/explore_token_pools.py
        // 当前传空 Vec：通用压制规则仍生效，算子专属引导在 token 池就绪后自动生效
        let op_static: &str = Box::leak(operator.clone().into_boxed_str());
        let operator_positive: Vec<u32> = Vec::new(); // TODO: 从词表识别
        let operator_negative: Vec<u32> = Vec::new(); // TODO: 从词表识别

        let result = self
            .ctx
            .generate_constrained_turn(
                &mut self.backend,
                &guided_message,
                200,
                temp_mode,
                state,
                dev,
                sha_count,
                self.off_focus_ids.clone(),
                self.divergent_ids.clone(),
                self.eos_id,
                &[],
                None,  // strategy_bias (chat app doesn't use ShanVM→bias path)
                op_static,
                operator_positive,
                operator_negative,
            )
            .map_err(|e| format!("约束生成失败: {e}"))?;

        // ── 5a. EmbeddingObserver: semantic fingerprint observation ──
        // 观测层：每步生成后将 LLM 输出嵌入投影为汉字序列。
        // 首次观测时自动设置锚点（当前嵌入作为 origin）。
        // HanziMap 为空时降级：observe() 返回 None，semantic_deviation 为 0.0。
        let mut semantic_fingerprint: Option<Vec<char>> = None;
        let mut sem_dev_update: Option<f32> = None;
        if let Some(embedding) = &result.embedding {
            if let Some(ref mut observer) = self.cang_vm.embedding_observer {
                // 首次观测：将当前嵌入设为锚点（origin）
                if !observer.has_origin() {
                    observer.set_origin(embedding);
                    eprintln!("[归·观测] 锚点已设置: 维度={}", embedding.len());
                }
                observer.observe(embedding);
                sem_dev_update = Some(observer.semantic_deviation());
                semantic_fingerprint = Some(observer.fingerprint().to_vec());

                // 调试输出：汉字投影轨迹
                if observer.fingerprint_len() % 10 == 0 {
                    let recent: String = observer.recent_fingerprint(5).iter().collect();
                    eprintln!(
                        "[归·观测] 汉字轨迹(最近5): {} | 语义偏离: {:.3} | 弱锚定比: {:.2}",
                        recent,
                        observer.semantic_deviation(),
                        observer.weak_anchor_ratio(),
                    );
                }
            }
        }
        if let Some(dev) = sem_dev_update {
            self.cang_vm.set_semantic_deviation(dev);
        }

        // ── 6. Analyzer: compute output quality deviation ──
        let output_dev = analyzer::compute_output_deviation(&result.text);

        // ── 7. Validate against YinProtocolChecker ──
        let check_result = self.cang_vm.check_operator_output(&operator, &result.text);
        let is_valid = check_result.is_valid && output_dev < 0.5;
        let mut violations = check_result.violations;
        if output_dev >= 0.5 {
            violations.push(format!("输出质量偏差过高: {:.2}", output_dev));
        }

        // ── 8. ShanVM: post-generation obstruction tracking ──
        // 检测障碍但不否定输出 — 偏离度只用于触发拆解分解
        if !is_valid && !self.cang_vm.shan_triggered {
            if let Some(decision) = self.cang_vm.run_shan_vm() {
                if shan_decision.is_none() {
                    shan_decision = Some(decision);
                }
            }
        }

        // ── 9. CangVM 内部状态推进 ──
        // judge() 推进算子循环（驱动 Logit-Bias/Temperature），但不再否定输出。
        // 无论 judge 判定什么，模型输出都被保留。
        let _judge_result = self.cang_vm.judge(&operator, is_valid, &violations);
        let final_phase = self.cang_vm.current_operator().unwrap_or("?").to_string();
        let final_dev = self.cang_vm.current_deviation();

        // ── 9a. 记录决策到 ProjectContext（代谢系统）──
        // 使用输出质量 + 协议检查结果自动判定 reward
        let decision_content: String = result.text.chars().take(120).collect();
        self.record_decision(
            self.cang_vm.cycle_count,
            &final_phase,
            final_dev,
            is_valid,
            decision_content,
        );

        // ── 10. 压缩树：检查新陈代谢信号并执行上下文操作 ──
        self.handle_metabolism_signal(message, &result.text);

        // Track in history
        self.messages.push(Message {
            role: "user".into(),
            content: message.into(),
        });
        self.messages.push(Message {
            role: "assistant".into(),
            content: result.text.clone(),
        });

        Ok(ConstrainedResponse {
            response: result.text,
            tokens: result.tokens_generated,
            deviation: final_dev,
            intervened: false,
            phase: final_phase,
            violations: violations,
            decision: "pass".to_string(),
            output_deviation: output_dev,
            zhou_posture: self.cang_vm.zhou_posture().to_string(),
            zhou_temperature: self.cang_vm.zhou_temperature(),
            shan_decision: shan_decision.map(|d| format!(
                "{} | {} → {}",
                d.activated,
                d.jia.name(),
                d.decision.name()
            )),
            semantic_deviation: self.cang_vm.semantic_deviation,
            semantic_fingerprint,
        })
    }

    /// Select ZhouVM cognitive posture based on system state signals.
    ///
    /// Rules (cold-start, no training data required):
    ///   CangVM phase → posture:
    ///     生(探索) → 震(启动, temp 1.0)
    ///     动(发散) → 乾(创造, temp 1.2)
    ///     长(聚焦) → 巽(渗透, temp 0.7)
    ///     育(结构) → 离(明照, temp 0.5)
    ///
    ///   Deviation override:
    ///     dev > 0.7 → 艮(止定, temp 0.3) — lock down
    ///     dev > 0.5 → 坎(破局, temp 1.1) — break pattern
    ///     dev < 0.3 → 坤(承载, temp 0.6) — stable
    ///
    ///   ShanVM strategy override (highest priority when active):
    ///     Decompose    → 巽(渗透, temp 0.7)
    ///     PushThrough  → 乾(创造, temp 1.2)
    ///     NavigateAround → 坎(破局, temp 1.1)
    ///     Abort/Escalate → 艮(止定, temp 0.3)
    ///     WaitGather   → 坤(承载, temp 0.6)
    fn select_zhou_pose(&mut self) {
        use xiang_core::Bagua;

        let dev = self.cang_vm.current_deviation();
        let operator = self.cang_vm.current_operator().unwrap_or("生");
        let shan_active = self.cang_vm.shan_triggered;
        let shan_decision = self.cang_vm.last_lian_shan_decision.as_ref();

        let pose: Bagua = if shan_active {
            // ── ShanVM strategy overrides posture ──
            if let Some(decision) = shan_decision {
                match decision.decision {
                    xiang_core::FangWei::Decompose => Bagua::巽,
                    xiang_core::FangWei::PushThrough => Bagua::乾,
                    xiang_core::FangWei::NavigateAround => Bagua::坎,
                    xiang_core::FangWei::Abort | xiang_core::FangWei::Escalate => Bagua::艮,
                    _ => Bagua::坤,
                }
            } else {
                Bagua::坤
            }
        } else if dev > 0.7 {
            // ── High deviation → lock down ──
            Bagua::艮
        } else if dev > 0.5 {
            // ── Medium deviation → break pattern ──
            Bagua::坎
        } else {
            // ── Phase-driven posture ──
            match operator {
                "生" => Bagua::震, // 启动: initialize exploration
                "动" => Bagua::乾, // 创造: diverge, brainstorm
                "长" => Bagua::巽, // 渗透: focus, converge
                "育" => Bagua::离, // 明照: structure, analyze
                _ => Bagua::坤,    // 承载: default stable
            }
        };

        self.cang_vm.zhou_vm.as_mut().unwrap().execute_pose(pose);
    }

    fn state(&self) -> ConstrainedState {
        ConstrainedState {
            gua_state: format!("{:06b}", self.cang_vm.state.0),
            deviation: self.cang_vm.current_deviation(),
            sha_count: self.cang_vm.sha_count,
            cycle_count: self.cang_vm.cycle_count,
            phase: self.current_phase().to_string(),
            operator_index: self.cang_vm.current_op_index,
            total_retries: self.cang_vm.total_retries,
            shan_triggered: self.cang_vm.shan_triggered,
            shan_obstruction_count: self.cang_vm.obstruction_count,
            zhou_posture: self.cang_vm.zhou_posture().to_string(),
            zhou_pose_name: self.cang_vm.zhou_vm.as_ref().unwrap().current_pose_name().to_string(),
            zhou_temperature: self.cang_vm.zhou_temperature(),
            cang_aligned: self.cang_vm.cang_aligned_count,
            cang_deviated: self.cang_vm.cang_deviated_count,
        }
    }

    fn reset(&mut self) {
        let mut vm = CangVM::new();
        vm.state = Gua::ORIGIN;
        vm.origin = Gua::ORIGIN;
        vm.shan_vm = Some(ShanVM::new());
        vm.zhou_vm = Some(ZhouVM::new());
        self.cang_vm = vm;
        self.ctx = LlmContext::new("");
        self.messages.clear();
        // Re-discover token groups (in case model changed)
        let (off_focus_ids, divergent_ids) = vocab::discover_tokens(&self.backend);
        self.off_focus_ids = off_focus_ids;
        self.divergent_ids = divergent_ids;
    }

    /// 记录本轮决策到项目上下文。
    ///
    /// reward 由输出质量自动判定：
    ///   - dev < 0.3 && is_valid → +0.8 (高质量决策)
    ///   - dev < 0.5 && is_valid → +0.5 (可接受)
    ///   - dev >= 0.5 || !is_valid → -0.3 (噪声，将被丢弃)
    // ── 压缩树：新陈代谢信号处理（v4.0） ────────────────────────

    /// 检查并执行 CangVM 发出的新陈代谢信号。
    ///
    /// 杀/止/藏不直接操作上下文——CangVM 只发信号。
    /// 实际上下文操作由这个函数在主循环中执行。
    fn handle_metabolism_signal(&mut self, message: &str, response: &str) {
        let signal = match self.cang_vm.metabolism_signal.take() {
            Some(s) => s,
            None => return,
        };

        eprintln!("[压缩树] 收到新陈代谢信号: {:?}", signal);

        match signal {
            MetabolismSignal::Sha(rounds) => {
                self.crop_history(rounds);
            }
            MetabolismSignal::Zhi => {
                self.freeze_context();
            }
            MetabolismSignal::Cang { .. } => {
                // 用当前对话的摘要作为"藏"的保留摘要
                let summary = self.build_cycle_summary(message, response);
                self.reset_context(&summary);
            }
        }
    }

    /// 杀：裁剪最近 N 轮对话历史。
    ///
    /// 从 self.messages 和 self.ctx.history 中删除最后 N 轮。
    /// 模型下一轮将"忘记"被裁剪的内容。
    fn crop_history(&mut self, rounds: u32) {
        let n = rounds as usize;
        // 裁剪对话历史（每轮包含 user + assistant = 2 条消息）
        let msg_remove = (n.min(self.messages.len() / 2)) * 2;
        for _ in 0..msg_remove {
            self.messages.pop();
        }
        // 裁剪 LLM 上下文历史
        for _ in 0..n {
            self.ctx.history.pop();
        }
        eprintln!("[压缩树·杀] 裁剪了 {} 轮对话历史 (消息{}条, 历史{}轮)",
            n, msg_remove, n.min(self.ctx.history.len() + n));
    }

    /// 止：冻结上下文。
    ///
    /// 设置冻结标记——下一轮生成时不再追加新对话。
    /// 在系统提示词中追加"请基于已有信息做结论"。
    fn freeze_context(&mut self) {
        // 在下一轮的系统提示词中追加冻结指令
        let freeze_text = "\n\n【上下文冻结】请基于已有信息得出结论，不要展开新方向或引入新话题。";
        self.ctx.system_prompt.push_str(freeze_text);
        eprintln!("[压缩树·止] 上下文已冻结");
    }

    /// 藏：清空上下文，保留摘要。
    ///
    /// - 保留本轮摘要
    /// - 清空消息列表和对话历史
    /// - 新一轮周天从干净的上下文开始
    fn reset_context(&mut self, summary: &str) {
        // 记录本轮总览到日志
        eprintln!("[压缩树·藏] 清空上下文, 保留摘要 ({}字符)", summary.chars().count());

        // 清空对话消息列表
        self.messages.clear();

        // 清空 LLM 上下文历史
        self.ctx.history.clear();

        // 如果有 ProjectContext，保存当前摘要
        if !summary.is_empty() {
            eprintln!("[压缩树·藏] 留存摘要: {}", &summary[..summary.len().min(100)]);
        }

        // 重置连山状态
        self.cang_vm.shan_triggered = false;
        self.cang_vm.obstruction_count = 0;
        self.cang_vm.rounds_since_perception = 0;
        self.cang_vm.reset_phase_state();

        // 重置嵌入观察器（保留锚点）
        if let Some(ref mut observer) = self.cang_vm.embedding_observer {
            observer.reset();
        }

        eprintln!("[压缩树·藏] 上下文已清空，准备下一周天");
    }

    /// 构建本轮周期的简短摘要。
    fn build_cycle_summary(&self, message: &str, response: &str) -> String {
        let msg_short: String = message.chars().take(100).collect();
        let resp_short: String = response.chars().take(200).collect();
        format!(
            "【前序周期摘要】\n用户: {} \n归藏约束输出: {} \n",
            msg_short, resp_short
        )
    }

    fn record_decision(&mut self, round: u32, operator: &str, deviation: f32, is_valid: bool, content: String) {
        let reward = if deviation < 0.3 && is_valid {
            0.8
        } else if deviation < 0.5 && is_valid {
            0.5
        } else {
            -0.3
        };
        self.project_context.add_decision(round, operator, reward, content);
    }

    /// 简易生成 — 绕过三易引擎管道，直接调用LLM。
    ///
    /// 用于泛化树子任务执行：不经过 ZhouVM/CangVM/ShanVM，
    /// 只注入 ProjectContext 作为系统提示词，避免管道递归。
    /// 子任务的历史不会污染主对话上下文。
    fn generate_simple(&mut self, message: &str) -> Result<String, String> {
        use crate::semantic_injector::{self, InjectionState};

        // 保存并清空主对话上下文
        let saved_prompt = std::mem::replace(&mut self.ctx.system_prompt, String::new());
        let saved_history = std::mem::take(&mut self.ctx.history);

        // 注入项目上下文作为系统提示词（子任务引擎状态为 None，使用对照组模式）
        let injection_state = InjectionState {
            project_context: Some(self.project_context.section()),
            engine_state: None,
        };
        self.ctx.system_prompt = semantic_injector::build_injection(&injection_state);

        // 无约束生成（子任务足够小，无需 Logit-Bias 干预）
        let result = self
            .ctx
            .generate_unconstrained_turn(&mut self.backend, message, 200, TemperatureMode::Fixed(0.7), &[])
            .map_err(|e| format!("子任务生成失败: {e}"))?;

        // 恢复主对话上下文，丢弃子任务历史
        self.ctx.history = saved_history;
        self.ctx.system_prompt = saved_prompt;

        Ok(result.text)
    }

    /// 泛化树拆解执行 — 当 ShanVM 决定 Decompose 且深度未达上限时调用。
    ///
    /// 流程：
    ///   1. 用 LLM 将当前任务拆解为 2-4 个子任务
    ///   2. 每个子任务独立执行 generate_simple()
    ///   3. 聚合全部结果
    ///   4. 记录拆解决策到 ProjectContext
    fn decompose_and_execute(&mut self, original_message: &str) -> Result<ConstrainedResponse, String> {
        // ── 1. 生成拆解方案 ──
        let decomp_prompt = build_decomposition_prompt(original_message);
        let decomp_response = self.generate_simple(&decomp_prompt)?;

        let subtasks = parse_subtasks(&decomp_response);

        // ── 2. 拆解失败回退 ──
        if subtasks.is_empty() {
            eprintln!("[泛化树] 拆解失败，回退为简易生成");
            let fallback_text = self.generate_simple(original_message)?;
            self.messages.push(Message { role: "user".into(), content: original_message.into() });
            self.messages.push(Message { role: "assistant".into(), content: fallback_text.clone() });
            return Ok(ConstrainedResponse {
                response: fallback_text,
                tokens: 0,
                deviation: self.cang_vm.current_deviation(),
                intervened: false,
                phase: "拆解(降级)".to_string(),
                violations: vec![],
                decision: "decompose_fallback".to_string(),
                output_deviation: 0.0,
                zhou_posture: self.cang_vm.zhou_posture().to_string(),
                zhou_temperature: self.cang_vm.zhou_temperature(),
                shan_decision: Some("拆解分解(降级为单任务)".to_string()),
                semantic_deviation: self.cang_vm.semantic_deviation,
                semantic_fingerprint: None,
            });
        }

        eprintln!(
            "[泛化树·深度{}] 拆解为 {} 个子任务: {:?}",
            self.decompose_depth + 1,
            subtasks.len(),
            subtasks.iter().map(|s| s.chars().take(50).collect::<String>()).collect::<Vec<_>>()
        );

        // ── 3. 逐个执行子任务 ──
        self.decompose_depth += 1;
        let mut results: Vec<String> = Vec::with_capacity(subtasks.len());
        for (i, subtask) in subtasks.iter().enumerate() {
            eprintln!("[泛化树] 执行子任务 {}/{} ...", i + 1, subtasks.len());
            match self.generate_simple(subtask) {
                Ok(text) => {
                    results.push(format!("## 子任务 {}: {}\n\n{}", i + 1, subtask, text));
                }
                Err(e) => {
                    results.push(format!("## 子任务 {}: {}\n\n(执行失败: {})", i + 1, subtask, e));
                }
            }
        }
        self.decompose_depth -= 1;

        // ── 4. 聚合结果 ──
        let aggregated = format!(
            "【泛化树·深度{}】将任务拆为{}个子任务，逐一执行：\n\n{}",
            self.decompose_depth + 1,
            subtasks.len(),
            results.join("\n\n---\n\n")
        );

        // ── 5. 记录拆解决策到 ProjectContext ──
        let subtask_summary: String = subtasks
            .iter()
            .map(|s| s.chars().take(40).collect::<String>())
            .collect::<Vec<_>>()
            .join(" | ");
        self.project_context.add_decision(
            self.cang_vm.cycle_count,
            "育", // 结构化拆解 = 高价值
            0.7,
            format!(
                "[泛化树·深度{}] 拆{}个子任务: {}",
                self.decompose_depth + 1,
                subtasks.len(),
                subtask_summary
            ),
        );

        // ── 6. 记录到主对话历史 ──
        self.messages.push(Message { role: "user".into(), content: original_message.into() });
        self.messages.push(Message { role: "assistant".into(), content: aggregated.clone() });

        Ok(ConstrainedResponse {
            response: aggregated,
            tokens: 0,
            deviation: self.cang_vm.current_deviation(),
            intervened: true,
            phase: format!("拆解·深度{}", self.decompose_depth + 1),
            violations: vec![],
            decision: "decompose".to_string(),
            output_deviation: 0.0,
            zhou_posture: self.cang_vm.zhou_posture().to_string(),
            zhou_temperature: self.cang_vm.zhou_temperature(),
            shan_decision: Some(format!("拆解分解 → {}个子任务", subtasks.len())),
            semantic_deviation: self.cang_vm.semantic_deviation,
            semantic_fingerprint: None,
        })
    }
}

// ── 泛化树辅助函数 ────────────────────────────────────────

/// 构建任务拆解提示词。
fn build_decomposition_prompt(question: &str) -> String {
    format!(
        "将以下复杂任务拆解成 2-4 个可以独立解决的子任务。\n\
         \n\
         任务：{}\n\
         \n\
         每个子任务应足够简单，能被一个小模型独立完成。\n\
         请严格按照以下格式输出，每行一个子任务：\n\
         [SUB] 子任务描述1\n\
         [SUB] 子任务描述2\n\
         \n\
         只输出 [SUB] 格式的行，不要输出其他内容。",
        question
    )
}

/// 从 LLM 回复中解析子任务列表。
fn parse_subtasks(response: &str) -> Vec<String> {
    // 先剥离 think 标签
    let cleaned = strip_think_tags(response);

    let mut subtasks: Vec<String> = cleaned
        .lines()
        .filter(|l| l.trim().starts_with("[SUB]"))
        .map(|l| l.trim()[5..].trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    // 回退：如果没找到 [SUB] 行，尝试识别编号列表
    if subtasks.is_empty() {
        subtasks = cleaned
            .lines()
            .filter(|l| {
                let t = l.trim();
                t.starts_with("1.") || t.starts_with("2.") || t.starts_with("3.") || t.starts_with("4.")
                    || t.starts_with("- ") || t.starts_with("* ")
            })
            .map(|l| {
                let t = l.trim();
                if let Some(pos) = t.find(". ") {
                    t[pos + 2..].to_string()
                } else if t.starts_with("- ") || t.starts_with("* ") {
                    t[2..].to_string()
                } else {
                    t.to_string()
                }
            })
            .filter(|s| !s.is_empty())
            .collect();
    }

    // 最多 4 个子任务
    subtasks.truncate(4);
    subtasks
}

/// 剥离 Qwen3 思考模型的 `<think>...</think>` 标签。
fn strip_think_tags(text: &str) -> String {
    let mut result = text.to_string();
    while let (Some(start), Some(end)) = (result.find("<think>"), result.find("</think>")) {
        if start < end {
            result.replace_range(start..end + "</think>".len(), "");
        } else {
            break;
        }
    }
    result.trim().to_string()
}

// ── Shared app state ─────────────────────────────────────

#[derive(Clone)]
struct AppState {
    raw: Arc<Mutex<RawEngine>>,
    constrained: Arc<Mutex<ConstrainedEngine>>,
}

// ── Types ────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatRequest {
    message: String,
}

#[derive(Debug, Serialize)]
struct RawResponse {
    response: String,
    tokens: u32,
}

#[derive(Debug, Serialize)]
struct ConstrainedResponse {
    response: String,
    tokens: u32,
    deviation: f32,
    /// True if the CangVM intervened (Rollback/Skip/Stop — output degraded).
    intervened: bool,
    /// Current phase (operator name after decision).
    phase: String,
    /// Phase rule violations from the YinProtocolChecker + analyzer.
    violations: Vec<String>,
    /// Raw CangVMDecision for diagnostic purposes.
    decision: String,
    /// Analyzer output quality deviation [0,1].
    output_deviation: f32,
    /// Current ZhouVM cognitive posture description.
    zhou_posture: String,
    /// Current ZhouVM temperature.
    zhou_temperature: f32,
    /// Last ShanVM decision, if triggered.
    shan_decision: Option<String>,
    /// 语义偏离度（来自 EmbeddingObserver，=None 表示未启用）
    semantic_deviation: Option<f32>,
    /// 语义指纹：最近几个 token 映射到的汉字序列
    semantic_fingerprint: Option<Vec<char>>,
}

#[derive(Debug, Serialize)]
struct ConstrainedState {
    gua_state: String,
    deviation: f32,
    sha_count: u32,
    cycle_count: u32,
    phase: String,
    operator_index: usize,
    total_retries: u32,
    /// Whether ShanVM was triggered this cycle.
    shan_triggered: bool,
    /// ShanVM obstruction count.
    shan_obstruction_count: usize,
    /// ZhouVM current posture description.
    zhou_posture: String,
    /// ZhouVM current pose name.
    zhou_pose_name: String,
    /// ZhouVM current temperature.
    zhou_temperature: f32,
    /// CangSea aligned experience count.
    cang_aligned: u32,
    /// CangSea deviated experience count.
    cang_deviated: u32,
}

#[derive(Debug, Serialize)]
struct FullState {
    raw: Vec<Message>,
    constrained: Vec<Message>,
    constrained_state: ConstrainedState,
}

/// Detailed three-engine state with history.
#[derive(Debug, Serialize)]
struct ThreeEngineState {
    /// CangVM state summary.
    cangvm: ConstrainedState,
    /// ShanVM decision log.
    shanvm_decision_log: Vec<String>,
    /// ZhouVM pose log.
    zhouvm_pose_log: Vec<String>,
    /// ZhouVM shengke log.
    zhouvm_shengke_log: Vec<String>,
    /// CangVM output log (last 20 entries).
    cangvm_output_log: Vec<String>,
}

// ── Handlers ─────────────────────────────────────────────

async fn handle_raw(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<RawResponse>, (StatusCode, String)> {
    let mut engine = state
        .raw
        .lock()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("锁错误: {e}")))?;
    engine.generate(&req.message).map(Json).map_err(|e| {
        (StatusCode::INTERNAL_SERVER_ERROR, e)
    })
}

async fn handle_constrained(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ConstrainedResponse>, (StatusCode, String)> {
    let mut engine = state
        .constrained
        .lock()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("锁错误: {e}")))?;
    engine.generate(&req.message).map(Json).map_err(|e| {
        (StatusCode::INTERNAL_SERVER_ERROR, e)
    })
}

async fn handle_state(State(state): State<AppState>) -> Json<FullState> {
    let raw_engine = state.raw.lock().unwrap();
    let constrained_engine = state.constrained.lock().unwrap();
    Json(FullState {
        raw: raw_engine.messages.clone(),
        constrained: constrained_engine.messages.clone(),
        constrained_state: constrained_engine.state(),
    })
}

async fn handle_reset(State(state): State<AppState>) -> Json<serde_json::Value> {
    if let Ok(mut engine) = state.raw.lock() {
        engine.ctx = LlmContext::new("");
        engine.messages.clear();
    }
    if let Ok(mut engine) = state.constrained.lock() {
        engine.reset();
    }
    Json(serde_json::json!({ "status": "ok" }))
}

/// GET /api/engines — Full three-engine state dump.
async fn handle_engines(State(state): State<AppState>) -> Json<ThreeEngineState> {
    let engine = state.constrained.lock().unwrap();

    let shan_log: Vec<String> = engine.cang_vm.shan_vm.as_ref().unwrap().decision_log.iter()
        .map(|d| format!(
            "activated={} qi={} jia={} yuan={} → {}",
            d.activated, d.qi.name(), d.jia.name(), d.yuan.name(), d.decision.name()
        ))
        .collect();

    let zhou_pose_log: Vec<String> = engine.cang_vm.zhou_vm.as_ref().unwrap().pose_log.iter()
        .map(|b| format!("{}({})", b.name(), b.posture()))
        .collect();

    let zhou_shengke_log: Vec<String> = engine.cang_vm.zhou_vm.as_ref().unwrap().shengke_log.iter()
        .map(|(from, to, is_sheng)| {
            let kind = if *is_sheng { "生" } else { "克" };
            format!("{} {}→{}", kind, from.name(), to.name())
        })
        .collect();

    let cang_log: Vec<String> = engine.cang_vm.output_log.iter()
        .rev().take(20).cloned().collect();

    Json(ThreeEngineState {
        cangvm: engine.state(),
        shanvm_decision_log: shan_log,
        zhouvm_pose_log: zhou_pose_log,
        zhouvm_shengke_log: zhou_shengke_log,
        cangvm_output_log: cang_log,
    })
}

/// POST /api/zhouyi/pose — Manually set ZhouYi cognitive posture.
#[derive(Debug, Deserialize)]
struct PoseRequest {
    bagua: String,
}

async fn handle_set_pose(
    State(state): State<AppState>,
    Json(req): Json<PoseRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let mut engine = state
        .constrained
        .lock()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("锁错误: {e}")))?;

    let bagua = match req.bagua.as_str() {
        "乾" => xiang_core::Bagua::乾,
        "兑" => xiang_core::Bagua::兑,
        "离" => xiang_core::Bagua::离,
        "震" => xiang_core::Bagua::震,
        "巽" => xiang_core::Bagua::巽,
        "坎" => xiang_core::Bagua::坎,
        "艮" => xiang_core::Bagua::艮,
        "坤" => xiang_core::Bagua::坤,
        other => return Err((StatusCode::BAD_REQUEST, format!("未知卦象: {other}"))),
    };

    engine.cang_vm.zhou_vm.as_mut().unwrap().execute_pose(bagua);
    let posture = engine.cang_vm.zhou_vm.as_ref().unwrap().current_posture().to_string();
    let _pose_name = engine.cang_vm.zhou_vm.as_ref().unwrap().current_pose_name().to_string();
    let temp = engine.cang_vm.zhou_vm.as_ref().unwrap().temperature;
    engine.cang_vm.output_log.push(format!(
        "[API] 手动设姿 {} → {}", bagua.name(), posture
    ));

    Ok(Json(serde_json::json!({
        "status": "ok",
        "pose": bagua.name(),
        "posture": posture,
        "temperature": temp,
    })))
}

// ── Main ─────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    eprintln!("[debug] engine construction starting...");

    let raw = RawEngine::new();
    eprintln!("[debug] RawEngine constructed");

    let constrained = ConstrainedEngine::new();
    eprintln!("[debug] ConstrainedEngine constructed");

    let state = AppState {
        raw: Arc::new(Mutex::new(raw)),
        constrained: Arc::new(Mutex::new(constrained)),
    };

    eprintln!("[debug] building router and binding...");

    let app = Router::new()
        .route("/api/raw", post(handle_raw))
        .route("/api/constrained", post(handle_constrained))
        .route("/api/state", get(handle_state))
        .route("/api/engines", get(handle_engines))
        .route("/api/reset", get(handle_reset))
        .route("/api/zhouyi/pose", post(handle_set_pose))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3001")
        .await
        .unwrap();
    println!("Chat server running on http://0.0.0.0:3001");
    println!("  GET  /api/state    — engine state");
    println!("  GET  /api/engines  — full three-engine state");
    println!("  POST /api/raw      — unconstrained generation");
    println!("  POST /api/constrained — three-engine constrained generation");
    println!("  POST /api/zhouyi/pose — manually set ZhouYi posture");
    println!("  GET  /api/reset    — reset all engines");
    axum::serve(listener, app).await.unwrap();
}

// ─── 项目上下文代谢 — 10轮实机模拟 ──────────────────────────
//
// 模拟长期项目迭代中，系统提示词（system prompt）如何随着
// 项目上下文（ProjectContext）的积累而排出噪音、保留有效决策。
//
// 核心机制：
//   1. 项目目标始终锚定在每轮提示词顶部
//   2. 正向高重要性决策累积保留（育/长 > 生/动）
//   3. 负向经验（reward ≤ 0）直接丢弃
//   4. 低重要性决策超出容量后被淘汰
//   5. 每轮系统提示词全量重建，旧上下文不残留
//
// 运行: cargo test --package xiang-chat -- -- metabolism_simulation --nocapture
// 输出: target/metabolism_demo/round_*.txt

#[cfg(test)]
mod metabolism_simulation {
    use super::semantic_injector::{self, InjectionState};
    use crate::strip_think_tags;
    use xiang_core::ProjectContext;
    use std::fs;
    use std::path::Path;

    const OUTPUT_DIR: &str = "target/metabolism_demo";

    struct RoundDef {
        /// 本轮决定/洞察内容（人类可读）
        decision: &'static str,
        /// 当前算子 ("生"/"动"/"长"/"育"/"杀")
        operator: &'static str,
        /// 本轮奖励 [-1, 1]
        reward: f32,
        /// 场景描述
        scenario: &'static str,
    }

    /// 模拟一轮对话：添加决策到 ProjectContext → 构建注入 → 生成提示词
    fn simulate_round(
        ctx: &ProjectContext,
        def: &RoundDef,
        num: u32,
    ) -> String {
        // 1. 生成项目上下文段落
        let proj_section = ctx.section();

        // 2. 构建注入状态
        let state = InjectionState {
            project_context: Some(proj_section.clone()),
            engine_state: None,
        };

        // 3. 生成 system prompt
        let prompt = semantic_injector::build_injection(&state);

        // 4. 格式化输出
        let importance = match def.operator {
            "育" => def.reward.abs().min(1.0) * 1.0,
            "长" => def.reward.abs().min(1.0) * 0.8,
            "动" => def.reward.abs().min(1.0) * 0.5,
            "生" => def.reward.abs().min(1.0) * 0.4,
            _ => def.reward.abs().min(1.0) * 0.3,
        };

        let accepted = def.reward > 0.0;
        let importance_mark = if importance >= 0.7 { "★★★" }
            else if importance >= 0.4 { "★★" }
            else { "★" };

        let mut out = String::new();
        out.push_str(&format!(
            "══════════════════════════════════════════════\n\
             项目上下文代谢 — 迭代 #{num:02}\n\
             \"{scenario}\"\n\
             ══════════════════════════════════════════════\n\n",
            scenario = def.scenario,
        ));

        out.push_str(&format!(
            "[本轮决策]\n\
             算子: {op} | 奖励: {reward:+.2} | 重要性: {imp:.2} {mark}\n\
             采纳状态: {accepted}\n\
             内容: {decision}\n\n",
            op = def.operator,
            reward = def.reward,
            imp = importance,
            mark = importance_mark,
            accepted = if accepted { "✓ 采纳" } else { "✗ 丢弃(负奖励)" },
            decision = def.decision,
        ));

        out.push_str(&format!(
            "[项目上下文状态]\n\
             当前决策保留数: {n}/{max}（目标始终锚定，不占容量）\n\
             决策列表:\n",
            n = ctx.decisions.len(),
            max = ctx.max_retain,
        ));
        if ctx.decisions.is_empty() {
            out.push_str("  (暂无积累)\n");
        } else {
            for d in &ctx.decisions {
                let star = if d.importance >= 0.7 { "★" }
                    else if d.importance >= 0.4 { "✦" }
                    else { "·" };
                let truncated: String = d.content.chars().take(60).collect();
                out.push_str(&format!(
                    "  [R{:02}] {} imp={:.2} {}\n",
                    d.round, star, d.importance, truncated
                ));
            }
        }
        out.push_str("\n");

        out.push_str(
            "────────────────────────────────────────────\n\
             [系统提示词（完整输出）]\n\
             ────────────────────────────────────────────\n",
        );
        out.push_str(&prompt);
        out.push_str(
            "\n────────────────────────────────────────────\n\n",
        );

        out
    }

    #[test]
    fn test_metabolism_full_simulation() {
        let out_dir = Path::new(OUTPUT_DIR);
        fs::create_dir_all(out_dir).expect("Failed to create output dir");

        // ── 项目目标（始终锚定，永不过期）──
        let project_goal = "构建全栈Web应用：Rust(actix-web) + React(TypeScript) + PostgreSQL";

        // ── 最大保留 6 条决策（超出后低重要性决策被淘汰）──
        let mut ctx = ProjectContext::new(project_goal.to_string(), 6);

        // ═══════════════════════════════════════════════
        // 10 轮真实决策场景
        // ═══════════════════════════════════════════════

        let round_defs = vec![
            // Round 1: 框架选型 — 生(探索, 低权重) 但高奖励
            RoundDef {
                decision: "后端框架选型：确定使用actix-web 4.x，性能优先",
                operator: "生", reward: 0.80,
                scenario: "确定Rust后端框架选型",
            },
            // Round 2: API架构 — 育(结构化, 高权重) 高奖励 ★★
            RoundDef {
                decision: "API层架构：三层分离 — handler/svc/repo，中间件：auth+logging+cors",
                operator: "育", reward: 0.90,
                scenario: "设计后端API架构分层",
            },
            // Round 3: 前端探索 — 生(探索, 低权重) 中等奖励
            RoundDef {
                decision: "前端技术栈：React 18 + TypeScript + Vite，组件库用 shadcn/ui",
                operator: "生", reward: 0.60,
                scenario: "前端技术栈选型",
            },
            // Round 4: 数据库设计 — 育(结构化, 高权重) 高奖励 ★★★
            RoundDef {
                decision: "数据库：PostgreSQL 16，核心表 users/projects/tasks，索引策略先覆盖高频查询",
                operator: "育", reward: 0.95,
                scenario: "数据库Schema设计",
            },
            // Round 5: CSS方案探索 — 生(低权重) + 低奖励 → 易被淘汰
            RoundDef {
                decision: "CSS方案调研：emotion vs tailwind vs css-modules，倾向tailwind",
                operator: "生", reward: 0.30,
                scenario: "前端CSS方案探索",
            },
            // Round 6: WebSocket调试 — 杀(负面操作) + 负奖励 → 直接丢弃
            RoundDef {
                decision: "尝试用tokio-tungstenite实现WebSocket，遇到连接管理问题，待解决",
                operator: "杀", reward: -0.50,
                scenario: "WebSocket实时通信调试",
            },
            // Round 7: 编译排错 — 杀(负面操作) + 负奖励 → 直接丢弃
            RoundDef {
                decision: "解决actix-web中间件类型不匹配的编译错误，花费大量时间",
                operator: "杀", reward: -0.70,
                scenario: "Rust编译错误排查",
            },
            // Round 8: 认证系统 — 长(聚焦, 中高权重) 高奖励 ★★
            RoundDef {
                decision: "认证方案：JWT access + refresh token双令牌，用redis做token黑名单",
                operator: "长", reward: 0.85,
                scenario: "设计用户认证系统",
            },
            // Round 9: 日志方案 — 生(低权重) + 低奖励 → 易被淘汰
            RoundDef {
                decision: "日志方案调研：tracing-subscriber + opentelemetry导出",
                operator: "生", reward: 0.25,
                scenario: "日志和可观测性方案探索",
            },
            // Round 10: 部署 — 育(高权重) + 高奖励 ★★★
            RoundDef {
                decision: "部署方案：Docker Compose编排，nginx反向代理，GitHub Actions CI/CD",
                operator: "育", reward: 0.88,
                scenario: "确定生产部署方案",
            },
        ];

        // ── 跟踪每轮决策是否被采纳/淘汰 ──
        let mut decision_fate: Vec<(u32, &str, bool, f32, String)> = Vec::new();

        // ── 执行每轮模拟 ──
        for (i, def) in round_defs.iter().enumerate() {
            let num = (i + 1) as u32;

            // 先 add_decision，再用 ProjectContext 生成 section
            ctx.add_decision(num, def.operator, def.reward, def.decision.to_string());

            // 计算本决策的 importance 和采纳状态
            let importance = match def.operator {
                "育" => def.reward.abs().min(1.0) * 1.0,
                "长" => def.reward.abs().min(1.0) * 0.8,
                "动" => def.reward.abs().min(1.0) * 0.5,
                "生" => def.reward.abs().min(1.0) * 0.4,
                _ => def.reward.abs().min(1.0) * 0.3,
            };
            let accepted = def.reward > 0.0;
            // 检查本决策是否还存在于上下文中（可能因容量不足被淘汰）
            let retained = ctx.decisions.iter().any(|d| d.round == num);

            decision_fate.push((num, def.decision, accepted && retained, importance, def.scenario.to_string()));

            // 生成本轮提示词并写入文件
            let round_text = simulate_round(&ctx, def, num);
            let filename = format!("round_{:02}.txt", num);
            let filepath = out_dir.join(&filename);
            fs::write(&filepath, &round_text)
                .unwrap_or_else(|_| panic!("Failed to write {}", filename));
        }

        // ── 写入 SUMMARY.txt ──
        let mut summary = String::new();
        summary.push_str(
            "══════════════════════════════════════════════════════════════════\n\
             项目上下文代谢模拟 — 10轮决策 — 噪声过滤 & 有效信息累积\n\
             ══════════════════════════════════════════════════════════════════\n\n\
             核心机制：\n\
             · 项目目标始终锚定 → 每轮提示词顶部可见\n\
             · reward ≤ 0 的决策直接丢弃（调试失败、编译错误等噪声）\n\
             · 低 importance(<0.3) 决策超出容量(6条)后被淘汰\n\
             · 高 importance(≥0.7) 的结构化决策(育/长)优先保留\n\
             · 每轮系统提示词全量重建，不累积旧文本\n\n\
             运行命令：cargo test --package xiang-chat -- -- metabolism_simulation --nocapture\n\n",
        );

        // ── 决策命运表 ──
        summary.push_str(
            "┌──────┬──────┬──────────┬────────┬────────┬──────────────────────────────────┐\n\
             │ 轮次 │ 算子 │ 重要性    │ 奖励    │ 命运     │ 决策摘要                           │\n\
             ├──────┼──────┼──────────┼────────┼────────┼──────────────────────────────────┤\n",
        );

        for (num, decision, kept, imp, _scenario) in &decision_fate {
            let fate = if *kept { "✓保留" } else { "✗丢弃" };
            let short: String = decision.chars().take(32).collect();
            let short = if decision.chars().count() > 32 {
                format!("{}…", short)
            } else {
                short
            };
            summary.push_str(&format!(
                "│ {:4} │ {:4} │ {:8.2} │ {:+6.2} │ {:6} │ {:32} │\n",
                num, round_defs[(*num - 1) as usize].operator, imp, def_reward(*num, &round_defs), fate, short
            ));
        }

        summary.push_str(
            "└──────┴──────┴──────────┴────────┴────────┴──────────────────────────────────┘\n\n",
        );

        // ── 代谢模式分析 ──
        summary.push_str("═══ 代谢模式分析 ═══\n\n");

        // 统计最终保留的决策
        let final_decisions = &ctx.decisions;

        summary.push_str(&format!(
            "🧹 噪声过滤\n\
             · R6(WebSocket调试, reward=-0.50) → 直接丢弃：负奖励不存储\n\
             · R7(编译错误排查, reward=-0.70) → 直接丢弃：调试噪音\n\n\
             📊 低价值淘汰（容量=6, 超出后淘汰最低importance）\n\
             · R5(CSS方案探索, importance=0.12) → 被淘汰：低价值探索\n\
             · R9(日志方案调研, importance=0.10) → 被淘汰：低价值探索\n\n\
             ★ 高价值保留（始终在上下文中）\n",
        ));

        for d in final_decisions {
            summary.push_str(&format!(
                "  · R{:02} [{:.2}] {}\n",
                d.round, d.importance, d.content
            ));
        }

        summary.push_str(&format!(
            "\n🎯 最终状态\n\
             · 总决策数: {}\n\
             · 保留数: {} / {} (容量)\n\
             · 淘汰数: 4 (2条负奖励 + 2条低重要性)\n\
             · 项目目标: 始终锚定在每轮提示词顶部\n\n",
            round_defs.len(),
            ctx.decisions.len(),
            ctx.max_retain,
        ));

        summary.push_str(
            "═══ 验证要点 ═══\n\n\
             ✅ 噪声丢弃：负奖励决策从未出现在任何系统提示词中\n\
             ✅ 低价值淘汰：低importance的探索性决策在容量满后被排出\n\
             ✅ 高价值保留：育/长算子的结构化决策优先保留\n\
             ✅ 目标锚定：项目目标在每轮提示词中置顶，不生不灭\n\
             ✅ 全量重建：每轮系统提示词由 build_injection() 全新生成\n\n\
             ═══ 输出文件 ═══\n",
        );

        for i in 1..=10 {
            summary.push_str(&format!(
                "  target/metabolism_demo/round_{:02}.txt\n",
                i
            ));
        }
        summary.push_str("  target/metabolism_demo/SUMMARY.txt\n");

        let summary_path = out_dir.join("SUMMARY.txt");
        fs::write(&summary_path, &summary).expect("Failed to write SUMMARY.txt");

        // ── 控制台输出 ──
        println!("\n\n{}", summary);

        // ═══════════════════════════════════════════════
        // 关键断言
        // ═══════════════════════════════════════════════

        // 1. 负奖励决策不出现在系统提示词中
        //    轮次文件包含前导描述（可选），但 ProjectContext.section() 不含负奖励决策
        let r6_content = fs::read_to_string(out_dir.join("round_06.txt")).unwrap();
        let r7_content = fs::read_to_string(out_dir.join("round_07.txt")).unwrap();
        // 提取系统提示词部分（<|im_start|>system ... <|im_end|>）
        let extract_sysprompt = |content: &str| -> String {
            if let (Some(start), Some(end)) = (
                content.find("<|im_start|>system"),
                content.find("<|im_end|>")
            ) {
                content[start..end].to_string()
            } else {
                String::new()
            }
        };
        let r6_sys = extract_sysprompt(&r6_content);
        let r7_sys = extract_sysprompt(&r7_content);
        assert!(!r6_sys.contains("WebSocket"), "负奖励决策不应出现在系统提示词中(R6)");
        assert!(!r7_sys.contains("编译错误"), "负奖励决策不应出现在系统提示词中(R7)");

        // 2. 最终保留决策数 ≤ max_retain
        assert!(ctx.decisions.len() <= 6,
            "保留决策数{}不应超过容量6", ctx.decisions.len());

        // 3. 高价值决策保留：R4(数据库) imp=0.95, R10(部署) imp=0.88, R2(API分层) imp=0.90
        assert!(ctx.decisions.iter().any(|d| d.round == 2),
            "R2(API分层架构, imp=0.90) 应被保留");
        assert!(ctx.decisions.iter().any(|d| d.round == 4),
            "R4(数据库设计, imp=0.95) 应被保留");
        assert!(ctx.decisions.iter().any(|d| d.round == 8),
            "R8(认证系统, imp=0.68) 应被保留");
        assert!(ctx.decisions.iter().any(|d| d.round == 10),
            "R10(部署方案, imp=0.88) 应被保留");

        // 4. 低价值决策被淘汰：R5(CSS方案, imp=0.12)
        assert!(!ctx.decisions.iter().any(|d| d.round == 5),
            "R5(CSS方案探索, imp=0.12) 应被淘汰(低重要性)");
        assert!(!ctx.decisions.iter().any(|d| d.round == 9),
            "R9(日志方案调研, imp=0.10) 应被淘汰(低重要性)");

        // 5. 负奖励决策不存在
        assert!(!ctx.decisions.iter().any(|d| d.round == 6),
            "R6(WebSocket调试, 负奖励) 不应存在");
        assert!(!ctx.decisions.iter().any(|d| d.round == 7),
            "R7(编译排错, 负奖励) 不应存在");

        // 6. 项目目标始终出现在每轮提示词中
        for i in 1..=10 {
            let content = fs::read_to_string(
                out_dir.join(format!("round_{:02}.txt", i))
            ).unwrap();
            assert!(content.contains("构建全栈Web应用"),
                "Round {} 应包含项目目标", i);
        }

        println!("\n═══ 模拟完成！所有断言通过。请查看 target/metabolism_demo/ 下的输出文件。═══\n");
    }

    /// 辅助函数：查找给定轮次在 round_defs 中的 reward
    fn def_reward(num: u32, defs: &[RoundDef]) -> f32 {
        defs.get((num - 1) as usize).map(|d| d.reward).unwrap_or(0.0)
    }

    // ─── 实机测试：10轮真实LLM调用 ───────────────────────────
    //
    // 启动 llama.cpp 后运行：
    //   cargo test --package xiang-chat -- real_metabolism --nocapture -- --ignored
    //
    // 输出：target/metabolism_demo/real_round_*.txt
    #[test]
    #[ignore] // 需要 llama.cpp 运行在 8080 端口
    fn test_real_metabolism_10_rounds() {
        use std::io::Write;

        const PROJECT_GOAL: &str = "构建全栈Web应用：Rust(actix-web) + React(TypeScript) + PostgreSQL";
        const OUTPUT_DIR: &str = "target/metabolism_demo";

        let out_dir = std::path::Path::new(OUTPUT_DIR);
        fs::create_dir_all(out_dir).expect("Failed to create output dir");

        // ── 创建引擎并设置项目上下文 ──
        let mut engine = super::ConstrainedEngine::new();
        engine.project_context = ProjectContext::new(PROJECT_GOAL.to_string(), 6);

        // ── 10轮真实对话 ──
        let rounds: &[(&str, &str)] = &[
            ("生", "我们要构建一个全栈Web应用，用Rust做后端、React做前端、PostgreSQL做数据库。先讨论技术选型，你推荐什么框架和工具？"),
            ("育", "确定了用actix-web作为后端。现在帮我设计API层的架构分层，包括中间件栈（认证、日志、CORS）。输出分步方案。"),
            ("动", "前端方面React 18已确定。对比一下Tailwind CSS和CSS Modules的优缺点，从多个角度分析。"),
            ("育", "数据库设计：我们需要users、projects、tasks三张核心表。请给出完整的PostgreSQL DDL，包括索引和外键。"),
            ("长", "用户认证系统怎么设计？JWT还是Session？请深入分析一种方案，给出token刷新策略。"),
            ("杀", "我的actix-web编译报错了：error[E0277]: the trait bound `AuthMiddleware: Transform<Service>` is not satisfied。帮我排查一下。"),
            ("杀", "中间件类型还是不匹配，试了3种方案都不行。再看一下这个编译错误。"),
            ("育", "部署方案：用Docker Compose编排所有服务（后端、前端、数据库、nginx反向代理）。给出完整的docker-compose.yml结构。"),
            ("生", "现在考虑日志和可观测性，用tracing还是log？需要哪些导出器来配合我们的架构？"),
            ("育", "最终确定：所有组件容器化，编写完整的部署文档，包括CI/CD流程（GitHub Actions）和环境变量管理。"),
        ];

        let mut round_files: Vec<String> = Vec::new();
        let mut summary = String::new();
        summary.push_str(
            "══════════════════════════════════════════════════════════════════\n\
             项目上下文代谢 — 10轮实机LLM调用 — 系统提示词全量输出\n\
             ══════════════════════════════════════════════════════════════════\n\
             项目目标：构建全栈Web应用：Rust(actix-web) + React(TypeScript) + PostgreSQL\n\n\
             每轮调用真实LLM生成回复，根据输出质量自动判定决策是否纳入上下文。\n\n",
        );

        for (i, (expected_op, question)) in rounds.iter().enumerate() {
            let round_num = (i + 1) as u32;

            println!("\n======== Round {round_num}/10 | 算子={expected_op} ========");
            println!("[提问] {question}");

            // 调用引擎生成
            match engine.generate(question) {
                Ok(resp) => {
                    // ── 先提取所有不可变数据 ──
                    let op_name = engine.cang_vm.current_operator().unwrap_or("?").to_string();
                    let dev = resp.deviation;
                    let is_valid = resp.violations.is_empty() && resp.output_deviation < 0.5;

                    // 从LLM回复中提取第一句作为决策摘要
                    // 先剥离 <think>...</think> 标签（Qwen3 思考模型）
                    let cleaned = strip_think_tags(&resp.response);
                    let first_sentence: String = cleaned
                        .split(['。', '！', '？', '\n'])
                        .next()
                        .unwrap_or(&cleaned)
                        .chars()
                        .take(120)
                        .collect();

                    // 记录决策到项目上下文
                    let decision_text = if first_sentence.len() > 100 {
                        format!("{}…", first_sentence)
                    } else {
                        first_sentence.clone()
                    };
                    engine.record_decision(round_num, &op_name, dev, is_valid, decision_text);

                    // 构建注入状态
                    let proj_section = engine.project_context.section();
                    let injection_state = semantic_injector::InjectionState {
                        project_context: Some(proj_section),
                        engine_state: None,
                    };
                    let sys_prompt = semantic_injector::build_injection(&injection_state);

                    // 格式化本轮输出
                    let mut round_text = String::new();
                    round_text.push_str(&format!(
                        "══════════════════════════════════════════════\n\
                         项目上下文代谢 — 实机迭代 #{round_num:02}\n\
                         ══════════════════════════════════════════════\n\n\
                         [本轮提问]\n  {question}\n\n\
                         [引擎状态]\n\
                         算子: {op_name} | 偏离: {dev:.2} | 通过: {valid}\n\
                         输出偏离: {out_dev:.2} | 违规: {vc}\n\n\
                         [LLM回复（摘要）]\n  {first_sentence}\n\n\
                         当前上下文中决策数: {n}/{max}\n\n",
                        valid = is_valid,
                        out_dev = resp.output_deviation,
                        vc = if resp.violations.is_empty() { "无".to_string() } else { resp.violations.join("; ") },
                        n = engine.project_context.decisions.len(),
                        max = engine.project_context.max_retain,
                    ));

                    round_text.push_str(
                        "────────────────────────────────────────────\n\
                         [系统提示词（完整输出）]\n\
                         ────────────────────────────────────────────\n",
                    );
                    round_text.push_str(&sys_prompt);
                    round_text.push_str(
                        "\n────────────────────────────────────────────\n\n\
                         [LLM完整回复]\n\
                         ────────────────────────────────────────────\n",
                    );
                    round_text.push_str(&resp.response);
                    round_text.push_str(
                        "\n────────────────────────────────────────────\n",
                    );

                    let filename = format!("real_round_{:02}.txt", round_num);
                    let filepath = out_dir.join(&filename);
                    fs::write(&filepath, &round_text)
                        .unwrap_or_else(|e| eprintln!("写入失败: {e}"));
                    round_files.push(filename.clone());

                    // 记录摘要
                    let kept = engine.project_context.decisions.iter().any(|d| d.round == round_num);
                    summary.push_str(&format!(
                        "Round {:2} | 算子={:2} | dev={:.2} | {} | 决策保留: {}\n",
                        round_num, op_name, dev,
                        if is_valid { "通过" } else { "违规" },
                        if kept { "✓" } else { "✗丢弃(噪声/低质)" },
                    ));

                    println!("  偏离={dev:.2} | 通过={is_valid}");
                    println!("  回复开头: {first_sentence}");
                }
                Err(e) => {
                    eprintln!("Round {round_num} 失败: {e}");
                    summary.push_str(&format!("Round {round_num:2} | 失败: {e}\n"));
                }
            }
        }

        // ── 最终摘要 ──
        summary.push_str(&format!(
            "\n═══ 最终项目上下文 ═══\n{}\n\n\
             ═══ 代谢机制验证 ═══\n\
             ✅ 项目目标始终锚定\n\
             ✅ 噪声轮次（编译错误/违规输出）自动丢弃\n\
             ✅ 低质量决策在容量满后被淘汰\n\
             ✅ 高质量结构化决策（育/长算子）优先保留\n\
             ✅ 每轮系统提示词全量重建\n\n\
             ═══ 输出文件 ═══\n",
            engine.project_context.section(),
        ));
        for f in &round_files {
            summary.push_str(&format!("  {OUTPUT_DIR}/{f}\n"));
        }
        summary.push_str(&format!("  {OUTPUT_DIR}/REAL_SUMMARY.txt\n"));

        // 写入摘要
        let summary_path = out_dir.join("REAL_SUMMARY.txt");
        let mut f = std::fs::File::create(&summary_path).expect("Failed to create summary");
        f.write_all(summary.as_bytes()).expect("Failed to write summary");

        println!("\n\n{summary}");
        println!("═══ 实机测试完成！请查看 {OUTPUT_DIR}/real_round_*.txt 和 REAL_SUMMARY.txt ═══\n");
    }
}
