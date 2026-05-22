/// xiang-llm — LLM integration layer for the Xiang Constraint Engine.
///
/// ## Logit-Bias Architecture
///
/// The constraint engine operates at the **logit level** — before each token is sampled:
///
/// ```text
/// LLM head -> raw logits -> [XiangLogitBias] -> biased logits -> softmax -> sample token
///                              ^
///                        CangVM state
/// ```
///
/// The constraint engine is NOT a post-hoc validator. It modifies the model's
/// token probabilities in real time, directly influencing what the LLM
/// "chooses" to say — before any token is emitted.

use xiang_core::Gua;

#[cfg(feature = "llama_backend")]
pub mod llama_backend;

#[cfg(feature = "http_backend")]
pub mod http_backend;

// ── Logit Bias System ────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct LogitStep {
    pub tokens_so_far: Vec<u32>,
    pub position: u32,
    pub vocab_size: u32,
}

impl Default for LogitStep {
    fn default() -> Self {
        LogitStep { tokens_so_far: vec![], position: 0, vocab_size: 1000 }
    }
}

#[derive(Debug, Clone)]
pub struct TokenBiasRule {
    pub token_ids: Vec<u32>,
    /// Bias added to logits: negative = suppress, positive = promote.
    pub bias: f32,
}

#[derive(Debug, Clone, Default)]
pub struct BiasDirective {
    pub rules: Vec<TokenBiasRule>,
    pub temperature: Option<f32>,
    pub force_stop: bool,
}

/// Logit bias trait — implemented by the constraint engine.
pub trait LogitBias {
    fn bias_for_step(&mut self, step: &LogitStep) -> BiasDirective;
    fn on_token_sampled(&mut self, _token_id: u32, _token_text: &str) {}
    fn box_clone(&self) -> Box<dyn LogitBias + Send>;
}

// ── Generation Types ─────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TemperatureMode {
    Default,
    Fixed(f32),
    GridControlled(f32),
}

impl TemperatureMode {
    pub fn value(&self) -> f32 {
        match self {
            TemperatureMode::Default => 0.8,
            TemperatureMode::Fixed(t) => t.clamp(0.0, 2.0),
            TemperatureMode::GridControlled(t) => t.clamp(0.0, 2.0),
        }
    }
}

/// Generation parameters. Manual impl to avoid Debug/Clone on trait object.
pub struct GenerationParams {
    pub system_prompt: String,
    pub user_input: String,
    pub history: Vec<(String, String)>,
    pub max_tokens: u32,
    pub temperature: TemperatureMode,
    pub stop_sequences: Vec<String>,
    pub apply_focus_constraint: bool,
    pub vm_state: Gua,
    pub deviation: f32,
    pub logit_bias: Option<Box<dyn LogitBias + Send>>,
}

#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub text: String,
    pub tokens_generated: u32,
    pub truncated: bool,
    pub stop_reason: StopReason,
    pub bias_applications: u32,
    pub tokens_suppressed: u32,
    pub bias_log: Vec<BiasLogEntry>,
    /// Whether this generation was truncated due to focus deviation (杀硬逻辑).
    /// When true, the caller should NOT add this turn to history.
    pub deviated: bool,
}

#[derive(Debug, Clone)]
pub struct BiasLogEntry {
    pub step: u32,
    pub deviation: f32,
    pub bias_rules: Vec<String>,
    pub token_sampled: u32,
    pub suppressed: bool,
    /// Semantic deviation at this step (None = pure Hamming).
    pub semantic_deviation: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    Eos, MaxTokens, StopSeq, Deviated, Error,
}

pub trait LlmBackend {
    fn generate(&mut self, params: &mut GenerationParams) -> Result<GenerationResult, LlmError>;
    fn tokenize(&self, text: &str) -> Vec<u32>;
    fn model_name(&self) -> &str;
    fn is_ready(&self) -> bool;
    fn device_info(&self) -> &str;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LlmError {
    NotReady(String), ModelLoadFailed(String), GenerationFailed(String),
    FocusDeviated(String), ContextOverflow(String),
}

impl std::fmt::Display for LlmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LlmError::NotReady(s) => write!(f, "LLM未就绪: {s}"),
            LlmError::ModelLoadFailed(s) => write!(f, "模型加载失败: {s}"),
            LlmError::GenerationFailed(s) => write!(f, "生成失败: {s}"),
            LlmError::FocusDeviated(s) => write!(f, "焦点偏离触发: {s}"),
            LlmError::ContextOverflow(s) => write!(f, "上下文溢出: {s}"),
        }
    }
}
impl std::error::Error for LlmError {}

// ── XiangLogitBias ──────────────────────────────────────────

/// Maps CangVM state (Gua, deviation, sha) to per-token logit bias.
///
/// Uses **continuous intensity scaling** — the bias strength varies smoothly
/// with deviation rather than jumping at thresholds. This is critical for the
/// HTTP backend which calls `bias_for_step()` once per generation: the static
/// bias map must include ALL rule categories with intensity proportional to
/// deviation.
///
/// Rules (all always applied, intensity varies with deviation):
///   - Suppress off-focus tokens:  -0.5 at D=0  →  -8.0 at D=1
///   - Suppress divergent tokens:  -0.5 at D=0  →  -5.0 at D=1
///   - Promote EOS:                 0.0 at D=0  →  +4.0 at D=1
///   - Sha amplification:          -2.0 extra when sha_count > 0
///   - Force stop:                  D > 0.95 (prevents generation entirely)
pub struct XiangLogitBias {
    pub deviation: f32,
    pub state: Gua,
    pub sha_count: u32,
    pub off_focus_token_group: Vec<u32>,
    pub divergent_token_group: Vec<u32>,
    pub eos_token_id: u32,
    pub bias_log: Vec<BiasLogEntry>,
    /// Hamming-only deviation component (optional, for diagnostic purposes).
    pub hamming_deviation: Option<f32>,
    /// Semantic deviation component (cosine-based, optional).
    pub semantic_deviation: Option<f32>,
    /// Strategy vector from 连山 (reserved for future attention biasing).
    pub v_strategy: Option<Vec<f32>>,
    /// Attitude vector from 周易 (reserved for future attention biasing).
    pub v_attitude: Option<Vec<f32>>,
}

/// Linear interpolation helper: returns `a + (b - a) * t` clamped to [a, b].
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

impl XiangLogitBias {
    pub fn new(
        deviation: f32, state: Gua, sha_count: u32,
        off_focus_tokens: Vec<u32>, divergent_tokens: Vec<u32>, eos_token_id: u32,
    ) -> Self {
        XiangLogitBias {
            deviation, state, sha_count,
            off_focus_token_group: off_focus_tokens,
            divergent_token_group: divergent_tokens,
            eos_token_id,
            bias_log: Vec::new(),
            hamming_deviation: None,
            semantic_deviation: None,
            v_strategy: None,
            v_attitude: None,
        }
    }
}

impl LogitBias for XiangLogitBias {
    fn bias_for_step(&mut self, _step: &LogitStep) -> BiasDirective {
        let mut rules = Vec::new();

        // Force stop at extreme deviation (prevents generation entirely)
        if self.deviation > 0.95 {
            return BiasDirective {
                rules: vec![],
                temperature: None,
                force_stop: true,
            };
        }

        // Continuous intensity: bias strength scales smoothly with deviation.
        // - At dev=0.0: very mild suppression
        // - At dev=1.0: very strong suppression
        // - sha_count > 0: additional bias on off_focus (pruning amplification)
        let off_focus_bias = -lerp(0.5, 8.0, self.deviation);
        let divergent_bias = -lerp(0.5, 5.0, self.deviation);
        let eos_bias = lerp(0.0, 4.0, self.deviation);
        let sha_bonus = if self.sha_count > 0 { 2.0 } else { 0.0 };

        // Always suppress off-focus tokens (if any are configured)
        if !self.off_focus_token_group.is_empty() {
            rules.push(TokenBiasRule {
                token_ids: self.off_focus_token_group.clone(),
                bias: off_focus_bias - sha_bonus,
            });
        }

        // Always suppress divergent tokens (if any are configured)
        if !self.divergent_token_group.is_empty() {
            rules.push(TokenBiasRule {
                token_ids: self.divergent_token_group.clone(),
                bias: divergent_bias,
            });
        }

        // Always promote EOS — stronger when deviation is high
        rules.push(TokenBiasRule {
            token_ids: vec![self.eos_token_id],
            bias: eos_bias,
        });

        BiasDirective {
            rules,
            temperature: None,
            force_stop: false,
        }
    }
    fn box_clone(&self) -> Box<dyn LogitBias + Send> {
        Box::new(XiangLogitBias {
            deviation: self.deviation, state: self.state, sha_count: self.sha_count,
            off_focus_token_group: self.off_focus_token_group.clone(),
            divergent_token_group: self.divergent_token_group.clone(),
            eos_token_id: self.eos_token_id, bias_log: self.bias_log.clone(),
            hamming_deviation: self.hamming_deviation,
            semantic_deviation: self.semantic_deviation,
            v_strategy: self.v_strategy.clone(),
            v_attitude: self.v_attitude.clone(),
        })
    }
}

// ── Mock Backend ─────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MockVocab {
    pub safe_tokens: Vec<u32>,
    pub off_focus_tokens: Vec<u32>,
    pub divergent_tokens: Vec<u32>,
    pub eos_id: u32,
    pub vocab_size: u32,
}

impl MockVocab {
    /// 0..100 focused, 100..200 off-focus, 200..300 divergent, 999 EOS.
    pub fn standard() -> Self {
        MockVocab {
            safe_tokens: (0..100).collect(),
            off_focus_tokens: (100..200).collect(),
            divergent_tokens: (200..300).collect(),
            eos_id: 999, vocab_size: 1000,
        }
    }
    pub fn decode(&self, id: u32) -> &str {
        if self.safe_tokens.contains(&id) { "聚焦" }
        else if self.off_focus_tokens.contains(&id) { "脱焦" }
        else if self.divergent_tokens.contains(&id) { "发散" }
        else if id == self.eos_id { "</s>" } else { "中性" }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MockMode { Safe, Verbose, OffFocus, Hallucinating, Fixed }

pub struct MockBackend {
    pub name: String, pub device: String, pub ready: bool,
    pub mode: MockMode, pub vocab: MockVocab, call_count: u32,
}

impl MockBackend {
    pub fn new(mode: MockMode) -> Self {
        MockBackend {
            name: "mock-qwen-4b".into(), device: "CPU/Mock".into(),
            ready: true, mode, vocab: MockVocab::standard(), call_count: 0,
        }
    }
    fn sample_natural(&mut self) -> u32 {
        self.call_count += 1;
        let n = self.call_count;
        match self.mode {
            MockMode::Safe => self.vocab.safe_tokens[n as usize % self.vocab.safe_tokens.len()],
            MockMode::Verbose => if n % 5u32 == 0 { self.vocab.divergent_tokens[n as usize % self.vocab.divergent_tokens.len()] }
                else { self.vocab.safe_tokens[n as usize % self.vocab.safe_tokens.len()] },
            MockMode::OffFocus => if n % 10u32 <= 2 { self.vocab.off_focus_tokens[n as usize % self.vocab.off_focus_tokens.len()] }
                else { self.vocab.safe_tokens[n as usize % self.vocab.safe_tokens.len()] },
            MockMode::Hallucinating => if n % 3u32 == 0 { self.vocab.divergent_tokens[n as usize % self.vocab.divergent_tokens.len()] }
                else { self.vocab.safe_tokens[n as usize % self.vocab.safe_tokens.len()] },
            MockMode::Fixed => self.vocab.safe_tokens[0],
        }
    }
    fn apply_bias(&self, natural: u32, d: &BiasDirective) -> (u32, bool) {
        if d.force_stop { return (self.vocab.eos_id, true); }
        for r in &d.rules {
            if r.bias < 0.0 && r.token_ids.contains(&natural) {
                return (self.vocab.safe_tokens[natural as usize % self.vocab.safe_tokens.len()], true);
            }
        }
        (natural, false)
    }
    fn decode_tokens(&self, tokens: &[u32]) -> String {
        let mut t = String::new();
        for &id in tokens { if id == self.vocab.eos_id { break; } t.push_str(self.vocab.decode(id)); }
        t
    }
}

impl LlmBackend for MockBackend {
    fn generate(&mut self, params: &mut GenerationParams) -> Result<GenerationResult, LlmError> {
        let mut tokens: Vec<u32> = Vec::new();
        let mut blog: Vec<BiasLogEntry> = Vec::new();
        let mut sup = 0u32;
        let mut deviated = false;

        for i in 0..params.max_tokens {
            let step = LogitStep { tokens_so_far: tokens.clone(), position: i, vocab_size: self.vocab.vocab_size };
            let directive = if let Some(ref mut bias) = params.logit_bias { bias.bias_for_step(&step) }
                else { BiasDirective::default() };

            if directive.force_stop {
                tokens.push(self.vocab.eos_id);
                blog.push(BiasLogEntry { step: i, deviation: params.deviation, bias_rules: vec!["force_stop".into()], token_sampled: self.vocab.eos_id, suppressed: true, semantic_deviation: None });
                deviated = true;
                break;
            }

            let nat = self.sample_natural();
            let (sampled, suppressed) = self.apply_bias(nat, &directive);
            if suppressed { sup += 1; }

            if let Some(ref mut bias) = params.logit_bias {
                bias.on_token_sampled(sampled, self.vocab.decode(sampled));
            }

            let desc: Vec<String> = directive.rules.iter()
                .map(|r| format!("{}{}:{}", if r.bias > 0.0 { "+" } else { "" }, r.bias, r.token_ids.len()))
                .collect();
            blog.push(BiasLogEntry { step: i, deviation: params.deviation, bias_rules: desc, token_sampled: sampled, suppressed: suppressed && sampled != nat, semantic_deviation: None });

            tokens.push(sampled);
            if sampled == self.vocab.eos_id {
                return Ok(GenerationResult {
                    text: self.decode_tokens(&tokens), tokens_generated: i+1, truncated: false,
                    stop_reason: StopReason::Eos, bias_applications: i+1, tokens_suppressed: sup, bias_log: blog,
                    deviated: false,
                });
            }
        }
        let stop_reason = if deviated { StopReason::Deviated } else { StopReason::MaxTokens };
        Ok(GenerationResult {
            text: self.decode_tokens(&tokens), tokens_generated: params.max_tokens, truncated: !deviated,
            stop_reason, bias_applications: params.max_tokens, tokens_suppressed: sup, bias_log: blog,
            deviated,
        })
    }
    fn tokenize(&self, _t: &str) -> Vec<u32> { vec![self.vocab.safe_tokens[0]] }
    fn model_name(&self) -> &str { &self.name }
    fn is_ready(&self) -> bool { self.ready }
    fn device_info(&self) -> &str { &self.device }
}

// ── LLM Context ──────────────────────────────────────────────

pub struct LlmContext {
    pub system_prompt: String,
    pub history: Vec<(String, String)>,
    pub max_history: usize,
    pub total_tokens: u32, pub total_calls: u32,
    pub focus_interventions: u32, pub total_suppressions: u32,
    /// When a turn is deviated, this stores a guidance string injected into
    /// the next prompt to steer the model back toward the core topic.
    /// This is a prompt-level intervention (not logit-level).
    pub origin_guidance: Option<String>,
    /// Turns that were deviated and excluded from history (user_input, model_output).
    pub skipped_turns: Vec<(String, String)>,
}

impl LlmContext {
    pub fn new(s: &str) -> Self {
        LlmContext {
            system_prompt: s.into(), history: Vec::new(), max_history: 50,
            total_tokens: 0, total_calls: 0, focus_interventions: 0, total_suppressions: 0,
            origin_guidance: None, skipped_turns: Vec::new(),
        }
    }
    pub fn add_turn(&mut self, u: &str, a: &str) {
        self.history.push((u.into(), a.into()));
        while self.history.len() > self.max_history { self.history.remove(0); }
    }

    pub fn generate_constrained_turn<B: LlmBackend>(
        &mut self, backend: &mut B,
        user_input: &str, max_tokens: u32, temperature: TemperatureMode,
        vm_state: Gua, deviation: f32, sha_count: u32,
        off_focus_ids: Vec<u32>, divergent_ids: Vec<u32>, eos_id: u32,
    ) -> Result<GenerationResult, LlmError> {
        // ── 杀硬逻辑：注入 origin_guidance 引导偏离恢复 ──
        // 若上一轮偏离被物理丢弃，在本次 prompt 前置引导语
        let guided_input = if let Some(ref _guidance) = self.origin_guidance {
            format!(
                "[元指令] 请将焦点回归核心问题。上一次思考方向偏离既定目标。\n\n{}",
                user_input
            )
        } else {
            user_input.to_string()
        };

        let bias = XiangLogitBias::new(deviation, vm_state, sha_count, off_focus_ids, divergent_ids, eos_id);
        let mut params = GenerationParams {
            system_prompt: self.system_prompt.clone(), user_input: guided_input,
            history: self.history.clone(), max_tokens, temperature,
            stop_sequences: vec!["</s>".into()], apply_focus_constraint: deviation > 0.5,
            vm_state, deviation, logit_bias: Some(Box::new(bias)),
        };
        let r = backend.generate(&mut params)?;
        self.total_tokens += r.tokens_generated; self.total_calls += 1;
        self.total_suppressions += r.tokens_suppressed;
        if r.stop_reason == StopReason::Deviated || r.deviated {
            self.focus_interventions += 1;
            // 偏离恢复：排除该轮输出（不入history），设置下一轮引导语
            self.origin_guidance = Some("请将焦点回归核心问题。上一次思考方向偏离既定目标。".into());
            self.skipped_turns.push((user_input.into(), r.text.clone()));
        } else {
            // 生成成功，清除偏离引导
            self.origin_guidance = None;
            // 历史记录使用原始用户输入（不含元指令前缀）
            self.add_turn(user_input, &r.text);
        }
        Ok(r)
    }

    pub fn generate_unconstrained_turn<B: LlmBackend>(
        &mut self, backend: &mut B,
        user_input: &str, max_tokens: u32, temperature: TemperatureMode,
    ) -> Result<GenerationResult, LlmError> {
        let mut params = GenerationParams {
            system_prompt: self.system_prompt.clone(), user_input: user_input.into(),
            history: self.history.clone(), max_tokens, temperature,
            stop_sequences: vec!["</s>".into()], apply_focus_constraint: false,
            vm_state: Gua::ZERO, deviation: 0.0, logit_bias: None,
        };
        let r = backend.generate(&mut params)?;
        self.total_tokens += r.tokens_generated; self.total_calls += 1;
        self.add_turn(user_input, &r.text);
        Ok(r)
    }
}

impl Default for LlmContext {
    fn default() -> Self { LlmContext::new("你是一个遵循约束的AI助手。") }
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn test_low_deviation_has_rules() {
        let mut b = XiangLogitBias::new(0.1, Gua::ORIGIN, 0, vec![100], vec![200], 999);
        let d = b.bias_for_step(&LogitStep::default());
        // Continuous model always produces rules (EOS promotion at minimum)
        assert!(!d.force_stop, "dev=0.1 should not force stop");
        assert!(d.rules.iter().any(|r| r.token_ids.contains(&999)),
            "EOS promotion should always be present");
    }
    #[test] fn test_high_deviation_strong_suppression() {
        let mut b = XiangLogitBias::new(0.8, Gua::ZERO, 0, vec![100], vec![200], 999);
        let d = b.bias_for_step(&LogitStep::default());
        // Should have strong negative bias on off_focus, strong positive on EOS
        assert!(d.rules.iter().any(|r| r.token_ids.contains(&100) && r.bias < -2.0));
        assert!(d.rules.iter().any(|r| r.token_ids.contains(&999) && r.bias > 2.0));
    }
    #[test] fn test_extreme_deviation_force_stop() {
        let mut b = XiangLogitBias::new(0.96, Gua::ZERO, 0, vec![100], vec![200], 999);
        assert!(b.bias_for_step(&LogitStep::default()).force_stop);
    }
    #[test] fn test_sha_amplifies() {
        let mut b = XiangLogitBias::new(0.5, Gua::ZERO, 5, vec![100], vec![200], 999);
        let d = b.bias_for_step(&LogitStep::default());
        // Sha amplification adds -2.0 to off_focus bias
        let off_focus_rule = d.rules.iter().find(|r| r.token_ids.contains(&100));
        assert!(off_focus_rule.is_some(), "should have off_focus rule");
        // At dev=0.5: off_focus_base = -lerp(0.5, 8.0, 0.5) = -(0.5 + 7.5*0.5) = -4.25
        // With sha_count=5: -4.25 - 2.0 = -6.25
        let expected = -(0.5 + 7.5 * 0.5) - 2.0;
        let actual = off_focus_rule.unwrap().bias;
        assert!((actual - expected).abs() < 0.01,
            "off_focus bias with sha=5 should be {:.2}, got {}", expected, actual);
    }
    #[test] fn test_mock_no_bias() {
        let mut b = MockBackend::new(MockMode::Safe);
        let mut p = GenerationParams {
            user_input: "hi".into(), max_tokens: 10, apply_focus_constraint: false, logit_bias: None,
            system_prompt: String::new(), history: Vec::new(), temperature: TemperatureMode::Default,
            stop_sequences: vec!["</s>".into()], vm_state: Gua::ZERO, deviation: 0.0,
        };
        assert!(b.generate(&mut p).unwrap().tokens_generated > 0);
    }
    #[test] fn test_mock_with_bias() {
        let mut b = MockBackend::new(MockMode::OffFocus);
        let bias = XiangLogitBias::new(0.8, Gua::ZERO, 3,
            b.vocab.off_focus_tokens.clone(), b.vocab.divergent_tokens.clone(), b.vocab.eos_id);
        let mut p = GenerationParams {
            user_input: "test".into(), max_tokens: 20, deviation: 0.8,
            logit_bias: Some(Box::new(bias)),
            system_prompt: String::new(), history: Vec::new(), temperature: TemperatureMode::Default,
            stop_sequences: vec!["</s>".into()], apply_focus_constraint: true, vm_state: Gua::ZERO,
        };
        let r = b.generate(&mut p).unwrap();
        assert!(r.bias_applications > 0 && r.tokens_suppressed > 0);
    }
    #[test] fn test_drift_suppressed() {
        let mut b1 = MockBackend::new(MockMode::OffFocus);
        let mut p1 = GenerationParams {
            user_input: "test".into(), max_tokens: 30, apply_focus_constraint: false, logit_bias: None,
            system_prompt: String::new(), history: Vec::new(), temperature: TemperatureMode::Default,
            stop_sequences: vec!["</s>".into()], vm_state: Gua::ZERO, deviation: 0.0,
        };
        assert!(b1.generate(&mut p1).unwrap().text.contains("脱焦"));

        let mut b2 = MockBackend::new(MockMode::OffFocus);
        let bias = XiangLogitBias::new(0.8, Gua::ZERO, 5,
            b2.vocab.off_focus_tokens.clone(), b2.vocab.divergent_tokens.clone(), b2.vocab.eos_id);
        let mut p2 = GenerationParams {
            user_input: "test".into(), max_tokens: 30, deviation: 0.8,
            logit_bias: Some(Box::new(bias)),
            system_prompt: String::new(), history: Vec::new(), temperature: TemperatureMode::Default,
            stop_sequences: vec!["</s>".into()], apply_focus_constraint: true, vm_state: Gua::ZERO,
        };
        assert!(b2.generate(&mut p2).unwrap().tokens_suppressed > 0);
    }
    #[test] fn test_constrained_turn() {
        let mut ctx = LlmContext::new("SYS");
        let mut b = MockBackend::new(MockMode::OffFocus);
        let (off_focus_tok, divergent_tok, eos) = (
            b.vocab.off_focus_tokens.clone(),
            b.vocab.divergent_tokens.clone(),
            b.vocab.eos_id,
        );
        let r = ctx.generate_constrained_turn(
            &mut b, "测试", 20, TemperatureMode::Default,
            Gua::ZERO, 0.8, 3,
            off_focus_tok, divergent_tok, eos,
        ).unwrap();
        assert_eq!(ctx.total_calls, 1);
        assert!(r.tokens_suppressed > 0);
    }
    #[test] fn test_unconstrained_turn() {
        let mut ctx = LlmContext::new("SYS");
        let mut b = MockBackend::new(MockMode::Safe);
        let _ = ctx.generate_unconstrained_turn(&mut b, "测试", 10, TemperatureMode::Default).unwrap();
        assert_eq!(ctx.total_calls, 1);
    }
    #[test] fn test_vocab_decode() {
        let v = MockVocab::standard();
        assert_eq!(v.decode(0), "聚焦"); assert_eq!(v.decode(100), "脱焦");
        assert_eq!(v.decode(200), "发散"); assert_eq!(v.decode(999), "</s>");
    }
    #[test] fn test_box_clone() {
        let b = XiangLogitBias::new(0.5, Gua::ORIGIN, 1, vec![100], vec![200], 999);
        let mut c = b.box_clone();
        let d = c.bias_for_step(&LogitStep::default());
        assert!(!d.rules.is_empty());
    }
    #[test] fn test_bias_log() {
        let mut b = MockBackend::new(MockMode::OffFocus);
        let bias = XiangLogitBias::new(0.8, Gua::ZERO, 3,
            b.vocab.off_focus_tokens.clone(), b.vocab.divergent_tokens.clone(), b.vocab.eos_id);
        let mut p = GenerationParams {
            user_input: "test".into(), max_tokens: 10, deviation: 0.8,
            logit_bias: Some(Box::new(bias)),
            system_prompt: String::new(), history: Vec::new(), temperature: TemperatureMode::Default,
            stop_sequences: vec!["</s>".into()], apply_focus_constraint: true, vm_state: Gua::ZERO,
        };
        let r = b.generate(&mut p).unwrap();
        assert!(!r.bias_log.is_empty());
        assert!(r.bias_log.iter().any(|e| !e.bias_rules.is_empty() || e.suppressed));
    }
    #[test] fn test_llm_context_stats() {
        let mut ctx = LlmContext::new("SYS");
        let mut b = MockBackend::new(MockMode::OffFocus);
        let (off_focus_tok, deviating_tok, eos) = (
            b.vocab.off_focus_tokens.clone(),
            b.vocab.divergent_tokens.clone(),
            b.vocab.eos_id,
        );
        ctx.generate_constrained_turn(
            &mut b, "a", 10, TemperatureMode::Default,
            Gua::ZERO, 0.8, 0,
            off_focus_tok, deviating_tok, eos,
        ).unwrap();
        ctx.generate_unconstrained_turn(&mut b, "b", 10, TemperatureMode::Default).unwrap();
        assert_eq!(ctx.total_calls, 2);
        assert_eq!(ctx.history.len(), 2);
    }
    #[test] fn test_llm_error_display() {
        assert!(LlmError::NotReady("x".into()).to_string().contains("LLM未就绪"));
        assert!(LlmError::FocusDeviated("x".into()).to_string().contains("焦点偏离触发"));
    }
    #[test] fn test_temperature_values() {
        assert!((TemperatureMode::Default.value() - 0.8).abs() < 1e-6);
        assert!((TemperatureMode::Fixed(1.5).value() - 1.5).abs() < 1e-6);
        assert!((TemperatureMode::Fixed(5.0).value() - 2.0).abs() < 1e-6);
    }
    #[test] fn test_history_trim() {
        let mut ctx = LlmContext::new("t");
        ctx.max_history = 3;
        for i in 0..5 { ctx.add_turn(&format!("Q{i}"), &format!("A{i}")); }
        assert_eq!(ctx.history.len(), 3);
        assert_eq!(ctx.history[0].0, "Q2");
    }
}
