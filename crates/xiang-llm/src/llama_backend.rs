/// Real llama.cpp backend using FFI.
///
/// Links against `libllama.so` built from the llama.cpp repository.
/// Supports CPU inference with automatic Vulkan GPU offload if available.
///
/// ## Logit Bias Integration
///
/// After each `llama_decode`, we retrieve the raw logits via `llama_get_logits_ith`,
/// apply the XiangLogitBias rules in-place on the logits array, then sample.
/// This gives us dynamic per-token constraint without requiring custom C samplers.
///
/// ## GPU Setup (AMD RX 6650 XT)
///
/// For GPU acceleration, rebuild llama.cpp with `-DGGML_VULKAN=ON`.
/// The Vulkan backend automatically detects AMD GPUs via Mesa RADV driver (gfx1032).

use std::ffi::CString;
use std::path::Path;
use std::collections::HashSet;

use crate::{
    LlmBackend, LlmError, GenerationParams, GenerationResult,
    StopReason, BiasDirective, BiasLogEntry,
};

// ── Vocab pattern constants ─────────────────────────────────

/// English text patterns for off-focus token discovery.
const ENGLISH_PATTERNS: &[&str] = &[
    "a","b","c","d","e","f","g","h","i","j","k","l","m",
    "n","o","p","q","r","s","t","u","v","w","x","y","z",
    "A","B","C","D","E","F","G","H","I","J","K","L","M",
    "N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
    "the","and","for","are","but","not","you","all","can",
    "have","with","this","that","from","they","been","were",
    "when","what","which","their","about","would","could",
    "should","there","other","into","than","then","them",
    "these","some","more","also","very","just","over",
    "such","each","well","here","where","after","before",
    "between","through","during","without","because","under",
    "might","shall","will","must","still","already","even",
    "first","second","third","last","next","much","many",
    "Hello","World","This","That","What","How","Why",
    "English","response","answer","question","please",
    "sorry","thank","yes","no","maybe","help",
    "I","you","he","she","we","they","me","him","her",
    "us","them","my","your","his","its","our","their",
];

/// Chinese transition phrases for divergent token discovery.
const TRANSITION_PATTERNS: &[&str] = &[
    "但是","然而","不过","可是","虽然","尽管","即使",
    "另一方面","相比之下","反之",
    "此外","另外","还有","再者",
    "总的来说","综上所述","总之",
    "首先","其次","最后",
    "因此","所以","从而","进而",
    "换句话说","也就是说",
];

// ── FFI types ────────────────────────────────────────────────

#[repr(C)]
struct llama_batch {
    n_tokens: i32,
    token: *mut i32,
    embd: *mut f32,
    pos: *mut i32,
    n_seq_id: *mut i32,
    seq_id: *mut *mut i32,
    logits: *mut i8,
}

#[repr(C)]
struct llama_model_params {
    devices: *mut *mut std::ffi::c_void,
    tensor_buft_overrides: *const std::ffi::c_void,
    n_gpu_layers: i32,
    split_mode: i32,
    main_gpu: i32,
    tensor_split: *const f32,
    progress_callback: Option<unsafe extern "C" fn(f32, *mut std::ffi::c_void) -> bool>,
    progress_callback_user_data: *mut std::ffi::c_void,
    kv_overrides: *const std::ffi::c_void,
    vocab_only: bool,
    use_mmap: bool,
    use_direct_io: bool,
    use_mlock: bool,
    check_tensors: bool,
    use_extra_bufts: bool,
    no_host: bool,
    no_alloc: bool,
}

#[repr(C)]
struct llama_context_params {
    n_ctx: u32,
    n_batch: u32,
    n_ubatch: u32,
    n_seq_max: u32,
    n_rs_seq: u32,
    n_threads: i32,
    n_threads_batch: i32,
    ctx_type: i32,
    rope_scaling_type: i32,
    pooling_type: i32,
    attention_type: i32,
    flash_attn_type: i32,
    rope_freq_base: f32,
    rope_freq_scale: f32,
    yarn_ext_factor: f32,
    yarn_attn_factor: f32,
    yarn_beta_fast: f32,
    yarn_beta_slow: f32,
    yarn_orig_ctx: u32,
    defrag_thold: f32,
    cb_eval: Option<unsafe extern "C" fn(*mut std::ffi::c_void) -> bool>,
    cb_eval_user_data: *mut std::ffi::c_void,
    type_k: i32,
    type_v: i32,
    abort_callback: Option<unsafe extern "C" fn(*mut std::ffi::c_void) -> bool>,
    abort_callback_data: *mut std::ffi::c_void,
    embeddings: bool,
    offload_kqv: bool,
    no_perf: bool,
    op_offload: bool,
    swa_full: bool,
    kv_unified: bool,
    samplers: *const std::ffi::c_void,
    n_samplers: usize,
}

#[repr(C)]
struct llama_sampler_chain_params {
    no_perf: bool,
}

#[repr(C)]
#[allow(dead_code)]
struct llama_logit_bias {
    token: i32,
    bias: f32,
}

// ── FFI function declarations ───────────────────────────────

#[link(name = "llama")]
#[allow(dead_code)]
unsafe extern "C" {
    fn llama_backend_init();
    fn llama_backend_free();
    fn llama_model_default_params() -> llama_model_params;
    fn llama_context_default_params() -> llama_context_params;
    fn llama_sampler_chain_default_params() -> llama_sampler_chain_params;
    fn llama_model_load_from_file(
        path: *const std::os::raw::c_char,
        params: llama_model_params,
    ) -> *mut std::ffi::c_void;
    fn llama_model_free(model: *mut std::ffi::c_void);
    fn llama_init_from_model(
        model: *mut std::ffi::c_void,
        params: llama_context_params,
    ) -> *mut std::ffi::c_void;
    fn llama_free(ctx: *mut std::ffi::c_void);
    fn llama_synchronize(ctx: *mut std::ffi::c_void);
    fn llama_n_ctx(ctx: *const std::ffi::c_void) -> u32;
    fn llama_get_model(ctx: *const std::ffi::c_void) -> *mut std::ffi::c_void;
    fn llama_model_get_vocab(model: *const std::ffi::c_void) -> *const std::ffi::c_void;
    fn llama_vocab_n_tokens(vocab: *const std::ffi::c_void) -> i32;
    fn llama_vocab_bos(vocab: *const std::ffi::c_void) -> i32;
    fn llama_vocab_eos(vocab: *const std::ffi::c_void) -> i32;
    fn llama_vocab_get_add_bos(vocab: *const std::ffi::c_void) -> bool;
    fn llama_vocab_get_text(vocab: *const std::ffi::c_void, token: i32) -> *const std::os::raw::c_char;
    fn llama_vocab_is_eog(vocab: *const std::ffi::c_void, token: i32) -> bool;
    fn llama_tokenize(
        vocab: *const std::ffi::c_void,
        text: *const std::os::raw::c_char,
        text_len: i32,
        tokens: *mut i32,
        n_tokens_max: i32,
        add_special: bool,
        parse_special: bool,
    ) -> i32;
    fn llama_token_to_piece(
        vocab: *const std::ffi::c_void,
        token: i32,
        buf: *mut std::os::raw::c_char,
        length: i32,
        lstrip: i32,
        special: bool,
    ) -> i32;
    fn llama_batch_get_one(
        tokens: *mut i32,
        n_tokens: i32,
    ) -> llama_batch;
    fn llama_decode(
        ctx: *mut std::ffi::c_void,
        batch: llama_batch,
    ) -> i32;
    fn llama_get_logits_ith(
        ctx: *mut std::ffi::c_void,
        i: i32,
    ) -> *mut f32;
    fn llama_sampler_chain_init(params: llama_sampler_chain_params) -> *mut std::ffi::c_void;
    fn llama_sampler_chain_add(chain: *mut std::ffi::c_void, smpl: *mut std::ffi::c_void);
    fn llama_sampler_init_greedy() -> *mut std::ffi::c_void;
    fn llama_sampler_init_dist(seed: u32) -> *mut std::ffi::c_void;
    fn llama_sampler_init_top_k(k: i32) -> *mut std::ffi::c_void;
    fn llama_sampler_init_top_p(p: f32, min_keep: usize) -> *mut std::ffi::c_void;
    fn llama_sampler_init_temp(t: f32) -> *mut std::ffi::c_void;
    fn llama_sampler_sample(
        smpl: *mut std::ffi::c_void,
        ctx: *mut std::ffi::c_void,
        idx: i32,
    ) -> i32;
    fn llama_sampler_free(smpl: *mut std::ffi::c_void);
    fn llama_supports_gpu_offload() -> bool;
    fn llama_print_system_info() -> *const std::os::raw::c_char;
    fn llama_model_desc(
        model: *const std::ffi::c_void,
        buf: *mut std::os::raw::c_char,
        buf_size: i32,
    ) -> i32;
    // ── Memory API (KV cache management, replaces deprecated llama_kv_cache_*) ──
    fn llama_get_memory(ctx: *const std::ffi::c_void) -> *mut std::ffi::c_void;
    fn llama_memory_clear(mem: *mut std::ffi::c_void, data: bool);
    fn llama_memory_seq_rm(mem: *mut std::ffi::c_void, seq_id: i32, p0: i32, p1: i32) -> bool;
    fn llama_memory_seq_keep(mem: *mut std::ffi::c_void, seq_id: i32);
    // ── Vocab API ──
    fn llama_vocab_get_type(vocab: *const std::ffi::c_void) -> i32;
}

// ── LlamaCppBackend ─────────────────────────────────────────

pub struct LlamaCppBackend {
    model: *mut std::ffi::c_void,
    ctx: *mut std::ffi::c_void,
    memory: *mut std::ffi::c_void,
    vocab: *const std::ffi::c_void,
    sampler: *mut std::ffi::c_void,
    model_path: String,
    n_ctx: u32,
    n_vocab: i32,
    #[allow(dead_code)]
    bos_id: i32,
    eos_id: i32,
    gpu_offload: bool,
}

// Safe to send across threads (raw pointers protected by access patterns)
unsafe impl Send for LlamaCppBackend {}

impl LlamaCppBackend {
    /// Load a GGUF model and initialize the llama.cpp backend.
    ///
    /// `n_ctx`: context size (e.g. 32768 for 32K context)
    /// `n_gpu_layers`: number of layers to offload to GPU (-1 = all, 0 = CPU only)
    pub fn new(
        model_path: &str,
        n_ctx: u32,
        n_gpu_layers: i32,
    ) -> Result<Self, LlmError> {
        if !Path::new(model_path).exists() {
            return Err(LlmError::ModelLoadFailed(format!(
                "模型文件不存在: {model_path}"
            )));
        }

        // Initialize backend
        unsafe { llama_backend_init() };

        // Load model
        let mut model_params = unsafe { llama_model_default_params() };
        model_params.n_gpu_layers = n_gpu_layers;

        let c_path = CString::new(model_path).map_err(|e| {
            LlmError::ModelLoadFailed(format!("路径包含NUL字节: {e}"))
        })?;

        let model = unsafe { llama_model_load_from_file(c_path.as_ptr(), model_params) };
        if model.is_null() {
            unsafe { llama_backend_free() };
            return Err(LlmError::ModelLoadFailed(
                "模型加载失败：请确认 GGUF 文件格式正确".into()
            ));
        }

        // Create context
        let mut ctx_params = unsafe { llama_context_default_params() };
        ctx_params.n_ctx = n_ctx;
        ctx_params.n_batch = 512;
        ctx_params.n_ubatch = 512;
        ctx_params.n_threads = num_cpus::get() as i32;
        ctx_params.n_threads_batch = num_cpus::get() as i32;

        let ctx = unsafe { llama_init_from_model(model, ctx_params) };
        if ctx.is_null() {
            unsafe { llama_model_free(model) };
            unsafe { llama_backend_free() };
            return Err(LlmError::ModelLoadFailed(
                "上下文创建失败：请检查显存是否充足".into()
            ));
        }

        // Get vocab
        let vocab = unsafe { llama_model_get_vocab(model) };
        let n_vocab = unsafe { llama_vocab_n_tokens(vocab) };
        let bos_id = unsafe { llama_vocab_bos(vocab) };
        let eos_id = unsafe { llama_vocab_eos(vocab) };

        // Get memory handle for KV cache management (上下文新陈代谢)
        let memory = unsafe { llama_get_memory(ctx) };

        // Create initial sampler chain (without temp — added per-call in generate())
        let sampler = Self::build_base_sampler();

        let gpu_offload = unsafe { llama_supports_gpu_offload() };
        let n_ctx_actual = unsafe { llama_n_ctx(ctx) };

        Ok(LlamaCppBackend {
            model,
            ctx,
            memory,
            vocab,
            sampler,
            model_path: model_path.to_string(),
            n_ctx: n_ctx_actual,
            n_vocab,
            bos_id,
            eos_id,
            gpu_offload,
        })
    }

    /// Build base sampler chain (top_k + top_p + dist), without temperature.
    fn build_base_sampler() -> *mut std::ffi::c_void {
        unsafe {
            let sparams = llama_sampler_chain_default_params();
            let sampler = llama_sampler_chain_init(sparams);
            llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
            llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.95, 1));
            llama_sampler_chain_add(sampler, llama_sampler_init_dist(0xFFFFFFFF));
            sampler
        }
    }

    /// Reset for a new generation.
    /// Frees and rebuilds the sampler, and clears the KV cache (上下文新陈代谢).
    ///
    /// This avoids the expensive context recreation (`llama_free` + `llama_init_from_model`)
    /// which also has a known Vulkan issue on repeated create/destroy.
    /// Instead we reuse the same context but wipe its memory, enabling clean
    /// per-trial generations without KV cache accumulation across trials.
    ///
    /// Without this, KV cache accumulates across runs, causing:
    /// - "decode: failed to find a memory slot" errors when n_ctx is exceeded
    /// - Progressively slower inference as cache grows
    pub fn reset_context(&mut self) {
        unsafe {
            if !self.sampler.is_null() {
                llama_sampler_free(self.sampler);
                self.sampler = std::ptr::null_mut();
            }
        }
        self.sampler = Self::build_base_sampler();

        // 上下文新陈代谢: clear KV cache so next generation starts fresh
        if !self.memory.is_null() {
            unsafe { llama_memory_clear(self.memory, true); }
        }
    }

    /// Rebuild the sampler chain with the given temperature.
    /// Must be called before each `generate()` to support dynamic temperature.
    fn rebuild_sampler_chain(&mut self, temp: f32) {
        unsafe {
            if !self.sampler.is_null() {
                llama_sampler_free(self.sampler);
            }
        }
        let sparams = unsafe { llama_sampler_chain_default_params() };
        let sampler = unsafe {
            let s = llama_sampler_chain_init(sparams);
            llama_sampler_chain_add(s, llama_sampler_init_top_k(40));
            llama_sampler_chain_add(s, llama_sampler_init_top_p(0.95, 1));
            if temp > 0.0 {
                llama_sampler_chain_add(s, llama_sampler_init_temp(temp));
            }
            llama_sampler_chain_add(s, llama_sampler_init_dist(0xFFFFFFFF));
            s
        };
        self.sampler = sampler;
    }

    fn decode_tokens(&self, tokens: &[i32]) -> String {
        let mut result = String::new();
        for &tok in tokens {
            if tok == self.eos_id || tok < 0 {
                continue;
            }
            let mut buf = [0i8; 256];
            let len = unsafe {
                llama_token_to_piece(
                    self.vocab, tok,
                    buf.as_mut_ptr(), 255, 0, false,
                )
            };
            if len > 0 {
                let bytes: Vec<u8> = buf[..len as usize].iter().map(|&b| b as u8).collect();
                if let Ok(s) = String::from_utf8(bytes) {
                    result.push_str(&s);
                }
            }
        }
        result
    }

    fn decode_token_str(&self, token: i32) -> String {
        if token == self.eos_id {
            return "</s>".to_string();
        }
        let mut buf = [0i8; 256];
        let len = unsafe {
            llama_token_to_piece(self.vocab, token, buf.as_mut_ptr(), 255, 0, false)
        };
        if len > 0 {
            let bytes: Vec<u8> = buf[..len as usize].iter().map(|&b| b as u8).collect();
            String::from_utf8(bytes).unwrap_or_default()
        } else {
            String::new()
        }
    }

    // ── Vocab Scanning ──────────────────────────────────────────

    /// Get the EOS token ID detected from the model's vocabulary.
    pub fn eos_token_id(&self) -> u32 {
        self.eos_id as u32
    }

    /// Scan the model's vocabulary and classify tokens into off-focus and divergent groups.
    ///
    /// Returns `(off_focus_ids, divergent_ids)`.
    /// Uses the same pattern-based approach as `xiang-chat/src/vocab.rs`.
    pub fn discover_vocab(&self) -> (Vec<u32>, Vec<u32>) {
        let off_focus = self.scan_vocab_category(ENGLISH_PATTERNS);
        let divergent = self.scan_vocab_category(TRANSITION_PATTERNS);
        (off_focus, divergent)
    }

    /// Tokenize each pattern and collect unique token IDs.
    fn scan_vocab_category(&self, patterns: &[&str]) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut seen = HashSet::new();
        let eos = self.eos_id as u32;

        for pattern in patterns {
            let tokens = self.tokenize(pattern);
            for tid in tokens {
                if tid == eos {
                    continue;
                }
                if seen.insert(tid) {
                    ids.push(tid);
                }
            }
        }
        ids
    }

    // ── KV Cache Management (上下文新陈代谢) ──────────────────────

    /// Snapshot the current KV cache state (placeholder — seq_id tracking not yet implemented).
    pub fn kv_cache_snapshot(&self) -> i32 {
        -1
    }

    /// Rollback KV cache (placeholder).
    pub fn kv_cache_rollback(&self, _snapshot_seq_id: i32) {}

    /// Clear the KV cache entirely via llama_memory_clear.
    /// Called by `reset_context()` between trials to prevent cache overflow.
    pub fn kv_cache_clear(&self) {
        if !self.memory.is_null() {
            unsafe { llama_memory_clear(self.memory, true); }
        }
    }
}

impl Drop for LlamaCppBackend {
    fn drop(&mut self) {
        unsafe {
            if !self.sampler.is_null() {
                llama_sampler_free(self.sampler);
            }
            if !self.ctx.is_null() {
                llama_free(self.ctx);
            }
            if !self.model.is_null() {
                llama_model_free(self.model);
            }
            llama_backend_free();
        }
    }
}

impl LlmBackend for LlamaCppBackend {
    fn generate(&mut self, params: &mut GenerationParams) -> Result<GenerationResult, LlmError> {
        // Rebuild sampler chain with current temperature (supports dynamic temp)
        self.rebuild_sampler_chain(params.temperature.value());

        // Tokenize input
        let c_input = CString::new(params.user_input.clone())
            .map_err(|_| LlmError::GenerationFailed("输入包含NUL字节".into()))?;

        let add_bos = unsafe { llama_vocab_get_add_bos(self.vocab) };
        let mut prompt_tokens = vec![0i32; 65536];
        let n_tokens = unsafe {
            llama_tokenize(
                self.vocab,
                c_input.as_ptr(),
                params.user_input.len() as i32,
                prompt_tokens.as_mut_ptr(),
                65536,
                add_bos,
                false,
            )
        };

        if n_tokens < 0 {
            return Err(LlmError::GenerationFailed("Tokenization失败".into()));
        }
        prompt_tokens.truncate(n_tokens as usize);

        // Check context capacity
        let max_gen = params.max_tokens.min(self.n_ctx - prompt_tokens.len() as u32);

        // Generation loop
        let mut output_tokens: Vec<i32> = Vec::new();
        let mut blog: Vec<BiasLogEntry> = Vec::new();
        let mut suppressed_count: u32 = 0;
        let n_vocab = self.n_vocab;

        // Process prompt tokens
        let n_prompt = prompt_tokens.len() as i32;
        let batch = unsafe {
            llama_batch_get_one(prompt_tokens.as_mut_ptr(), n_prompt)
        };
        let ret = unsafe { llama_decode(self.ctx, batch) };
        if ret < 0 {
            return Err(LlmError::GenerationFailed("Prompt处理失败".into()));
        }

        // Also process the prompt through the sampler (accept tokens)
        // The sampler chain needs to know about the prompt tokens for some samplers
        for _tok in &prompt_tokens {
            // We don't sample during prompt processing, just accept
        }

        let _n_past = n_prompt;

        for step in 0..max_gen as i32 {
            // ── Logit Bias ──
            // Get logits from the last token, apply constraint engine bias
            let logits_ptr = unsafe { llama_get_logits_ith(self.ctx, -1) };
            if logits_ptr.is_null() {
                break;
            }

            // Compute bias directive from logit_bias (if present)
            let directive = if let Some(ref mut bias) = params.logit_bias {
                let logit_step = crate::LogitStep {
                    tokens_so_far: output_tokens.iter().map(|&t| t as u32).collect(),
                    position: step as u32,
                    vocab_size: n_vocab as u32,
                };
                bias.bias_for_step(&logit_step)
            } else {
                BiasDirective::default()
            };

            if directive.force_stop {
                output_tokens.push(self.eos_id);
                blog.push(BiasLogEntry {
                    step: step as u32,
                    deviation: params.deviation,
                    bias_rules: vec!["force_stop".into()],
                    token_sampled: self.eos_id as u32,
                    suppressed: true,
                    semantic_deviation: None,
                });
                break;
            }

            // Apply bias rules directly to logits array
            let mut suppressed = false;
            let mut sampled_id;

            if !directive.rules.is_empty() {
                let logits_slice = unsafe {
                    std::slice::from_raw_parts_mut(logits_ptr, n_vocab as usize)
                };

                for rule in &directive.rules {
                    for &tid in &rule.token_ids {
                        if (tid as i32) < n_vocab {
                            logits_slice[tid as usize] += rule.bias;
                        }
                    }
                }

                // Check if the natural (highest probability) token would be suppressed
                // Find current highest logit
                let mut max_logit = f32::NEG_INFINITY;
                let mut max_idx = 0i32;
                for (i, &l) in logits_slice.iter().enumerate() {
                    if l > max_logit {
                        max_logit = l;
                        max_idx = i as i32;
                    }
                }

                // Check if the highest logit is a suppressed token
                for rule in &directive.rules {
                    if rule.bias < 0.0 && rule.token_ids.contains(&(max_idx as u32)) {
                        suppressed = true;
                        break;
                    }
                }
            }

            // Sample token using the sampler chain
            sampled_id = unsafe { llama_sampler_sample(self.sampler, self.ctx, -1) };

            if suppressed {
                suppressed_count += 1;
            }

            // Accept token into sampler chain
            // Note: accept is handled by llama_sampler_sample

            // Notify the bias engine about the sampled token
            if let Some(ref mut bias) = params.logit_bias {
                let token_text = self.decode_token_str(sampled_id);
                bias.on_token_sampled(sampled_id as u32, &token_text);
            }

            // Log bias application
            let desc: Vec<String> = directive.rules.iter()
                .map(|r| format!("{}{}:{}",
                    if r.bias > 0.0 { "+" } else { "" },
                    r.bias, r.token_ids.len()))
                .collect();
            blog.push(BiasLogEntry {
                step: step as u32,
                deviation: params.deviation,
                bias_rules: desc,
                token_sampled: sampled_id as u32,
                suppressed: suppressed && sampled_id != self.eos_id,
                semantic_deviation: None,
            });

            output_tokens.push(sampled_id);

            // Check for EOS
            if unsafe { llama_vocab_is_eog(self.vocab, sampled_id) } {
                let text = self.decode_tokens(&output_tokens);
                return Ok(GenerationResult {
                    text,
                    tokens_generated: (step + 1) as u32,
                    truncated: false,
                    stop_reason: StopReason::Eos,
                    bias_applications: (step + 1) as u32,
                    tokens_suppressed: suppressed_count,
                    bias_log: blog,
                    deviated: false,
                });
            }

            // Decode the next token
            let next_batch = unsafe {
                llama_batch_get_one(&mut sampled_id as *mut i32, 1)
            };
            let ret = unsafe { llama_decode(self.ctx, next_batch) };
            if ret < 0 {
                break;
            }
        }

        let text = self.decode_tokens(&output_tokens);
        let n_gen = output_tokens.len() as u32;

        if n_gen >= max_gen {
            Ok(GenerationResult {
                text,
                tokens_generated: n_gen,
                truncated: true,
                stop_reason: StopReason::MaxTokens,
                bias_applications: n_gen,
                tokens_suppressed: suppressed_count,
                bias_log: blog,
                deviated: false,
            })
        } else {
            Ok(GenerationResult {
                text,
                tokens_generated: n_gen,
                truncated: false,
                stop_reason: StopReason::Eos,
                bias_applications: n_gen,
                tokens_suppressed: suppressed_count,
                bias_log: blog,
                deviated: false,
            })
        }
    }

    fn tokenize(&self, text: &str) -> Vec<u32> {
        let c_text = CString::new(text).unwrap_or_default();
        let mut tokens = vec![0i32; 65536];
        let n = unsafe {
            llama_tokenize(
                self.vocab, c_text.as_ptr(), text.len() as i32,
                tokens.as_mut_ptr(), 65536, false, false,
            )
        };
        if n <= 0 { return vec![]; }
        tokens.truncate(n as usize);
        tokens.into_iter().map(|t| t as u32).collect()
    }

    fn model_name(&self) -> &str {
        // Extract model name from path
        Path::new(&self.model_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
    }

    fn is_ready(&self) -> bool {
        !self.model.is_null() && !self.ctx.is_null()
    }

    fn device_info(&self) -> &str {
        if self.gpu_offload {
            "llama.cpp + GPU (Vulkan)"
        } else {
            "llama.cpp + CPU"
        }
    }
}
