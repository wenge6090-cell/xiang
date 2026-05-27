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
    fn llama_get_embeddings_ith(
        ctx: *mut std::ffi::c_void,
        i: i32,
    ) -> *mut f32;
    fn llama_model_n_embd(
        model: *const std::ffi::c_void,
    ) -> i32;
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
    n_batch: u32,
    n_vocab: i32,
    n_embd: i32,
    #[allow(dead_code)]
    bos_id: i32,
    eos_id: i32,
    gpu_offload: bool,
    // ── KV Cache 管理（v4.0） ──
    /// 当前 seq_id，每次 generate() 调用递增。
    /// 用于基于 seq_id 的 KV cache snapshot/rollback。
    current_seq_id: i32,
    /// 跟踪当前生成中已处理的 token 数（含 prompt 处理）。
    /// 用于 kv_cache_snapshot() 返回位置。
    tokens_processed: i32,
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
        ctx_params.embeddings = true;  // Enable embedding extraction for HanziMap
        let n_batch = ctx_params.n_batch as u32;

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
        let n_embd = unsafe { llama_model_n_embd(model) };

        // Get memory handle for KV cache management (上下文新陈代谢)
        let memory = unsafe { llama_get_memory(ctx) };

        // Create initial sampler chain (without temp — added per-call in generate())
        let sampler = Self::build_base_sampler();

        let gpu_offload = unsafe { llama_supports_gpu_offload() };
        let n_ctx_actual = unsafe { llama_n_ctx(ctx) };

        // ── v4.0: seq_id 初始化 ──
        // 每次新连接使用递增 seq_id，支持基于 seq_id 的 KV cache rollback
        let current_seq_id = 0;
        let tokens_processed = 0;

        Ok(LlamaCppBackend {
            model,
            ctx,
            memory,
            vocab,
            sampler,
            model_path: model_path.to_string(),
            n_ctx: n_ctx_actual,
            n_batch: n_batch,
            n_vocab,
            n_embd,
            bos_id,
            eos_id,
            gpu_offload,
            current_seq_id,
            tokens_processed,
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
    /// Frees and rebuilds the sampler.
    /// Clears both SSM recurrent state and attention KV cache.
    pub fn reset_context(&mut self) {
        unsafe {
            if !self.sampler.is_null() {
                llama_sampler_free(self.sampler);
                self.sampler = std::ptr::null_mut();
            }
        }
        self.sampler = Self::build_base_sampler();

        // Clear memory metadata (data=false) — resets cell tracking & head pointers.
        // We do NOT zero GPU buffers (data=true) because that would invalidate
        // the recurrent state slots and cause "failed to find a memory slot" errors.
        // SSM state persistence between turns is mitigated by Fix 1:
        // each turn now includes full chat history in the prompt, making turns
        // self-contained regardless of residual SSM state.
        if !self.memory.is_null() {
            unsafe { llama_memory_clear(self.memory, false); }
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

    /// Get the model's embedding dimension.
    pub fn n_embd(&self) -> i32 {
        self.n_embd
    }

    /// Extract the last token's output embedding from the current context.
    ///
    /// Uses `llama_get_embeddings_ith(ctx, -1)` to get the embedding for the
    /// most recently decoded token position. Requires `ctx_params.embeddings = true`.
    ///
    /// Returns `None` if embeddings are not available (n_embd = 0) or if the
    /// underlying FFI call returns null.
    pub fn get_last_embedding(&self) -> Option<Vec<f32>> {
        if self.n_embd <= 0 {
            return None;
        }
        let ptr = unsafe { llama_get_embeddings_ith(self.ctx, -1) };
        if ptr.is_null() {
            return None;
        }
        let slice = unsafe { std::slice::from_raw_parts(ptr, self.n_embd as usize) };
        Some(slice.to_vec())
    }

    /// Encode text and return the last token's output embedding.
    ///
    /// Clears the KV cache first to prevent cross-contamination between calls.
    /// Used by the semantic navigation system to build output-space operator centroids
    /// by running anchor characters through the model at startup.
    pub fn embed_text(&mut self, text: &str) -> Option<Vec<f32>> {
        // Clear KV cache (use memory_clear for compatibility across model architectures)
        unsafe {
            llama_memory_clear(self.memory, true);
        }

        // Tokenize
        let c_text = CString::new(text).ok()?;
        let mut tokens = vec![0i32; 1024];
        let add_bos = unsafe { llama_vocab_get_add_bos(self.vocab) };
        let n = unsafe {
            llama_tokenize(
                self.vocab,
                c_text.as_ptr(),
                text.len() as i32,
                tokens.as_mut_ptr(),
                1024,
                add_bos,
                false,
            )
        };
        if n <= 0 {
            return None;
        }
        tokens.truncate(n as usize);

        // Decode in batches (same pattern as generate())
        let n_batch_size = self.n_batch as usize;
        for chunk_start in (0..tokens.len()).step_by(n_batch_size) {
            let chunk_end = (chunk_start + n_batch_size).min(tokens.len());
            let chunk = &tokens[chunk_start..chunk_end];
            let batch = unsafe {
                llama_batch_get_one(chunk.as_ptr() as *mut i32, chunk.len() as i32)
            };
            let ret = unsafe { llama_decode(self.ctx, batch) };
            if ret < 0 {
                return None;
            }
        }

        self.get_last_embedding()
    }

    /// Scan the model's vocabulary and classify tokens into off-focus and divergent groups.
    ///
    /// Returns `(off_focus_ids, divergent_ids)`.
    /// Uses pattern-based approach: tokenizes each pattern string and collects unique token IDs.
    /// This provides targeted suppression (~200 tokens) instead of suppressing half the vocab.
    /// Patterns cover common English vocabulary to suppress English text generation.
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

    /// 创建带 seq_id 的 batch，所有 token 共享同一 seq_id。
    ///
    /// `positions_start`：该 batch 中第一个 token 在完整序列中的位置（用于 KV cache 定位）
    /// seq_id 机制允许每个生成回合使用独立序列号，
    /// 使得 kv_cache_rollback 可以精确删除某次生成的 KV cache 条目。
    fn make_batch_with_seq_id(
        tokens: &[i32], n_tokens: i32,
        positions_start: i32, seq_id: i32,
    ) -> (llama_batch, Vec<i32>, Vec<i32>, Vec<Box<[i32]>>) {
        let n = n_tokens as usize;
        let positions: Vec<i32> = (0..n).map(|i| positions_start + i as i32).collect();
        let n_seq_ids: Vec<i32> = vec![1; n];
        let seq_id_arrays: Vec<Box<[i32]>> = (0..n).map(|_| vec![seq_id].into_boxed_slice()).collect();
        let seq_id_ptrs: Vec<*mut i32> = seq_id_arrays.iter().map(|a| a.as_ptr() as *mut i32).collect();

        let batch = llama_batch {
            n_tokens: n_tokens,
            token: tokens.as_ptr() as *mut i32,
            embd: std::ptr::null_mut(),
            pos: positions.as_ptr() as *mut i32,
            n_seq_id: n_seq_ids.as_ptr() as *mut i32,
            seq_id: seq_id_ptrs.as_ptr() as *mut *mut i32,
            logits: std::ptr::null_mut(),
        };
        (batch, positions, n_seq_ids, seq_id_arrays)
    }

    /// 单 token 的 seq_id batch（用于生成循环的每一步）。
    fn make_single_batch_with_seq_id(
        token: i32, position: i32, seq_id: i32,
    ) -> (llama_batch, i32, i32, Box<[i32]>) {
        let n_seq_id: i32 = 1;
        let seq_id_arr: Box<[i32]> = vec![seq_id].into_boxed_slice();
        let batch = llama_batch {
            n_tokens: 1,
            token: &token as *const i32 as *mut i32,
            embd: std::ptr::null_mut(),
            pos: &position as *const i32 as *mut i32,
            n_seq_id: &n_seq_id as *const i32 as *mut i32,
            seq_id: &(seq_id_arr.as_ptr() as *mut i32) as *const *mut i32 as *mut *mut i32,
            logits: std::ptr::null_mut(),
        };
        (batch, n_seq_id, position, seq_id_arr)
    }

    /// Snapshot the current KV cache state — 返回当前 seq_id 和 token 位置。
    ///
    /// 调用后，如果生成偏离，可以用返回的 seq_id 做 kv_cache_rollback()，
    /// 删除从该位置开始的所有 KV cache 条目。
    pub fn kv_cache_snapshot(&self) -> i32 {
        self.current_seq_id
    }

    /// Rollback KV cache — 删除指定 seq_id 之后的所有 KV cache 条目。
    ///
    /// 基于 llama.cpp 的 seq_id 机制：
    /// - `llama_memory_seq_rm(mem, seq_id, p0, p1)` 删除 [p0, p1) 位置
    /// - 传 p0=0, p1=MAX 删除该 seq_id 的所有条目
    pub fn kv_cache_rollback(&self, snapshot_seq_id: i32) {
        if snapshot_seq_id < 0 || self.memory.is_null() {
            return;
        }
        // 删除 seq_id = snapshot_seq_id 之后的所有 seq_id 条目
        // 从位置 0 开始删到 MAX，覆盖该 seq_id 的所有 KV cache
        unsafe {
            // 保留 snapshot_seq_id 的条目，删除所有 >= snapshot_seq_id + 1 的 seq_id
            for sid in (snapshot_seq_id + 1)..(snapshot_seq_id + 100) {
                if !llama_memory_seq_rm(self.memory, sid, 0, i32::MAX) {
                    break; // seq_id 不存在则停止
                }
            }
        }
    }

    /// Clear the KV cache — metadata-only, no GPU buffer clear.
    pub fn kv_cache_clear(&self) {
        if !self.memory.is_null() {
            unsafe { llama_memory_clear(self.memory, false); }
        }
    }

    /// Build a full prompt with chat template for Qwen3.5, including conversation history.
    /// Format: <|im_start|>system\n{system}<|im_end|>\n
    ///         <|im_start|>user\n{u1}<|im_end|>\n<|im_start|>assistant\n{a1}<|im_end|>\n...
    ///         <|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n
    pub fn build_chat_prompt(
        system_prompt: &str,
        history: &[(String, String)],
        user_input: &str,
    ) -> String {
        let mut prompt = String::new();
        if !system_prompt.is_empty() {
            prompt.push_str(&format!(
                "<|im_start|>system\n{}<|im_end|>\n", system_prompt
            ));
        }
        for (user, asst) in history {
            prompt.push_str(&format!(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}<|im_end|>\n",
                user, asst
            ));
        }
        prompt.push_str(&format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            user_input
        ));
        prompt
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

        // Build prompt with chat template (includes conversation history)
        let full_prompt = Self::build_chat_prompt(
            &params.system_prompt,
            &params.history,
            &params.user_input,
        );
        let c_input = CString::new(full_prompt.clone())
            .map_err(|_| LlmError::GenerationFailed("输入包含NUL字节".into()))?;

        let add_bos = unsafe { llama_vocab_get_add_bos(self.vocab) };
        let mut prompt_tokens = vec![0i32; 65536];
        let n_tokens = unsafe {
            llama_tokenize(
                self.vocab,
                c_input.as_ptr(),
                full_prompt.len() as i32,
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

        // Process prompt tokens in batches to stay within n_batch limit
        // v4.0: 使用 seq_id 管理 KV cache
        self.current_seq_id += 1;
        let gen_seq_id = self.current_seq_id;
        let mut n_past = 0i32;
        let n_batch_size = self.n_batch as usize;
        for chunk_start in (0..prompt_tokens.len()).step_by(n_batch_size) {
            let chunk_end = (chunk_start + n_batch_size).min(prompt_tokens.len());
            let chunk = &prompt_tokens[chunk_start..chunk_end];
            let n_tok = (chunk_end - chunk_start) as i32;
            let (batch, _pos, _ns, _sa) = Self::make_batch_with_seq_id(
                chunk, n_tok, n_past, gen_seq_id,
            );
            let ret = unsafe { llama_decode(self.ctx, batch) };
            if ret < 0 {
                return Err(LlmError::GenerationFailed("Prompt处理失败".into()));
            }
            n_past += n_tok;
        }
        self.tokens_processed = n_past;

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

            // ── Check stop sequences ──
            // Decode so far and check against each stop sequence pattern.
            if !params.stop_sequences.is_empty() {
                let partial = self.decode_tokens(&output_tokens);
                for seq in &params.stop_sequences {
                    if partial.contains(seq.as_str()) {
                        let text = self.decode_tokens(&output_tokens);
                        return Ok(GenerationResult {
                            text,
                            tokens_generated: (step + 1) as u32,
                            truncated: false,
                            stop_reason: StopReason::StopSeq,
                            bias_applications: (step + 1) as u32,
                            tokens_suppressed: suppressed_count,
                            bias_log: blog,
                            deviated: false,
                            embedding: self.get_last_embedding(),
                        });
                    }
                }
            }

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
                    embedding: self.get_last_embedding(),
                });
            }

            // Decode the next token (v4.0: 使用 seq_id 管理 KV cache)
            let (next_batch, _ns, _pos, _sa) = Self::make_single_batch_with_seq_id(
                sampled_id, n_past, gen_seq_id,
            );
            n_past += 1;
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
                embedding: self.get_last_embedding(),
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
                embedding: self.get_last_embedding(),
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
