/// HTTP backend for llama.cpp — connects to a remote llama.cpp server
/// running on Windows (or any host) with GPU acceleration.
///
/// Uses `ureq` — a pure synchronous HTTP client — to avoid nested
/// tokio runtime issues.

use crate::{
    GenerationParams, GenerationResult,
    LlmBackend, LlmError, LogitBias, LogitStep,
    StopReason,
};
use std::collections::{HashMap, HashSet};
use std::sync::{LazyLock, Mutex};
use std::time::Duration;

// ── Vocab pattern constants ─────────────────────────────────

/// Comprehensive English corpus for off-focus token discovery.
///
/// The HTTP backend cannot iterate the full vocabulary, so we tokenize a dense
/// English text covering diverse vocabulary — common words, technical terms,
/// academic language — to discover the English-related token IDs the model uses.
///
/// BPE tokenization splits words into subword units, so this corpus covers
/// far more token IDs than the old 80-word pattern list.
const ENGLISH_CORPUS: &str = "\
The system processes and analyzes complex data patterns through machine learning algorithms \
and neural network architectures for generating comprehensive responses about artificial \
intelligence and deep learning optimization strategies. Information retrieval and natural \
language understanding require sophisticated tokenization and embedding techniques with \
attention mechanisms and transformer models. Database management involves structured query \
processing transaction handling consistency guarantees replication fault tolerance and \
load balancing across distributed computing environments. Software engineering practices \
include testing debugging refactoring deployment continuous integration monitoring \
performance profiling security auditing vulnerability assessment and code review. \
Cryptographic protocols implement encryption decryption authentication authorization \
digital signatures key exchange hashing functions and access control mechanisms. \
Network architecture encompasses routing switching firewalls load balancers proxy servers \
content delivery edge computing and microservices orchestration with containerization. \
Scientific computing utilizes numerical methods statistical analysis data visualization \
hypothesis testing confidence intervals regression classification clustering dimensionality \
reduction feature engineering model validation and hyperparameter tuning. \
The fundamental principles of computer science include abstraction encapsulation inheritance \
polymorphism modularity concurrency parallelism synchronization memory management garbage \
collection type systems formal verification and program analysis. \
Critical thinking involves reasoning evaluation interpretation explanation inference \
deduction induction abduction analysis synthesis comparison contrast critique reflection \
metacognition problem solving decision making creativity innovation and design thinking.";

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

/// Global vocab discovery cache shared across all HttpBackend instances.
/// Keyed by server URL to support multiple endpoints if needed.
static VOCAB_CACHE: LazyLock<Mutex<HashMap<String, Option<(Vec<u32>, Vec<u32>)>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// HTTP backend that proxies inference to a remote llama.cpp server.
pub struct HttpBackend {
    server_url: String,
    name: String,
    ready: bool,
    device: String,
}

impl std::fmt::Debug for HttpBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpBackend")
            .field("server_url", &self.server_url)
            .field("name", &self.name)
            .field("ready", &self.ready)
            .field("device", &self.device)
            .finish()
    }
}

impl HttpBackend {
    pub fn new(server_url: &str) -> Self {
        let url = server_url.trim_end_matches('/').to_string();
        let ready = Self::check_health(&url);
        HttpBackend {
            name: if ready { "remote-qwen" } else { "unknown" }.into(),
            device: "AMD RX 6650 XT (Vulkan) via HTTP".into(),
            server_url: url,
            ready,
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.server_url, path)
    }

    fn agent() -> ureq::Agent {
        ureq::config::Config::builder()
            .timeout_connect(Some(Duration::from_secs(10)))
            .timeout_global(Some(Duration::from_secs(120)))
            .build()
            .into()
    }

    fn check_health(url: &str) -> bool {
        let health_url = format!("{}/health", url);
        match Self::agent().get(&health_url).call() {
            Ok(resp) => resp.status() == 200,
            Err(_) => false,
        }
    }

    fn build_prompt(
        system_prompt: &str,
        history: &[(String, String)],
        user_input: &str,
    ) -> String {
        let mut prompt = String::new();
        prompt.push_str(&format!(
            "<|im_start|>system\n{}<|im_end|>\n", system_prompt
        ));
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

    fn bias_to_logit_bias(
        bias: &mut Option<Box<dyn LogitBias + Send>>,
        vocab_size: u32,
    ) -> (Option<HashMap<String, f64>>, bool, u32) {
        let Some(bias) = bias else {
            return (None, false, 0);
        };

        let step = LogitStep {
            tokens_so_far: vec![],
            position: 0,
            vocab_size,
        };
        let directive = bias.bias_for_step(&step);

        if directive.force_stop {
            return (None, true, 0);
        }

        let mut map = HashMap::new();
        let total_applications = directive.rules.iter().map(|r| r.token_ids.len()).sum::<usize>() as u32;

        for rule in &directive.rules {
            for &tid in &rule.token_ids {
                map.insert(tid.to_string(), rule.bias as f64);
            }
        }

        (Some(map), false, total_applications)
    }

    /// Get the EOS token ID reported by the server (hardcoded for Qwen3.5).
    pub fn eos_token_id(&self) -> u32 {
        248046
    }

    /// Scan the model's vocabulary and classify tokens into off-focus and divergent groups.
    /// Results are cached globally (per server URL) to avoid repeated HTTP tokenize requests.
    ///
    /// Off-focus tokens: discovered by tokenizing a comprehensive English corpus.
    /// The corpus covers diverse vocabulary — BPE subword tokenization ensures broad coverage.
    pub fn discover_vocab(&self) -> (Vec<u32>, Vec<u32>) {
        // Check global cache first
        {
            let cache = VOCAB_CACHE.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(entry) = cache.get(&self.server_url) {
                if let Some(cached) = entry {
                    return cached.clone();
                }
            }
        }
        // Not cached — scan and store
        let off_focus = self.scan_vocab_corpus(ENGLISH_CORPUS);
        let divergent = self.scan_vocab_category(TRANSITION_PATTERNS);
        let result = (off_focus, divergent);
        // Store in cache
        if let Ok(mut cache) = VOCAB_CACHE.lock() {
            cache.insert(self.server_url.clone(), Some(result.clone()));
        }
        result
    }

    /// Tokenize a corpus string and collect all unique token IDs (excluding EOS).
    fn scan_vocab_corpus(&self, corpus: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut seen = HashSet::new();
        let eos = self.eos_token_id();

        let tokens = self.tokenize(corpus);
        for tid in tokens {
            if tid == eos {
                continue;
            }
            if seen.insert(tid) {
                ids.push(tid);
            }
        }
        ids
    }

    fn scan_vocab_category(&self, patterns: &[&str]) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut seen = HashSet::new();
        let eos = self.eos_token_id();

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
}

impl LlmBackend for HttpBackend {
    fn generate(&mut self, params: &mut GenerationParams) -> Result<GenerationResult, LlmError> {
        if !self.ready {
            return Err(LlmError::NotReady(
                "HTTP后端未就绪：请确认 llama.cpp server 正在运行".into(),
            ));
        }

        let (logit_bias_map, force_stop, bias_apps) =
            Self::bias_to_logit_bias(&mut params.logit_bias, 152064);

        if force_stop {
            return Ok(GenerationResult {
                text: String::new(),
                tokens_generated: 0,
                truncated: false,
                stop_reason: StopReason::Deviated,
                bias_applications: 1,
                tokens_suppressed: 0,
                bias_log: vec![],
                deviated: false,
                embedding: None,
            });
        }

        let prompt = Self::build_prompt(
            &params.system_prompt,
            &params.history,
            &params.user_input,
        );

        let mut body = serde_json::json!({
            "prompt": prompt,
            "n_predict": params.max_tokens,
            "temperature": params.temperature.value(),
            "top_k": 40,
            "top_p": 0.95,
            "stop": params.stop_sequences,
            "cache_prompt": true,
        });

        if let Some(bias_map) = logit_bias_map {
            body["logit_bias"] = serde_json::json!(bias_map);
        }

        let resp = Self::agent()
            .post(&self.url("/completion"))
            .send_json(&body)
            .map_err(|e| LlmError::GenerationFailed(format!("HTTP请求失败: {e}")))?;

        if resp.status() != 200 {
            let status = resp.status();
            let text = resp.into_body().read_to_string().unwrap_or_default();
            return Err(LlmError::GenerationFailed(format!(
                "服务器返回 {status}: {text}"
            )));
        }

        let body_str = resp.into_body().read_to_string()
            .map_err(|e| LlmError::GenerationFailed(format!("读取响应失败: {e}")))?;
        let data: serde_json::Value = serde_json::from_str(&body_str)
            .map_err(|e| LlmError::GenerationFailed(format!("解析响应失败: {e}")))?;

        let content = data["content"].as_str().unwrap_or("").to_string();
        let tokens = data["tokens_predicted"].as_u64().unwrap_or(0) as u32;
        let truncated = data["truncated"].as_bool().unwrap_or(false);
        let stop_type = data["stop_type"].as_str().unwrap_or("");
        let stop_str = data["stop_reason"].as_str().unwrap_or(stop_type);

        let stop_reason = match stop_str {
            "stop" | "eos" => StopReason::Eos,
            "limit" => StopReason::MaxTokens,
            _ => StopReason::Eos,
        };

        Ok(GenerationResult {
            text: content,
            tokens_generated: tokens,
            truncated,
            stop_reason,
            bias_applications: bias_apps,
            tokens_suppressed: 0,
            bias_log: vec![],
            deviated: false,
            embedding: None,
        })
    }

    fn tokenize(&self, text: &str) -> Vec<u32> {
        // NOTE: 不检查 self.ready（构造时的缓存值）；让 HTTP 请求本身决定能否连接，
        // 否则 llama.cpp 在构造后启动时，tokenize() 会永远返回空（词汇发现死循环）。
        //
        // Agent 已配置 connect=5s / global=30s 超时，不会挂死。
        let body = serde_json::json!({ "content": text });
        let url = self.url("/tokenize");

        let resp = match HttpBackend::agent().post(&url).send_json(&body) {
            Ok(r) => r,
            Err(_) => return vec![],
        };

        let body_str = match resp.into_body().read_to_string() {
            Ok(s) => s,
            Err(_) => return vec![],
        };
        let data: serde_json::Value = match serde_json::from_str(&body_str) {
            Ok(v) => v,
            Err(_) => return vec![],
        };

        data["tokens"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_u64())
                    .map(|v| v as u32)
                    .collect()
            })
            .unwrap_or_default()
    }

    fn model_name(&self) -> &str {
        &self.name
    }

    fn is_ready(&self) -> bool {
        // Live health check — the cached `self.ready` only reflects the initial
        // connection state and goes stale if the server starts after us.
        let health_url = format!("{}/health", self.server_url);
        match Self::agent().get(&health_url).call() {
            Ok(resp) => resp.status() == 200,
            Err(_) => false,
        }
    }

    fn device_info(&self) -> &str {
        &self.device
    }
}
