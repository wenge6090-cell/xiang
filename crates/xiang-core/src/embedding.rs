/// 嵌入向量 (Embedding) — semantic vector representation.
///
/// Type alias for a dense float vector (typically 8–1024 dimensions).
/// Used by the Yin encoder (阴仪) to represent text as continuous vectors.
pub type Embedding = Vec<f32>;

/// Compute cosine similarity between two embeddings.
///
/// Returns a value in [-1.0, 1.0].
/// - 1.0 = identical direction
/// - 0.0 = orthogonal
/// - -1.0 = opposite direction
///
/// # Panics
/// Panics if the two vectors have different lengths.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Embedding dimension mismatch");
    if a.is_empty() {
        return 0.0;
    }
    let (dot, norm_a, norm_b) = a
        .iter()
        .zip(b.iter())
        .fold((0.0f32, 0.0f32, 0.0f32), |(d, na, nb), (&x, &y)| {
            (d + x * y, na + x * x, nb + y * y)
        });
    let denom = (norm_a * norm_b).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// TextEncoder trait — the Yin (阴仪) interface.
///
/// Implementations encode arbitrary text into a dense `Embedding` vector.
/// The encoder is stateless: `encode(text)` always produces the same vector
/// for the same input (deterministic).
pub trait TextEncoder: Send + Sync {
    /// Encode text into a dense embedding vector.
    fn encode(&self, text: &str) -> Embedding;

    /// Dimension of the output embedding.
    fn dimension(&self) -> usize;

    /// Human-readable model name (e.g. "mock-8d", "bge-m3").
    fn model_name(&self) -> &str;
}

/// Mock encoder mode — controls the focus behaviour of the MockEncoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MockEncoderMode {
    /// Low deviation — encoder stays close to origin.
    Focused,
    /// Increasing deviation — encoder drifts over successive calls.
    Drifting,
    /// High deviation — encoder produces hallucinated, off-topic vectors.
    Hallucinating,
}

/// MockEncoder — deterministic pseudo-random encoder for testing.
///
/// Uses a simple hash of the input text to produce a unit vector of
/// dimension `dim`. The `mode` controls the focus behaviour:
///
/// - `Focused`: hash → normalise → unit vector (always on-topic)
/// - `Drifting`: each `encode()` call accumulates drift via a counter
/// - `Hallucinating`: produces random-like vectors with high cosine distance
///   from the origin embedding
///
/// Same text always produces the same vector for a given mode step.
pub struct MockEncoder {
    dim: usize,
    mode: MockEncoderMode,
    /// Drift counter (incremented per encode call in Drifting mode)
    drift_step: u32,
    /// Seed for deterministic hashing
    seed: u64,
}

impl MockEncoder {
    /// Create a new MockEncoder with the given dimension and mode.
    pub fn new(dim: usize, mode: MockEncoderMode) -> Self {
        MockEncoder {
            dim,
            mode,
            drift_step: 0,
            seed: 42,
        }
    }

    /// Create with a custom seed for deterministic reproducibility.
    pub fn with_seed(dim: usize, mode: MockEncoderMode, seed: u64) -> Self {
        MockEncoder {
            dim,
            mode,
            drift_step: 0,
            seed,
        }
    }

    /// Reset the drift counter (useful for starting new experiment runs).
    pub fn reset_drift(&mut self) {
        self.drift_step = 0;
    }

    /// Simple FNV-1a hash for deterministic pseudo-random generation.
    fn hash_text(&self, text: &str) -> u64 {
        let mut hash: u64 = self.seed.wrapping_mul(0xcbf29ce484222325);
        for byte in text.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        // Mix in the drift step for Drifting mode
        if self.mode == MockEncoderMode::Drifting {
            hash ^= self.drift_step as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    /// Convert a u64 hash into a unit vector of dimension `dim`.
    fn hash_to_unit_vector(&self, hash: u64) -> Embedding {
        let mut state = hash;
        let mut vec = Vec::with_capacity(self.dim);
        for _ in 0..self.dim {
            // Simple xorshift to generate deterministic "random" floats
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Map to [-1.0, 1.0]
            let val = ((state as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32;
            vec.push(val);
        }
        // Normalise to unit vector
        let norm: f32 = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut vec {
                *v /= norm;
            }
        }
        vec
    }
}

impl TextEncoder for MockEncoder {
    fn encode(&self, text: &str) -> Embedding {
        let hash = self.hash_text(text);
        let base = self.hash_to_unit_vector(hash);

        match self.mode {
            MockEncoderMode::Focused => base,
            MockEncoderMode::Drifting => {
                // For Drifting, we need to simulate a counter. Since `encode`
                // takes `&self`, we use interior mutability via a static-like
                // approach: the drift_step is included in the hash (see hash_text).
                // No further alteration needed — the hash already incorporates drift_step.
                base
            }
            MockEncoderMode::Hallucinating => {
                // Produce a vector that is deliberately far from "focused" origin.
                // Flip sign of every second component to reduce cosine similarity.
                base.iter()
                    .enumerate()
                    .map(|(i, &v)| if i % 2 == 0 { v } else { -v })
                    .collect()
            }
        }
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn model_name(&self) -> &str {
        match self.mode {
            MockEncoderMode::Focused => "mock-8d-focused",
            MockEncoderMode::Drifting => "mock-8d-drifting",
            MockEncoderMode::Hallucinating => "mock-8d-hallucinating",
        }
    }
}

/// Increment the drift counter for MockEncoders in Drifting mode.
///
/// This should be called after each encode cycle by the owning code.
pub fn advance_drift(encoder: &mut MockEncoder) {
    encoder.drift_step = encoder.drift_step.wrapping_add(1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_same() {
        let v = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_cosine_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_mock_deterministic() {
        let enc = MockEncoder::new(8, MockEncoderMode::Focused);
        let v1 = enc.encode("hello");
        let v2 = enc.encode("hello");
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_mock_unit_vector() {
        let enc = MockEncoder::new(8, MockEncoderMode::Focused);
        let v = enc.encode("test");
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_mock_focused_close() {
        let enc = MockEncoder::new(8, MockEncoderMode::Focused);
        let origin = enc.encode("origin");
        let similar = enc.encode("origin"); // same text → same vector
        let sim = cosine_similarity(&origin, &similar);
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_mock_drifting_diverges() {
        let mut enc = MockEncoder::new(8, MockEncoderMode::Drifting);
        let origin = enc.encode("origin");
        // After drift advance, same text should produce different vector
        advance_drift(&mut enc);
        let drifted = enc.encode("origin");
        let sim = cosine_similarity(&origin, &drifted);
        // Should be different (not 1.0)
        assert!(sim < 0.99, "Drifted vector should differ from origin, got sim={sim}");
    }

    #[test]
    fn test_dimension() {
        let enc = MockEncoder::new(128, MockEncoderMode::Focused);
        assert_eq!(enc.dimension(), 128);
        assert_eq!(enc.encode("x").len(), 128);
    }
}
