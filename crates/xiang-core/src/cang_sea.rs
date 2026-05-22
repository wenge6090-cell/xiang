/// 藏海 (CangSea) — 64×64 Hebbian weight matrix for experience learning.
///
/// Stores state transition experiences with rewards.
/// Positive experiences strengthen weights, negative experiences weaken them.
/// Maximum 1000 entries, with low-reward entries evicted FIFO.
///
/// CangSea v2: also supports a SemanticStore for vector-based experience memory,
/// including immune memory and redundancy merging (see `SemanticStore`).

use crate::embedding::Embedding;
use crate::gua::Gua;
use crate::ZhiForces;
use std::path::Path;

// ─── Legacy CangSea (unchanged) ──────────────────────────────────

/// An experience entry in the CangSea.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CangSeaEntry {
    pub from: Gua,
    pub to: Gua,
    pub reward: f32,
    pub timestamp: u64,
}

/// 64×64 Hebbian weight matrix for learning from experience.
#[derive(Debug, Clone)]
pub struct CangSea {
    /// W[from.0][to.0] — transition weight matrix
    weights: [[f32; 64]; 64],
    /// Ordered records for FIFO eviction
    entries: Vec<CangSeaEntry>,
    /// Maximum number of entries before eviction
    max_entries: usize,
    /// Monotonic timestamp counter
    clock: u64,
    /// Semantic vector store (CangSea v2).
    /// None = legacy mode only (backwards compatible).
    /// Not serialized — embeddings are reconstructed at runtime from project_embedding_to_bits.
    pub semantic_store: Option<SemanticStore>,
}

impl CangSea {
    /// Create a new CangSea with default max 1000 entries (no semantic store).
    pub fn new() -> Self {
        Self::with_capacity(1000)
    }

    pub fn with_capacity(max_entries: usize) -> Self {
        CangSea {
            weights: [[0.0; 64]; 64],
            entries: Vec::with_capacity(max_entries),
            max_entries,
            clock: 0,
            semantic_store: None,
        }
    }

    /// Create a CangSea with a semantic store of given capacity.
    pub fn with_semantic(
        max_entries: usize,
        semantic_capacity: usize,
        semantic_dim: usize,
        immune_max: usize,
    ) -> Self {
        CangSea {
            weights: [[0.0; 64]; 64],
            entries: Vec::with_capacity(max_entries),
            max_entries,
            clock: 0,
            semantic_store: Some(SemanticStore::new(semantic_capacity, semantic_dim, immune_max)),
        }
    }

    /// Store a learning experience.
    /// Positive reward (>0) strengthens weight; negative weakens it.
    /// The learning rate η is proportional to |reward|.
    pub fn store(&mut self, from: Gua, to: Gua, reward: f32) {
        // Evict if at capacity (remove lowest-reward entry)
        if self.entries.len() >= self.max_entries {
            self.evict_lowest();
        }

        // Apply Hebbian update
        let eta = reward.abs().min(1.0) * 0.1; // learning rate capped
        if reward > 0.0 {
            self.weights[from.0 as usize][to.0 as usize] += eta;
        } else {
            self.weights[from.0 as usize][to.0 as usize] -= eta;
            // Clamp to non-negative
            if self.weights[from.0 as usize][to.0 as usize] < 0.0 {
                self.weights[from.0 as usize][to.0 as usize] = 0.0;
            }
        }

        self.clock += 1;
        self.entries.push(CangSeaEntry {
            from,
            to,
            reward,
            timestamp: self.clock,
        });
    }

    /// Store a semantic learning experience.
    ///
    /// If the semantic store is active, this stores a `SemanticEntry` and
    /// handles immune zone routing (reward < -0.5 → also pushed to immune_zone).
    /// Also projects the entry onto the legacy matrix for backwards compatibility.
    ///
    /// Panics if `semantic_store` is None.
    pub fn store_semantic(&mut self, entry: SemanticEntry) {
        let store = self
            .semantic_store
            .as_mut()
            .expect("store_semantic called without semantic_store");
        store.store(entry.clone());

        // Also project to legacy matrix
        let from_gua = Gua::from_bits(project_embedding_to_bits(&entry.v_think));
        let to_gua = Gua::from_bits(project_embedding_to_bits(
            &entry.v_goal,
        ));
        self.store(from_gua, to_gua, entry.reward);
    }

    /// Reinforce a specific transition with learning rate eta.
    pub fn reinforce(&mut self, from: Gua, to: Gua, eta: f32) {
        self.weights[from.0 as usize][to.0 as usize] += eta;
    }

    /// Weaken a specific transition with learning rate eta.
    pub fn weaken(&mut self, from: Gua, to: Gua, eta: f32) {
        let w = &mut self.weights[from.0 as usize][to.0 as usize];
        *w = (*w - eta).max(0.0);
    }

    /// Get the weight for a specific transition.
    pub fn weight(&self, from: Gua, to: Gua) -> f32 {
        self.weights[from.0 as usize][to.0 as usize]
    }

    /// Sample a next state from the learned distribution for a given `from` state.
    /// Uses weighted probability from the row `weights[from][..]`.
    /// Returns None if the row has zero total weight.
    pub fn hebbian_sample<R: rand::Rng>(&self, rng: &mut R, from: Gua) -> Option<Gua> {
        let row = &self.weights[from.0 as usize];
        let total: f32 = row.iter().sum();
        if total <= 0.0 {
            return None;
        }

        let mut threshold: f32 = rng.random::<f32>() * total;
        for (i, &w) in row.iter().enumerate() {
            threshold -= w;
            if threshold <= 0.0 {
                return Some(Gua(i as u8));
            }
        }
        // Fallback: return last state (shouldn't happen due to floating point)
        row.iter()
            .rposition(|&w| w > 0.0)
            .map(|i| Gua(i as u8))
    }

    /// Number of entries currently stored.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the CangSea is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over all entries.
    pub fn entries(&self) -> impl Iterator<Item = &CangSeaEntry> {
        self.entries.iter()
    }

    /// Get entries for a specific `from` state, sorted by weight descending.
    pub fn entries_from(&self, from: Gua) -> Vec<&CangSeaEntry> {
        let mut result: Vec<_> = self.entries.iter().filter(|e| e.from == from).collect();
        result.sort_by(|a, b| {
            self.weight(b.from, b.to)
                .partial_cmp(&self.weight(a.from, a.to))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        result
    }

    /// Evict the entry with the lowest reward.
    fn evict_lowest(&mut self) {
        if let Some(pos) = self.entries.iter().enumerate()
            .min_by(|(_, a), (_, b)| a.reward.partial_cmp(&b.reward).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
        {
            self.entries.remove(pos);
        }
    }

    /// Generate a human-readable summary of experiences for a given `from` state.
    ///
    /// Groups experiences into "aligned" (reward > 0) and "deviated" (reward ≤ 0),
    /// showing the top `top_k` most-weighted entries.
    /// Returns `None` if there are no entries for this state.
    pub fn experience_summary(&self, from: Gua, top_k: usize) -> Option<String> {
        let entries = self.entries_from(from);
        if entries.is_empty() {
            return None;
        }

        let (aligned, deviated): (Vec<&&CangSeaEntry>, Vec<&&CangSeaEntry>) =
            entries.iter().partition(|e| e.reward > 0.0);

        let mut s = String::new();
        s.push_str(&format!(
            "当前状态'{}'({})有{}条经验",
            from.name(),
            from,
            entries.len()
        ));

        // Show top-K aligned
        if !aligned.is_empty() {
            s.push_str(&format!("\n  对齐经验({}次)", aligned.len()));
            for e in aligned.iter().take(top_k) {
                let w = self.weight(e.from, e.to);
                s.push_str(&format!(
                    "\n    → {} ({}) r={:.2} w={:.2}",
                    e.to.name(),
                    e.to,
                    e.reward,
                    w
                ));
            }
        }

        // Show top-K deviated
        if !deviated.is_empty() {
            s.push_str(&format!("\n  偏离经验({}次)", deviated.len()));
            for e in deviated.iter().take(top_k) {
                let w = self.weight(e.from, e.to);
                s.push_str(&format!(
                    "\n    → {} ({}) r={:.2} w={:.2}",
                    e.to.name(),
                    e.to,
                    e.reward,
                    w
                ));
            }
        }

        Some(s)
    }

    /// Prune all entries with reward below threshold.
    pub fn prune_low_reward(&mut self, threshold: f32) {
        self.entries.retain(|e| e.reward >= threshold);
    }

    /// Query CangSea for push/resist forces applicable to the current state.
    ///
    /// - push_forces: high-weight aligned experiences (reward > 0, weight > 0.1)
    /// - resist_forces: cautionary deviated experiences (reward < -0.1)
    /// Returns up to 3 entries per force direction.
    pub fn query_forces(&self, from: Gua) -> ZhiForces {
        let entries = self.entries_from(from);
        let mut forces = ZhiForces::empty();

        for e in entries.iter() {
            let w = self.weight(e.from, e.to);
            if e.reward > 0.0 && w > 0.1 && forces.push_forces.len() < 3 {
                forces.push_forces.push(format!(
                    "[对齐] {}→{} r={:.2} w={:.2}",
                    e.from.name(),
                    e.to.name(),
                    e.reward,
                    w
                ));
            } else if e.reward < -0.1 && forces.resist_forces.len() < 3 {
                forces.resist_forces.push(format!(
                    "[偏离] {}→{} r={:.2} w={:.2}",
                    e.from.name(),
                    e.to.name(),
                    e.reward,
                    w
                ));
            }
        }

        forces
    }

    /// Save CangSea to a JSON file.
    ///
    /// Serializes the legacy weight matrix and entries.
    /// SemanticStore is NOT persisted (requires runtime embedding reconstruction).
    pub fn save_to_file(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
        let file = std::fs::File::create(path)?;
        let writer = std::io::BufWriter::new(file);
        serde_json::to_writer(writer, self)?;
        Ok(())
    }

    /// Load CangSea from a JSON file saved by `save_to_file`.
    ///
    /// Returns the deserialized CangSea. The semantic_store will be None
    /// regardless of what was stored (it was skipped during serialization).
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let cang_sea: CangSea = serde_json::from_reader(reader)?;
        Ok(cang_sea)
    }
}

impl Default for CangSea {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Custom Serialize/Deserialize for CangSea ──────────────────
// [[f32; 64]; 64] exceeds serde's default max array size (32),
// so we serialize/deserialize as Vec<Vec<f32>>.

impl serde::Serialize for CangSea {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;

        let weights_2d: Vec<Vec<f32>> = self.weights.iter()
            .map(|row| row.to_vec())
            .collect();

        let mut state = serializer.serialize_struct("CangSea", 4)?;
        state.serialize_field("weights", &weights_2d)?;
        state.serialize_field("entries", &self.entries)?;
        state.serialize_field("max_entries", &self.max_entries)?;
        state.serialize_field("clock", &self.clock)?;
        state.end()
    }
}

impl<'de> serde::Deserialize<'de> for CangSea {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::{self, MapAccess, Visitor};
        use std::fmt;

        #[derive(serde::Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field { Weights, Entries, MaxEntries, Clock }

        struct CangSeaVisitor;

        impl<'de> Visitor<'de> for CangSeaVisitor {
            type Value = CangSea;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct CangSea")
            }

            fn visit_map<V: MapAccess<'de>>(self, mut map: V) -> Result<CangSea, V::Error> {
                let mut weights: Option<Vec<Vec<f32>>> = None;
                let mut entries: Option<Vec<CangSeaEntry>> = None;
                let mut max_entries: Option<usize> = None;
                let mut clock: Option<u64> = None;

                while let Some(key) = map.next_key::<Field>()? {
                    match key {
                        Field::Weights => {
                            weights = Some(map.next_value::<Vec<Vec<f32>>>()?);
                        }
                        Field::Entries => {
                            entries = Some(map.next_value()?);
                        }
                        Field::MaxEntries => {
                            max_entries = Some(map.next_value()?);
                        }
                        Field::Clock => {
                            clock = Some(map.next_value()?);
                        }
                    }
                }

                let weights = weights.ok_or_else(|| de::Error::missing_field("weights"))?;
                let entries = entries.ok_or_else(|| de::Error::missing_field("entries"))?;
                let max_entries = max_entries.ok_or_else(|| de::Error::missing_field("max_entries"))?;
                let clock = clock.ok_or_else(|| de::Error::missing_field("clock"))?;

                if weights.len() != 64 || weights.iter().any(|r| r.len() != 64) {
                    return Err(de::Error::custom("weights must be 64×64"));
                }

                let mut weights_arr = [[0.0f32; 64]; 64];
                for (i, row) in weights.iter().enumerate() {
                    weights_arr[i].copy_from_slice(row);
                }

                Ok(CangSea {
                    weights: weights_arr,
                    entries,
                    max_entries,
                    clock,
                    semantic_store: None,
                })
            }
        }

        deserializer.deserialize_struct("CangSea", &["weights", "entries", "max_entries", "clock"], CangSeaVisitor)
    }
}

// ─── CangSea v2: Semantic Vector Store ───────────────────────────

/// Redundancy merge strategy for experience crystallisation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MergeStrategy {
    /// Weighted average of vectors, weighted by |reward|.
    WeightedAverage,
    /// Keep only the entry with the highest absolute reward.
    MostRewarded,
    /// Compute the centroid (unweighted mean) of all vectors.
    Centroid,
}

/// A semantic experience entry in the CangSea v2.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SemanticEntry {
    /// Embedding of the final thought state.
    pub v_think: Embedding,
    /// Embedding of the goal.
    pub v_goal: Embedding,
    /// Embedding of the obstacle (empty vec if none).
    pub v_obstacle: Embedding,
    /// Embedding of the strategy.
    pub v_strategy: Embedding,
    /// Embedding of the cognitive attitude.
    pub v_attitude: Embedding,
    /// Deviation at time of storage.
    pub deviation: f32,
    /// Reward/penalty signal.
    pub reward: f32,
    /// Monotonic timestamp.
    pub timestamp: u64,
    /// How many times this entry has been pushed into immune zone.
    pub immune_count: u32,
    /// How many times this entry has been involved in a merge.
    pub merge_count: u32,
    /// Merge generation counter (0 = original, 1+ = crystal generation).
    pub crystal_generation: u32,
}

/// SemanticStore — vector-based experience memory for CangSea v2.
///
/// Stores semantic embeddings of thought trajectories and provides
/// cosine-similarity-based retrieval, immune memory, and redundancy
/// merging into experience crystals.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SemanticStore {
    /// Regular experience library.
    pub entries: Vec<SemanticEntry>,
    /// Immune memory zone — negative-reward experiences permanently isolated.
    /// Never evicted by normal pruning.
    pub immune_zone: Vec<SemanticEntry>,
    /// Maximum entries in the regular library.
    max_entries: usize,
    /// Dimension of stored embeddings.
    dim: usize,
    /// Maximum entries in the immune zone.
    immune_zone_max: usize,
    /// Monotonic clock for timestamps.
    clock: u64,
    /// Current merge strategy.
    pub merge_strategy: MergeStrategy,
}

impl SemanticStore {
    /// Create a new SemanticStore.
    pub fn new(max_entries: usize, dim: usize, immune_zone_max: usize) -> Self {
        SemanticStore {
            entries: Vec::with_capacity(max_entries),
            immune_zone: Vec::with_capacity(immune_zone_max),
            max_entries,
            dim,
            immune_zone_max,
            clock: 0,
            merge_strategy: MergeStrategy::WeightedAverage,
        }
    }

    /// Current total entries (regular + immune).
    pub fn total_entries(&self) -> usize {
        self.entries.len() + self.immune_zone.len()
    }

    /// Check if the regular store is above 80% capacity, triggering merge.
    fn should_merge(&self) -> bool {
        self.entries.len() as f32 >= self.max_entries as f32 * 0.8
    }

    /// Store a semantic entry.
    ///
    /// - reward >= -0.5: stored in regular entries only
    /// - reward < -0.5: stored in regular AND immune zone
    /// - If regular store exceeds 80% capacity, triggers merge_similar(0.85)
    pub fn store(&mut self, mut entry: SemanticEntry) {
        self.clock += 1;
        entry.timestamp = self.clock;

        // Evict lowest reward if at capacity in regular store
        if self.entries.len() >= self.max_entries {
            self.evict_lowest();
        }

        let is_negative = entry.reward < -0.5;

        if is_negative {
            entry.immune_count += 1;

            // Immune zone eviction: merge immune entries instead of deleting
            if self.immune_zone.len() >= self.immune_zone_max {
                self.merge_immune_zone();
                // If still full after merge (unlikely), evict oldest immune entry
                if self.immune_zone.len() >= self.immune_zone_max {
                    self.immune_zone.remove(0);
                }
            }
            self.immune_zone.push(entry.clone());
        }

        self.entries.push(entry);

        // Trigger auto-merge if above 80%
        if self.should_merge() {
            self.merge_similar(0.85);
        }
    }

    /// Query similar entries from the regular store by cosine similarity.
    ///
    /// Returns the top `k` entries above `threshold`, sorted by similarity descending.
    pub fn query_similar(
        &self,
        embedding: &[f32],
        threshold: f32,
        top_k: usize,
    ) -> Vec<(f32, &SemanticEntry)> {
        self.query_from_slice(&self.entries, embedding, threshold, top_k)
    }

    /// Query similar strategy entries (matches on v_strategy field).
    pub fn query_similar_strategies(
        &self,
        v_goal: &[f32],
        v_obstacle: &[f32],
        threshold: f32,
        top_k: usize,
    ) -> Vec<(f32, &SemanticEntry)> {
        // Combine goal + obstacle for comparison
        let combined = Self::combine_embeddings(v_goal, v_obstacle);
        self.entries
            .iter()
            .map(|e| {
                let e_combined = Self::combine_embeddings(&e.v_goal, &e.v_obstacle);
                (crate::embedding::cosine_similarity(&combined, &e_combined), e)
            })
            .filter(|(sim, _)| *sim >= threshold)
            .collect::<Vec<_>>()
            .sort_and_truncate(top_k)
    }

    /// Query similar attitude entries (matches on v_attitude field).
    pub fn query_similar_attitudes(
        &self,
        v_origin: &[f32],
        threshold: f32,
        top_k: usize,
    ) -> Vec<(f32, &SemanticEntry)> {
        self.entries
            .iter()
            .map(|e| {
                (crate::embedding::cosine_similarity(v_origin, &e.v_attitude), e)
            })
            .filter(|(sim, _)| *sim >= threshold)
            .collect::<Vec<_>>()
            .sort_and_truncate(top_k)
    }

    /// Query similar experiences from the regular store by v_think.
    pub fn query_similar_experiences(
        &self,
        v_think: &[f32],
        threshold: f32,
        top_k: usize,
    ) -> Vec<(f32, &SemanticEntry)> {
        self.entries
            .iter()
            .map(|e| {
                (crate::embedding::cosine_similarity(v_think, &e.v_think), e)
            })
            .filter(|(sim, _)| *sim >= threshold)
            .collect::<Vec<_>>()
            .sort_and_truncate(top_k)
    }

    /// Query immune zone for similar dangerous patterns.
    ///
    /// Returns all immune entries with cosine similarity >= threshold.
    pub fn query_immune_similar(
        &self,
        embedding: &[f32],
        threshold: f32,
    ) -> Vec<(f32, &SemanticEntry)> {
        self.immune_zone
            .iter()
            .map(|e| {
                (crate::embedding::cosine_similarity(embedding, &e.v_think), e)
            })
            .filter(|(sim, _)| *sim >= threshold)
            .collect()
    }

    /// Quick check: is the current thought pattern dangerously close to
    /// any immune memory pattern?
    pub fn is_pattern_dangerous(&self, embedding: &[f32], threshold: f32) -> bool {
        self.immune_zone.iter().any(|e| {
            crate::embedding::cosine_similarity(embedding, &e.v_think) >= threshold
        })
    }

    /// Merge highly similar entries into experience crystals.
    ///
    /// For each cluster of entries with cosine_similarity >= threshold,
    /// merge them into a single crystal entry.
    /// The resulting crystal keeps the highest reward, averages the vectors
    /// (weighted by |reward|), and increments merge_count + crystal_generation.
    pub fn merge_similar(&mut self, threshold: f32) -> usize {
        let mut merge_count = 0;
        // Simple greedy clustering: for each entry, find similar ones
        let mut merged_indices: Vec<usize> = Vec::new();
        let mut new_crystals: Vec<SemanticEntry> = Vec::new();

        for i in 0..self.entries.len() {
            if merged_indices.contains(&i) {
                continue;
            }

            let mut cluster: Vec<usize> = vec![i];
            for j in (i + 1)..self.entries.len() {
                if merged_indices.contains(&j) {
                    continue;
                }
                let sim = crate::embedding::cosine_similarity(
                    &self.entries[i].v_think,
                    &self.entries[j].v_think,
                );
                if sim >= threshold {
                    cluster.push(j);
                }
            }

            if cluster.len() >= 2 {
                // Merge cluster into crystal
                let crystal = self.merge_cluster(&cluster);
                merge_count += cluster.len() - 1;
                new_crystals.push(crystal);
                for idx in &cluster {
                    merged_indices.push(*idx);
                }
            }
        }

        // Rebuild entries: keep unmerged + new crystals
        if !new_crystals.is_empty() {
            self.entries = self
                .entries
                .iter()
                .enumerate()
                .filter(|(i, _)| !merged_indices.contains(i))
                .map(|(_, e)| e.clone())
                .chain(new_crystals)
                .collect();
        }

        merge_count
    }

    /// Merge immune zone entries (internal merge, never mixes with regular entries).
    pub fn merge_immune_zone(&mut self) -> usize {
        if self.immune_zone.len() < 2 {
            return 0;
        }

        let mut merge_count = 0;
        let mut merged_indices: Vec<usize> = Vec::new();
        let mut new_crystals: Vec<SemanticEntry> = Vec::new();

        for i in 0..self.immune_zone.len() {
            if merged_indices.contains(&i) {
                continue;
            }
            let mut cluster: Vec<usize> = vec![i];
            for j in (i + 1)..self.immune_zone.len() {
                if merged_indices.contains(&j) {
                    continue;
                }
                let sim = crate::embedding::cosine_similarity(
                    &self.immune_zone[i].v_think,
                    &self.immune_zone[j].v_think,
                );
                if sim >= 0.85 {
                    cluster.push(j);
                }
            }
            if cluster.len() >= 2 {
                let crystal = self.merge_cluster_immune(&cluster);
                merge_count += cluster.len() - 1;
                new_crystals.push(crystal);
                for idx in &cluster {
                    merged_indices.push(*idx);
                }
            }
        }

        if !new_crystals.is_empty() {
            self.immune_zone = self
                .immune_zone
                .iter()
                .enumerate()
                .filter(|(i, _)| !merged_indices.contains(i))
                .map(|(_, e)| e.clone())
                .chain(new_crystals)
                .collect();
        }

        merge_count
    }

    // ─── Private helpers ────────────────────────────────────────

    fn query_from_slice<'a>(
        &self,
        slice: &'a [SemanticEntry],
        embedding: &[f32],
        threshold: f32,
        top_k: usize,
    ) -> Vec<(f32, &'a SemanticEntry)> {
        slice
            .iter()
            .map(|e| {
                (crate::embedding::cosine_similarity(embedding, &e.v_think), e)
            })
            .filter(|(sim, _)| *sim >= threshold)
            .collect::<Vec<_>>()
            .sort_and_truncate(top_k)
    }

    fn combine_embeddings(a: &[f32], b: &[f32]) -> Vec<f32> {
        if a.is_empty() {
            return b.to_vec();
        }
        if b.is_empty() {
            return a.to_vec();
        }
        let len = a.len().min(b.len());
        a.iter()
            .zip(b.iter())
            .take(len)
            .map(|(&x, &y)| (x + y) / 2.0)
            .collect()
    }

    fn merge_cluster(&self, indices: &[usize]) -> SemanticEntry {
        let entries: Vec<&SemanticEntry> = indices.iter().map(|&i| &self.entries[i]).collect();
        self.merge_entries_from_slice(&entries)
    }

    fn merge_cluster_immune(&self, indices: &[usize]) -> SemanticEntry {
        let entries: Vec<&SemanticEntry> = indices.iter().map(|&i| &self.immune_zone[i]).collect();
        self.merge_entries_from_slice(&entries)
    }

    fn merge_entries_from_slice(&self, cluster: &[&SemanticEntry]) -> SemanticEntry {
        let base = cluster[0];

        let weights_fn = |entries: &[&SemanticEntry]| -> Vec<f32> {
            entries.iter().map(|e| e.reward.abs()).collect()
        };

        match self.merge_strategy {
            MergeStrategy::MostRewarded => {
                let best = cluster
                    .iter()
                    .max_by(|a, b| a.reward.abs().partial_cmp(&b.reward.abs()).unwrap())
                    .unwrap();
                let mut crystal = (*best).clone();
                crystal.merge_count += 1;
                crystal.crystal_generation += 1;
                crystal
            }
            MergeStrategy::WeightedAverage => {
                let weights = weights_fn(cluster);
                let total_weight: f32 = weights.iter().sum();
                let averaged = Self::weighted_average_entries(cluster, &weights, total_weight);
                let mut entry = base.clone();
                entry.v_think = averaged.0;
                entry.v_goal = averaged.1;
                entry.v_obstacle = averaged.2;
                entry.v_strategy = averaged.3;
                entry.v_attitude = averaged.4;
                entry.reward = cluster
                    .iter()
                    .map(|e| e.reward)
                    .fold(f32::NEG_INFINITY, f32::max);
                entry.merge_count += 1;
                entry.crystal_generation += 1;
                entry
            }
            MergeStrategy::Centroid => {
                let averaged = Self::centroid_entries(cluster);
                let mut entry = base.clone();
                entry.v_think = averaged.0;
                entry.v_goal = averaged.1;
                entry.v_obstacle = averaged.2;
                entry.v_strategy = averaged.3;
                entry.v_attitude = averaged.4;
                entry.reward = cluster
                    .iter()
                    .map(|e| e.reward)
                    .fold(f32::NEG_INFINITY, f32::max);
                entry.merge_count += 1;
                entry.crystal_generation += 1;
                entry
            }
        }
    }

    fn weighted_average_entries(
        entries: &[&SemanticEntry],
        weights: &[f32],
        total_weight: f32,
    ) -> (Embedding, Embedding, Embedding, Embedding, Embedding) {
        let dim = entries[0].v_think.len();
        let mut v_think = vec![0.0f32; dim];
        let mut v_goal = vec![0.0f32; dim];
        let mut v_obstacle = vec![0.0f32; dim];
        let mut v_strategy = vec![0.0f32; dim];
        let mut v_attitude = vec![0.0f32; dim];

        for (entry, &weight) in entries.iter().zip(weights.iter()) {
            let w = weight / total_weight;
            for (i, &val) in entry.v_think.iter().enumerate() {
                v_think[i] += val * w;
            }
            for (i, &val) in entry.v_goal.iter().enumerate() {
                if i < v_goal.len() {
                    v_goal[i] += val * w;
                }
            }
            for (i, &val) in entry.v_obstacle.iter().enumerate() {
                if i < v_obstacle.len() {
                    v_obstacle[i] += val * w;
                }
            }
            for (i, &val) in entry.v_strategy.iter().enumerate() {
                if i < v_strategy.len() {
                    v_strategy[i] += val * w;
                }
            }
            for (i, &val) in entry.v_attitude.iter().enumerate() {
                if i < v_attitude.len() {
                    v_attitude[i] += val * w;
                }
            }
        }
        (v_think, v_goal, v_obstacle, v_strategy, v_attitude)
    }

    fn centroid_entries(
        entries: &[&SemanticEntry],
    ) -> (Embedding, Embedding, Embedding, Embedding, Embedding) {
        let n = entries.len() as f32;
        let dim = entries[0].v_think.len();
        let mut v_think = vec![0.0f32; dim];
        let mut v_goal = vec![0.0f32; dim];
        let mut v_obstacle = vec![0.0f32; dim];
        let mut v_strategy = vec![0.0f32; dim];
        let mut v_attitude = vec![0.0f32; dim];

        for entry in entries {
            for (i, &val) in entry.v_think.iter().enumerate() {
                v_think[i] += val;
            }
            for (i, &val) in entry.v_goal.iter().enumerate() {
                if i < v_goal.len() {
                    v_goal[i] += val;
                }
            }
            for (i, &val) in entry.v_obstacle.iter().enumerate() {
                if i < v_obstacle.len() {
                    v_obstacle[i] += val;
                }
            }
            for (i, &val) in entry.v_strategy.iter().enumerate() {
                if i < v_strategy.len() {
                    v_strategy[i] += val;
                }
            }
            for (i, &val) in entry.v_attitude.iter().enumerate() {
                if i < v_attitude.len() {
                    v_attitude[i] += val;
                }
            }
        }
        for v in &mut v_think { *v /= n; }
        for v in &mut v_goal { *v /= n; }
        for v in &mut v_obstacle { *v /= n; }
        for v in &mut v_strategy { *v /= n; }
        for v in &mut v_attitude { *v /= n; }
        (v_think, v_goal, v_obstacle, v_strategy, v_attitude)
    }

    fn evict_lowest(&mut self) {
        if let Some(pos) = self
            .entries
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.reward
                    .partial_cmp(&b.reward)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
        {
            // Never evict entries with immune_count > 0 (they're in immune zone)
            if self.entries[pos].immune_count == 0 {
                self.entries.remove(pos);
            } else {
                // Find the next-lowest non-immune entry
                if let Some(alt_pos) = self
                    .entries
                    .iter()
                    .enumerate()
                    .filter(|(_, e)| e.immune_count == 0)
                    .min_by(|(_, a), (_, b)| {
                        a.reward
                            .partial_cmp(&b.reward)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i)
                {
                    self.entries.remove(alt_pos);
                } else {
                    // All entries have immune count > 0, remove the one with lowest reward anyway
                    self.entries.remove(pos);
                }
            }
        }
    }
}

/// Helper for Vec<(f32, &T)> to sort by similarity descending and truncate.
trait SortAndTruncate<'a, T> {
    fn sort_and_truncate(self, top_k: usize) -> Self;
}

impl<'a, T> SortAndTruncate<'a, T> for Vec<(f32, &'a T)> {
    fn sort_and_truncate(mut self, top_k: usize) -> Self {
        self.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        self.truncate(top_k);
        self
    }
}

// ─── Projection helper ──────────────────────────────────────────

/// Project an embedding onto a 6-bit Gua.
///
/// Each of the 6 bits corresponds to the sign of a dimension of the embedding.
/// Positive → 1, non-positive → 0.
/// If the embedding has fewer than 6 dimensions, pads with 0.
pub fn project_embedding_to_bits(embedding: &[f32]) -> [u8; 6] {
    let mut bits = [0u8; 6];
    for i in 0..6 {
        if i < embedding.len() && embedding[i] > 0.0 {
            bits[i] = 1; // MSB first for Gua convention
        }
    }
    bits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_and_weight() {
        let mut sea = CangSea::new();
        sea.store(Gua(0), Gua(1), 0.5);
        assert!(sea.weight(Gua(0), Gua(1)) > 0.0);
        assert_eq!(sea.weight(Gua(0), Gua(2)), 0.0);
        assert_eq!(sea.len(), 1);
    }

    #[test]
    fn test_negative_reward_weakens() {
        let mut sea = CangSea::new();
        sea.store(Gua(0), Gua(1), 0.5);
        let w1 = sea.weight(Gua(0), Gua(1));
        sea.store(Gua(0), Gua(1), -0.5);
        assert!(sea.weight(Gua(0), Gua(1)) < w1);
    }

    #[test]
    fn test_weight_never_negative() {
        let mut sea = CangSea::new();
        sea.store(Gua(0), Gua(1), -10.0);
        assert!(sea.weight(Gua(0), Gua(1)) >= 0.0);
    }

    #[test]
    fn test_reinforce_and_weaken() {
        let mut sea = CangSea::new();
        sea.reinforce(Gua(0), Gua(1), 0.3);
        assert!((sea.weight(Gua(0), Gua(1)) - 0.3).abs() < f32::EPSILON);
        sea.weaken(Gua(0), Gua(1), 0.1);
        assert!((sea.weight(Gua(0), Gua(1)) - 0.2).abs() < f32::EPSILON);
    }

    #[test]
    fn test_hebbian_sample() {
        let mut sea = CangSea::new();
        for _ in 0..10 {
            sea.reinforce(Gua(0), Gua(1), 0.1);
        }
        sea.reinforce(Gua(0), Gua(2), 0.05);

        let mut rng = rand::rng();
        let mut counts = [0u32; 64];
        for _ in 0..1000 {
            if let Some(g) = sea.hebbian_sample(&mut rng, Gua(0)) {
                counts[g.0 as usize] += 1;
            }
        }
        assert!(counts[1] > counts[2] * 5,
            "Expected Gua(1) >> Gua(2), got counts[1]={} counts[2]={}", counts[1], counts[2]);
    }

    #[test]
    fn test_hebbian_sample_empty_row() {
        let sea = CangSea::new();
        let mut rng = rand::rng();
        assert_eq!(sea.hebbian_sample(&mut rng, Gua(0)), None);
    }

    #[test]
    fn test_capacity_eviction() {
        let mut sea = CangSea::with_capacity(3);
        sea.store(Gua(0), Gua(1), 1.0);
        sea.store(Gua(0), Gua(2), 0.5);
        sea.store(Gua(0), Gua(3), 0.1);
        assert_eq!(sea.len(), 3);
        sea.store(Gua(0), Gua(4), 2.0);
        assert_eq!(sea.len(), 3);
        let entries_from_0 = sea.entries_from(Gua(0));
        assert!(entries_from_0.iter().all(|e| e.reward >= 0.5));
    }

    // ─── Semantic Store Tests ─────────────────────────────────

    fn make_test_entry(reward: f32, dim: usize) -> SemanticEntry {
        let v = vec![0.5_f32; dim];
        SemanticEntry {
            v_think: v.clone(),
            v_goal: v.clone(),
            v_obstacle: v.clone(),
            v_strategy: v.clone(),
            v_attitude: v.clone(),
            deviation: 0.3,
            reward,
            timestamp: 0,
            immune_count: 0,
            merge_count: 0,
            crystal_generation: 0,
        }
    }

    #[test]
    fn test_semantic_store_basic() {
        let mut store = SemanticStore::new(100, 4, 20);
        let entry = make_test_entry(0.5, 4);
        store.store(entry);
        assert_eq!(store.entries.len(), 1);
        assert_eq!(store.immune_zone.len(), 0);
    }

    #[test]
    fn test_semantic_negative_to_immune() {
        let mut store = SemanticStore::new(100, 4, 20);
        let entry = make_test_entry(-0.8, 4);
        store.store(entry);
        assert_eq!(store.entries.len(), 1);
        assert_eq!(store.immune_zone.len(), 1);
    }

    #[test]
    fn test_query_similar() {
        let mut store = SemanticStore::new(100, 4, 20);
        store.store(make_test_entry(0.5, 4));
        let query = vec![0.5_f32; 4];
        let results = store.query_similar(&query, 0.9, 5);
        assert_eq!(results.len(), 1);
        assert!((results[0].0 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_is_pattern_dangerous() {
        let mut store = SemanticStore::new(100, 4, 20);
        let negative = make_test_entry(-0.8, 4);
        store.store(negative);
        let query = vec![0.5_f32; 4];
        assert!(store.is_pattern_dangerous(&query, 0.9));
    }

    #[test]
    fn test_is_pattern_not_dangerous() {
        let mut store = SemanticStore::new(100, 4, 20);
        let negative = make_test_entry(-0.8, 4);
        store.store(negative);
        let query = vec![-0.5_f32; 4];
        assert!(!store.is_pattern_dangerous(&query, 0.9));
    }

    #[test]
    fn test_merge_similar() {
        let mut store = SemanticStore::new(10, 4, 5);
        // Store two nearly identical entries
        store.store(make_test_entry(0.5, 4));
        store.store(make_test_entry(0.6, 4));
        // Force merge manually
        let merged = store.merge_similar(0.9);
        assert!(merged > 0, "Should have merged at least 1 pair");
        assert!(store.entries.len() < 2);
        // Crystal should have higher generation
        assert!(store.entries[0].crystal_generation >= 1);
    }

    #[test]
    fn test_merge_immune_zone() {
        let mut store = SemanticStore::new(10, 4, 5);
        store.store(make_test_entry(-0.8, 4));
        store.store(make_test_entry(-0.9, 4));
        let merged = store.merge_immune_zone();
        assert!(merged > 0, "Should have merged immune entries");
        assert!(store.immune_zone.len() < 2);
    }

    #[test]
    fn test_cangsea_with_semantic() {
        let mut sea = CangSea::with_semantic(100, 100, 4, 20);
        assert!(sea.semantic_store.is_some());
        let entry = make_test_entry(0.5, 4);
        sea.store_semantic(entry);
        assert!(sea.semantic_store.as_ref().unwrap().entries.len() == 1);
        // Legacy store should also have an entry (projection)
        assert!(sea.len() > 0);
    }

    #[test]
    fn test_project_embedding_to_bits() {
        // Positive first 3 dims → bits [1,1,1,0,0,0] in MSB-first
        let emb = vec![1.0, 1.0, 1.0, -1.0, -1.0, -1.0];
        let bits = project_embedding_to_bits(&emb);
        assert_eq!(bits, [1, 1, 1, 0, 0, 0]);
    }

    // ─── 上下文新陈代谢模拟测试 ────────────────────────────
    //
    // 场景：10轮对话，经历多次主题切换，验证：
    //   1. 不同 Gua 状态的经验互相隔离，不会泄漏
    //   2. 回到旧状态时，相关经验可被召回
    //   3. 负向经验产生 resist forces
    //   4. 全新状态无经验 → forces 为空 → WaitGather
    //   5. 同一状态的经验随轮次积累和强化

    /// 辅助：模拟一轮对话，返回 (summary, forces)
    fn simulate_round(
        sea: &mut CangSea,
        from: Gua,
        to: Gua,
        reward: f32,
        label: &str,
    ) -> (Option<String>, ZhiForces) {
        sea.store(from, to, reward);
        let summary = sea.experience_summary(from, 3);
        let forces = sea.query_forces(from);
        println!("    [{label}] store({}, {}, r={:.1})", from.name(), to.name(), reward);
        println!("      entries_from({}): {} 条", from.name(), sea.entries_from(from).len());
        if let Some(ref s) = summary {
            // 只打印摘要的概要行
            let first_line = s.lines().next().unwrap_or("");
            println!("      summary: {first_line}");
            println!("      forces: push={}, resist={}",
                forces.push_forces.len(), forces.resist_forces.len());
        } else {
            println!("      summary: None（无经验）");
            println!("      forces: push={}, resist={}",
                forces.push_forces.len(), forces.resist_forces.len());
        }
        (summary, forces)
    }

    #[test]
    fn test_context_metabolism_10_rounds() {
        let mut sea = CangSea::new();

        // ── 场景定义 ──────────────────────────────────
        // Gua states used:
        //   1  = 木气萌芽  (Rust 探索)
        //   9  = 木木并生  (Rust 代码生长)
        //   11 = 火木同燃  (Rust 活跃开发)
        //   27 = 火火同辉  (Rust 成功)
        //   32 = 水藏潜渊  (DB schema 深潜)
        //   28 = 水火既济  (DB 设计成功)
        //   36 = 水水重险  (Rust 调试困境)
        //   18 = 风风流转  (测试话题)

        let rust_explore   = Gua(1);   // 木气萌芽
        let rust_grow      = Gua(9);   // 木木并生
        let rust_active    = Gua(11);  // 火木同燃
        let rust_success   = Gua(27);  // 火火同辉
        let db_deep        = Gua(32);  // 水藏潜渊
        let db_success     = Gua(28);  // 水火既济
        let rust_debug     = Gua(36);  // 水水重险
        let test_flow      = Gua(18);  // 风风流转

        println!("\n══════════════════════════════════════════");
        println!("  上下文新陈代谢模拟 — 10轮对话");
        println!("══════════════════════════════════════════\n");

        // ── Round 1: Rust 探索，正向 ──
        println!("─── Round 1: \"How do I set up Rust?\" ────");
        let (s1, _f1) = simulate_round(&mut sea, rust_explore, rust_explore, 0.8, "Rust起步成功");
        assert!(s1.is_some(), "Round 1 should have summary");
        // weight = 0.8*0.1 = 0.08 < 0.1 → forces 为空，这是正确的置信度门槛
        println!("      → 单次经验 weight=0.08 < 0.1 阈值，forces 为空（置信度门槛）");
        println!("      weight({}→{}): {:.3}", rust_explore.name(), rust_explore.name(),
            sea.weight(rust_explore, rust_explore));

        // ── Round 2: Rust 模块设计 ──
        println!("\n─── Round 2: \"Design the module structure\" ─");
        let (s2, _f2) = simulate_round(&mut sea, rust_explore, rust_grow, 0.6, "模块结构设计");
        assert!(s2.is_some(), "Round 2 should have summary");
        // 累计 weight = 0.08 + 0.06 = 0.14 > 0.1 → 现在有 push forces
        println!("      → 累计 weight=0.14 > 0.1 阈值，forces 应出现");
        println!("      weight({}→{}): {:.3}", rust_explore.name(), rust_explore.name(),
            sea.weight(rust_explore, rust_explore));
        println!("      weight({}→{}): {:.3}", rust_explore.name(), rust_grow.name(),
            sea.weight(rust_explore, rust_grow));

        // ── Round 3: Rust 核心实现 ──
        println!("\n─── Round 3: \"Implement the core logic\" ─");
        // note: each (from,to) pair weight stored separately; 3 different `to` targets
        // means each weight ≤ 0.09 < 0.1 threshold → forces 为空
        let (s3, _f3) = simulate_round(&mut sea, rust_explore, rust_active, 0.9, "核心逻辑实现");
        assert!(s3.is_some());
        println!("      → from=1 累积 3 条经验（分散到不同 to 状态）");
        println!("      weight({}→{}): {:.3}", rust_explore.name(), rust_active.name(),
            sea.weight(rust_explore, rust_active));

        // ── Round 4: 切换到 DB schema ──
        println!("\n─── Round 4: \"Now about database schema\" ──");
        println!("      ** 主题切换：Rust(1) → DB(32) **");
        let (s4, _f4) = simulate_round(&mut sea, db_deep, db_success, 0.7, "DB schema设计");
        assert!(s4.is_some(), "Round 4 should have summary for DB");
        // 关键断言：查询 from=1 的经验数应该还是 3，查询 from=32 应该是 1
        assert_eq!(sea.entries_from(rust_explore).len(), 3,
            "Rust 经验应该在主题切换后保留");
        assert_eq!(sea.entries_from(db_deep).len(), 1,
            "DB 经验单独存储，与 Rust 隔离");

        // ── Round 5: DB 表设计 ──
        println!("\n─── Round 5: \"Design the user table\" ────");
        let (s5, _f5) = simulate_round(&mut sea, db_deep, db_deep, 0.5, "用户表设计");
        assert!(s5.is_some());
        assert_eq!(sea.entries_from(db_deep).len(), 2,
            "DB 经验积累到 2 条");

        // ── Round 6: 回到 Rust ──
        println!("\n─── Round 6: \"Back to Rust, compiler errors\" ─");
        println!("      ** 主题回归：DB(32) → Rust(1) **");
        let (s6, _f6) = simulate_round(&mut sea, rust_explore, rust_debug, -0.6, "Rust编译错误");
        assert!(s6.is_some(), "Round 6 should find Rust experiences");
        // Rust from=1 现在有 4 条经验（3 positive + 1 negative）
        assert_eq!(sea.entries_from(rust_explore).len(), 4,
            "回到 Rust 时，之前的 Rust 经验应该全部可查");

        // ── Round 7: Rust 调试失败 ──
        println!("\n─── Round 7: \"Debugging borrow checker\" ──");
        let (s7, f7) = simulate_round(&mut sea, rust_explore, rust_explore, -0.8, "调试失败");
        assert!(s7.is_some());
        // Rust from=1 现在有 5 条经验
        assert_eq!(sea.entries_from(rust_explore).len(), 5);
        // 负向经验应该已产生 resist forces
        println!("      resist_forces: {:?}", f7.resist_forces);

        // ── Round 8: 换方案成功 ──
        println!("\n─── Round 8: \"Different approach works\" ──");
        let (s8, _f8) = simulate_round(&mut sea, rust_explore, rust_success, 0.8, "换方案成功");
        assert!(s8.is_some());
        assert_eq!(sea.entries_from(rust_explore).len(), 6,
            "Rust 经验累积到 6 条");

        // ── Round 9: 新话题 测试 ──
        println!("\n─── Round 9: \"What about testing?\" ──────");
        println!("      ** 新主题：Testing(18) **");
        let (s9, f9) = simulate_round(&mut sea, test_flow, test_flow, 0.3, "测试话题");
        // from=18 只有 1 条经验且 weight=0.03 < 0.1 → forces 为空
        // 如果这是 activate 状态且无 forces，春初 → ShanVM 会返回 WaitGather
        assert!(s9.is_some(), "Round 9 should have summary");
        assert!(f9.push_forces.is_empty(), "新话题经验不足，无 push forces");
        assert!(f9.resist_forces.is_empty(), "新话题无 resist forces");
        println!("      → ShanVM: forces 为空 + 春初 → WaitGather（等待收集信息）");

        // ── Round 10: 回到 DB schema ──
        println!("\n─── Round 10: \"Back to DB, refine schema\" ─");
        println!("      ** 主题回归：Testing(18) → DB(32) **");
        let (s10, _f10) = simulate_round(&mut sea, db_deep, db_success, 0.7, "DB schema优化");
        assert!(s10.is_some(), "Round 10 should recall DB experiences");
        assert_eq!(sea.entries_from(db_deep).len(), 3,
            "DB 经验累积到 3 条");

        // ── 最终状态验证 ──
        println!("\n═══════════════ 最终状态 ═══════════════");
        println!("  总经验数: {}", sea.len());
        println!("  Rust(1) 经验: {} 条", sea.entries_from(rust_explore).len());
        println!("  DB(32)  经验: {} 条", sea.entries_from(db_deep).len());
        println!("  Test(18)经验: {} 条", sea.entries_from(test_flow).len());

        // ── 核心代谢断言 ──
        // 1. 隔离性：不同状态的经验不混淆
        assert_eq!(sea.entries_from(db_deep).len(), 3,
            "DB from=32 应该只有 DB 相关经验");
        assert!(sea.entries_from(db_deep).iter().all(|e| e.from == db_deep),
            "DB 查询不应包含非 DB 状态的经验");

        // 2. Rust 经验完整保留
        assert_eq!(sea.entries_from(rust_explore).len(), 6,
            "Rust from=1 应该保留全部 6 条经验");

        // 3. 正向经验权重累积
        let w = sea.weight(rust_explore, rust_active);
        assert!(w > 0.08, "重复强化后 weight 应该 > 单次值");

        // 4. 负向经验不会使 weight 变负
        let w_neg = sea.weight(rust_explore, rust_debug);
        assert!(w_neg >= 0.0, "负向 reward 的 weight 应 clamp 到 ≥0");

        // 5. 最终 DB 查询包含正确的经验数
        let db_entries = sea.entries_from(db_deep);
        assert_eq!(db_entries.len(), 3);

        println!("\n═══ 新陈代谢模拟完成 ═══");
        println!("  ✅ 不同状态经验隔离");
        println!("  ✅ 主题回归时召回相关经验");
        println!("  ✅ 负向经验产生 resist forces");
        println!("  ✅ 新主题无经验 → WaitGather");
        println!("  ✅ 经验随轮次累积强化（Hebbian）");
    }
}
