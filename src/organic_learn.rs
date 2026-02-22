//! Organic Learning — CLIP embeddings learn into signed holographic containers.
//!
//! Instead of binary SimHash (1-bit per hyperplane → Hamming distance),
//! this module uses **Signed(7) quantized projections** into an **OrganicWAL**
//! with X-Trans spatial sampling. The signed base preserves magnitude,
//! enabling destructive interference (Auslöschung) that separates stored
//! from unstored concepts.
//!
//! ## Architecture
//!
//! ```text
//! CLIP ViT-B/32 (512-dim float32)
//!        │
//!        ▼
//!  quantize_embedding()
//!  ├─ random projection to D dimensions
//!  └─ quantize to Signed(7): values in [-3, +3]
//!        │
//!        ▼
//!  OrganicWAL (X-Trans pattern, C channels)
//!  ├─ register_concept() for each class template
//!  ├─ write() / write_plastic() per training sample
//!  └─ surgical_extract() to recover learned coefficients
//!        │
//!        ▼
//!  Recognition: read concept similarity from container
//! ```
//!
//! ## Why Signed + Organic
//!
//! - **Signed(7)** range [-3..+3] enables destructive interference:
//!   unstored concepts cancel to ~0, stored concepts accumulate.
//! - **Organic absorption**: receptivity prevents saturation. Full positions
//!   reject new writes, keeping the container balanced.
//! - **Gram-Schmidt orthogonalization**: each write removes projections
//!   onto existing concepts before writing — no cross-talk.
//! - **Plasticity**: new concepts absorb quickly, stable ones resist change.

use rustynum_oracle::{
    XTransPattern, OrganicWAL, PlasticityTracker, AbsorptionTracker,
    organic_flush, OrganicFlushAction,
};

// ---------------------------------------------------------------------------
// Quantized Projection: float32 embedding → Signed(7) i8 template
// ---------------------------------------------------------------------------

/// Project a float32 CLIP embedding into a Signed(7) i8 vector of length D.
///
/// Each output position is a random projection (dot product with a random
/// hyperplane), quantized to [-3, +3]. The seed + position index determine
/// the hyperplane deterministically.
///
/// This is like SimHash but keeps 3 bits of magnitude instead of 1 bit of sign.
pub fn quantize_embedding(embedding: &[f32], d: usize, seed: u64) -> Vec<i8> {
    let dim = embedding.len();
    let mut result = vec![0i8; d];

    for j in 0..d {
        let mut dot = 0.0f64;
        let mut state = seed
            .wrapping_add(j as u64)
            .wrapping_mul(0x9e3779b97f4a7c15);

        for k in 0..dim {
            state ^= state >> 30;
            state = state.wrapping_mul(0xbf58476d1ce4e5b9);
            state ^= state >> 27;
            state = state.wrapping_mul(0x94d049bb133111eb);
            state ^= state >> 31;

            // Uniform [-1, +1] random component from top bits.
            let u = (state >> 33) as f64 / ((1u64 << 31) as f64) * 2.0 - 1.0;
            dot += embedding[k] as f64 * u;
        }

        // Quantize to Signed(7) range: [-3, +3].
        //
        // For uniform [-1,1] random projections against an embedding:
        //   Var(dot) = ||emb||² / 3, so σ = ||emb|| / √3
        // We normalize by σ and scale so ±2σ maps to ±3.
        // But we don't know ||emb|| per-position, so we use 1/√3
        // (correct for L2-normalized embeddings, conservative otherwise).
        let scaled = (dot * 3.0_f64.sqrt() * 1.5).round().clamp(-3.0, 3.0) as i8;
        result[j] = scaled;
    }

    result
}

/// Quantize a batch of embeddings with the same seed.
pub fn batch_quantize(embeddings: &[Vec<f32>], d: usize, seed: u64) -> Vec<Vec<i8>> {
    embeddings.iter().map(|emb| quantize_embedding(emb, d, seed)).collect()
}

// ---------------------------------------------------------------------------
// Concept Learner: accumulates per-class holographic representations
// ---------------------------------------------------------------------------

/// A concept class that accumulates training samples.
#[derive(Clone, Debug)]
pub struct ConceptClass {
    /// Human-readable name.
    pub name: String,
    /// Class identifier.
    pub class_id: String,
    /// Index in the OrganicWAL.
    pub wal_index: usize,
    /// Number of training samples written.
    pub sample_count: usize,
    /// Running absorption history.
    pub mean_absorption: f32,
}

/// Result of a learning step.
#[derive(Clone, Debug)]
pub struct LearnResult {
    pub concept_name: String,
    pub absorption: f32,
    pub projections_onto_others: Vec<(String, f32)>,
    pub coefficient_after: f32,
}

/// Result of a recognition query.
#[derive(Clone, Debug)]
pub struct RecognitionResult {
    pub concept_name: String,
    pub class_id: String,
    pub similarity: f32,
    pub amplitude: f32,
    pub coefficient: f32,
}

/// Organic concept learner.
///
/// Wraps an OrganicWAL with class metadata, plasticity tracking, and
/// absorption monitoring. Each training sample is written with organic
/// absorption and Gram-Schmidt orthogonalization.
pub struct ConceptLearner {
    /// The organic WAL engine.
    pub wal: OrganicWAL,
    /// Per-concept plasticity tracker.
    pub plasticity: PlasticityTracker,
    /// Absorption tracker (overexposure monitoring).
    pub absorption: AbsorptionTracker,
    /// Registered concept classes.
    pub classes: Vec<ConceptClass>,
    /// Container: the holographic storage.
    pub container: Vec<i8>,
    /// Container dimension.
    pub d: usize,
    /// Number of X-Trans channels.
    pub channels: usize,
    /// Projection seed (deterministic).
    pub seed: u64,
    /// Learning rate for cross-concept coefficient updates.
    pub learning_rate: f32,
    /// Whether to use plasticity modulation.
    pub use_plasticity: bool,
}

impl ConceptLearner {
    /// Create a new concept learner.
    ///
    /// - `d`: container dimension (e.g., 2048 or 4096)
    /// - `channels`: X-Trans channel count (e.g., 32 or 64)
    /// - `seed`: deterministic projection seed
    pub fn new(d: usize, channels: usize, seed: u64) -> Self {
        let pattern = XTransPattern::new(d, channels);
        let wal = OrganicWAL::new(pattern);
        Self {
            wal,
            plasticity: PlasticityTracker::new(0, 100),
            absorption: AbsorptionTracker::new(64),
            classes: Vec::new(),
            container: vec![0i8; d],
            d,
            channels,
            seed,
            learning_rate: 0.1,
            use_plasticity: true,
        }
    }

    /// Register a new concept class.
    ///
    /// The class template is generated from a representative embedding
    /// (e.g., the class centroid or first training sample).
    pub fn register_class(
        &mut self,
        class_id: &str,
        name: &str,
        representative_embedding: &[f32],
    ) -> usize {
        let template = quantize_embedding(representative_embedding, self.d, self.seed);
        let idx = self.classes.len();
        self.wal.register_concept(idx as u32, template);
        self.plasticity.add_concept();
        self.classes.push(ConceptClass {
            name: name.to_string(),
            class_id: class_id.to_string(),
            wal_index: idx,
            sample_count: 0,
            mean_absorption: 1.0,
        });
        idx
    }

    /// Register a class from a raw i8 template (for synthetic experiments).
    pub fn register_class_raw(
        &mut self,
        class_id: &str,
        name: &str,
        template: Vec<i8>,
    ) -> usize {
        let idx = self.classes.len();
        self.wal.register_concept(idx as u32, template);
        self.plasticity.add_concept();
        self.classes.push(ConceptClass {
            name: name.to_string(),
            class_id: class_id.to_string(),
            wal_index: idx,
            sample_count: 0,
            mean_absorption: 1.0,
        });
        idx
    }

    /// Learn: write a training sample for a concept class.
    ///
    /// The sample's template is projected and written into the container
    /// via organic absorption with Gram-Schmidt orthogonalization.
    pub fn learn(&mut self, class_index: usize, amplitude: f32) -> LearnResult {
        let result = if self.use_plasticity {
            self.wal.write_plastic(
                &mut self.container,
                class_index,
                amplitude,
                self.learning_rate,
                &mut self.plasticity,
            )
        } else {
            self.wal.write(
                &mut self.container,
                class_index,
                amplitude,
                self.learning_rate,
            )
        };

        self.absorption.record(result.absorption);
        self.classes[class_index].sample_count += 1;

        // Update running absorption
        let cls = &mut self.classes[class_index];
        cls.mean_absorption = 0.9 * cls.mean_absorption + 0.1 * result.absorption;

        // Build projection report
        let projections = result.projections.iter().enumerate()
            .filter(|(i, _)| *i != class_index)
            .map(|(i, &p)| (self.classes[i].name.clone(), p))
            .collect();

        LearnResult {
            concept_name: self.classes[class_index].name.clone(),
            absorption: result.absorption,
            projections_onto_others: projections,
            coefficient_after: self.wal.coefficients[class_index],
        }
    }

    /// Recognize: read all concepts from the container, ranked by similarity.
    pub fn recognize(&self) -> Vec<RecognitionResult> {
        let readbacks = self.wal.read_all(&self.container);
        let extracted = self.wal.surgical_extract(&self.container);

        let mut results: Vec<RecognitionResult> = readbacks.iter().enumerate()
            .map(|(i, &(_, sim, amp))| RecognitionResult {
                concept_name: self.classes[i].name.clone(),
                class_id: self.classes[i].class_id.clone(),
                similarity: sim,
                amplitude: amp,
                coefficient: if i < extracted.len() { extracted[i] } else { 0.0 },
            })
            .collect();

        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        results
    }

    /// Recognize a specific concept class.
    pub fn recognize_class(&self, class_index: usize) -> RecognitionResult {
        let (sim, amp) = self.wal.read(&self.container, class_index);
        let extracted = self.wal.surgical_extract(&self.container);
        RecognitionResult {
            concept_name: self.classes[class_index].name.clone(),
            class_id: self.classes[class_index].class_id.clone(),
            similarity: sim,
            amplitude: amp,
            coefficient: if class_index < extracted.len() { extracted[class_index] } else { 0.0 },
        }
    }

    /// Get the current flush action recommendation.
    pub fn flush_recommendation(&self) -> OrganicFlushAction {
        self.absorption.flush_decision()
    }

    /// Flush the container: extract coefficients, prune, re-materialize.
    pub fn flush(&mut self, keep_top_k: Option<usize>) {
        organic_flush(
            &mut self.wal,
            &mut self.container,
            &self.plasticity,
            keep_top_k,
        );
    }

    /// Number of registered classes.
    pub fn num_classes(&self) -> usize {
        self.classes.len()
    }

    /// Get all coefficients (running WAL values).
    pub fn coefficients(&self) -> &[f32] {
        &self.wal.coefficients
    }

    /// Get surgically extracted coefficients.
    pub fn extracted_coefficients(&self) -> Vec<f32> {
        self.wal.surgical_extract(&self.container)
    }

    /// Container saturation: 1.0 = all bytes at ±127, 0.0 = all zeros.
    pub fn saturation(&self) -> f32 {
        let total_abs: f64 = self.container.iter()
            .map(|&v| (v as f64).abs())
            .sum();
        (total_abs / (self.d as f64 * 127.0)) as f32
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rustynum_oracle::{Base, generate_templates};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn seeded_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    #[test]
    fn test_quantize_embedding_range() {
        let emb: Vec<f32> = (0..512).map(|i| (i as f32 * 0.01).sin()).collect();
        let template = quantize_embedding(&emb, 2048, 42);

        assert_eq!(template.len(), 2048);
        for &v in &template {
            assert!(v >= -3 && v <= 3, "value {} out of Signed(7) range", v);
        }
    }

    #[test]
    fn test_quantize_deterministic() {
        let emb: Vec<f32> = (0..512).map(|i| (i as f32 * 0.01).sin()).collect();
        let t1 = quantize_embedding(&emb, 2048, 42);
        let t2 = quantize_embedding(&emb, 2048, 42);
        assert_eq!(t1, t2, "quantization must be deterministic");
    }

    #[test]
    fn test_quantize_different_seeds_differ() {
        let emb: Vec<f32> = (0..512).map(|i| (i as f32 * 0.01).sin()).collect();
        let t1 = quantize_embedding(&emb, 2048, 42);
        let t2 = quantize_embedding(&emb, 2048, 99);
        assert_ne!(t1, t2, "different seeds should produce different templates");
    }

    #[test]
    fn test_similar_embeddings_similar_templates() {
        let a: Vec<f32> = (0..512).map(|i| (i as f32 * 0.01).sin()).collect();
        let mut b = a.clone();
        // Small perturbation
        for i in 0..10 {
            b[i] += 0.01;
        }

        let ta = quantize_embedding(&a, 2048, 42);
        let tb = quantize_embedding(&b, 2048, 42);

        // Count differences
        let diffs: usize = ta.iter().zip(tb.iter()).filter(|(a, b)| a != b).count();
        // Should be mostly the same (< 10% different for tiny perturbation)
        assert!(
            diffs < 200,
            "similar embeddings: {} / 2048 positions differ (expected < 200)",
            diffs,
        );
    }

    #[test]
    fn test_opposite_embeddings_produce_opposite_templates() {
        // Two anti-parallel embeddings (maximally different direction)
        let a: Vec<f32> = (0..512).map(|i| (i as f32 * 0.1).sin()).collect();
        let b: Vec<f32> = a.iter().map(|x| -x).collect();

        let ta = quantize_embedding(&a, 2048, 42);
        let tb = quantize_embedding(&b, 2048, 42);

        // Opposite embeddings: every non-zero position should flip sign
        let sign_flips: usize = ta.iter().zip(tb.iter())
            .filter(|&(va, vb)| *va != 0 && *vb != 0 && va.signum() != vb.signum())
            .count();
        let nonzero: usize = ta.iter().zip(tb.iter())
            .filter(|&(va, vb)| *va != 0 && *vb != 0)
            .count();

        assert!(
            nonzero > 100,
            "too few non-zero positions: {} (quantization too aggressive?)",
            nonzero,
        );
        // Nearly all non-zero positions should have opposite signs
        let flip_ratio = sign_flips as f64 / nonzero as f64;
        assert!(
            flip_ratio > 0.9,
            "opposite embeddings: only {:.1}% of non-zero positions flipped (expected > 90%)",
            flip_ratio * 100.0,
        );
    }

    // --- ConceptLearner Tests ---

    #[test]
    fn test_learner_register_and_learn() {
        let mut learner = ConceptLearner::new(2048, 16, 42);
        let emb: Vec<f32> = (0..512).map(|i| (i as f32 * 0.01).sin()).collect();

        let idx = learner.register_class("cat", "Cat", &emb);
        assert_eq!(idx, 0);
        assert_eq!(learner.num_classes(), 1);

        let result = learner.learn(0, 0.8);
        assert!(result.absorption > 0.9, "empty container absorption = {}", result.absorption);
        assert!(result.coefficient_after.abs() > 0.01, "coefficient should be non-zero");
    }

    #[test]
    fn test_learner_two_classes_recognition() {
        let mut learner = ConceptLearner::new(4096, 32, 42);

        // Two very different embeddings
        let emb_cat: Vec<f32> = (0..512).map(|i| (i as f32 * 0.01).sin()).collect();
        let emb_dog: Vec<f32> = (0..512).map(|i| (i as f32 * 0.03 + 1.5).cos()).collect();

        learner.register_class("cat", "Cat", &emb_cat);
        learner.register_class("dog", "Dog", &emb_dog);

        // Train: write each class multiple times
        for _ in 0..5 {
            learner.learn(0, 0.5);
        }
        for _ in 0..5 {
            learner.learn(1, 0.5);
        }

        // Recognize
        let results = learner.recognize();
        assert_eq!(results.len(), 2);

        // Both should have positive coefficients
        for r in &results {
            assert!(
                r.coefficient > 0.0,
                "concept '{}' has coefficient {:.4} (expected > 0)",
                r.concept_name, r.coefficient,
            );
        }
    }

    #[test]
    fn test_learner_unstored_concept_near_zero() {
        let mut learner = ConceptLearner::new(4096, 32, 42);
        let mut rng = seeded_rng();

        // Register 4 concepts with random templates
        let templates = generate_templates(4, 4096, Base::Signed(7), &mut rng);
        for (i, t) in templates.iter().enumerate() {
            learner.register_class_raw(
                &format!("c{}", i),
                &format!("Concept {}", i),
                t.clone(),
            );
        }

        // Only train concepts 0 and 1
        for _ in 0..5 {
            learner.learn(0, 0.6);
            learner.learn(1, 0.6);
        }

        // Extract coefficients
        let coeffs = learner.extracted_coefficients();

        // Concepts 0 and 1 should have significant coefficients
        assert!(
            coeffs[0].abs() > 0.1,
            "trained concept 0 coefficient = {:.4} (expected > 0.1)",
            coeffs[0],
        );
        assert!(
            coeffs[1].abs() > 0.1,
            "trained concept 1 coefficient = {:.4} (expected > 0.1)",
            coeffs[1],
        );

        // Concepts 2 and 3 should have near-zero coefficients (Auslöschung!)
        assert!(
            coeffs[2].abs() < 0.3,
            "unstored concept 2 coefficient = {:.4} (expected < 0.3)",
            coeffs[2],
        );
        assert!(
            coeffs[3].abs() < 0.3,
            "unstored concept 3 coefficient = {:.4} (expected < 0.3)",
            coeffs[3],
        );
    }

    #[test]
    fn test_learner_flush_and_rematerialize() {
        let mut learner = ConceptLearner::new(2048, 16, 42);
        let mut rng = seeded_rng();

        let templates = generate_templates(4, 2048, Base::Signed(7), &mut rng);
        for (i, t) in templates.iter().enumerate() {
            learner.register_class_raw(
                &format!("c{}", i),
                &format!("Concept {}", i),
                t.clone(),
            );
        }

        for _ in 0..3 {
            for i in 0..4 {
                learner.learn(i, 0.5);
            }
        }

        let coeffs_before = learner.extracted_coefficients();
        learner.flush(None);
        let coeffs_after = learner.extracted_coefficients();

        // Coefficients should survive the flush
        for i in 0..4 {
            let drift = (coeffs_before[i] - coeffs_after[i]).abs();
            assert!(
                drift < 0.5,
                "concept {} flush drift = {:.4} (expected < 0.5)",
                i, drift,
            );
        }
    }

    #[test]
    fn test_learner_saturation_increases() {
        let mut learner = ConceptLearner::new(2048, 16, 42);
        let mut rng = seeded_rng();

        let templates = generate_templates(8, 2048, Base::Signed(7), &mut rng);
        for (i, t) in templates.iter().enumerate() {
            learner.register_class_raw(
                &format!("c{}", i),
                &format!("Concept {}", i),
                t.clone(),
            );
        }

        let sat_before = learner.saturation();
        assert!(sat_before < 0.01, "empty container saturation = {}", sat_before);

        for _ in 0..10 {
            for i in 0..8 {
                learner.learn(i, 0.8);
            }
        }

        let sat_after = learner.saturation();
        assert!(
            sat_after > sat_before,
            "saturation should increase: {} → {}",
            sat_before, sat_after,
        );
    }

    #[test]
    fn test_learner_plasticity_effect() {
        let mut rng = seeded_rng();
        let templates = generate_templates(2, 4096, Base::Signed(7), &mut rng);

        // Learner with plasticity
        let mut learner_plastic = ConceptLearner::new(4096, 32, 42);
        learner_plastic.use_plasticity = true;
        for (i, t) in templates.iter().enumerate() {
            learner_plastic.register_class_raw(
                &format!("c{}", i), &format!("C{}", i), t.clone(),
            );
        }

        // Learner without plasticity
        let mut learner_rigid = ConceptLearner::new(4096, 32, 42);
        learner_rigid.use_plasticity = false;
        for (i, t) in templates.iter().enumerate() {
            learner_rigid.register_class_raw(
                &format!("c{}", i), &format!("C{}", i), t.clone(),
            );
        }

        // Same training
        for _ in 0..5 {
            learner_plastic.learn(0, 0.5);
            learner_rigid.learn(0, 0.5);
        }

        // Both should learn something
        let c_plastic = learner_plastic.wal.coefficients[0];
        let c_rigid = learner_rigid.wal.coefficients[0];
        assert!(c_plastic.abs() > 0.01, "plastic should learn");
        assert!(c_rigid.abs() > 0.01, "rigid should learn");
        // They should differ (plasticity modulates amplitude)
        assert!(
            (c_plastic - c_rigid).abs() > 0.001,
            "plasticity should create a difference: plastic={:.4} rigid={:.4}",
            c_plastic, c_rigid,
        );
    }
}
