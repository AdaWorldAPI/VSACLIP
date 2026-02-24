//! Hybrid Recognition — wires 1024-D Jina BF16 + bitpacked Hamming + spatial resonance.
//!
//! The key insight: 1024-D Jina embeddings exist in two simultaneous spaces:
//!
//! 1. **Analog BF16 space** (2048 bytes): sign/exponent/mantissa-aware structured distance
//!    via AVX-512 BITALG — knows the *geometry* of the vector space
//! 2. **Binary Hamming space** (2048 bytes): XOR + POPCNT — pure resonance match
//!    via AVX-512 VPOPCNTDQ — knows the *topology* of the space
//!
//! These two representations are the SAME bytes interpreted differently:
//! - BF16 interpretation: `[sign:1][exp:7][man:8]` × 1024 dims = 2048 bytes
//! - Binary interpretation: 16384 bits = bitpacked fingerprint container
//!
//! The hybrid pipeline:
//! ```text
//! Jina 1024-D float32
//!   │
//!   ├─ fp32_to_bf16_bytes() → 2048 bytes (BF16 quantization)
//!   │   └─ ALSO a 16384-bit binary fingerprint (same bytes!)
//!   │
//!   ├─ K0→K1→K2 binary Hamming (kernels.rs) → reject 95%
//!   │   Uses VPOPCNTDQ: treats BF16 bytes as raw bits
//!   │
//!   ├─ BF16 structured scoring (bf16_hamming.rs) → quality-rank 5% survivors
//!   │   Uses AVX-512 BITALG: respects sign/exp/man structure
//!   │
//!   ├─ VNNI int8 dot product → optional Tier 0 prefilter on quantized embeddings
//!   │   Uses VPDPBUSD: fast dot product for embedding similarity
//!   │
//!   └─ Spatial resonance (spatial_resonance.rs) → 3D Crystal4K-aligned awareness
//!       Uses per-axis BF16 decomposition for SPO/grammar integration
//! ```
//!
//! ## Why This Works
//!
//! Jina v3 produces normalized unit vectors. After BF16 truncation:
//! - Most dimensions have similar exponents (normalized magnitude)
//! - Sign bits carry the dominant similarity signal
//! - Hamming distance on the raw BF16 bytes correlates with cosine distance
//! - The BF16 structured weighting (sign=256, exp=32, man=1) amplifies
//!   the geometrically meaningful bits
//!
//! The binary Hamming path prunes 95% of candidates at 2ns/container.
//! The BF16 path quality-ranks the 5% survivors at ~10ns/container.
//! Together: sublinear search with full geometric awareness.

#[cfg(feature = "rustynum")]
use rustynum_core::{
    bf16_hamming::{
        self, AwarenessThresholds, BF16Weights, JINA_WEIGHTS, TRAINING_WEIGHTS,
    },
    hybrid::{self, HybridConfig, HybridStats, LearningSignal},
    kernels::{SKU_16K_BITS, SKU_16K_BYTES},
    spatial_resonance::{
        self, CrystalAxis, SpatialCrystal3D, SpatialLearningSignal,
    },
};

use ladybug_contract::wide_container::WideContainer;
use ladybug_contract::cogrecord8k::CogRecord8K;

// ============================================================================
// Jina 1024-D Embedding Wrapper
// ============================================================================

/// A Jina 1024-D embedding stored as BF16 bytes.
///
/// This is simultaneously:
/// - A 1024-D BF16 vector (analog geometry: sign/exp/man per dimension)
/// - A 16384-bit binary fingerprint (Hamming topology: XOR + POPCNT)
/// - A WideContainer (ladybug-rs Container 1: CAM channel)
///
/// The SAME 2048 bytes serve all three interpretations.
#[derive(Clone, Debug)]
pub struct JinaEmbedding {
    /// BF16 bytes: 1024 dims × 2 bytes = 2048 bytes = 16384 bits.
    pub bf16_bytes: Vec<u8>,
}

impl JinaEmbedding {
    /// Quantize from float32 Jina embedding to BF16.
    #[cfg(feature = "rustynum")]
    pub fn from_f32(embedding: &[f32]) -> Self {
        assert_eq!(embedding.len(), 1024, "Jina v3 embeddings are 1024-D");
        let bf16_bytes = bf16_hamming::fp32_to_bf16_bytes(embedding);
        Self { bf16_bytes }
    }

    /// Create from raw BF16 bytes.
    pub fn from_bf16_bytes(bytes: Vec<u8>) -> Self {
        assert_eq!(bytes.len(), 2048, "1024 dims × 2 bytes = 2048");
        Self { bf16_bytes: bytes }
    }

    /// View as a WideContainer (zero-copy interpretation).
    pub fn to_wide_container(&self) -> WideContainer {
        let bytes: [u8; 2048] = self.bf16_bytes.as_slice().try_into().unwrap();
        WideContainer::from_bytes(&bytes)
    }

    /// Create from a WideContainer's CAM slot.
    pub fn from_wide_container(wc: &WideContainer) -> Self {
        let bytes = wc.as_bytes().to_vec();
        Self { bf16_bytes: bytes }
    }

    /// Widen back to float32 (lossless in BF16→FP32 direction).
    #[cfg(feature = "rustynum")]
    pub fn to_f32(&self) -> Vec<f32> {
        bf16_hamming::bf16_bytes_to_fp32(&self.bf16_bytes)
    }

    /// BF16 structured distance to another embedding.
    #[cfg(feature = "rustynum")]
    pub fn bf16_distance(&self, other: &JinaEmbedding, weights: &BF16Weights) -> u64 {
        let f = bf16_hamming::select_bf16_hamming_fn();
        f(&self.bf16_bytes, &other.bf16_bytes, weights)
    }

    /// Binary Hamming distance (treats BF16 bytes as raw 16384-bit fingerprint).
    pub fn hamming_distance(&self, other: &JinaEmbedding) -> u32 {
        let mut dist = 0u32;
        for (a, b) in self.bf16_bytes.chunks_exact(8).zip(other.bf16_bytes.chunks_exact(8)) {
            let wa = u64::from_le_bytes(a.try_into().unwrap());
            let wb = u64::from_le_bytes(b.try_into().unwrap());
            dist += (wa ^ wb).count_ones();
        }
        dist
    }

    /// Split into 3 axes for Crystal4K-aligned spatial resonance.
    ///
    /// Splits the 1024-D embedding into 3 groups of ~341 dims each,
    /// mapping to X (dims 0-340), Y (dims 341-681), Z (dims 682-1023).
    /// This aligns with Crystal4K's 3-axis projection model.
    #[cfg(feature = "rustynum")]
    pub fn to_spatial_crystal(&self) -> SpatialCrystal3D {
        let axis_dims = 341; // 341 + 341 + 342 = 1024
        let axis_bytes = axis_dims * 2; // 682 bytes per axis

        let x = CrystalAxis::from_bf16_bytes(self.bf16_bytes[..axis_bytes].to_vec());
        let y = CrystalAxis::from_bf16_bytes(self.bf16_bytes[axis_bytes..axis_bytes * 2].to_vec());
        let z = CrystalAxis::from_bf16_bytes(self.bf16_bytes[axis_bytes * 2..].to_vec());

        SpatialCrystal3D::new(x, y, z)
    }
}

// ============================================================================
// Hybrid Recognizer — the full pipeline
// ============================================================================

/// Configuration for hybrid recognition.
#[cfg(feature = "rustynum")]
#[derive(Clone, Debug)]
pub struct HybridRecognizerConfig {
    /// Hybrid pipeline config (K0/K1/K2 gates + BF16 weights).
    pub hybrid: HybridConfig,
    /// BF16 weights for scoring (default: JINA_WEIGHTS).
    pub bf16_weights: BF16Weights,
    /// Whether to also compute spatial learning signal.
    pub compute_spatial: bool,
    /// Awareness thresholds for learning feedback.
    pub awareness_thresholds: AwarenessThresholds,
    /// Maximum matches to return.
    pub top_k: usize,
}

#[cfg(feature = "rustynum")]
impl Default for HybridRecognizerConfig {
    fn default() -> Self {
        Self {
            hybrid: HybridConfig::sku_16k(),
            bf16_weights: JINA_WEIGHTS,
            compute_spatial: true,
            awareness_thresholds: AwarenessThresholds::default(),
            top_k: 10,
        }
    }
}

#[cfg(feature = "rustynum")]
impl HybridRecognizerConfig {
    /// Config optimized for learning (TRAINING_WEIGHTS ignore mantissa).
    pub fn learning() -> Self {
        Self {
            hybrid: HybridConfig::learning(SKU_16K_BITS),
            bf16_weights: TRAINING_WEIGHTS,
            compute_spatial: true,
            awareness_thresholds: AwarenessThresholds::default(),
            top_k: 20,
        }
    }
}

/// A match from hybrid recognition.
#[cfg(feature = "rustynum")]
#[derive(Clone, Debug)]
pub struct RecognitionMatch {
    /// Index in the database.
    pub index: usize,
    /// Combined score (lower = better).
    pub combined_score: f64,
    /// Binary Hamming distance.
    pub hamming_distance: u32,
    /// BF16 structured distance.
    pub bf16_distance: u64,
    /// Number of sign-flip dimensions.
    pub sign_flips: usize,
    /// Whether this is a strong match (HDR hot).
    pub is_hot: bool,
}

/// Result of hybrid recognition.
#[cfg(feature = "rustynum")]
#[derive(Clone, Debug)]
pub struct RecognitionResult {
    /// Top matches sorted by combined score.
    pub matches: Vec<RecognitionMatch>,
    /// Pipeline statistics.
    pub stats: HybridStats,
    /// Learning signal (if top-K > 0).
    pub learning_signal: Option<LearningSignal>,
    /// 3D spatial learning signal (if compute_spatial=true and top-K > 0).
    pub spatial_signal: Option<SpatialLearningSignal>,
}

/// Run hybrid recognition: Jina 1024-D query against a database.
///
/// Pipeline:
/// 1. Binary Hamming K0→K1→K2 prunes 95% (VPOPCNTDQ on raw BF16 bytes)
/// 2. BF16 structured scoring quality-ranks survivors (AVX-512 BITALG)
/// 3. Learning signal extraction from query vs top-K (awareness decomposition)
/// 4. Spatial signal extraction for Crystal4K/SPO integration (optional)
#[cfg(feature = "rustynum")]
pub fn hybrid_recognize(
    query: &JinaEmbedding,
    database_bytes: &[u8],
    n_candidates: usize,
    config: &HybridRecognizerConfig,
) -> RecognitionResult {
    // Phase 1+2: hybrid pipeline (binary pruning + BF16 scoring)
    let (hybrid_scores, stats) = hybrid::hybrid_pipeline(
        &query.bf16_bytes,
        database_bytes,
        n_candidates,
        &config.hybrid,
    );

    // Build recognition matches
    let matches: Vec<RecognitionMatch> = hybrid_scores
        .iter()
        .take(config.top_k)
        .map(|hs| RecognitionMatch {
            index: hs.index,
            combined_score: hs.combined_score,
            hamming_distance: hs.hamming_distance,
            bf16_distance: hs.bf16_distance,
            sign_flips: hs.structural_diff.sign_flips,
            is_hot: hs.hdr.hot > 0,
        })
        .collect();

    // Phase 3: Learning signal from top-K
    let learning_signal = if !hybrid_scores.is_empty() {
        let top_match_offset = hybrid_scores[0].index * SKU_16K_BYTES;
        let top_bytes = &database_bytes[top_match_offset..top_match_offset + SKU_16K_BYTES];
        Some(hybrid::extract_learning_signal(
            &query.bf16_bytes,
            &[top_bytes],
            &config.awareness_thresholds,
        ))
    } else {
        None
    };

    // Phase 4: Spatial signal for Crystal4K/SPO integration
    let spatial_signal = if config.compute_spatial && !hybrid_scores.is_empty() {
        let query_crystal = query.to_spatial_crystal();
        let top_match_offset = hybrid_scores[0].index * SKU_16K_BYTES;
        let top_bytes = &database_bytes[top_match_offset..top_match_offset + SKU_16K_BYTES];
        let top_embedding = JinaEmbedding::from_bf16_bytes(top_bytes.to_vec());
        let result_crystal = top_embedding.to_spatial_crystal();
        Some(spatial_resonance::extract_spatial_learning_signal(
            &query_crystal,
            &result_crystal,
            &config.awareness_thresholds,
        ))
    } else {
        None
    };

    RecognitionResult {
        matches,
        stats,
        learning_signal,
        spatial_signal,
    }
}

/// Pack a database of JinaEmbeddings into flat bytes for hybrid_recognize().
#[cfg(feature = "rustynum")]
pub fn pack_database(embeddings: &[JinaEmbedding]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(embeddings.len() * SKU_16K_BYTES);
    for emb in embeddings {
        // Pad BF16 bytes (2048) to SKU-16K (2048) — they're the same size!
        bytes.extend_from_slice(&emb.bf16_bytes);
    }
    bytes
}

/// Build a database from float32 Jina embeddings.
#[cfg(feature = "rustynum")]
pub fn build_database(embeddings: &[Vec<f32>]) -> (Vec<JinaEmbedding>, Vec<u8>) {
    let jina_embeddings: Vec<JinaEmbedding> = embeddings
        .iter()
        .map(|e| JinaEmbedding::from_f32(e))
        .collect();
    let packed = pack_database(&jina_embeddings);
    (jina_embeddings, packed)
}

// ============================================================================
// CogRecord8K Integration — wire into 4-container record
// ============================================================================

/// Store a Jina embedding into a CogRecord8K's CAM container.
///
/// The BF16 bytes go into Container 1 (CAM) for binary Hamming resonance.
/// The same bytes serve BF16 structured distance queries.
pub fn embed_into_cam(record: &mut CogRecord8K, embedding: &JinaEmbedding) {
    record.cam = embedding.to_wide_container();
}

/// Store a Jina embedding into a CogRecord8K's EMBED container.
///
/// Uses the EMBED container (Container 3) for float-space queries
/// while CAM (Container 1) holds the binary resonance version.
pub fn embed_into_embed(record: &mut CogRecord8K, embedding: &JinaEmbedding) {
    record.embed = embedding.to_wide_container();
}

/// Extract a JinaEmbedding from a CogRecord8K's CAM container.
pub fn extract_from_cam(record: &CogRecord8K) -> JinaEmbedding {
    JinaEmbedding::from_wide_container(&record.cam)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embedding(seed: u32) -> Vec<f32> {
        (0..1024)
            .map(|i| ((i as f32 + seed as f32) * 0.01).sin())
            .collect()
    }

    #[test]
    fn test_jina_embedding_creation() {
        let emb = make_embedding(42);
        let jina = JinaEmbedding::from_f32(&emb);
        assert_eq!(jina.bf16_bytes.len(), 2048);
    }

    #[test]
    fn test_jina_embedding_self_distance_zero() {
        let jina = JinaEmbedding::from_f32(&make_embedding(42));
        assert_eq!(jina.hamming_distance(&jina), 0);
        assert_eq!(jina.bf16_distance(&jina, &JINA_WEIGHTS), 0);
    }

    #[test]
    fn test_jina_embedding_similar_close() {
        let emb_a = make_embedding(42);
        let mut emb_b = emb_a.clone();
        // Tiny perturbation
        for i in 0..10 {
            emb_b[i] += 0.001;
        }
        let a = JinaEmbedding::from_f32(&emb_a);
        let b = JinaEmbedding::from_f32(&emb_b);

        let hamming = a.hamming_distance(&b);
        assert!(hamming < 1000, "similar embeddings should be close in Hamming: {}", hamming);

        let bf16 = a.bf16_distance(&b, &JINA_WEIGHTS);
        assert!(bf16 < 5000, "similar embeddings should be close in BF16: {}", bf16);
    }

    #[test]
    fn test_jina_embedding_different_far() {
        let a = JinaEmbedding::from_f32(&make_embedding(1));
        let b = JinaEmbedding::from_f32(&make_embedding(99999));

        let hamming = a.hamming_distance(&b);
        assert!(hamming > 1000, "different embeddings should be far in Hamming: {}", hamming);
    }

    #[test]
    fn test_jina_to_wide_container_roundtrip() {
        let jina = JinaEmbedding::from_f32(&make_embedding(42));
        let wc = jina.to_wide_container();
        let back = JinaEmbedding::from_wide_container(&wc);
        assert_eq!(jina.bf16_bytes, back.bf16_bytes);
    }

    #[test]
    fn test_jina_to_f32_roundtrip() {
        let emb = make_embedding(42);
        let jina = JinaEmbedding::from_f32(&emb);
        let back = jina.to_f32();

        for (orig, recovered) in emb.iter().zip(back.iter()) {
            if orig.is_finite() && *orig != 0.0 {
                let rel_err = ((orig - recovered) / orig).abs();
                assert!(
                    rel_err < 0.02,
                    "BF16 roundtrip error: {} → {} (err {})",
                    orig, recovered, rel_err
                );
            }
        }
    }

    #[test]
    fn test_jina_to_spatial_crystal() {
        let jina = JinaEmbedding::from_f32(&make_embedding(42));
        let crystal = jina.to_spatial_crystal();

        // 341 + 341 + 342 = 1024 dims, but bytes: 682 + 682 + 684 = 2048
        assert_eq!(crystal.x.n_dims, 341);
        assert_eq!(crystal.y.n_dims, 341);
        // Z axis gets remaining dims (342)
        assert_eq!(crystal.z.n_dims, 342);
        assert_eq!(
            crystal.total_bytes(),
            2048,
            "total bytes should equal BF16 embedding size"
        );
    }

    #[test]
    fn test_hybrid_recognize_exact_match() {
        let config = HybridRecognizerConfig::default();
        let query_emb = make_embedding(42);
        let query = JinaEmbedding::from_f32(&query_emb);

        // Database: exact match + 9 random
        let mut embeddings = vec![query_emb.clone()];
        for i in 1..10 {
            embeddings.push(make_embedding(i * 1000));
        }
        let (_, db_bytes) = build_database(&embeddings);

        let result = hybrid_recognize(&query, &db_bytes, 10, &config);

        assert!(!result.matches.is_empty(), "should find matches");
        assert_eq!(result.matches[0].index, 0, "best match should be exact copy");
        assert_eq!(result.matches[0].hamming_distance, 0);
        assert!(result.learning_signal.is_some());
    }

    #[test]
    fn test_hybrid_recognize_learning_signal() {
        let config = HybridRecognizerConfig::learning();
        let query = JinaEmbedding::from_f32(&make_embedding(42));

        let mut embeddings = vec![make_embedding(42)]; // exact match
        for i in 1..5 {
            embeddings.push(make_embedding(i * 1000));
        }
        let (_, db_bytes) = build_database(&embeddings);

        let result = hybrid_recognize(&query, &db_bytes, 5, &config);

        if let Some(signal) = &result.learning_signal {
            assert!(
                signal.crystallized_ratio > 0.5,
                "exact match should be mostly crystallized: {}",
                signal.crystallized_ratio
            );
        }
    }

    #[test]
    fn test_hybrid_recognize_spatial_signal() {
        let config = HybridRecognizerConfig {
            compute_spatial: true,
            ..Default::default()
        };
        let query = JinaEmbedding::from_f32(&make_embedding(42));

        let embeddings = vec![make_embedding(42), make_embedding(999)];
        let (_, db_bytes) = build_database(&embeddings);

        let result = hybrid_recognize(&query, &db_bytes, 2, &config);

        assert!(result.spatial_signal.is_some(), "should compute spatial signal");
        if let Some(spatial) = &result.spatial_signal {
            assert!(
                spatial.awareness.spatial_coherence > 0.5,
                "exact match should have high spatial coherence: {}",
                spatial.awareness.spatial_coherence
            );
        }
    }

    #[test]
    fn test_embed_into_cogrecord8k() {
        let jina = JinaEmbedding::from_f32(&make_embedding(42));
        let mut record = CogRecord8K::new();

        embed_into_cam(&mut record, &jina);
        let extracted = extract_from_cam(&record);
        assert_eq!(jina.bf16_bytes, extracted.bf16_bytes);
    }

    #[test]
    fn test_pack_database_size() {
        let embeddings: Vec<JinaEmbedding> = (0..5)
            .map(|i| JinaEmbedding::from_f32(&make_embedding(i)))
            .collect();
        let packed = pack_database(&embeddings);
        assert_eq!(packed.len(), 5 * 2048);
    }
}
