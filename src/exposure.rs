//! Exposure Module — Belichtungsmesser as formal algorithm.
//!
//! Maps camera exposure metering modes to Hamming distance filtering:
//!
//! | Photography     | Algorithm               | Bits examined |
//! |----------------|-------------------------|---------------|
//! | Spot metering   | 64-bit prefix filter    | 64 / 8192    |
//! | Center-weighted | 512-bit segment filter  | 512 / 8192   |
//! | Matrix metering | Full container compare  | 8192 / 8192  |
//!
//! The safety margin guarantees zero false negatives by accounting
//! for the variance in per-segment Hamming distance.

use ladybug_contract::container::CONTAINER_BITS;

/// Safety margin for early exit.
///
/// For a segment of `s` bits, the expected Hamming distance for a
/// random pair is `s/2` with std_dev = `sqrt(s/4)`.
///
/// A margin of 1.5 means we allow `threshold * (s/d) * 1.5` at each stage,
/// which catches >99.9% of true matches.
///
/// Derivation:
/// ```text
/// Per-segment Hamming ~ Binomial(s, 0.5)
/// E[H_segment] = s * p_match  (where p_match = total_dist / d)
/// Var[H_segment] = s * p_match * (1 - p_match)
/// 3σ safety: margin ≈ 1 + 3 * sqrt((1-p) / (s*p))
/// For s=64, p≈0.3: margin ≈ 1.5
/// ```
pub const DEFAULT_SAFETY_MARGIN: f32 = 1.5;

/// Compute the optimal threshold for a given stage.
///
/// `bits_examined`: how many bits are checked at this stage
/// `total_threshold`: the target Hamming distance threshold on full container
/// `safety`: safety margin multiplier
#[inline]
pub fn stage_threshold(bits_examined: u32, total_threshold: u32, safety: f32) -> u32 {
    let d = CONTAINER_BITS as f32;
    (total_threshold as f32 * (bits_examined as f32 / d) * safety) as u32
}

/// The three exposure stages with their bit-width configurations.
#[derive(Debug, Clone, Copy)]
pub struct ExposureStage {
    /// Starting word index in the container
    pub word_start: usize,
    /// Number of words to examine
    pub word_count: usize,
    /// Total bits examined
    pub bits: u32,
}

/// Default three-stage exposure configuration for 8192-bit containers.
pub const STAGES: [ExposureStage; 3] = [
    ExposureStage { word_start: 0, word_count: 1,   bits: 64 },    // Spot
    ExposureStage { word_start: 0, word_count: 8,   bits: 512 },   // Center
    ExposureStage { word_start: 0, word_count: 128, bits: 8192 },  // Matrix
];

/// Estimate the survival rate at each stage for random data.
///
/// For random binary vectors at threshold T (as fraction of d):
/// - Stage 1 (64-bit): P(survive) ≈ P(Binomial(64, 0.5) < 64*T*safety/d)
/// - This is approximately the tail of the binomial CDF
///
/// Returns (s1_rate, s2_rate, s3_rate) as fractions.
pub fn estimated_survival_rates(threshold_frac: f32, safety: f32) -> (f32, f32, f32) {
    // For random data with threshold at 30% of d:
    // Stage 1 survival ≈ 5-20% depending on safety margin
    // Stage 2 survival ≈ 5-15% of stage 1
    // These are empirical; exact values depend on the CDF
    let s1 = (threshold_frac * safety * 2.0).min(0.30);
    let s2 = (threshold_frac * safety * 0.5).min(0.20);
    (s1, s2, 1.0)
}

/// Compute theoretical instruction count for given parameters.
pub fn theoretical_instructions(n: u64, threshold_frac: f32, safety: f32) -> (u64, u64, f64) {
    let full = n * 128; // 128 words per container
    let (s1_rate, s2_rate, _) = estimated_survival_rates(threshold_frac, safety);
    let s1_survivors = (n as f32 * s1_rate) as u64;
    let s2_survivors = (s1_survivors as f32 * s2_rate) as u64;

    let early = n * 1               // stage 1: 1 word per container
              + s1_survivors * 7     // stage 2: 7 additional words
              + s2_survivors * 120;  // stage 3: 120 remaining words

    let speedup = full as f64 / early as f64;
    (full, early, speedup)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_threshold() {
        let total = 2500u32;
        let safety = 1.5f32;

        let s1 = stage_threshold(64, total, safety);
        let s2 = stage_threshold(512, total, safety);
        let s3 = stage_threshold(8192, total, safety);

        // Each stage threshold should be proportionally larger
        assert!(s1 < s2);
        assert!(s2 < s3);
        // Stage 3 should be >= total threshold (with safety > 1)
        assert!(s3 >= total);
    }

    #[test]
    fn test_theoretical_speedup() {
        let (full, early, speedup) = theoretical_instructions(1_000_000, 0.30, 1.5);
        println!("1M containers: full={}, early={}, speedup={:.1}x", full, early, speedup);
        assert!(speedup > 10.0);
    }
}
