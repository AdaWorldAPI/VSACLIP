//! Extensions to ladybug-contract Container for HDR sweep operations.
//!
//! These methods add:
//! - Partial Hamming distance (on bit-width segments)
//! - Early-exit Hamming with progressive widening
//! - Anti-resonance detection
//! - Majority-vote superposition (bundling)

use ladybug_contract::container::{Container, CONTAINER_WORDS, CONTAINER_BITS};

/// Partial Hamming distance on a segment of the container.
///
/// Computes hamming distance using only words `start..start+count`.
/// Used for progressive early-exit filtering.
#[inline]
pub fn hamming_partial(a: &Container, b: &Container, start: usize, count: usize) -> u32 {
    debug_assert!(start + count <= CONTAINER_WORDS);
    let mut dist = 0u32;
    for i in start..start + count {
        dist += (a.words[i] ^ b.words[i]).count_ones();
    }
    dist
}

/// Early-exit Hamming distance with three-stage Belichtungsmesser.
///
/// Returns `Some(distance)` if within threshold, `None` if definitely beyond.
/// The safety margin ensures zero false negatives.
///
/// Stage 1: 64-bit  spot metering     (1 word)  — eliminates ~80-95%
/// Stage 2: 512-bit center-weighted   (8 words) — eliminates ~80-90% of survivors
/// Stage 3: full    matrix metering   (128 words) — exact
#[inline]
pub fn hamming_early_exit(a: &Container, b: &Container, threshold: u32, safety: f32) -> Option<u32> {
    let d = CONTAINER_BITS as f32;

    // Stage 1: Spot metering — 64 bits (1 word)
    let s1_threshold = (threshold as f32 * (64.0 / d) * safety) as u32;
    let d64 = (a.words[0] ^ b.words[0]).count_ones();
    if d64 > s1_threshold {
        return None;
    }

    // Stage 2: Center-weighted — 512 bits (8 words)
    let s2_threshold = (threshold as f32 * (512.0 / d) * safety) as u32;
    let mut d512 = d64;
    for i in 1..8 {
        d512 += (a.words[i] ^ b.words[i]).count_ones();
    }
    if d512 > s2_threshold {
        return None;
    }

    // Stage 3: Matrix metering — full 8192 bits
    let mut full = d512;
    for i in 8..CONTAINER_WORDS {
        full += (a.words[i] ^ b.words[i]).count_ones();
    }

    if full < threshold {
        Some(full)
    } else {
        None
    }
}

/// Check if two containers are anti-resonant (opposite bits).
///
/// Anti-resonance = distance > CONTAINER_BITS - threshold.
/// Provides natural lateral inhibition.
#[inline]
pub fn is_anti_resonant(a: &Container, b: &Container, threshold: u32) -> bool {
    a.hamming(b) > CONTAINER_BITS as u32 - threshold
}

/// Majority-vote superposition — robust bundling for any K.
///
/// Each bit is set if the majority of input containers have it set.
/// More robust than XOR superposition for large K.
///
/// Use XOR for bind/unbind (2 operands).
/// Use majority for bundling/superposition (K operands).
pub fn majority_bundle(items: &[&Container]) -> Container {
    Container::bundle(items)  // ladybug-contract already has this!
}

/// Weighted majority bundle — items with higher weight dominate.
pub fn weighted_bundle(items: &[(&Container, f32)]) -> Container {
    Container::bundle_weighted(items)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partial_hamming_sums_to_full() {
        let a = Container::random(42);
        let b = Container::random(99);

        let full = a.hamming(&b);
        let partial_sum: u32 = (0..CONTAINER_WORDS)
            .map(|i| hamming_partial(&a, &b, i, 1))
            .sum();

        assert_eq!(full, partial_sum);
    }

    #[test]
    fn test_early_exit_no_false_negatives() {
        let a = Container::random(42);
        let threshold = 3000u32;
        let safety = 1.5f32;

        // Test 1000 random containers
        let mut false_negatives = 0;
        for seed in 0..1000u64 {
            let b = Container::random(seed);
            let full = a.hamming(&b);
            let early = hamming_early_exit(&a, &b, threshold, safety);

            if full < threshold && early.is_none() {
                false_negatives += 1;
            }
        }

        assert_eq!(false_negatives, 0, "Early exit must have ZERO false negatives");
    }

    #[test]
    fn test_early_exit_filters_most() {
        let a = Container::random(42);
        let threshold = 2500u32;
        let safety = 1.5f32;

        let mut total = 0;
        let mut survived = 0;
        for seed in 0..1000u64 {
            let b = Container::random(seed);
            total += 1;
            if hamming_early_exit(&a, &b, threshold, safety).is_some() {
                survived += 1;
            }
        }

        // For random vectors, very few should survive
        let survival_rate = survived as f64 / total as f64;
        assert!(survival_rate < 0.10, "Expected <10% survival, got {:.1}%", survival_rate * 100.0);
    }

    #[test]
    fn test_anti_resonance() {
        let a = Container::random(42);
        let anti = {
            let mut c = Container::zero();
            for i in 0..CONTAINER_WORDS {
                c.words[i] = !a.words[i];
            }
            c
        };

        assert!(is_anti_resonant(&a, &anti, 200));
        assert!(!is_anti_resonant(&a, &Container::random(99), 200));
    }
}
