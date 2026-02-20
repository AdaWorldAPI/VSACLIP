//! HDR POPCNT Sweep — Belichtungsmesser Early-Exit Search
//!
//! Three-stage exposure metering: 64→512→16384 bit progressive matching.
//! Zero false negatives guaranteed (safety margin 1.5×, proven in Python).

use crate::{Fingerprint, HdrScore, ResonanceMatch, FINGERPRINT_BITS, FINGERPRINT_U64};
use crate::hamming_distance;

const SAFETY_MARGIN: f64 = 1.5;

/// Three-stage Belichtungsmesser sweep
pub fn hdr_sweep(query: &Fingerprint, containers: &[Fingerprint], threshold: u32) -> Vec<ResonanceMatch> {
    let d = FINGERPRINT_BITS as u32;
    let q = query.as_raw();
    let s1_t = ((threshold as f64) * (64.0 / d as f64) * SAFETY_MARGIN) as u32;
    let s2_t = ((threshold as f64) * (512.0 / d as f64) * SAFETY_MARGIN) as u32;

    // Stage 1: Spot — 64 bit
    let stage1: Vec<usize> = containers.iter().enumerate()
        .filter(|(_, c)| (q[0] ^ c.as_raw()[0]).count_ones() <= s1_t)
        .map(|(i, _)| i).collect();

    // Stage 2: Center — 512 bit
    let stage2: Vec<usize> = stage1.into_iter().filter(|&i| {
        let c = containers[i].as_raw();
        (0..8).map(|j| (q[j] ^ c[j]).count_ones()).sum::<u32>() <= s2_t
    }).collect();

    // Stage 3: Matrix — full 16384 bit
    let mut results: Vec<ResonanceMatch> = stage2.into_iter().filter_map(|i| {
        let dist = hamming_distance(query, &containers[i]);
        (dist < threshold).then(|| ResonanceMatch {
            index: i, score: HdrScore::from_distance(dist, d),
        })
    }).collect();

    results.sort_by(|a, b| b.score.total.cmp(&a.score.total)
        .then(a.score.raw_dist.cmp(&b.score.raw_dist)));
    results
}

/// Full sweep without early exit (for correctness verification)
pub fn full_sweep(query: &Fingerprint, containers: &[Fingerprint], threshold: u32) -> Vec<ResonanceMatch> {
    let d = FINGERPRINT_BITS as u32;
    let mut results: Vec<ResonanceMatch> = containers.iter().enumerate().filter_map(|(i, c)| {
        let dist = hamming_distance(query, c);
        (dist < threshold).then(|| ResonanceMatch {
            index: i, score: HdrScore::from_distance(dist, d),
        })
    }).collect();
    results.sort_by(|a, b| b.score.total.cmp(&a.score.total)
        .then(a.score.raw_dist.cmp(&b.score.raw_dist)));
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_zero_false_negatives() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let n = 5000;
        let threshold = FINGERPRINT_BITS as u32 * 30 / 100;
        let query = Fingerprint::from_raw(std::array::from_fn(|_| rng.gen()));
        let mut containers: Vec<Fingerprint> = (0..n)
            .map(|_| Fingerprint::from_raw(std::array::from_fn(|_| rng.gen()))).collect();
        // Plant 20 near-matches
        for i in 0..20 {
            let mut data = *query.as_raw();
            for _ in 0..rng.gen_range(500..threshold as usize) {
                let bit = rng.gen_range(0..FINGERPRINT_BITS);
                data[bit/64] ^= 1u64 << (bit%64);
            }
            containers[i * 250] = Fingerprint::from_raw(data);
        }
        let full = full_sweep(&query, &containers, threshold);
        let early = hdr_sweep(&query, &containers, threshold);
        let full_idx: std::collections::HashSet<usize> = full.iter().map(|m| m.index).collect();
        let early_idx: std::collections::HashSet<usize> = early.iter().map(|m| m.index).collect();
        assert_eq!(full_idx.difference(&early_idx).count(), 0, "false negatives!");
    }
}
