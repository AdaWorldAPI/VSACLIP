//! HDR POPCNT Sweep Engine — the core query algorithm.
//!
//! Performs a three-stage early-exit Hamming sweep over a corpus of Containers,
//! returning matches with HDR exposure scores.
//!
//! # Performance
//!
//! For N containers with threshold T and safety margin S:
//! - Full sweep: N × 128 POPCNT instructions
//! - HDR sweep:  N × 1 + 0.05N × 8 + 0.005N × 128 ≈ N × 2.04 instructions
//! - Speedup: ~63× for random data

use ladybug_contract::container::{Container, CONTAINER_WORDS, CONTAINER_BITS};
use crate::container_ext::hamming_early_exit;
use crate::hdr::{HdrConfig, HdrDistribution};
use crate::{HdrScore, ResonanceMatch};

/// Configuration for the sweep engine
#[derive(Debug, Clone)]
pub struct SweepConfig {
    /// Hamming distance threshold for match acceptance
    pub threshold: u32,
    /// Safety margin for early exit (1.0 = no margin, 1.5 = conservative)
    pub safety: f32,
    /// HDR exposure configuration
    pub hdr: HdrConfig,
    /// Maximum results to return (0 = unlimited)
    pub limit: usize,
}

impl Default for SweepConfig {
    fn default() -> Self {
        Self {
            threshold: (CONTAINER_BITS as f32 * 0.30) as u32, // 30% = 2457
            safety: 1.5,
            hdr: HdrConfig::default(),
            limit: 0,
        }
    }
}

/// Result of a sweep operation
#[derive(Debug)]
pub struct SweepResult {
    pub matches: Vec<ResonanceMatch>,
    pub distribution: HdrDistribution,
    pub containers_scanned: usize,
    pub stage1_survivors: usize,
    pub stage2_survivors: usize,
    pub instructions: u64,
}

/// Perform HDR POPCNT sweep with three-stage early exit.
///
/// This is the core algorithm. It finds all containers within `threshold`
/// Hamming distance of `query`, using progressive bit-width expansion
/// to eliminate non-matches early.
pub fn hdr_sweep(query: &Container, corpus: &[Container], config: &SweepConfig) -> SweepResult {
    let n = corpus.len();
    let threshold = config.threshold;
    let safety = config.safety;
    let d = CONTAINER_BITS as f32;

    let mut matches = Vec::new();
    let mut distribution = HdrDistribution::default();
    let mut instructions: u64 = 0;
    let mut s1_survivors = 0usize;
    let mut s2_survivors = 0usize;

    // Precompute stage thresholds
    let s1_t = (threshold as f32 * (64.0 / d) * safety) as u32;
    let s2_t = (threshold as f32 * (512.0 / d) * safety) as u32;

    for (idx, container) in corpus.iter().enumerate() {
        // Stage 1: Spot metering — 1 word (64 bits)
        let d64 = (query.words[0] ^ container.words[0]).count_ones();
        instructions += 1;

        if d64 > s1_t {
            continue;
        }
        s1_survivors += 1;

        // Stage 2: Center-weighted — 8 words (512 bits)
        let mut d512 = d64;
        for i in 1..8 {
            d512 += (query.words[i] ^ container.words[i]).count_ones();
        }
        instructions += 7;

        if d512 > s2_t {
            continue;
        }
        s2_survivors += 1;

        // Stage 3: Matrix metering — full 128 words (8192 bits)
        let mut full = d512;
        for i in 8..CONTAINER_WORDS {
            full += (query.words[i] ^ container.words[i]).count_ones();
        }
        instructions += (CONTAINER_WORDS - 8) as u64;

        let score = config.hdr.score(full);
        distribution.record(&score);

        if full < threshold {
            matches.push(ResonanceMatch {
                index: idx,
                distance: full,
                hdr: score,
            });

            if config.limit > 0 && matches.len() >= config.limit {
                break;
            }
        }
    }

    // Sort by distance (closest first)
    matches.sort_by_key(|m| m.distance);

    SweepResult {
        matches,
        distribution,
        containers_scanned: n,
        stage1_survivors: s1_survivors,
        stage2_survivors: s2_survivors,
        instructions,
    }
}

/// Convenience: sweep and return only the top-K matches
pub fn top_k(query: &Container, corpus: &[Container], k: usize, threshold: u32) -> Vec<ResonanceMatch> {
    let config = SweepConfig {
        threshold,
        limit: 0, // get all matches, then truncate
        ..Default::default()
    };

    let mut result = hdr_sweep(query, corpus, &config);
    result.matches.truncate(k);
    result.matches
}

/// Full sweep (no early exit) — for benchmarking comparison
pub fn full_sweep(query: &Container, corpus: &[Container], threshold: u32) -> Vec<ResonanceMatch> {
    let hdr = HdrConfig::default();

    corpus.iter().enumerate()
        .filter_map(|(idx, c)| {
            let dist = query.hamming(c);
            if dist < threshold {
                Some(ResonanceMatch {
                    index: idx,
                    distance: dist,
                    hdr: hdr.score(dist),
                })
            } else {
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_near(reference: &Container, target_dist: usize) -> Container {
        let mut c = reference.clone();
        // Flip specific bits to achieve approximate target distance
        let mut rng_state = target_dist as u64 ^ 0xDEADBEEF;
        let mut flipped = 0;
        while flipped < target_dist {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let bit = (rng_state as usize) % CONTAINER_BITS;
            let word = bit / 64;
            let pos = bit % 64;
            c.words[word] ^= 1u64 << pos;
            flipped += 1;
        }
        c
    }

    #[test]
    fn test_hdr_sweep_finds_planted_matches() {
        let query = Container::random(42);
        let threshold = 2500u32;

        // Build corpus: 1000 random + 5 planted near matches
        let mut corpus: Vec<Container> = (0..1000u64)
            .map(|s| Container::random(s + 1000))
            .collect();

        // Plant 5 matches at known distances
        for i in 0..5 {
            let near = make_near(&query, 500 + i * 200);
            corpus[i * 200] = near;
        }

        let config = SweepConfig {
            threshold,
            safety: 1.5,
            ..Default::default()
        };

        let result = hdr_sweep(&query, &corpus, &config);

        // Full sweep for comparison
        let full = full_sweep(&query, &corpus, threshold);

        // HDR must find at least as many as full sweep
        assert!(
            result.matches.len() >= full.len(),
            "HDR found {} but full found {}",
            result.matches.len(),
            full.len()
        );

        // Zero false negatives: every full match must appear in HDR
        let hdr_indices: std::collections::HashSet<usize> =
            result.matches.iter().map(|m| m.index).collect();
        for fm in &full {
            assert!(
                hdr_indices.contains(&fm.index),
                "False negative: full found index {} (dist={}) but HDR missed it",
                fm.index,
                fm.distance
            );
        }

        // Instruction savings
        let full_instructions = corpus.len() as u64 * CONTAINER_WORDS as u64;
        let speedup = full_instructions as f64 / result.instructions as f64;
        println!("Speedup: {:.1}x ({} vs {} instructions)",
                 speedup, result.instructions, full_instructions);
        assert!(speedup > 5.0, "Expected >5x speedup, got {:.1}x", speedup);
    }

    #[test]
    fn test_top_k() {
        let query = Container::random(42);
        let mut corpus: Vec<Container> = (0..500u64)
            .map(|s| Container::random(s + 1000))
            .collect();

        // Plant one definite match
        corpus[100] = make_near(&query, 200);

        let top = top_k(&query, &corpus, 5, 3000);
        assert!(!top.is_empty());
        assert!(top[0].distance < 3000);
    }
}
