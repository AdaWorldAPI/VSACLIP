//! Wide HDR POPCNT Sweep — 16,384-bit WideContainer sweep engine.
//!
//! The same three-stage early-exit algorithm scaled to 2 KB containers.
//! Full sweep: 32 VPOPCNTDQ instructions per container.
//! HDR early-exit: ~2-4 instructions average.
//!
//! # Performance
//!
//! | Stage | Bits  | Words | AVX-512 Regs | Eliminates |
//! |-------|-------|-------|-------------|------------|
//! | 1     | 64    | 1     | -           | ~80-95%    |
//! | 2     | 1024  | 16    | 2 zmm       | ~80-90%    |
//! | 3     | 16384 | 256   | 32 zmm      | exact      |
//!
//! Stage 2 is wider (1024 vs 512 bits) because 16384-bit containers
//! have more variance at small sample sizes.

use ladybug_contract::wide_container::{WideContainer, WIDE_WORDS, WIDE_BITS};
use crate::hdr::{HdrConfig, HdrDistribution};
use crate::ResonanceMatch;

/// Configuration for the wide sweep engine.
#[derive(Debug, Clone)]
pub struct WideSweepConfig {
    /// Hamming distance threshold for match acceptance.
    pub threshold: u32,
    /// Safety margin for early exit (1.0 = no margin, 1.5 = conservative).
    pub safety: f32,
    /// HDR exposure configuration (scaled to 16384-bit).
    pub hdr: HdrConfig,
    /// Maximum results to return (0 = unlimited).
    pub limit: usize,
}

impl Default for WideSweepConfig {
    fn default() -> Self {
        Self {
            threshold: (WIDE_BITS as f32 * 0.30) as u32, // 30% = 4915
            safety: 1.5,
            hdr: HdrConfig::for_bits(WIDE_BITS as u32),
            limit: 0,
        }
    }
}

/// Result of a wide sweep operation.
#[derive(Debug)]
pub struct WideSweepResult {
    pub matches: Vec<ResonanceMatch>,
    pub distribution: HdrDistribution,
    pub containers_scanned: usize,
    pub stage1_survivors: usize,
    pub stage2_survivors: usize,
    pub instructions: u64,
}

/// Three-stage HDR POPCNT sweep on 16,384-bit WideContainers.
///
/// Same algorithm as `hdr_sweep` but scaled for 2KB containers:
/// - Stage 1: 64-bit spot metering (1 word)
/// - Stage 2: 1024-bit center-weighted (16 words)
/// - Stage 3: full 16384-bit matrix metering (256 words)
pub fn wide_hdr_sweep(
    query: &WideContainer,
    corpus: &[WideContainer],
    config: &WideSweepConfig,
) -> WideSweepResult {
    let n = corpus.len();
    let threshold = config.threshold;
    let safety = config.safety;
    let d = WIDE_BITS as f32;

    let mut matches = Vec::new();
    let mut distribution = HdrDistribution::default();
    let mut instructions: u64 = 0;
    let mut s1_survivors = 0usize;
    let mut s2_survivors = 0usize;

    // Precompute stage thresholds
    let s1_t = (threshold as f32 * (64.0 / d) * safety) as u32;
    let s2_t = (threshold as f32 * (1024.0 / d) * safety) as u32;

    for (idx, container) in corpus.iter().enumerate() {
        // Stage 1: Spot metering — 1 word (64 bits)
        let d64 = (query.words[0] ^ container.words[0]).count_ones();
        instructions += 1;

        if d64 > s1_t {
            continue;
        }
        s1_survivors += 1;

        // Stage 2: Center-weighted — 16 words (1024 bits)
        let mut d1024 = d64;
        for i in 1..16 {
            d1024 += (query.words[i] ^ container.words[i]).count_ones();
        }
        instructions += 15;

        if d1024 > s2_t {
            continue;
        }
        s2_survivors += 1;

        // Stage 3: Matrix metering — full 256 words (16384 bits)
        let mut full = d1024;
        for i in 16..WIDE_WORDS {
            full += (query.words[i] ^ container.words[i]).count_ones();
        }
        instructions += (WIDE_WORDS - 16) as u64;

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

    matches.sort_by_key(|m| m.distance);

    WideSweepResult {
        matches,
        distribution,
        containers_scanned: n,
        stage1_survivors: s1_survivors,
        stage2_survivors: s2_survivors,
        instructions,
    }
}

/// Full sweep on WideContainers (no early exit) — for benchmarking.
pub fn wide_full_sweep(
    query: &WideContainer,
    corpus: &[WideContainer],
    threshold: u32,
) -> Vec<ResonanceMatch> {
    let hdr = HdrConfig::for_bits(WIDE_BITS as u32);

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

/// Top-K on WideContainers.
pub fn wide_top_k(
    query: &WideContainer,
    corpus: &[WideContainer],
    k: usize,
    threshold: u32,
) -> Vec<ResonanceMatch> {
    let config = WideSweepConfig {
        threshold,
        limit: 0,
        ..Default::default()
    };

    let mut result = wide_hdr_sweep(query, corpus, &config);
    result.matches.truncate(k);
    result.matches
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_near_wide(reference: &WideContainer, target_dist: usize) -> WideContainer {
        let mut c = reference.clone();
        let mut rng_state = target_dist as u64 ^ 0xDEADBEEF;
        let mut flipped = 0;
        while flipped < target_dist {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let bit = (rng_state as usize) % WIDE_BITS;
            let word = bit / 64;
            let pos = bit % 64;
            c.words[word] ^= 1u64 << pos;
            flipped += 1;
        }
        c
    }

    #[test]
    fn test_wide_sweep_finds_planted_matches() {
        let query = WideContainer::random(42);
        let threshold = 5000u32; // ~30% of 16384

        let mut corpus: Vec<WideContainer> = (0..1000u64)
            .map(|s| WideContainer::random(s + 1000))
            .collect();

        // Plant 5 matches at known distances
        for i in 0..5 {
            let near = make_near_wide(&query, 1000 + i * 400);
            corpus[i * 200] = near;
        }

        let config = WideSweepConfig {
            threshold,
            safety: 1.5,
            ..Default::default()
        };

        let result = wide_hdr_sweep(&query, &corpus, &config);
        let full = wide_full_sweep(&query, &corpus, threshold);

        // Zero false negatives
        let hdr_indices: std::collections::HashSet<usize> =
            result.matches.iter().map(|m| m.index).collect();
        for fm in &full {
            assert!(
                hdr_indices.contains(&fm.index),
                "False negative: full found index {} (dist={}) but wide HDR missed it",
                fm.index, fm.distance
            );
        }

        let full_instructions = corpus.len() as u64 * WIDE_WORDS as u64;
        let speedup = full_instructions as f64 / result.instructions as f64;
        assert!(speedup > 3.0, "Expected >3x speedup, got {:.1}x", speedup);
    }
}
