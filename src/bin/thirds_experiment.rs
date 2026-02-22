//! Rule of Thirds Projection — The Lines ARE the Signal
//!
//! Photographers place subjects ON the 1/3 lines, at the 4 crossings.
//! Not in the center. Not in zones. On the EDGES between regions.
//!
//!   +-------------+-------------+
//!   |             |             |
//!   |          *--+--*          |   * = power point (on the line)
//!   |          |  |  |          |
//!   +----------+--+--+----------+   The lines carry the subject
//!   |          |  |  |          |   The zones are background
//!   |          *--+--*          |
//!   |             |             |
//!   +-------------+-------------+
//!
//! Translation to SimHash:
//!
//! The 8192 output bits are arranged on a conceptual 1/3 grid.
//! Bits at the 1/3 and 2/3 BOUNDARIES get cross-projected:
//!   - Two hyperplanes at each boundary, XOR-bound (differential)
//!   - This catches what SITS on the line — the transition, the edge
//!   - Like edge detection: boundaries carry more info than interiors
//!
//! The 4 power points (where vertical and horizontal 1/3 lines cross)
//! get QUAD cross-projection: 4 hyperplanes, golden-angle spaced.
//!
//! Strategies tested:
//!   1. BASELINE — Uniform random hyperplanes
//!   2. THIRDS LINES — XOR-bind pairs at 1/3 and 2/3 bit boundaries
//!   3. POWER POINTS — Golden quad at the 4 crossing intersections
//!   4. FULL COMPOSITION — Lines + power points + golden background
//!   5. DOUBLE-LINED INNER RECTANGLE — 2x bind center zone
//!
//! Run: cargo run --release --bin thirds

use ladybug_contract::container::{Container, CONTAINER_BITS, CONTAINER_WORDS};
use std::time::Instant;

const DIM: usize = 512;
const GOLDEN_ANGLE: f64 = 2.399_963_229_728_653;
const PHI: f64 = 1.618_033_988_749_895;
const N_CLASSES: usize = 30;
const SAMPLES_PER_CLASS: usize = 80;
const N_TOTAL: usize = N_CLASSES * SAMPLES_PER_CLASS;

// 1/3 and 2/3 boundaries in the 8192-bit space
const THIRD_1: usize = CONTAINER_BITS / 3;      // 2730
const THIRD_2: usize = CONTAINER_BITS * 2 / 3;  // 5461

// Width of the "line" — how many bits around each boundary get the crosshair
const LINE_WIDTH: usize = 256; // +/-128 bits around each 1/3 line

// ============================================================================
// PRNG
// ============================================================================

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self { Self(seed) }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() as i64 as f64 / i64::MAX as f64) as f32
    }

    fn next_gaussian(&mut self) -> f32 {
        let mut s = 0.0f32;
        for _ in 0..6 { s += self.next_f32(); }
        s / 3.0
    }
}

// ============================================================================
// HYPERPLANE PRIMITIVES
// ============================================================================

/// Random hyperplane dot product
#[inline]
fn hp_dot_random(embedding: &[f32], bit_idx: usize, seed: u64) -> f32 {
    let mut state = seed
        .wrapping_add(bit_idx as u64)
        .wrapping_mul(0x9e3779b97f4a7c15);
    let mut dot = 0.0f32;
    for d in 0..embedding.len() {
        state ^= state >> 30;
        state = state.wrapping_mul(0xbf58476d1ce4e5b9);
        state ^= state >> 27;
        state = state.wrapping_mul(0x94d049bb133111eb);
        state ^= state >> 31;
        let sign = if state & 1 == 0 { 1.0f32 } else { -1.0f32 };
        dot += embedding[d] * sign;
    }
    dot
}

/// Golden-angle hyperplane dot product
#[inline]
fn hp_dot_golden(embedding: &[f32], bit_idx: usize) -> f32 {
    let mut dot = 0.0f32;
    let base = bit_idx as f64 * GOLDEN_ANGLE;
    for d in 0..embedding.len() {
        let w = (base + d as f64 * PHI).cos() as f32;
        dot += embedding[d] * w;
    }
    dot
}

/// Is this bit position on a 1/3 line?
#[inline]
fn on_thirds_line(bit_idx: usize) -> bool {
    let d1 = bit_idx.abs_diff(THIRD_1);
    let d2 = bit_idx.abs_diff(THIRD_2);
    d1 < LINE_WIDTH / 2 || d2 < LINE_WIDTH / 2
}

/// Is this bit position at a power point (crossing of TWO 1/3 lines)?
/// We use the conceptual 2D layout: bit_idx maps to (row, col) in a ~90x91 grid
/// The 1/3 lines in this 2D view cross at 4 points
#[inline]
fn at_power_point(bit_idx: usize) -> bool {
    let cols = 91usize;
    let row = bit_idx / cols;
    let col = bit_idx % cols;
    let rows = CONTAINER_BITS / cols;
    let row_third_1 = rows / 3;
    let row_third_2 = rows * 2 / 3;
    let col_third_1 = cols / 3;
    let col_third_2 = cols * 2 / 3;
    let on_row_line = row.abs_diff(row_third_1) < 4 || row.abs_diff(row_third_2) < 4;
    let on_col_line = col.abs_diff(col_third_1) < 4 || col.abs_diff(col_third_2) < 4;
    on_row_line && on_col_line
}

// ============================================================================
// STRATEGY 1: BASELINE — Uniform random
// ============================================================================

fn simhash_baseline(embedding: &[f32], seed: u64) -> Container {
    let mut result = Container::zero();
    for w in 0..CONTAINER_WORDS {
        let mut word = 0u64;
        for b in 0..64u32 {
            let idx = w * 64 + b as usize;
            if hp_dot_random(embedding, idx, seed) >= 0.0 {
                word |= 1u64 << b;
            }
        }
        result.words[w] = word;
    }
    result
}

// ============================================================================
// STRATEGY 2: THIRDS LINES — XOR-bind at 1/3 boundaries
//
// Bits ON the 1/3 lines: two hyperplanes, XOR-bound (differential)
//   -> catches what SITS on the line (the subject)
// Bits OFF the lines: standard single hyperplane (background)
//
// The XOR-bind means: instead of "which side of a plane?"
// it's "are you on the SAME side of two DIFFERENT planes?"
// This is a 2nd-order feature — relationships, not raw position.
// ============================================================================

fn simhash_thirds_lines(embedding: &[f32], seed: u64) -> Container {
    let seed2 = seed ^ 0xFACE_CAFE_BABE_F00D;
    let mut result = Container::zero();
    for w in 0..CONTAINER_WORDS {
        let mut word = 0u64;
        for b in 0..64u32 {
            let idx = w * 64 + b as usize;
            if on_thirds_line(idx) {
                // ON the line: XOR-bind two projections (differential)
                let s1 = hp_dot_random(embedding, idx, seed) >= 0.0;
                let s2 = hp_dot_random(embedding, idx, seed2) >= 0.0;
                if s1 ^ s2 {
                    word |= 1u64 << b;
                }
            } else {
                // Background: single projection
                if hp_dot_random(embedding, idx, seed) >= 0.0 {
                    word |= 1u64 << b;
                }
            }
        }
        result.words[w] = word;
    }
    result
}

// ============================================================================
// STRATEGY 3: POWER POINTS — Golden quad at 4 crossings
//
// At the 4 points where 1/3 lines cross: 4 golden-angle hyperplanes,
// majority vote. Like a photographer's focus point — highest concentration
// of measurement. Elsewhere: golden-angle single hyperplanes.
// ============================================================================

fn simhash_power_points(embedding: &[f32], _seed: u64) -> Container {
    let mut result = Container::zero();
    for w in 0..CONTAINER_WORDS {
        let mut word = 0u64;
        for b in 0..64u32 {
            let idx = w * 64 + b as usize;
            if at_power_point(idx) {
                // Power point: 4 golden-angle hyperplanes, majority >= 3
                let base = idx * 4;
                let d0 = hp_dot_golden(embedding, base);
                let d1 = hp_dot_golden(embedding, base + 1);
                let d2 = hp_dot_golden(embedding, base + 2);
                let d3 = hp_dot_golden(embedding, base + 3);
                let votes = (d0 >= 0.0) as u8 + (d1 >= 0.0) as u8
                          + (d2 >= 0.0) as u8 + (d3 >= 0.0) as u8;
                if votes >= 3 {
                    word |= 1u64 << b;
                }
            } else {
                // Background: golden-angle single hyperplane
                if hp_dot_golden(embedding, idx) >= 0.0 {
                    word |= 1u64 << b;
                }
            }
        }
        result.words[w] = word;
    }
    result
}

// ============================================================================
// STRATEGY 4: FULL COMPOSITION — Lines + power points + golden background
//
// The whole viewfinder works together:
//   Power points catch the subject (eye of the bird)
//   Lines catch the composition (horizon, leading lines)
//   Background fills in context (sky, grass, water)
// ============================================================================

fn simhash_full_composition(embedding: &[f32], seed: u64) -> Container {
    let seed2 = seed ^ 0xFACE_CAFE_BABE_F00D;
    let mut result = Container::zero();
    for w in 0..CONTAINER_WORDS {
        let mut word = 0u64;
        for b in 0..64u32 {
            let idx = w * 64 + b as usize;
            let bit_set = if at_power_point(idx) {
                // POWER POINT: golden quad, majority >= 3
                let base = idx * 4;
                let votes = (hp_dot_golden(embedding, base) >= 0.0) as u8
                          + (hp_dot_golden(embedding, base + 1) >= 0.0) as u8
                          + (hp_dot_golden(embedding, base + 2) >= 0.0) as u8
                          + (hp_dot_golden(embedding, base + 3) >= 0.0) as u8;
                votes >= 3
            } else if on_thirds_line(idx) {
                // ON THE LINE: XOR-bind differential
                let s1 = hp_dot_random(embedding, idx, seed) >= 0.0;
                let s2 = hp_dot_random(embedding, idx, seed2) >= 0.0;
                s1 ^ s2
            } else {
                // BACKGROUND: golden-angle structured
                hp_dot_golden(embedding, idx) >= 0.0
            };
            if bit_set {
                word |= 1u64 << b;
            }
        }
        result.words[w] = word;
    }
    result
}

// ============================================================================
// STRATEGY 5: DOUBLE-LINED INNER RECTANGLE
//
// The inner rectangle (between the 1/3 lines) gets TWO full projections,
// XOR-bound. The entire inner third x inner third.
//
//   +-----+===========+=====+
//   |     |  DOUBLE   |     |   Double = 2 projections, XOR-bound
//   |     |  BIND     |     |   Single = 1 projection (background)
//   +-----+===========+=====+
//   |     |           |     |
//   +-----+-----------+-----+
// ============================================================================

fn simhash_double_inner(embedding: &[f32], seed: u64) -> Container {
    let seed2 = seed ^ 0xBADC_0DED_EADB_EEF0;
    let cols = 91usize;
    let mut result = Container::zero();
    for w in 0..CONTAINER_WORDS {
        let mut word = 0u64;
        for b in 0..64u32 {
            let idx = w * 64 + b as usize;
            let row = idx / cols;
            let col = idx % cols;
            let rows = CONTAINER_BITS / cols;
            let in_inner = row >= rows / 3 && row < rows * 2 / 3
                        && col >= cols / 3 && col < cols * 2 / 3;
            let bit_set = if in_inner {
                // INNER RECTANGLE: double projection, XOR-bind
                let s1 = hp_dot_random(embedding, idx, seed) >= 0.0;
                let s2 = hp_dot_random(embedding, idx, seed2) >= 0.0;
                s1 ^ s2
            } else {
                // OUTER: single projection
                hp_dot_random(embedding, idx, seed) >= 0.0
            };
            if bit_set {
                word |= 1u64 << b;
            }
        }
        result.words[w] = word;
    }
    result
}

// ============================================================================
// STRATEGY 6: RESONANCE RADAR — 4 HDR POPCNTs at corners
//
// Place 4 "radar stations" at the power points (1/3 crossings).
// Each station computes a local Hamming neighborhood:
//   - Station projects its local quadrant with a unique seed
//   - Quadrants are then XOR-bound at the seams
//
// Like 4 radar dishes at the corners of a field, each scanning
// a different arc. The combined signal covers 360 degrees.
//
//   R1 ---- line ---- R2
//   |                  |
//   line    center   line
//   |                  |
//   R3 ---- line ---- R4
//
// Each Ri uses a different seed → different hyperplane family.
// The XOR at boundaries creates interference patterns that
// encode the RELATIVE structure between quadrants.
// ============================================================================

fn simhash_resonance_radar(embedding: &[f32], seed: u64) -> Container {
    let cols = 91usize;
    let rows = CONTAINER_BITS / cols;
    let row_mid = rows / 2;
    let col_mid = cols / 2;

    // 4 different seeds for 4 quadrants (radar stations)
    let seeds = [
        seed,
        seed ^ 0xDEAD_BEEF_0000_0001,
        seed ^ 0xCAFE_BABE_0000_0002,
        seed ^ 0xFACE_F00D_0000_0003,
    ];

    let mut result = Container::zero();
    for w in 0..CONTAINER_WORDS {
        let mut word = 0u64;
        for b in 0..64u32 {
            let idx = w * 64 + b as usize;
            let row = idx / cols;
            let col = idx % cols;

            // Determine which quadrant (radar station)
            let q = if row < row_mid {
                if col < col_mid { 0 } else { 1 }
            } else {
                if col < col_mid { 2 } else { 3 }
            };

            // On a quadrant boundary? XOR-bind two adjacent quadrant projections
            let on_row_boundary = row.abs_diff(row_mid) < 3;
            let on_col_boundary = col.abs_diff(col_mid) < 3;

            let bit_set = if on_row_boundary && on_col_boundary {
                // Center crosshair: XOR all 4 quadrant projections
                let s0 = hp_dot_random(embedding, idx, seeds[0]) >= 0.0;
                let s1 = hp_dot_random(embedding, idx, seeds[1]) >= 0.0;
                let s2 = hp_dot_random(embedding, idx, seeds[2]) >= 0.0;
                let s3 = hp_dot_random(embedding, idx, seeds[3]) >= 0.0;
                // 4-way XOR: true if odd number of positives
                s0 ^ s1 ^ s2 ^ s3
            } else if on_row_boundary {
                // Horizontal boundary: XOR top and bottom
                let q_top = if col < col_mid { 0 } else { 1 };
                let q_bot = if col < col_mid { 2 } else { 3 };
                let s_top = hp_dot_random(embedding, idx, seeds[q_top]) >= 0.0;
                let s_bot = hp_dot_random(embedding, idx, seeds[q_bot]) >= 0.0;
                s_top ^ s_bot
            } else if on_col_boundary {
                // Vertical boundary: XOR left and right
                let q_left = if row < row_mid { 0 } else { 2 };
                let q_right = if row < row_mid { 1 } else { 3 };
                let s_left = hp_dot_random(embedding, idx, seeds[q_left]) >= 0.0;
                let s_right = hp_dot_random(embedding, idx, seeds[q_right]) >= 0.0;
                s_left ^ s_right
            } else {
                // Interior: single quadrant projection
                hp_dot_random(embedding, idx, seeds[q]) >= 0.0
            };

            if bit_set {
                word |= 1u64 << b;
            }
        }
        result.words[w] = word;
    }

    result
}

// ============================================================================
// STRATEGY 7: 60-DEGREE HEX CROSS
//
// A crosshair with 6 arms at 60-degree spacing (hexagonal pattern).
// Like the Fujifilm X-Trans 6x6 grid — aperiodic, every direction
// sampled without repetition.
//
// For each bit, take 6 measurements at 60-degree golden-angle offsets
// and majority-vote (4 out of 6 must agree).
//
//         *
//        / \
//   *---+---*      6-pointed star measurement pattern
//        \ /       Each arm = a different hyperplane direction
//         *        Majority vote = robust bit decision
//
// The 60-degree spacing is irrational relative to the golden angle,
// so the pattern never repeats. This is the hexagonal X-Trans idea.
// ============================================================================

fn simhash_hex_cross(embedding: &[f32], seed: u64) -> Container {
    let total_bits = CONTAINER_WORDS * 64;
    // 6 arms at 60-degree spacing: offsets are total_bits/6 apart
    let arm_offset = total_bits / 6;

    // Pre-compute all raw signs
    let mut signs = vec![false; total_bits];
    for bit_idx in 0..total_bits {
        signs[bit_idx] = hp_dot_random(embedding, bit_idx, seed) >= 0.0;
    }

    let mut result = Container::zero();
    for w in 0..CONTAINER_WORDS {
        let mut word = 0u64;
        for b in 0..64u32 {
            let i = w * 64 + b as usize;

            // 6-point hex cross
            let s0 = signs[i];
            let s1 = signs[(i + arm_offset) % total_bits];
            let s2 = signs[(i + 2 * arm_offset) % total_bits];
            let s3 = signs[(i + 3 * arm_offset) % total_bits];
            let s4 = signs[(i + 4 * arm_offset) % total_bits];
            let s5 = signs[(i + 5 * arm_offset) % total_bits];

            // Majority vote: 4 out of 6 must agree
            let count = s0 as u8 + s1 as u8 + s2 as u8
                      + s3 as u8 + s4 as u8 + s5 as u8;
            if count >= 4 {
                word |= 1u64 << b;
            }
        }
        result.words[w] = word;
    }

    result
}

// ============================================================================
// STRATEGY 8: INNER RECTANGLE FINGERPRINT — bind vs XOR vs hyperposition
//
// Take the inner rectangle (between the 1/3 lines in 2D layout).
// Project it THREE different ways and combine with different operators:
//   View A: random hyperplanes (seed 1)
//   View B: random hyperplanes (seed 2)
//   View C: golden-angle hyperplanes
//
// Inner rectangle: A XOR B XOR C (hyperposition — all three views bound)
// Lines: A XOR B (differential — two views)
// Background: A only (simple projection)
//
// The hyperposition at the center creates the densest encoding:
// 3 independent measurements per bit. The lines get 2. Background gets 1.
// Information density follows the rule of thirds: subject > composition > bg.
// ============================================================================

fn simhash_inner_fingerprint(embedding: &[f32], seed: u64) -> Container {
    let seed_b = seed ^ 0x1234_5678_9ABC_DEF0;
    let cols = 91usize;
    let rows = CONTAINER_BITS / cols;

    let mut result = Container::zero();
    for w in 0..CONTAINER_WORDS {
        let mut word = 0u64;
        for b in 0..64u32 {
            let idx = w * 64 + b as usize;
            let row = idx / cols;
            let col = idx % cols;

            let in_inner = row >= rows / 3 && row < rows * 2 / 3
                        && col >= cols / 3 && col < cols * 2 / 3;

            let bit_set = if in_inner {
                // INNER: hyperposition of 3 views (A XOR B XOR C)
                let a = hp_dot_random(embedding, idx, seed) >= 0.0;
                let b_sign = hp_dot_random(embedding, idx, seed_b) >= 0.0;
                let c = hp_dot_golden(embedding, idx) >= 0.0;
                a ^ b_sign ^ c
            } else if on_thirds_line(idx) {
                // LINE: differential of 2 views (A XOR B)
                let a = hp_dot_random(embedding, idx, seed) >= 0.0;
                let b_sign = hp_dot_random(embedding, idx, seed_b) >= 0.0;
                a ^ b_sign
            } else {
                // BACKGROUND: single view (A)
                hp_dot_random(embedding, idx, seed) >= 0.0
            };

            if bit_set {
                word |= 1u64 << b;
            }
        }
        result.words[w] = word;
    }

    result
}

// ============================================================================
// SYNTHETIC DATASET
// ============================================================================

struct Dataset {
    embeddings: Vec<Vec<f32>>,
    labels: Vec<usize>,
}

impl Dataset {
    fn generate(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let mut embeddings = Vec::with_capacity(N_TOTAL);
        let mut labels = Vec::with_capacity(N_TOTAL);
        let mut centroids = Vec::with_capacity(N_CLASSES);

        for _ in 0..N_CLASSES {
            let mut c: Vec<f32> = (0..DIM).map(|_| rng.next_f32()).collect();
            let norm: f32 = c.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut c { *x /= norm; }
            centroids.push(c);
        }

        for (ci, centroid) in centroids.iter().enumerate() {
            for _ in 0..SAMPLES_PER_CLASS {
                let mut s: Vec<f32> = centroid.iter()
                    .map(|&c| c + rng.next_gaussian() * 0.15)
                    .collect();
                let norm: f32 = s.iter().map(|x| x * x).sum::<f32>().sqrt();
                for x in &mut s { *x /= norm; }
                embeddings.push(s);
                labels.push(ci);
            }
        }

        Self { embeddings, labels }
    }
}

// ============================================================================
// MEASUREMENT
// ============================================================================

struct Stats {
    intra_mean: f64,
    intra_std: f64,
    inter_mean: f64,
    inter_std: f64,
    sep_ratio: f64,
    overlap: f64,
    pop_mean: f64,
    pop_std: f64,
}

fn measure(containers: &[Container], labels: &[usize]) -> Stats {
    let n = containers.len();
    let d = CONTAINER_BITS as f64;

    let mut intra = Vec::new();
    let mut inter = Vec::new();
    let mut rng = Rng::new(0xBEEF);

    for class in 0..N_CLASSES {
        let idx: Vec<usize> = (0..n).filter(|&i| labels[i] == class).collect();
        for i in 0..idx.len() {
            for j in (i+1)..idx.len() {
                intra.push(containers[idx[i]].hamming(&containers[idx[j]]) as f64);
            }
        }
    }

    let n_inter = intra.len().max(10_000);
    for _ in 0..n_inter {
        let i = (rng.next_u64() as usize) % n;
        let mut j = (rng.next_u64() as usize) % n;
        for _ in 0..20 {
            if labels[j] != labels[i] { break; }
            j = (rng.next_u64() as usize) % n;
        }
        if labels[j] != labels[i] {
            inter.push(containers[i].hamming(&containers[j]) as f64);
        }
    }

    let intra_m = intra.iter().sum::<f64>() / intra.len() as f64;
    let inter_m = inter.iter().sum::<f64>() / inter.len() as f64;
    let intra_s = (intra.iter().map(|&x| (x - intra_m).powi(2)).sum::<f64>()
        / intra.len() as f64).sqrt();
    let inter_s = (inter.iter().map(|&x| (x - inter_m).powi(2)).sum::<f64>()
        / inter.len() as f64).sqrt();

    let gap = (inter_m - intra_m).abs();
    let combined = (intra_s * intra_s + inter_s * inter_s).sqrt();
    let overlap = if combined > 0.0 {
        1.0 / (1.0 + (gap / combined * 1.7).exp())
    } else { 1.0 };

    let pops: Vec<f64> = containers.iter().map(|c| c.popcount() as f64).collect();
    let pm = pops.iter().sum::<f64>() / pops.len() as f64;
    let ps = (pops.iter().map(|&p| (p - pm).powi(2)).sum::<f64>() / pops.len() as f64).sqrt();

    Stats {
        intra_mean: intra_m / d,
        intra_std: intra_s / d,
        inter_mean: inter_m / d,
        inter_std: inter_s / d,
        sep_ratio: inter_m / intra_m,
        overlap,
        pop_mean: pm,
        pop_std: ps,
    }
}

fn run<F: Fn(&[f32]) -> Container>(name: &str, data: &Dataset, f: F) -> Stats {
    let t0 = Instant::now();
    let containers: Vec<Container> = data.embeddings.iter().map(|e| f(e)).collect();
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    let stats = measure(&containers, &data.labels);
    let d = CONTAINER_BITS as f64;

    println!("+--------------------------------------------------------------+");
    println!("|  {:<58} |", name);
    println!("+--------------------------------------------------------------+");
    println!("|  Intra: {:5.2}% +/- {:4.2}%  ({:4.0} +/- {:3.0} bits)               |",
        stats.intra_mean*100.0, stats.intra_std*100.0,
        stats.intra_mean*d, stats.intra_std*d);
    println!("|  Inter: {:5.2}% +/- {:4.2}%  ({:4.0} +/- {:3.0} bits)               |",
        stats.inter_mean*100.0, stats.inter_std*100.0,
        stats.inter_mean*d, stats.inter_std*d);
    println!("|  Separation: {:.4}x       Overlap: {:.4}                  |",
        stats.sep_ratio, stats.overlap);
    println!("|  Popcount: {:4.0} +/- {:3.0}    Time: {:6.1} ms                    |",
        stats.pop_mean, stats.pop_std, ms);
    println!("+--------------------------------------------------------------+");
    println!();
    stats
}

fn main() {
    println!();
    println!("+==============================================================+");
    println!("|   VSACLIP — Rule of Thirds Projection Experiment            |");
    println!("|   {} classes x {} samples = {} vectors x {}d            |",
        N_CLASSES, SAMPLES_PER_CLASS, N_TOTAL, DIM);
    println!("|   Container: {} bits, Line width: {} bits                |",
        CONTAINER_BITS, LINE_WIDTH);
    println!("+==============================================================+");
    println!();

    let data = Dataset::generate(42);
    let seed = 0xADA_2025;

    let s1 = run("1. BASELINE — Random hyperplanes",
        &data, |e| simhash_baseline(e, seed));

    let s2 = run("2. THIRDS LINES — XOR-bind at 1/3 boundaries",
        &data, |e| simhash_thirds_lines(e, seed));

    let s3 = run("3. POWER POINTS — Golden quad at crossings",
        &data, |e| simhash_power_points(e, seed));

    let s4 = run("4. FULL COMPOSITION — Lines + power + golden bg",
        &data, |e| simhash_full_composition(e, seed));

    let s5 = run("5. DOUBLE-LINED INNER RECTANGLE — 2x bind center",
        &data, |e| simhash_double_inner(e, seed));

    let s6 = run("6. RESONANCE RADAR — 4 corners XOR at seams",
        &data, |e| simhash_resonance_radar(e, seed));

    let s7 = run("7. HEX CROSS — 60-degree 6-arm majority",
        &data, |e| simhash_hex_cross(e, seed));

    let s8 = run("8. INNER FINGERPRINT — 3-view hyperposition",
        &data, |e| simhash_inner_fingerprint(e, seed));

    // Comparison
    println!("+==============================+=======+=======+=========+=========+========+");
    println!("| Strategy                     | Intra | Inter | Sep.    | Overlap | Pop    |");
    println!("+==============================+=======+=======+=========+=========+========+");

    let all: Vec<(&str, &Stats)> = vec![
        ("Baseline (random)",        &s1),
        ("Thirds lines (XOR)",       &s2),
        ("Power points (golden)",    &s3),
        ("Full composition",         &s4),
        ("Double inner rect",        &s5),
        ("Resonance radar",          &s6),
        ("Hex cross (60 deg)",       &s7),
        ("Inner fingerprint (3x)",   &s8),
    ];

    for (name, s) in &all {
        println!("| {:28} |{:5.1}% |{:5.1}% | {:6.4}x |  {:5.4} | {:4.0}  |",
            name, s.intra_mean*100.0, s.inter_mean*100.0,
            s.sep_ratio, s.overlap, s.pop_mean);
    }
    println!("+==============================+=======+=======+=========+=========+========+");

    // Winner
    let best_sep = all.iter()
        .max_by(|a, b| a.1.sep_ratio.partial_cmp(&b.1.sep_ratio).unwrap()).unwrap();
    let best_ovl = all.iter()
        .min_by(|a, b| a.1.overlap.partial_cmp(&b.1.overlap).unwrap()).unwrap();

    println!();
    println!("  Best separation: {} ({:.4}x)", best_sep.0, best_sep.1.sep_ratio);
    println!("  Lowest overlap:  {} ({:.4})", best_ovl.0, best_ovl.1.overlap);

    // Popcount balance
    let ideal = CONTAINER_BITS as f64 / 2.0;
    println!();
    println!("  Popcount balance (ideal = {:.0}):", ideal);
    for (name, s) in &all {
        let dev = (s.pop_mean - ideal).abs() / ideal * 100.0;
        println!("    {:28}: {:4.0} +/- {:3.0}  ({:.1}% off)",
            name, s.pop_mean, s.pop_std, dev);
    }
    println!();
}
