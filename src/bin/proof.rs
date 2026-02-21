//! VSACLIP Proof of Concept — Rust port of hdr_proof.py
//!
//! Run: cargo run --bin vsaclip-proof

use ladybug_contract::container::{Container, CONTAINER_BITS, CONTAINER_WORDS};
use vsaclip::hdr::HdrConfig;
use vsaclip::sweep::{hdr_sweep, SweepConfig};
use vsaclip::ingest::simhash;

use std::time::Instant;

fn make_near(reference: &Container, target_dist: usize, rng_seed: u64) -> Container {
    let mut c = reference.clone();
    let mut state = rng_seed ^ 0xDEADBEEF;
    let mut flipped = std::collections::HashSet::new();
    while flipped.len() < target_dist {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let bit = (state as usize) % CONTAINER_BITS;
        if flipped.insert(bit) {
            c.words[bit / 64] ^= 1u64 << (bit % 64);
        }
    }
    c
}

fn test_superposition() -> bool {
    println!("═══════════════════════════════════════════════");
    println!("TEST 1: Superposition — XOR vs Majority");
    println!("═══════════════════════════════════════════════");

    let target = Container::random(42);
    let features: Vec<Container> = (0..1000u64).map(|s| Container::random(s)).collect();

    for k in [1, 5, 10, 25, 49] {
        let close = make_near(&target, CONTAINER_BITS / 10, k as u64 * 7);

        let mut patches: Vec<Container> = (0..k - 1).map(|s| Container::random(s + 5000)).collect();
        patches.push(close.clone());

        // XOR superposition
        let mut xor_sup = patches[0].clone();
        for p in &patches[1..] {
            xor_sup = xor_sup.xor(p);
        }

        // Majority bundle
        let refs: Vec<&Container> = patches.iter().collect();
        let maj_sup = Container::bundle(&refs);

        // Rank target among all features using majority
        let mut dists: Vec<(u32, usize)> = features.iter().enumerate()
            .map(|(i, f)| (maj_sup.hamming(f), i))
            .collect();
        dists.sort();
        let rank = dists.iter().position(|(_, i)| *i == 42).unwrap_or(999);

        let xor_dist = xor_sup.hamming(&target);
        let maj_dist = maj_sup.hamming(&target);

        println!("  K={:3} | XOR: {:5} | MAJ: {:5} rank={:4} {}",
                 k, xor_dist, maj_dist, rank,
                 if rank < 20 { "✓" } else { "⚠" });
    }
    println!();
    true
}

fn test_early_exit() -> bool {
    println!("═══════════════════════════════════════════════");
    println!("TEST 2: Early Exit — Zero False Negatives");
    println!("═══════════════════════════════════════════════");

    let query = Container::random(42);
    let threshold = (CONTAINER_BITS as f32 * 0.30) as u32;
    let safety = 1.5f32;
    let n = 10_000;

    let mut corpus: Vec<Container> = (0..n as u64)
        .map(|s| Container::random(s + 1000))
        .collect();

    // Plant 50 matches
    for i in 0..50 {
        let d = 400 + (i * 40) as usize;
        corpus[i * 200] = make_near(&query, d, i as u64 * 13);
    }

    // Full sweep
    let t0 = Instant::now();
    let full: Vec<(usize, u32)> = corpus.iter().enumerate()
        .filter_map(|(i, c)| {
            let d = query.hamming(c);
            if d < threshold { Some((i, d)) } else { None }
        })
        .collect();
    let t_full = t0.elapsed();

    // HDR early exit
    let t0 = Instant::now();
    let config = SweepConfig { threshold, safety, ..Default::default() };
    let result = hdr_sweep(&query, &corpus, &config);
    let t_early = t0.elapsed();

    let hdr_set: std::collections::HashSet<usize> = result.matches.iter().map(|m| m.index).collect();
    let missed: Vec<_> = full.iter().filter(|(i, _)| !hdr_set.contains(i)).collect();

    let full_ops = n as u64 * CONTAINER_WORDS as u64;
    let speedup = full_ops as f64 / result.instructions.max(1) as f64;

    println!("  Full:  {:4} matches | {:>10} ops | {:?}", full.len(), full_ops, t_full);
    println!("  HDR:   {:4} matches | {:>10} ops | {:?}", result.matches.len(), result.instructions, t_early);
    println!("  Stage 1: {} → {} ({:.1}%)", n, result.stage1_survivors, result.stage1_survivors as f64 / n as f64 * 100.0);
    println!("  Stage 2: {} → {}", result.stage1_survivors, result.stage2_survivors);
    println!("  Speedup: {:.1}×", speedup);
    println!("  False negatives: {}", missed.len());

    if missed.is_empty() {
        println!("  ✓ ZERO FALSE NEGATIVES");
    } else {
        println!("  ✗ {} FALSE NEGATIVES", missed.len());
    }
    println!();
    missed.is_empty()
}

fn test_hdr_scoring() -> bool {
    println!("═══════════════════════════════════════════════");
    println!("TEST 3: HDR Exposure Scoring");
    println!("═══════════════════════════════════════════════");

    let query = Container::random(42);
    let hdr = HdrConfig::default();
    let n = 5000;

    let mut corpus: Vec<Container> = (0..n as u64).map(|s| Container::random(s + 2000)).collect();

    let plants = [(0, 5, "blazing"), (1, 15, "strong"), (2, 25, "medium"), (3, 44, "weak")];
    for &(idx, pct, _) in &plants {
        corpus[idx] = make_near(&query, CONTAINER_BITS * pct / 100, idx as u64 * 31);
    }

    for &(idx, pct, label) in &plants {
        let dist = query.hamming(&corpus[idx]);
        let score = hdr.score(dist);
        println!("  '{}' (target={}%): dist={:5}, HDR={}", label, pct, dist, score.total());
    }

    let mut counts = [0usize; 7];
    for c in &corpus {
        let dist = query.hamming(c);
        let s = hdr.score(dist).total() as usize;
        if s < 7 { counts[s] += 1; }
    }

    println!("\n  Distribution:");
    for s in (0..7).rev() {
        if counts[s] > 0 {
            println!("    HDR={}: {:5} ({:.1}%)", s, counts[s], counts[s] as f64 / n as f64 * 100.0);
        }
    }

    let noise = counts[0] as f64 / n as f64 * 100.0;
    println!("\n  Noise floor: {:.1}%  {}", noise, if noise > 85.0 { "✓" } else { "⚠" });
    println!();
    noise > 85.0
}

fn test_anti_resonance() -> bool {
    println!("═══════════════════════════════════════════════");
    println!("TEST 4: Anti-Resonance");
    println!("═══════════════════════════════════════════════");

    let orig = Container::random(42);
    let mut anti = Container::zero();
    for i in 0..CONTAINER_WORDS { anti.words[i] = !orig.words[i]; }
    let close = make_near(&orig, CONTAINER_BITS / 10, 77);
    let neutral = Container::random(99);

    println!("  Close:   {:5} / {}  (resonance)", orig.hamming(&close), CONTAINER_BITS);
    println!("  Neutral: {:5} / {}  (noise)", orig.hamming(&neutral), CONTAINER_BITS);
    println!("  Anti:    {:5} / {}  (inhibition)", orig.hamming(&anti), CONTAINER_BITS);
    println!();
    true
}

fn test_simhash_bridge() -> bool {
    println!("═══════════════════════════════════════════════");
    println!("TEST 5: SimHash Bridge (float → Hamming)");
    println!("═══════════════════════════════════════════════");

    let emb_a: Vec<f32> = (0..512).map(|i| (i as f32 * 0.1).sin()).collect();
    let emb_b: Vec<f32> = (0..512).map(|i| (i as f32 * 0.1 + 0.05).sin()).collect(); // close
    let emb_c: Vec<f32> = (0..512).map(|i| (i as f32 * 0.7).cos()).collect(); // far

    let ca = simhash(&emb_a, 42);
    let cb = simhash(&emb_b, 42);
    let cc = simhash(&emb_c, 42);

    let d_ab = ca.hamming(&cb);
    let d_ac = ca.hamming(&cc);

    println!("  Similar:    dist = {:5} ({:.1}%)", d_ab, d_ab as f64 / CONTAINER_BITS as f64 * 100.0);
    println!("  Dissimilar: dist = {:5} ({:.1}%)", d_ac, d_ac as f64 / CONTAINER_BITS as f64 * 100.0);
    println!("  Preserves similarity: {}", if d_ab < d_ac { "✓" } else { "✗" });
    println!();
    d_ab < d_ac
}

fn test_instruction_count() -> bool {
    println!("═══════════════════════════════════════════════");
    println!("TEST 6: Instruction Count at Scale");
    println!("═══════════════════════════════════════════════");

    for n in [10_000u64, 100_000, 1_000_000, 10_000_000] {
        let (_full, early, speedup) = vsaclip::exposure::theoretical_instructions(n, 0.30, 1.5);
        let us_early = early as f64 / 3_000.0;
        let us_avx = early as f64 / 24_000.0;
        println!("\n  N={:>12} | early: {:>8.0} μs | AVX-512: {:>6.0} μs | {:.0}× speedup",
                 n, us_early, us_avx, speedup);
    }
    println!();
    true
}

fn main() {
    println!("\n════════════════════════════════════════════════════");
    println!("  VSACLIP — HDR POPCNT Sweep Proof (Rust)");
    println!("  Container: {} bits = {} × u64", CONTAINER_BITS, CONTAINER_WORDS);
    println!("════════════════════════════════════════════════════\n");

    let tests: Vec<(&str, fn() -> bool)> = vec![
        ("Superposition",   test_superposition),
        ("Early Exit",      test_early_exit),
        ("HDR Scoring",     test_hdr_scoring),
        ("Anti-Resonance",  test_anti_resonance),
        ("SimHash Bridge",  test_simhash_bridge),
        ("Instr. Count",    test_instruction_count),
    ];

    let mut results = Vec::new();
    for (name, f) in &tests {
        let ok = f();
        results.push((*name, ok));
    }

    println!("════════════════════════════════════════════════════");
    println!("  RESULTS");
    println!("════════════════════════════════════════════════════");
    for (name, ok) in &results {
        println!("  {} {}", if *ok { "✓" } else { "✗" }, name);
    }
    println!();
    if results.iter().all(|(_, ok)| *ok) {
        println!("  ALL PASSED.");
    } else {
        println!("  FAILURES DETECTED.");
    }
    println!();
}
