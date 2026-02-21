//! CogRecord8K Experiment — 8 KB = 4 × 16,384-bit containers.
//!
//! Validates:
//! 1. WideContainer (16384-bit) operations match Container (8192-bit) semantics
//! 2. CogRecord8K layout: META / CAM / INDEX / EMBED
//! 3. Dual distance metrics: VPOPCNTDQ Hamming vs VNNI int8 dot-product
//! 4. Wide HDR sweep: 3-stage early exit on 16384-bit containers
//! 5. Edge encoding: XOR-bind with rotation addresses
//! 6. Legacy upgrade path: 2KB → 8KB
//!
//! Run: cargo run --release --bin cogrecord8k-experiment

use ladybug_contract::container::Container;
use ladybug_contract::wide_container::{WideContainer, EmbeddingFormat, WIDE_BITS, WIDE_WORDS, WIDE_BYTES};
use ladybug_contract::cogrecord8k::{
    CogRecord8K, RECORD8K_BITS, RECORD8K_BYTES, RECORD8K_AVX512_ITERS,
    SLOT_META, SLOT_CAM, SLOT_INDEX, SLOT_EMBED,
};
use vsaclip::wide_sweep::{wide_hdr_sweep, wide_full_sweep, WideSweepConfig};
use vsaclip::hdr::HdrConfig;

use std::time::Instant;

fn main() {
    println!();
    println!("CogRecord8K Experiment");
    println!("======================");
    println!("8 KB = 4 × 16,384-bit WideContainers = 65,536 bits");
    println!("128 AVX-512 VPOPCNTDQ instructions for full record");
    println!();

    let all_pass = [
        test_wide_container_basics(),
        test_cogrecord8k_layout(),
        test_dual_distance_metrics(),
        test_wide_hdr_sweep(),
        test_edge_encoding(),
        test_legacy_upgrade(),
        test_int8_embedding_search(),
    ]
    .iter()
    .all(|&p| p);

    println!();
    if all_pass {
        println!("ALL EXPERIMENTS PASSED");
    } else {
        println!("SOME EXPERIMENTS FAILED");
        std::process::exit(1);
    }
}

// =============================================================================
// TEST 1: WideContainer Basics
// =============================================================================

fn test_wide_container_basics() -> bool {
    println!("═══════════════════════════════════════════════════════════════");
    println!("TEST 1: WideContainer (16,384-bit) Basics");
    println!("═══════════════════════════════════════════════════════════════");

    let a = WideContainer::random(42);
    let b = WideContainer::random(99);

    // Size verification
    let size = std::mem::size_of::<WideContainer>();
    println!("  Size:     {} bytes = {} KB", size, size / 1024);
    assert_eq!(size, 2048, "WideContainer must be exactly 2 KB");

    // Hamming distance
    let d = a.hamming(&b);
    println!("  Hamming:  {} (expected ~{}, sigma ~64)", d, WIDE_BITS / 2);
    assert!(d > 7800 && d < 8600, "random hamming out of range");

    // XOR self-inverse
    let xored = a.xor(&a);
    assert!(xored.is_zero(), "a XOR a must be zero");
    println!("  XOR:      self-inverse ✓");

    // Permute roundtrip
    let rotated = a.permute(1337);
    let back = rotated.permute(-1337);
    assert_eq!(a, back, "permute roundtrip failed");
    println!("  Permute:  roundtrip ✓");

    // Popcount
    let pc = a.popcount();
    println!("  Popcount: {} (expected ~{})", pc, WIDE_BITS / 2);
    assert!(pc > 7800 && pc < 8600, "random popcount out of range");

    // Container upgrade
    let c = Container::random(42);
    let wide = WideContainer::from_container(&c);
    let back = wide.to_container_lower();
    assert_eq!(c, back, "container upgrade roundtrip failed");
    println!("  Upgrade:  8192→16384 roundtrip ✓");

    println!("  PASSED");
    println!();
    true
}

// =============================================================================
// TEST 2: CogRecord8K Layout
// =============================================================================

fn test_cogrecord8k_layout() -> bool {
    println!("═══════════════════════════════════════════════════════════════");
    println!("TEST 2: CogRecord8K Layout (4 × 16,384 = 65,536 bits = 8 KB)");
    println!("═══════════════════════════════════════════════════════════════");

    let size = std::mem::size_of::<CogRecord8K>();
    println!("  Size:       {} bytes = {} KB", size, size / 1024);
    assert_eq!(size, 8192, "CogRecord8K must be exactly 8 KB");

    println!("  Bits:       {}", RECORD8K_BITS);
    assert_eq!(RECORD8K_BITS, 65_536);

    println!("  Bytes:      {}", RECORD8K_BYTES);
    assert_eq!(RECORD8K_BYTES, 8_192);

    println!("  AVX-512:    {} VPOPCNTDQ iterations", RECORD8K_AVX512_ITERS);
    assert_eq!(RECORD8K_AVX512_ITERS, 128);

    // Container slots
    let mut r = CogRecord8K::new();
    r.cam = WideContainer::random(42);
    r.index = WideContainer::random(99);

    assert_eq!(r.container(SLOT_META), &r.meta);
    assert_eq!(r.container(SLOT_CAM), &r.cam);
    assert_eq!(r.container(SLOT_INDEX), &r.index);
    assert_eq!(r.container(SLOT_EMBED), &r.embed);
    println!("  Slots:      META=0 CAM=1 INDEX=2 EMBED=3 ✓");

    // Bytes roundtrip
    let bytes = r.as_bytes();
    let r2 = CogRecord8K::from_bytes(bytes);
    assert_eq!(r.cam, r2.cam);
    assert_eq!(r.index, r2.index);
    println!("  Serialize:  8192-byte roundtrip ✓");

    println!("  PASSED");
    println!();
    true
}

// =============================================================================
// TEST 3: Dual Distance Metrics
// =============================================================================

fn test_dual_distance_metrics() -> bool {
    println!("═══════════════════════════════════════════════════════════════");
    println!("TEST 3: Dual Distance — VPOPCNTDQ Hamming vs VNNI int8 dot");
    println!("═══════════════════════════════════════════════════════════════");

    // --- Binary Hamming path ---
    let mut a = CogRecord8K::new();
    let mut b = CogRecord8K::new();
    a.embed = WideContainer::random(1);
    b.embed = WideContainer::random(2);
    // Default format = Binary16K
    let (dist, is_hamming) = a.embed_distance(&b);
    assert!(is_hamming, "default should be Hamming");
    println!("  Binary16K:  hamming={} ✓", dist);

    // --- Int8 dot-product path ---
    let mut a8 = CogRecord8K::with_embedding_format(EmbeddingFormat::Int8x1024);
    let mut b8 = CogRecord8K::with_embedding_format(EmbeddingFormat::Int8x1024);

    // Create simple int8 embeddings
    let emb_a: Vec<i8> = (0..1024).map(|i| ((i % 127) as i8 - 64)).collect();
    let emb_b: Vec<i8> = (0..1024).map(|i| ((i % 127) as i8 - 64)).collect();
    a8.embed.store_int8(&emb_a);
    b8.embed.store_int8(&emb_b);

    let (dist8, is_hamming8) = a8.embed_distance(&b8);
    assert!(!is_hamming8, "Int8 should NOT be Hamming");
    println!("  Int8x1024:  neg_dot={} (same vector → large negative) ✓", dist8);

    // Cosine similarity should be ~1.0 for same vector
    let cosine = a8.embed_cosine(&b8);
    println!("  Cosine:     {:.4} (expected ~1.0)", cosine);
    assert!((cosine - 1.0).abs() < 0.01, "same vector cosine should be ~1.0");

    // --- Different vectors ---
    let emb_c: Vec<i8> = (0..1024).map(|i| (((i * 7 + 13) % 127) as i8 - 64)).collect();
    b8.embed.store_int8(&emb_c);
    let cosine_diff = a8.embed_cosine(&b8);
    println!("  Diff cos:   {:.4} (should be < 1.0)", cosine_diff);
    assert!(cosine_diff < 0.99, "different vectors should have cosine < 1.0");

    // --- Format switching ---
    println!("  Formats:    Binary16K={}, Int8x1024={}, Int8x2048={}, Int4x1024={}, Int4x4096={}",
             EmbeddingFormat::Binary16K.dimensions(),
             EmbeddingFormat::Int8x1024.dimensions(),
             EmbeddingFormat::Int8x2048.dimensions(),
             EmbeddingFormat::Int4x1024.dimensions(),
             EmbeddingFormat::Int4x4096.dimensions());

    println!("  PASSED");
    println!();
    true
}

// =============================================================================
// TEST 4: Wide HDR Sweep
// =============================================================================

fn test_wide_hdr_sweep() -> bool {
    println!("═══════════════════════════════════════════════════════════════");
    println!("TEST 4: Wide HDR Sweep (16,384-bit, 3-stage early exit)");
    println!("═══════════════════════════════════════════════════════════════");

    let n = 10_000;
    let query = WideContainer::random(42);
    let threshold = (WIDE_BITS as f32 * 0.30) as u32; // ~4915

    // Build corpus with planted matches
    let mut corpus: Vec<WideContainer> = (0..n as u64)
        .map(|s| WideContainer::random(s + 1000))
        .collect();

    // Plant 10 near matches
    let planted = 10;
    for i in 0..planted {
        let mut near = query.clone();
        let target_dist = 1000 + i * 300; // 1000 to 3700 bits
        let mut state = (i as u64) ^ 0xCAFEBABE;
        for _ in 0..target_dist {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let bit = (state as usize) % WIDE_BITS;
            near.words[bit / 64] ^= 1u64 << (bit % 64);
        }
        corpus[i * (n / planted)] = near;
    }

    // Wide HDR sweep
    let config = WideSweepConfig {
        threshold,
        safety: 1.5,
        hdr: HdrConfig::for_bits(WIDE_BITS as u32),
        limit: 0,
    };

    let t = Instant::now();
    let result = wide_hdr_sweep(&query, &corpus, &config);
    let hdr_time = t.elapsed();

    // Full sweep for comparison
    let t = Instant::now();
    let full = wide_full_sweep(&query, &corpus, threshold);
    let full_time = t.elapsed();

    println!("  Corpus:       {} WideContainers", n);
    println!("  Threshold:    {} / {} = {:.1}%", threshold, WIDE_BITS, 100.0 * threshold as f64 / WIDE_BITS as f64);
    println!("  HDR matches:  {}", result.matches.len());
    println!("  Full matches: {}", full.len());
    println!("  S1 survivors: {} ({:.1}%)", result.stage1_survivors,
             100.0 * result.stage1_survivors as f64 / n as f64);
    println!("  S2 survivors: {} ({:.1}%)", result.stage2_survivors,
             100.0 * result.stage2_survivors as f64 / n as f64);

    let full_instructions = n as u64 * WIDE_WORDS as u64;
    let speedup = full_instructions as f64 / result.instructions as f64;
    println!("  Instructions: {} HDR vs {} full = {:.1}× speedup",
             result.instructions, full_instructions, speedup);
    println!("  Time:         {:.1}μs HDR vs {:.1}μs full = {:.1}× speedup",
             hdr_time.as_secs_f64() * 1e6, full_time.as_secs_f64() * 1e6,
             full_time.as_secs_f64() / hdr_time.as_secs_f64());

    // Verify zero false negatives
    let hdr_indices: std::collections::HashSet<usize> =
        result.matches.iter().map(|m| m.index).collect();
    let mut false_negatives = 0;
    for fm in &full {
        if !hdr_indices.contains(&fm.index) {
            false_negatives += 1;
        }
    }
    println!("  False neg:    {} (must be 0)", false_negatives);
    assert_eq!(false_negatives, 0, "ZERO false negatives required");

    assert!(speedup > 3.0, "Expected >3× speedup");

    println!("  PASSED");
    println!();
    true
}

// =============================================================================
// TEST 5: Edge Encoding
// =============================================================================

fn test_edge_encoding() -> bool {
    println!("═══════════════════════════════════════════════════════════════");
    println!("TEST 5: Edge Encoding — XOR-bind with rotation addresses");
    println!("═══════════════════════════════════════════════════════════════");

    let src = WideContainer::random(10);
    let rel = WideContainer::random(20);
    let tgt = WideContainer::random(30);

    // Encode edge
    let edge = CogRecord8K::make_edge(&src, &rel, &tgt);

    // Recover target
    let recovered_tgt = CogRecord8K::recover_target(&edge, &src, &rel);
    assert_eq!(recovered_tgt, tgt, "target recovery failed");
    println!("  Target recovery:  ✓");

    // Recover source
    let recovered_src = CogRecord8K::recover_source(&edge, &rel, &tgt);
    assert_eq!(recovered_src, src, "source recovery failed");
    println!("  Source recovery:  ✓");

    // Edge is dissimilar to all components (XOR property)
    let d_src = edge.hamming(&src);
    let d_rel = edge.hamming(&rel);
    let d_tgt = edge.hamming(&tgt);
    println!("  Edge-to-src:  {} (expected ~{})", d_src, WIDE_BITS / 2);
    println!("  Edge-to-rel:  {} (expected ~{})", d_rel, WIDE_BITS / 2);
    println!("  Edge-to-tgt:  {} (expected ~{})", d_tgt, WIDE_BITS / 2);
    // All should be near expected distance (half bits different)
    for (name, d) in [("src", d_src), ("rel", d_rel), ("tgt", d_tgt)] {
        assert!(d > 7000 && d < 9400, "edge-to-{name} distance {d} out of range");
    }

    // Multiple edges in one INDEX container (bundled)
    let rel2 = WideContainer::random(21);
    let tgt2 = WideContainer::random(31);
    let edge2 = CogRecord8K::make_edge(&src, &rel2, &tgt2);

    // Bundle edges
    let bundled = WideContainer::bundle(&[&edge, &edge2]);
    // Bundled edge should be closer to both individual edges than to random
    let random = WideContainer::random(999);
    let d_edge1 = bundled.hamming(&edge);
    let d_edge2 = bundled.hamming(&edge2);
    let d_random = bundled.hamming(&random);
    println!("  Bundled-to-edge1: {} (should be < random)", d_edge1);
    println!("  Bundled-to-edge2: {} (should be < random)", d_edge2);
    println!("  Bundled-to-random: {}", d_random);
    assert!(d_edge1 < d_random, "bundle should be closer to edge1 than random");
    assert!(d_edge2 < d_random, "bundle should be closer to edge2 than random");

    println!("  PASSED");
    println!();
    true
}

// =============================================================================
// TEST 6: Legacy Upgrade
// =============================================================================

fn test_legacy_upgrade() -> bool {
    println!("═══════════════════════════════════════════════════════════════");
    println!("TEST 6: Legacy Upgrade (2 KB CogRecord → 8 KB CogRecord8K)");
    println!("═══════════════════════════════════════════════════════════════");

    let mut legacy = ladybug_contract::record::CogRecord::default();
    legacy.content = Container::random(42);
    legacy.meta = Container::random(1);

    // Upgrade
    let upgraded = CogRecord8K::from_legacy(&legacy);
    println!("  Upgraded:   2 KB → 8 KB");

    // Check CAM contains legacy content in lower half
    let cam_lower = upgraded.cam.to_container_lower();
    assert_eq!(cam_lower, legacy.content, "CAM lower half should match legacy content");
    println!("  CAM lower:  matches legacy content ✓");

    // Check META contains legacy meta in lower half
    let meta_lower = upgraded.meta.to_container_lower();
    assert_eq!(meta_lower, legacy.meta, "META lower half should match legacy meta");
    println!("  META lower: matches legacy meta ✓");

    // Check INDEX and EMBED are zeroed
    assert!(upgraded.index.is_zero(), "INDEX should be zeroed after upgrade");
    assert!(upgraded.embed.is_zero(), "EMBED should be zeroed after upgrade");
    println!("  INDEX:      zeroed ✓");
    println!("  EMBED:      zeroed ✓");

    // Downgrade roundtrip
    let downgraded = upgraded.to_legacy();
    assert_eq!(downgraded.meta, legacy.meta);
    assert_eq!(downgraded.content, legacy.content);
    println!("  Downgrade:  roundtrip ✓");

    println!("  PASSED");
    println!();
    true
}

// =============================================================================
// TEST 7: Int8 Embedding Search
// =============================================================================

fn test_int8_embedding_search() -> bool {
    println!("═══════════════════════════════════════════════════════════════");
    println!("TEST 7: Int8 Embedding Search (VNNI dot-product path)");
    println!("═══════════════════════════════════════════════════════════════");

    let dims = 1024;
    let n = 1000;

    // Create a query embedding
    let query_emb: Vec<i8> = (0..dims).map(|i| ((i * 3 + 7) % 255) as i8).collect();
    let mut query_record = CogRecord8K::with_embedding_format(EmbeddingFormat::Int8x1024);
    query_record.embed.store_int8(&query_emb);

    // Create corpus with one planted near-duplicate
    let mut records: Vec<CogRecord8K> = Vec::with_capacity(n);
    for i in 0..n {
        let mut r = CogRecord8K::with_embedding_format(EmbeddingFormat::Int8x1024);
        // Random embeddings
        let emb: Vec<i8> = (0..dims).map(|j| (((i * 7 + j * 13 + 42) % 255) as i8)).collect();
        r.embed.store_int8(&emb);
        r.cam = WideContainer::random(i as u64 + 5000); // random CAM
        records.push(r);
    }

    // Plant a near-duplicate at position 500
    let mut near_emb = query_emb.clone();
    // Perturb slightly
    for i in 0..10 {
        near_emb[i * 100] = near_emb[i * 100].wrapping_add(1);
    }
    records[500].embed.store_int8(&near_emb);

    // Search using dot product
    let t = Instant::now();
    let mut best_idx = 0;
    let mut best_dot = i32::MIN;
    for (idx, r) in records.iter().enumerate() {
        let dot = query_record.embed.int8_dot(&r.embed, dims);
        if dot > best_dot {
            best_dot = dot;
            best_idx = idx;
        }
    }
    let search_time = t.elapsed();

    println!("  Corpus:     {} records × {} dims int8", n, dims);
    println!("  Best match: index={} dot={}", best_idx, best_dot);
    println!("  Expected:   index=500 (planted near-duplicate)");
    println!("  Time:       {:.1}μs", search_time.as_secs_f64() * 1e6);

    assert_eq!(best_idx, 500, "should find planted near-duplicate");

    // Also verify cosine
    let cosine = query_record.embed_cosine(&records[500]);
    println!("  Cosine:     {:.4} (planted near-dup)", cosine);
    assert!(cosine > 0.99, "near-duplicate cosine should be > 0.99");

    // Embedding format summary
    println!();
    println!("  Container 3 Embedding Formats:");
    println!("  ┌──────────────┬──────────┬──────────┬─────────────┬──────────────────┐");
    println!("  │ Format       │ Dims     │ Bits/dim │ Total bits  │ Hardware         │");
    println!("  ├──────────────┼──────────┼──────────┼─────────────┼──────────────────┤");
    println!("  │ Binary16K    │ 16384    │ 1        │ 16384 (full)│ VPOPCNTDQ        │");
    println!("  │ Int8×1024    │ 1024     │ 8        │ 8192 (half) │ VPDPBUSD (VNNI)  │");
    println!("  │ Int8×2048    │ 2048     │ 8        │ 16384 (full)│ VPDPBUSD (VNNI)  │");
    println!("  │ Int4×1024    │ 1024     │ 4        │ 4096 (¼)    │ Software         │");
    println!("  │ Int4×4096    │ 4096     │ 4        │ 16384 (full)│ Software         │");
    println!("  └──────────────┴──────────┴──────────┴─────────────┴──────────────────┘");

    println!("  PASSED");
    println!();
    true
}
