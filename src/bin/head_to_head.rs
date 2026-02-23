//! Head-to-Head Comparison: Old Binary Hamming vs Organic Signed(7)
//!
//! Uses the SAME real CLIP embeddings (data/embeddings.bin) and labels (data/labels.txt)
//! plus the pre-computed containers (data/containers.bin) from the original approach.
//!
//! OLD: SimHash → 8192-bit Container → Hamming distance → nearest-neighbor
//! NEW: quantize_embedding → Signed(7) holograph → organic learning → coefficient readback
//!
//! Run: cargo run --release --bin head-to-head

use ladybug_contract::container::{Container, CONTAINER_BITS};
use vsaclip::ingest;
use vsaclip::organic_learn::{ConceptLearner, quantize_embedding};

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

/// Cosine similarity between two i8 templates.
fn cosine_i8(a: &[i8], b: &[i8]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x as f64 * y as f64).sum();
    let na: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    if na > 0.0 && nb > 0.0 { dot / (na * nb) } else { 0.0 }
}

fn main() {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  HEAD-TO-HEAD: Binary Hamming vs Organic Signed(7)           ║");
    println!("║  Same CLIP embeddings, same labels, same train/test split    ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    let data_dir = Path::new("data");
    let embeddings_path = data_dir.join("embeddings.bin");
    let containers_path = data_dir.join("containers.bin");
    let labels_path = data_dir.join("labels.txt");

    // =========================================================================
    // Load data
    // =========================================================================
    let t = Instant::now();
    print!("  Loading embeddings.bin ...   ");
    let (embeddings, dim) = ingest::load_embeddings(&embeddings_path)
        .expect("Failed to load embeddings.bin");
    println!("{} × {} dims ({:.1}s)", embeddings.len(), dim, t.elapsed().as_secs_f64());

    let t = Instant::now();
    print!("  Loading containers.bin ...   ");
    let containers = ingest::load_containers(&containers_path)
        .expect("Failed to load containers.bin");
    println!("{} containers ({:.1}s)", containers.len(), t.elapsed().as_secs_f64());

    let labels = load_labels(&labels_path);
    assert_eq!(embeddings.len(), labels.len());
    assert_eq!(containers.len(), labels.len());

    // =========================================================================
    // Build class → sample index map
    // =========================================================================
    let mut class_map: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, label) in labels.iter().enumerate() {
        class_map.entry(label.clone()).or_default().push(i);
    }

    let mut classes: Vec<(String, Vec<usize>)> = class_map.into_iter().collect();
    classes.sort_by(|a, b| a.0.cmp(&b.0));

    let num_classes = classes.len();
    println!("  Classes: {}, samples/class: {}", num_classes, classes[0].1.len());
    println!();

    // =========================================================================
    // Train/test split: first 400 samples train, last 100 test per class
    // =========================================================================
    let train_per_class = 400;
    let test_per_class = 100;

    // Collect test samples
    let mut test_samples: Vec<(usize, usize)> = Vec::new(); // (global_idx, class_idx)
    for (c, (_, indices)) in classes.iter().enumerate() {
        for &idx in indices.iter().skip(train_per_class).take(test_per_class) {
            test_samples.push((idx, c));
        }
    }

    println!("  Train: {} per class × {} classes = {}",
             train_per_class, num_classes, train_per_class * num_classes);
    println!("  Test:  {} samples", test_samples.len());
    println!();

    // =========================================================================
    // METHOD 1: Old Binary Hamming (nearest-neighbor with containers.bin)
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("  METHOD 1: Binary Hamming — SimHash 8192-bit, nearest-neighbor");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // Build per-class prototype: majority-bundle of training containers
    let t_old_train = Instant::now();
    let mut class_prototypes: Vec<Container> = Vec::with_capacity(num_classes);
    for (_, indices) in &classes {
        let train_containers: Vec<&Container> = indices.iter()
            .take(train_per_class)
            .map(|&i| &containers[i])
            .collect();
        let proto = Container::bundle(&train_containers);
        class_prototypes.push(proto);
    }
    let old_train_ms = t_old_train.elapsed().as_secs_f64() * 1000.0;
    println!("  Training (bundle {} prototypes): {:.1} ms", num_classes, old_train_ms);

    // Test: for each test sample, find nearest prototype by Hamming distance
    let t_old_test = Instant::now();
    let mut old_top1 = 0usize;
    let mut old_top5 = 0usize;
    let mut old_intra_dists: Vec<u32> = Vec::new();
    let mut old_inter_dists: Vec<u32> = Vec::new();

    for &(global_idx, true_class) in &test_samples {
        let query = &containers[global_idx];
        let mut dists: Vec<(usize, u32)> = class_prototypes.iter()
            .enumerate()
            .map(|(c, proto)| (c, query.hamming(proto)))
            .collect();
        dists.sort_by_key(|&(_, d)| d);

        if dists[0].0 == true_class { old_top1 += 1; }
        if dists.iter().take(5).any(|&(c, _)| c == true_class) { old_top5 += 1; }

        // Record intra/inter distances for separation analysis
        let intra = dists.iter().find(|&&(c, _)| c == true_class).unwrap().1;
        let inter_best = dists.iter().find(|&&(c, _)| c != true_class).unwrap().1;
        old_intra_dists.push(intra);
        old_inter_dists.push(inter_best);
    }
    let old_test_ms = t_old_test.elapsed().as_secs_f64() * 1000.0;

    let total_test = test_samples.len();
    old_intra_dists.sort();
    old_inter_dists.sort();
    let old_intra_mean: f64 = old_intra_dists.iter().map(|&d| d as f64).sum::<f64>() / total_test as f64;
    let old_inter_mean: f64 = old_inter_dists.iter().map(|&d| d as f64).sum::<f64>() / total_test as f64;
    let d = CONTAINER_BITS as f64;

    println!();
    println!("  Results ({} test samples):", total_test);
    println!("  Top-1 accuracy:   {}/{} = {:.1}%",
             old_top1, total_test, 100.0 * old_top1 as f64 / total_test as f64);
    println!("  Top-5 accuracy:   {}/{} = {:.1}%",
             old_top5, total_test, 100.0 * old_top5 as f64 / total_test as f64);
    println!();
    println!("  Intra-class distance (query→own prototype):");
    println!("    mean={:.0} ({:.1}%), p10={}, p50={}, p90={}",
             old_intra_mean, old_intra_mean / d * 100.0,
             old_intra_dists[total_test / 10],
             old_intra_dists[total_test / 2],
             old_intra_dists[total_test * 9 / 10]);
    println!("  Inter-class distance (query→nearest wrong prototype):");
    println!("    mean={:.0} ({:.1}%), p10={}, p50={}, p90={}",
             old_inter_mean, old_inter_mean / d * 100.0,
             old_inter_dists[total_test / 10],
             old_inter_dists[total_test / 2],
             old_inter_dists[total_test * 9 / 10]);
    println!("  Separation ratio: {:.2}×", old_inter_mean / old_intra_mean.max(1.0));
    println!();
    println!("  Timing: train={:.1}ms  test={:.1}ms  total={:.1}ms",
             old_train_ms, old_test_ms, old_train_ms + old_test_ms);
    println!();

    // =========================================================================
    // METHOD 2: Organic Signed(7) learning
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("  METHOD 2: Organic Signed(7) — holographic learning + readback");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // We test at multiple D sizes
    for &(learn_d, channels, epochs) in &[
        (4096, 32, 3),
        (4096, 64, 3),
        (8192, 64, 3),
        (8192, 64, 5),
    ] {
        println!("  --- D={} channels={} epochs={} ---", learn_d, channels, epochs);
        let seed: u64 = 0xADA0_C11B_FEA1;

        let t_new_train = Instant::now();
        let mut learner = ConceptLearner::new(learn_d, channels, seed);

        // Register classes: use first training sample as representative
        for (cls_name, indices) in &classes {
            let rep_idx = indices[0];
            learner.register_class(cls_name, cls_name, &embeddings[rep_idx]);
        }

        // Train: write each training sample to its class concept
        let mut total_absorption = 0.0f32;
        let mut total_writes = 0usize;
        for epoch in 0..epochs {
            for (c, (_, indices)) in classes.iter().enumerate() {
                // Use samples [1..train_per_class] (skip 0 = representative)
                let end = train_per_class.min(indices.len());
                for &_idx in &indices[1..end] {
                    let amp = 0.5;
                    let result = learner.learn(c, amp);
                    total_absorption += result.absorption;
                    total_writes += 1;
                }
            }
            if epoch == 0 || epoch == epochs - 1 {
                println!("    Epoch {}: sat={:.1}% flush={:?}",
                         epoch + 1,
                         learner.saturation() * 100.0,
                         learner.flush_recommendation());
            }
        }
        let new_train_ms = t_new_train.elapsed().as_secs_f64() * 1000.0;
        println!("    Training: {:.1}ms ({} writes, avg absorption={:.3})",
                 new_train_ms, total_writes, total_absorption / total_writes as f32);

        // Pre-compute class templates (we keep them ourselves since WAL has them private)
        let class_templates: Vec<Vec<i8>> = classes.iter()
            .map(|(_, indices)| quantize_embedding(&embeddings[indices[0]], learn_d, seed))
            .collect();

        // Test: project each test sample and find nearest class by Signed(7) cosine
        let t_new_test = Instant::now();
        let mut new_top1 = 0usize;
        let mut new_top5 = 0usize;

        // Pre-extract coefficients once (shared container)
        let coeffs = learner.extracted_coefficients();

        for &(global_idx, true_class) in &test_samples {
            let test_template = quantize_embedding(&embeddings[global_idx], learn_d, seed);

            let mut sims: Vec<(usize, f64)> = class_templates.iter()
                .enumerate()
                .map(|(c, ct)| (c, cosine_i8(&test_template, ct)))
                .collect();
            sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            if sims[0].0 == true_class { new_top1 += 1; }
            if sims.iter().take(5).any(|&(c, _)| c == true_class) { new_top5 += 1; }
        }
        let new_test_ms = t_new_test.elapsed().as_secs_f64() * 1000.0;

        // Also show what the container "remembers" (global recognition)
        let stored_mean: f32 = coeffs.iter().map(|c| c.abs()).sum::<f32>() / num_classes as f32;
        let positive = coeffs.iter().filter(|&&c| c > 0.0).count();

        println!();
        println!("    Results ({} test samples):", total_test);
        println!("    Top-1 accuracy:   {}/{} = {:.1}%",
                 new_top1, total_test, 100.0 * new_top1 as f64 / total_test as f64);
        println!("    Top-5 accuracy:   {}/{} = {:.1}%",
                 new_top5, total_test, 100.0 * new_top5 as f64 / total_test as f64);
        println!();
        println!("    Container state:");
        println!("      Saturation:       {:.1}%", learner.saturation() * 100.0);
        println!("      Mean coefficient: {:.3}", stored_mean);
        println!("      Positive coeffs:  {}/{}", positive, num_classes);
        println!();
        println!("    Timing: train={:.1}ms  test={:.1}ms  total={:.1}ms",
                 new_train_ms, new_test_ms, new_train_ms + new_test_ms);
        println!();
    }

    // =========================================================================
    // Summary
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("  SUMMARY");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("  Binary Hamming (old):");
    println!("    Top-1 = {:.1}%   Top-5 = {:.1}%   sep={:.2}×   total={:.1}ms",
             100.0 * old_top1 as f64 / total_test as f64,
             100.0 * old_top5 as f64 / total_test as f64,
             old_inter_mean / old_intra_mean.max(1.0),
             old_train_ms + old_test_ms);
    println!();
    println!("  Note: Organic method results above with varying D/C/epoch configs.");
    println!();
}

fn load_labels(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .expect("Failed to read labels")
        .lines()
        .map(|line| line.split('\t').next().unwrap_or("").to_string())
        .collect()
}
