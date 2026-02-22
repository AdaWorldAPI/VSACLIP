//! Recognition Experiment: Gram-Schmidt Projection vs Binary Hamming
//!
//! Uses real CLIP embeddings (data/embeddings.bin) and pre-computed
//! containers (data/containers.bin) to compare:
//!
//! 1. HAMMING BRUTE (old): SimHash 8192-bit → Hamming distance → nearest-neighbor
//! 2. PROJECTION READOUT: Gram-Schmidt projections as class scores (new)
//! 3. INDEPENDENT READOUT: Non-orthogonal projections
//! 4. TWO-STAGE: Hamming shortlist + Gram-Schmidt re-rank
//!
//! The key insight: learn path uses Gram-Schmidt, so recognize path MUST too.
//! Same operation, different interpretation of the output.
//!
//! Run: cargo run --release --bin recognize-experiment

use ladybug_contract::container::Container;
use vsaclip::ingest;
use rustynum_oracle::recognize::Recognizer;

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

fn main() {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  RECOGNITION EXPERIMENT: Gram-Schmidt Projection vs Hamming      ║");
    println!("║  learn path = recognize path = Gram-Schmidt projection           ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
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

    // =========================================================================
    // Train/test split
    // =========================================================================
    let train_per_class = 400;
    let test_per_class = 100;

    let test_samples_full: Vec<(usize, usize)> = classes.iter().enumerate()
        .flat_map(|(c, (_, indices))| {
            indices.iter().skip(train_per_class).take(test_per_class)
                .map(move |&idx| (idx, c))
        })
        .collect();

    println!("  Classes: {}, train: {}/class, test: {} total",
             num_classes, train_per_class, test_samples_full.len());
    println!();

    // =========================================================================
    // METHOD 1: Old Binary Hamming (baseline, from containers.bin)
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  METHOD 1: Binary Hamming — 8192-bit nearest-neighbor");
    println!("═══════════════════════════════════════════════════════════════════");

    let t = Instant::now();
    let mut class_prototypes: Vec<Container> = Vec::with_capacity(num_classes);
    for (_, indices) in &classes {
        let train_containers: Vec<&Container> = indices.iter()
            .take(train_per_class)
            .map(|&i| &containers[i])
            .collect();
        class_prototypes.push(Container::bundle(&train_containers));
    }
    let train_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let (top1, top5) = hamming_evaluate(&test_samples_full, &containers, &class_prototypes);
    let test_ms = t.elapsed().as_secs_f64() * 1000.0;
    let total_test_full = test_samples_full.len();

    println!("  Top-1: {}/{} = {:.1}%   Top-5: {}/{} = {:.1}%",
             top1, total_test_full, pct(top1, total_test_full),
             top5, total_test_full, pct(top5, total_test_full));
    println!("  Time: train={:.0}ms test={:.0}ms total={:.0}ms",
             train_ms, test_ms, train_ms + test_ms);
    println!();

    // =========================================================================
    // METHOD 2–4: Gram-Schmidt Recognition at various D
    // =========================================================================
    // (D, channels, label, max_test_per_class)
    // Larger D → slower projection → test fewer samples
    for &(d, channels, label, test_limit) in &[
        (8192, 64, "8K", 50),
        (16384, 128, "16K", 20),
        (32768, 200, "32K", 10),
        (65536, 200, "64K", 5),
    ] {
        // Reduce test samples for larger D
        let test_samples: Vec<(usize, usize)> = classes.iter().enumerate()
            .flat_map(|(c, (_, indices))| {
                indices.iter().skip(train_per_class).take(test_limit)
                    .map(move |&idx| (idx, c))
            })
            .collect();
        let total_test = test_samples.len();
        println!("═══════════════════════════════════════════════════════════════════");
        println!("  METHOD 2-4: Gram-Schmidt Recognition — D={} ({})", d, label);
        println!("═══════════════════════════════════════════════════════════════════");

        let seed: u64 = 0xADA0_C11B_FEA1;

        // Register + Learn
        let t = Instant::now();
        let mut rec = Recognizer::new(d, channels.min(num_classes), dim, seed);

        for (cls_name, indices) in &classes {
            let rep_idx = indices[0];
            rec.register_class(cls_name, &embeddings[rep_idx]);
        }

        // Train: accumulate samples
        for (c, (_, indices)) in classes.iter().enumerate() {
            let end = train_per_class.min(indices.len());
            for &idx in &indices[1..end] {
                rec.learn(c, &embeddings[idx], 0.5);
            }
        }
        let train_ms = t.elapsed().as_secs_f64() * 1000.0;

        println!("  Training: {:.0}ms ({} classes × {} samples, sat={:.1}%)",
                 train_ms, num_classes, train_per_class - 1, rec.saturation() * 100.0);

        // Test: Orthogonal (full Gram-Schmidt) recognition
        let t = Instant::now();
        let mut gs_top1 = 0usize;
        let mut gs_top5 = 0usize;
        for &(global_idx, true_class) in &test_samples {
            let result = rec.recognize(&embeddings[global_idx]);
            if result.top1_class == true_class { gs_top1 += 1; }
            if result.ranked.iter().take(5).any(|&(c, _)| c == true_class) { gs_top5 += 1; }
        }
        let gs_ms = t.elapsed().as_secs_f64() * 1000.0;

        println!("  Gram-Schmidt (orth): Top-1={:.1}%  Top-5={:.1}%  ({:.0}ms)",
                 pct(gs_top1, total_test), pct(gs_top5, total_test), gs_ms);

        // Test: Independent (no orthogonalization) recognition
        let t = Instant::now();
        let mut indep_top1 = 0usize;
        let mut indep_top5 = 0usize;
        for &(global_idx, true_class) in &test_samples {
            let result = rec.recognize_independent(&embeddings[global_idx]);
            if result.top1_class == true_class { indep_top1 += 1; }
            if result.ranked.iter().take(5).any(|&(c, _)| c == true_class) { indep_top5 += 1; }
        }
        let indep_ms = t.elapsed().as_secs_f64() * 1000.0;

        println!("  Independent:         Top-1={:.1}%  Top-5={:.1}%  ({:.0}ms)",
                 pct(indep_top1, total_test), pct(indep_top5, total_test), indep_ms);

        // Test: Two-stage (Hamming shortlist + GS re-rank)
        let t = Instant::now();
        let mut ts_top1 = 0usize;
        let mut ts_top5 = 0usize;
        for &(global_idx, true_class) in &test_samples {
            let result = rec.recognize_two_stage(&embeddings[global_idx], 20);
            if result.top1_class == true_class { ts_top1 += 1; }
            if result.ranked.iter().take(5).any(|&(c, _)| c == true_class) { ts_top5 += 1; }
        }
        let ts_ms = t.elapsed().as_secs_f64() * 1000.0;

        println!("  Two-stage (top-20):  Top-1={:.1}%  Top-5={:.1}%  ({:.0}ms)",
                 pct(ts_top1, total_test), pct(ts_top5, total_test), ts_ms);

        // Novelty detection: average residual energy for known vs unknown-ish
        let mut known_residuals = Vec::new();
        for &(global_idx, _) in test_samples.iter().take(200) {
            let result = rec.recognize(&embeddings[global_idx]);
            known_residuals.push(result.residual_energy);
        }
        let known_mean: f32 = known_residuals.iter().sum::<f32>() / known_residuals.len() as f32;
        println!("  Avg residual energy: {:.4}", known_mean);
        println!();
    }

    // =========================================================================
    // Summary table
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  SUMMARY — {} classes", num_classes);
    println!("═══════════════════════════════════════════════════════════════════");
    println!();
    println!("  Method                    Top-1%    Top-5%");
    println!("  ─────────────────────────────────────────");
    println!("  Binary Hamming (8K)       {:.1}%     {:.1}%  ({} test)",
             pct(top1, total_test_full), pct(top5, total_test_full), total_test_full);
    println!("  (see Gram-Schmidt results above for D=8K,16K,32K,64K)");
    println!();
}

fn hamming_evaluate(
    test_samples: &[(usize, usize)],
    containers: &[Container],
    prototypes: &[Container],
) -> (usize, usize) {
    let mut top1 = 0usize;
    let mut top5 = 0usize;
    for &(global_idx, true_class) in test_samples {
        let query = &containers[global_idx];
        let mut dists: Vec<(usize, u32)> = prototypes.iter()
            .enumerate()
            .map(|(c, proto)| (c, query.hamming(proto)))
            .collect();
        dists.sort_by_key(|&(_, d)| d);
        if dists[0].0 == true_class { top1 += 1; }
        if dists.iter().take(5).any(|&(c, _)| c == true_class) { top5 += 1; }
    }
    (top1, top5)
}

fn pct(n: usize, total: usize) -> f64 {
    100.0 * n as f64 / total.max(1) as f64
}

fn load_labels(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .expect("Failed to read labels")
        .lines()
        .map(|line| line.split('\t').next().unwrap_or("").to_string())
        .collect()
}
