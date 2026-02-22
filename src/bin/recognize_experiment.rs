//! Recognition Experiment: Gram-Schmidt Projection vs Binary Hamming
//!
//! Uses real CLIP embeddings (data/embeddings.bin) and pre-computed
//! containers (data/containers.bin) to compare:
//!
//! 1. HAMMING BRUTE (old): SimHash 8192-bit → Hamming distance → nearest-neighbor
//! 2. GRAM-SCHMIDT READOUT: project query → GS projections as class scores
//! 3. INDEPENDENT READOUT: project query → independent dot products
//!
//! Key insight: learn path uses Gram-Schmidt, recognize path must too.
//!
//! Run: cargo run --release --bin recognize-experiment

use ladybug_contract::container::Container;
use vsaclip::ingest;
use rustynum_oracle::recognize::{Recognizer, Projector64K};

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

    // Load data
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

    // Build class map
    let mut class_map: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, label) in labels.iter().enumerate() {
        class_map.entry(label.clone()).or_default().push(i);
    }
    let mut classes: Vec<(String, Vec<usize>)> = class_map.into_iter().collect();
    classes.sort_by(|a, b| a.0.cmp(&b.0));
    let num_classes = classes.len();

    let train_per_class = 20; // fast: 20 train samples
    let test_per_class = 50;  // 50 test per class = 10K test

    let test_samples: Vec<(usize, usize)> = classes.iter().enumerate()
        .flat_map(|(c, (_, indices))| {
            indices.iter().skip(train_per_class).take(test_per_class)
                .map(move |&idx| (idx, c))
        })
        .collect();
    let total_test = test_samples.len();

    println!("  {} classes, {} train/class, {} test samples", num_classes, train_per_class, total_test);
    println!();

    // =========================================================================
    // Baseline: Binary Hamming (8192-bit, 400 train samples for majority-bundle)
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  BASELINE: Binary Hamming — 8192-bit nearest-neighbor");
    println!("═══════════════════════════════════════════════════════════════════");

    let t = Instant::now();
    let class_protos_20: Vec<Container> = classes.iter()
        .map(|(_, indices)| {
            let refs: Vec<&Container> = indices.iter()
                .take(train_per_class)
                .map(|&i| &containers[i])
                .collect();
            Container::bundle(&refs)
        })
        .collect();
    let train_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let (h_top1, h_top5) = hamming_nn(&test_samples, &containers, &class_protos_20);
    let test_ms = t.elapsed().as_secs_f64() * 1000.0;

    println!("  Top-1: {}/{} = {:.1}%   Top-5: {:.1}%   ({:.0}ms train + {:.0}ms test)",
             h_top1, total_test, pct(h_top1, total_test),
             pct(h_top5, total_test), train_ms, test_ms);

    // Also test with 400 train (the full bundle — this is the "old" result)
    let class_protos_400: Vec<Container> = classes.iter()
        .map(|(_, indices)| {
            let refs: Vec<&Container> = indices.iter()
                .take(400)
                .map(|&i| &containers[i])
                .collect();
            Container::bundle(&refs)
        })
        .collect();
    let (h400_top1, h400_top5) = hamming_nn(&test_samples, &containers, &class_protos_400);
    println!("  (400 train bundle: Top-1={:.1}% Top-5={:.1}%)",
             pct(h400_top1, total_test), pct(h400_top5, total_test));
    println!();

    // =========================================================================
    // Gram-Schmidt Recognition at various D
    // =========================================================================
    // Strategy: compute class centroid in embedding space, project ONCE per class,
    // then recognition = project query + GS readout (no WAL writes needed for test).

    for &(d, label, test_limit) in &[
        (4096usize, "4K", 50usize),
        (8192, "8K", 50),
        (16384, "16K", 20),
        (32768, "32K", 10),
    ] {
        println!("═══════════════════════════════════════════════════════════════════");
        println!("  GRAM-SCHMIDT RECOGNITION — D={} ({})", d, label);
        println!("═══════════════════════════════════════════════════════════════════");

        let seed: u64 = 0xADA0_C11B_FEA1;
        let channels = d / 64; // reasonable channel count

        // Step 1: Compute class centroids in embedding space
        let t = Instant::now();
        let centroids: Vec<Vec<f32>> = classes.iter()
            .map(|(_, indices)| {
                let n = train_per_class.min(indices.len());
                let mut centroid = vec![0.0f32; dim];
                for &idx in &indices[..n] {
                    for (j, &v) in embeddings[idx].iter().enumerate() {
                        centroid[j] += v;
                    }
                }
                for v in &mut centroid { *v /= n as f32; }
                // L2 normalize
                let norm: f32 = centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 { for v in &mut centroid { *v /= norm; } }
                centroid
            })
            .collect();

        // Step 2: Register centroids with the Recognizer
        let mut rec = Recognizer::new(d, channels.min(num_classes), dim, seed);
        for (c, (cls_name, _)) in classes.iter().enumerate() {
            rec.register_class(cls_name, &centroids[c]);
        }

        // Step 3: (Optional) Train a few WAL writes to build the container
        for (c, (_, indices)) in classes.iter().enumerate() {
            let n = train_per_class.min(indices.len());
            for &idx in &indices[1..n.min(6)] { // 5 writes per class max
                rec.learn(c, &embeddings[idx], 0.5);
            }
        }
        let train_ms = t.elapsed().as_secs_f64() * 1000.0;
        println!("  Train: {:.0}ms (centroid + {} WAL writes, sat={:.1}%)",
                 train_ms, num_classes * 5, rec.saturation() * 100.0);

        // Step 4: Test with reduced samples for larger D
        let test_subset: Vec<&(usize, usize)> = test_samples.iter()
            .enumerate()
            .filter(|(i, _)| i % (test_per_class / test_limit).max(1) == 0)
            .map(|(_, s)| s)
            .collect();
        let n_test = test_subset.len();

        // Gram-Schmidt (orthogonal) recognition
        let t = Instant::now();
        let mut gs_top1 = 0usize;
        let mut gs_top5 = 0usize;
        for &&(global_idx, true_class) in &test_subset {
            let result = rec.recognize(&embeddings[global_idx]);
            if result.top1_class == true_class { gs_top1 += 1; }
            if result.ranked.iter().take(5).any(|&(c, _)| c == true_class) { gs_top5 += 1; }
        }
        let gs_ms = t.elapsed().as_secs_f64() * 1000.0;

        println!("  Gram-Schmidt (orth): Top-1={}/{} = {:.1}%  Top-5={:.1}%  ({:.0}ms, {:.1}ms/q)",
                 gs_top1, n_test, pct(gs_top1, n_test),
                 pct(gs_top5, n_test), gs_ms, gs_ms / n_test as f64);

        // Independent (non-orthogonal) recognition
        let t = Instant::now();
        let mut indep_top1 = 0usize;
        let mut indep_top5 = 0usize;
        for &&(global_idx, true_class) in &test_subset {
            let result = rec.recognize_independent(&embeddings[global_idx]);
            if result.top1_class == true_class { indep_top1 += 1; }
            if result.ranked.iter().take(5).any(|&(c, _)| c == true_class) { indep_top5 += 1; }
        }
        let indep_ms = t.elapsed().as_secs_f64() * 1000.0;

        println!("  Independent:         Top-1={}/{} = {:.1}%  Top-5={:.1}%  ({:.0}ms)",
                 indep_top1, n_test, pct(indep_top1, n_test),
                 pct(indep_top5, n_test), indep_ms);

        // Novelty: residual energy
        let mut residuals = Vec::new();
        for &&(global_idx, _) in test_subset.iter().take(100) {
            let result = rec.recognize(&embeddings[global_idx]);
            residuals.push(result.residual_energy);
        }
        let mean_res: f32 = residuals.iter().sum::<f32>() / residuals.len().max(1) as f32;
        println!("  Avg residual: {:.4}", mean_res);
        println!();
    }

    // Summary
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  SUMMARY — {} classes, Hamming baseline with {} train vs {} train",
             num_classes, train_per_class, 400);
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Hamming (20 train):   Top-1={:.1}%  Top-5={:.1}%",
             pct(h_top1, total_test), pct(h_top5, total_test));
    println!("  Hamming (400 train):  Top-1={:.1}%  Top-5={:.1}%",
             pct(h400_top1, total_test), pct(h400_top5, total_test));
    println!("  (Gram-Schmidt results above)");
    println!();
}

fn hamming_nn(
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
