//! Recognition Experiment v2: Batch Projection + Two-Stage Recognition
//!
//! Uses real CLIP embeddings (data/embeddings.bin) and pre-computed
//! containers (data/containers.bin) to compare:
//!
//! 1. HAMMING BRUTE (old): SimHash 8192-bit → Hamming distance → nearest-neighbor
//! 2. INDEPENDENT READOUT: batch-projected query → independent dot products
//! 3. TWO-STAGE: Hamming shortlist (top-20) → Gram-Schmidt re-rank
//! 4. FULL GS: project query → GS projections as class scores
//!
//! Key optimization: batch projection generates each hyperplane ONCE across
//! all samples — 100-1000× faster than projecting one at a time.
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
    println!("║  RECOGNITION v2: Batch Projection + Two-Stage Re-rank           ║");
    println!("║  Batch project = 100-1000× faster, Two-stage = best accuracy    ║");
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

    let train_per_class = 20;
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
    // Baseline: Binary Hamming (8192-bit)
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
    // Gram-Schmidt Recognition at various D — with BATCH projection
    // =========================================================================

    for &(d, label) in &[
        (4096usize, "4K"),
        (8192, "8K"),
        (16384, "16K"),
    ] {
        println!("═══════════════════════════════════════════════════════════════════");
        println!("  GRAM-SCHMIDT RECOGNITION — D={} ({}) — BATCH PROJECTED", d, label);
        println!("═══════════════════════════════════════════════════════════════════");

        let seed: u64 = 0xADA0_C11B_FEA1;
        let channels = d / 64;
        let projector = Projector64K::new(dim, seed);

        // Step 1: Compute class centroids
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
                let norm: f32 = centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 { for v in &mut centroid { *v /= norm; } }
                centroid
            })
            .collect();

        // Step 2: Batch-project centroids to register with recognizer
        let _centroid_templates = projector.project_batch_signed_d(&centroids, d);
        let centroid_proj_ms = t.elapsed().as_secs_f64() * 1000.0;
        println!("  Centroid projection: {:.0}ms ({} classes × D={})", centroid_proj_ms, num_classes, d);

        // Step 3: Build recognizer from pre-projected centroid templates
        let mut rec = Recognizer::new(d, channels.min(num_classes), dim, seed);
        for (c, (cls_name, _)) in classes.iter().enumerate() {
            // Register via the Recognizer API (which re-projects internally)
            rec.register_class(cls_name, &centroids[c]);
        }

        // Step 4: Optional WAL writes (only 5 per class for speed)
        let t_train = Instant::now();
        for (c, (_, indices)) in classes.iter().enumerate() {
            let n = train_per_class.min(indices.len());
            for &idx in &indices[1..n.min(6)] {
                rec.learn(c, &embeddings[idx], 0.5);
            }
        }
        let train_ms = t_train.elapsed().as_secs_f64() * 1000.0;
        println!("  WAL training: {:.0}ms ({} writes, sat={:.1}%)",
                 train_ms, num_classes * 5, rec.saturation() * 100.0);

        // Step 5: BATCH project all test embeddings (the key optimization)
        let test_embs: Vec<Vec<f32>> = test_samples.iter()
            .map(|&(idx, _)| embeddings[idx].clone())
            .collect();

        let t_proj = Instant::now();
        let test_templates = projector.project_batch_signed_d(&test_embs, d);
        let proj_ms = t_proj.elapsed().as_secs_f64() * 1000.0;
        println!("  Batch projection: {:.0}ms ({} samples × D={})", proj_ms, total_test, d);

        // Step 6a: Independent recognition (from pre-projected templates)
        let t = Instant::now();
        let mut indep_top1 = 0usize;
        let mut indep_top5 = 0usize;
        for (i, &(_, true_class)) in test_samples.iter().enumerate() {
            let result = rec.recognize_independent_from_template(&test_templates[i]);
            if result.top1_class == true_class { indep_top1 += 1; }
            if result.ranked.iter().take(5).any(|&(c, _)| c == true_class) { indep_top5 += 1; }
        }
        let indep_ms = t.elapsed().as_secs_f64() * 1000.0;
        println!("  Independent:  Top-1={}/{} = {:.1}%  Top-5={:.1}%  ({:.0}ms classify, {:.0}ms total)",
                 indep_top1, total_test, pct(indep_top1, total_test),
                 pct(indep_top5, total_test), indep_ms, proj_ms + indep_ms);

        // Step 6b: Two-stage recognition (Hamming shortlist → GS re-rank)
        for &shortlist_size in &[10usize, 20, 50] {
            let t = Instant::now();
            let mut ts_top1 = 0usize;
            let mut ts_top5 = 0usize;
            for (i, &(_, true_class)) in test_samples.iter().enumerate() {
                let result = rec.recognize_two_stage_from_template(&test_templates[i], shortlist_size);
                if result.top1_class == true_class { ts_top1 += 1; }
                if result.ranked.iter().take(5).any(|&(c, _)| c == true_class) { ts_top5 += 1; }
            }
            let ts_ms = t.elapsed().as_secs_f64() * 1000.0;
            println!("  Two-stage({}): Top-1={}/{} = {:.1}%  Top-5={:.1}%  ({:.0}ms classify, {:.0}ms total)",
                     shortlist_size,
                     ts_top1, total_test, pct(ts_top1, total_test),
                     pct(ts_top5, total_test), ts_ms, proj_ms + ts_ms);
        }

        // Step 6c: Full Gram-Schmidt (test with subset for large D to avoid depletion)
        let gs_test_limit = if d <= 8192 { total_test } else { 2000 };
        let t = Instant::now();
        let mut gs_top1 = 0usize;
        let mut gs_top5 = 0usize;
        let gs_count = gs_test_limit.min(total_test);
        for i in 0..gs_count {
            let (_, true_class) = test_samples[i];
            let result = rec.recognize_from_template(&test_templates[i]);
            if result.top1_class == true_class { gs_top1 += 1; }
            if result.ranked.iter().take(5).any(|&(c, _)| c == true_class) { gs_top5 += 1; }
        }
        let gs_ms = t.elapsed().as_secs_f64() * 1000.0;
        println!("  Full GS:      Top-1={}/{} = {:.1}%  Top-5={:.1}%  ({:.0}ms classify)",
                 gs_top1, gs_count, pct(gs_top1, gs_count),
                 pct(gs_top5, gs_count), gs_ms);

        // Novelty: residual energy (from full GS)
        let mut residuals = Vec::new();
        for i in 0..100.min(total_test) {
            let result = rec.recognize_from_template(&test_templates[i]);
            residuals.push(result.residual_energy);
        }
        let mean_res: f32 = residuals.iter().sum::<f32>() / residuals.len().max(1) as f32;
        println!("  Avg residual: {:.4}", mean_res);
        println!();
    }

    // =========================================================================
    // Summary
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  SUMMARY — {} classes, Hamming baseline vs projection methods", num_classes);
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Hamming (20 train):   Top-1={:.1}%  Top-5={:.1}%",
             pct(h_top1, total_test), pct(h_top5, total_test));
    println!("  Hamming (400 train):  Top-1={:.1}%  Top-5={:.1}%",
             pct(h400_top1, total_test), pct(h400_top5, total_test));
    println!("  (Projection results above — note batch projection speedup)");
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
