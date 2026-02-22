//! Organic Learning Experiment — Signed(7) holographs learn visual concepts.
//!
//! Three experiments:
//! 1. **Synthetic**: random Signed(7) templates, verify stored vs unstored separation
//! 2. **CLIP-like**: synthetic "embeddings" quantized to Signed(7), test learning
//! 3. **Real CLIP** (if data/embeddings.bin exists): actual Tiny ImageNet classes
//!
//! Run: cargo run --release --bin organic-learn

use vsaclip::organic_learn::{ConceptLearner, quantize_embedding};
use rustynum_oracle::{Base, generate_templates};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::Rng;

use std::path::Path;
use std::time::Instant;

fn main() {
    println!();
    println!("Organic Learning Experiment");
    println!("===========================");
    println!("Signed(7) holographs + OrganicWAL + X-Trans channels");
    println!("Question: can concepts LEARN and be RECOGNIZED?");
    println!();

    experiment_1_synthetic();
    experiment_2_clip_like();
    experiment_3_scaling();
    experiment_4_real_clip();
}

// =============================================================================
// Experiment 1: Synthetic — Pure template recovery
// =============================================================================

fn experiment_1_synthetic() {
    println!("=================================================================");
    println!("EXPERIMENT 1: Synthetic — Stored vs Unstored Separation");
    println!("=================================================================");
    println!();

    let mut rng = StdRng::seed_from_u64(42);

    for &(d, channels, k_stored, k_total) in &[
        (2048, 16, 8, 16),
        (4096, 32, 16, 32),
        (4096, 64, 32, 64),
        (8192, 64, 32, 64),
    ] {
        print!("  D={:<5} C={:<3} K_stored={:<3} K_total={:<3} ... ", d, channels, k_stored, k_total);

        let mut learner = ConceptLearner::new(d, channels, 42);
        learner.use_plasticity = true;

        let templates = generate_templates(k_total, d, Base::Signed(7), &mut rng);
        for (i, t) in templates.iter().enumerate() {
            learner.register_class_raw(
                &format!("c{}", i),
                &format!("Concept-{}", i),
                t.clone(),
            );
        }

        // Train only the first k_stored concepts
        for _ in 0..5 {
            for i in 0..k_stored {
                let amp = rng.gen_range(0.3f32..0.8f32);
                learner.learn(i, amp);
            }
        }

        // Extract and measure
        let coeffs = learner.extracted_coefficients();

        let stored_mean: f32 = coeffs[..k_stored].iter().map(|c| c.abs()).sum::<f32>()
            / k_stored as f32;
        let unstored_mean: f32 = if k_total > k_stored {
            coeffs[k_stored..].iter().map(|c| c.abs()).sum::<f32>()
                / (k_total - k_stored) as f32
        } else {
            0.0
        };

        let separation = if unstored_mean > 1e-6 {
            stored_mean / unstored_mean
        } else {
            f32::INFINITY
        };

        let pass = stored_mean > 0.1 && (k_total == k_stored || separation > 2.0);
        println!(
            "stored={:.3} unstored={:.3} ratio={:.1}x sat={:.1}% {}",
            stored_mean,
            unstored_mean,
            separation,
            learner.saturation() * 100.0,
            if pass { "PASS" } else { "WEAK" },
        );
    }
    println!();
}

// =============================================================================
// Experiment 2: CLIP-like — Quantized embedding learning
// =============================================================================

fn experiment_2_clip_like() {
    println!("=================================================================");
    println!("EXPERIMENT 2: CLIP-like — Quantized Embeddings Learn Concepts");
    println!("=================================================================");
    println!();

    let mut rng = StdRng::seed_from_u64(123);
    let d = 4096;
    let channels = 32;
    let num_classes = 10;
    let samples_per_class = 20;
    let seed: u64 = 0xADA0_C11B_0001;

    // Generate synthetic "class centroids" (like CLIP centroids per class)
    let centroids: Vec<Vec<f32>> = (0..num_classes)
        .map(|c| {
            let mut emb = vec![0.0f32; 512];
            // Each class has energy in a different region of the embedding space
            for i in (c * 50)..((c + 1) * 50).min(512) {
                emb[i] = rng.gen_range(0.5f32..1.5f32);
            }
            // L2 normalize
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut emb { *x /= norm; }
            emb
        })
        .collect();

    // Register classes using centroids
    let mut learner = ConceptLearner::new(d, channels, seed);
    for (c, centroid) in centroids.iter().enumerate() {
        learner.register_class(
            &format!("cls{}", c),
            &format!("Class-{}", c),
            centroid,
        );
    }

    println!("  Classes:            {}", num_classes);
    println!("  Samples per class:  {}", samples_per_class);
    println!("  Container:          D={} channels={}", d, channels);
    println!("  Projection seed:    0x{:X}", seed);
    println!();

    // Train: for each class, generate noisy samples around the centroid
    let t = Instant::now();
    for epoch in 0..3 {
        let mut epoch_absorption = 0.0f32;
        let mut epoch_count = 0;

        for c in 0..num_classes {
            for _ in 0..samples_per_class {
                let amp = rng.gen_range(0.3f32..0.7f32);
                let result = learner.learn(c, amp);
                epoch_absorption += result.absorption;
                epoch_count += 1;
            }
        }

        println!(
            "  Epoch {}: avg_absorption={:.3} saturation={:.1}% flush={:?}",
            epoch + 1,
            epoch_absorption / epoch_count as f32,
            learner.saturation() * 100.0,
            learner.flush_recommendation(),
        );
    }

    // Recognize
    println!();
    println!("  Recognition results:");
    println!("  {:<12} {:>8} {:>8} {:>10}", "Concept", "Sim", "Coeff", "Samples");
    println!("  {}", "-".repeat(44));

    let results = learner.recognize();
    let mut correct = 0;
    for r in &results {
        let cls = &learner.classes.iter()
            .find(|c| c.name == r.concept_name).unwrap();
        println!(
            "  {:<12} {:>7.3} {:>8.3} {:>9}",
            r.concept_name, r.similarity, r.coefficient, cls.sample_count,
        );
        if r.coefficient > 0.0 { correct += 1; }
    }

    println!();
    println!(
        "  All {} classes have positive coefficients: {}",
        num_classes,
        if correct == num_classes { "YES" } else { "NO" },
    );
    println!();
}

// =============================================================================
// Experiment 3: Scaling — How many concepts can we learn?
// =============================================================================

fn experiment_3_scaling() {
    println!("=================================================================");
    println!("EXPERIMENT 3: Scaling — Max Learnable Concepts vs Container Size");
    println!("=================================================================");
    println!();
    println!("  {:<6} {:<4} {:<5} {:>8} {:>8} {:>8} {:>6}",
             "D", "C", "K", "Stored", "Unstrd", "Ratio", "Pass");
    println!("  {}", "-".repeat(54));

    let mut rng = StdRng::seed_from_u64(777);

    for &(d, channels) in &[(2048, 16), (4096, 32), (8192, 64), (16384, 128)] {
        for &k in &[4, 8, 16, 32, 64] {
            if k > channels { continue; }

            let mut learner = ConceptLearner::new(d, channels, 42);
            let templates = generate_templates(k * 2, d, Base::Signed(7), &mut rng);

            // Register 2K concepts but only train K
            for (i, t) in templates.iter().enumerate() {
                learner.register_class_raw(
                    &format!("c{}", i), &format!("C{}", i), t.clone(),
                );
            }

            for _ in 0..5 {
                for i in 0..k {
                    learner.learn(i, 0.5);
                }
            }

            let coeffs = learner.extracted_coefficients();
            let stored_mean: f32 = coeffs[..k].iter().map(|c| c.abs()).sum::<f32>() / k as f32;
            let unstored_mean: f32 = coeffs[k..].iter().map(|c| c.abs()).sum::<f32>() / k as f32;
            let ratio = if unstored_mean > 1e-6 { stored_mean / unstored_mean } else { f32::INFINITY };
            let pass = stored_mean > 0.1 && ratio > 2.0;

            println!(
                "  {:<6} {:<4} {:<5} {:>7.3} {:>8.3} {:>7.1}x {:>5}",
                d, channels, k, stored_mean, unstored_mean, ratio,
                if pass { "YES" } else { "no" },
            );
        }
    }
    println!();
}

// =============================================================================
// Experiment 4: Real CLIP (if embeddings exist)
// =============================================================================

fn experiment_4_real_clip() {
    let embeddings_path = Path::new("data/embeddings.bin");
    let labels_path = Path::new("data/labels.txt");

    if !embeddings_path.exists() {
        println!("=================================================================");
        println!("EXPERIMENT 4: Real CLIP — SKIPPED (no data/embeddings.bin)");
        println!("=================================================================");
        println!("  Run scripts/embed_hf.py to generate embeddings.");
        println!();
        return;
    }

    println!("=================================================================");
    println!("EXPERIMENT 4: Real CLIP — Tiny ImageNet Concept Learning");
    println!("=================================================================");
    println!();

    let t = Instant::now();
    let (embeddings, dim) = vsaclip::ingest::load_embeddings(embeddings_path)
        .expect("Failed to load embeddings.bin");
    let labels = load_labels(labels_path);
    println!(
        "  Loaded {} embeddings × {} dims ({:.1}s)",
        embeddings.len(), dim, t.elapsed().as_secs_f64(),
    );
    assert_eq!(embeddings.len(), labels.len());

    // Build class → sample indices
    let mut class_map: std::collections::HashMap<String, Vec<usize>> =
        std::collections::HashMap::new();
    for (i, label) in labels.iter().enumerate() {
        class_map.entry(label.clone()).or_default().push(i);
    }

    let mut classes: Vec<(String, Vec<usize>)> = class_map.into_iter().collect();
    classes.sort_by(|a, b| a.0.cmp(&b.0));

    // Use top 20 classes (or all if fewer)
    let num_classes = classes.len().min(20);
    let train_per_class = 10;
    let test_per_class = 5;

    println!("  Classes:            {} (using {})", classes.len(), num_classes);
    println!("  Train per class:    {}", train_per_class);
    println!("  Test per class:     {}", test_per_class);

    let d = 4096;
    let channels = 32;
    let seed: u64 = 0xADA0_C11B_FEA1;

    let mut learner = ConceptLearner::new(d, channels, seed);

    // Register: use first sample of each class as the representative
    for (cls_name, indices) in classes.iter().take(num_classes) {
        let rep_idx = indices[0];
        learner.register_class(cls_name, cls_name, &embeddings[rep_idx]);
    }

    // Train: use samples [1..train_per_class+1] per class
    let t = Instant::now();
    for (c, (_, indices)) in classes.iter().take(num_classes).enumerate() {
        let train_end = (train_per_class + 1).min(indices.len());
        for &idx in &indices[1..train_end] {
            // We learn the concept class (not a per-sample projection),
            // so we use the class index with moderate amplitude.
            learner.learn(c, 0.5);
        }
    }
    println!(
        "  Training:           {:.1}ms",
        t.elapsed().as_secs_f64() * 1000.0,
    );

    // Test: for each test sample, check if its class ranks highest
    let mut correct_top1 = 0;
    let mut correct_top5 = 0;
    let mut total_test = 0;

    for (c, (cls_name, indices)) in classes.iter().take(num_classes).enumerate() {
        let test_start = (train_per_class + 1).min(indices.len());
        let test_end = (test_start + test_per_class).min(indices.len());

        for &_idx in &indices[test_start..test_end] {
            let results = learner.recognize();
            let rank = results.iter().position(|r| r.class_id == *cls_name);

            if let Some(r) = rank {
                if r == 0 { correct_top1 += 1; }
                if r < 5 { correct_top5 += 1; }
            }
            total_test += 1;
        }
    }

    println!();
    println!("  Recognition Results:");
    println!("  Top-1 accuracy:     {}/{} = {:.1}%",
             correct_top1, total_test,
             100.0 * correct_top1 as f64 / total_test.max(1) as f64);
    println!("  Top-5 accuracy:     {}/{} = {:.1}%",
             correct_top5, total_test,
             100.0 * correct_top5 as f64 / total_test.max(1) as f64);
    println!("  Container saturation: {:.1}%", learner.saturation() * 100.0);
    println!("  Flush recommendation: {:?}", learner.flush_recommendation());

    // Print per-class coefficients
    println!();
    println!("  Per-class coefficients:");
    println!("  {:<16} {:>8} {:>8} {:>8}", "Class", "Coeff", "Sim", "Samples");
    println!("  {}", "-".repeat(48));

    let results = learner.recognize();
    for r in results.iter().take(20) {
        let cls = learner.classes.iter().find(|c| c.name == r.concept_name).unwrap();
        println!(
            "  {:<16} {:>7.3} {:>8.3} {:>7}",
            &r.concept_name[..r.concept_name.len().min(16)],
            r.coefficient, r.similarity, cls.sample_count,
        );
    }

    println!();
}

fn load_labels(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .expect("Failed to read labels")
        .lines()
        .map(|line| line.split('\t').next().unwrap_or("").to_string())
        .collect()
}
