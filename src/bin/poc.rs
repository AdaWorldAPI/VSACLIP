//! VSACLIP Proof of Concept — Tiny ImageNet Recognition via Hamming Resonance
//!
//! Run: cargo run --release --bin poc
//!
//! Pipeline:
//! 1. Load CLIP embeddings (pre-computed or via fastembed)
//! 2. SimHash → 8192-bit Containers
//! 3. Resonance cascade: Features → Parts → Objects
//! 4. Ground truth evaluation: cluster purity
//! 5. Sweep benchmark: HDR vs full

use ladybug_contract::container::{Container, CONTAINER_BITS, CONTAINER_WORDS};
use vsaclip::ingest::{self, CLIP_SIMHASH_SEED};
use vsaclip::sweep::{hdr_sweep, full_sweep, SweepConfig};
use vsaclip::hdr::HdrConfig;

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

/// Resonance cascade thresholds (as fractions of d)
/// Tuned for CLIP ViT-B/32 SimHash (intra ~20%, inter ~25%)
const L1_THRESHOLD_FRAC: f32 = 0.20; // Features: tight — catches same-concept
const L2_THRESHOLD_FRAC: f32 = 0.22; // Parts: slightly wider
const L3_THRESHOLD_FRAC: f32 = 0.25; // Objects: captures inter-class boundary

/// Max features to propagate between layers
const TOP_K: usize = 20;

/// A cluster discovered by the resonance cascade
#[derive(Debug)]
struct Cluster {
    /// The prototype container for this cluster
    prototype: Container,
    /// Indices of images assigned to this cluster
    members: Vec<usize>,
}

fn main() {
    println!();
    println!("VSACLIP Proof-of-Concept");
    println!("========================");
    println!("Container: {} bits = {} × u64", CONTAINER_BITS, CONTAINER_WORDS);
    println!();

    let data_dir = Path::new("data");
    let embeddings_path = data_dir.join("embeddings.bin");
    let labels_path = data_dir.join("labels.txt");
    let containers_path = data_dir.join("containers.bin");

    // =========================================================================
    // Step 1: Load or compute containers
    // =========================================================================
    let (containers, labels) = if containers_path.exists() && labels_path.exists() {
        print!("[1/6] Loading cached containers...              ");
        let t = Instant::now();
        let containers = ingest::load_containers(&containers_path)
            .expect("Failed to load containers");
        let labels = load_labels(&labels_path);
        println!("✓ {} containers ({:.1}s)", containers.len(), t.elapsed().as_secs_f64());
        (containers, labels)
    } else if embeddings_path.exists() && labels_path.exists() {
        print!("[1/6] Loading CLIP embeddings...                ");
        let t = Instant::now();
        let (embeddings, dim) = ingest::load_embeddings(&embeddings_path)
            .expect("Failed to load embeddings");
        let labels = load_labels(&labels_path);
        println!("✓ {} embeddings (dim={}) ({:.1}s)", embeddings.len(), dim, t.elapsed().as_secs_f64());

        // SimHash projection
        print!("[2/6] SimHash projection...                     ");
        let t = Instant::now();
        let containers = ingest::batch_simhash(&embeddings, CLIP_SIMHASH_SEED);
        println!("✓ {} containers ({:.1}s)", containers.len(), t.elapsed().as_secs_f64());

        // Cache containers for next run
        print!("      Caching containers to disk...             ");
        let t = Instant::now();
        ingest::save_containers(&containers, &containers_path)
            .expect("Failed to save containers");
        println!("✓ ({:.1}s)", t.elapsed().as_secs_f64());

        (containers, labels)
    } else {
        eprintln!("ERROR: No embeddings found at {}", embeddings_path.display());
        eprintln!();
        eprintln!("Please run one of:");
        eprintln!("  1. cargo run --bin download-data   # download Tiny ImageNet");
        eprintln!("  2. python3 scripts/embed_images.py # pre-compute CLIP embeddings");
        eprintln!();
        eprintln!("Or with fastembed feature (if ONNX Runtime is available):");
        eprintln!("  cargo run --release --features fastembed --bin poc");
        std::process::exit(1);
    };

    let n = containers.len();
    assert_eq!(n, labels.len(), "Container/label count mismatch");

    // =========================================================================
    // Step 2: Verify SimHash quality
    // =========================================================================
    print!("[3/6] Verifying SimHash quality...               ");
    verify_simhash_quality(&containers, &labels);

    // =========================================================================
    // Step 3: Resonance cascade — discover features, parts, objects
    // =========================================================================
    print!("[4/6] Running resonance cascade...               ");
    let t = Instant::now();
    let (l1_features, l2_parts, l3_objects) = resonance_cascade(&containers);
    println!("✓ L1: {} features, L2: {} parts, L3: {} objects ({:.1}s)",
             l1_features.len(), l2_parts.len(), l3_objects.len(),
             t.elapsed().as_secs_f64());

    // =========================================================================
    // Step 4: Assign images to L3 objects and evaluate
    // =========================================================================
    print!("[5/6] Ground truth evaluation...                 ");
    let t = Instant::now();
    evaluate_clusters(&containers, &labels, &l3_objects, t);

    // =========================================================================
    // Step 5: Sweep benchmark
    // =========================================================================
    println!("[6/6] Sweep benchmark (N={})...", n);
    sweep_benchmark(&containers);

    println!();
    println!("Done. No training. No GPU. No numpy.");
    println!();
}

/// Load labels from labels.txt (class_id\tpath per line)
fn load_labels(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .expect("Failed to read labels file")
        .lines()
        .map(|line| {
            line.split('\t')
                .next()
                .unwrap_or("")
                .to_string()
        })
        .collect()
}

/// Check that same-class images are closer than cross-class images
fn verify_simhash_quality(containers: &[Container], labels: &[String]) {
    let n = containers.len();
    if n < 200 {
        println!("⚠ too few images for quality check");
        return;
    }

    // Build class → indices map
    let mut class_indices: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, label) in labels.iter().enumerate() {
        class_indices.entry(label.as_str()).or_default().push(i);
    }

    let classes: Vec<&&str> = class_indices.keys().collect();
    let num_classes = classes.len();

    // Sample intra-class and inter-class distances
    let mut intra_dists = Vec::new();
    let mut inter_dists = Vec::new();
    let mut rng_state = 0xDEADBEEFu64;

    for cls in &classes {
        let indices = &class_indices[**cls];
        if indices.len() < 2 { continue; }

        // Sample intra-class pairs
        for pair in 0..std::cmp::min(5, indices.len() - 1) {
            let i = indices[pair];
            let j = indices[pair + 1];
            intra_dists.push(containers[i].hamming(&containers[j]));
        }
    }

    // Sample inter-class pairs
    for _ in 0..std::cmp::min(1000, num_classes * 5) {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let c1 = (rng_state as usize) % num_classes;
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let c2 = (rng_state as usize) % num_classes;
        if c1 == c2 { continue; }

        let i1 = &class_indices[*classes[c1]];
        let i2 = &class_indices[*classes[c2]];
        if i1.is_empty() || i2.is_empty() { continue; }

        let idx1 = i1[0];
        let idx2 = i2[0];
        inter_dists.push(containers[idx1].hamming(&containers[idx2]));
    }

    if intra_dists.is_empty() || inter_dists.is_empty() {
        println!("⚠ insufficient data for quality check");
        return;
    }

    intra_dists.sort();
    inter_dists.sort();

    let intra_mean: f64 = intra_dists.iter().map(|&d| d as f64).sum::<f64>() / intra_dists.len() as f64;
    let inter_mean: f64 = inter_dists.iter().map(|&d| d as f64).sum::<f64>() / inter_dists.len() as f64;
    let d = CONTAINER_BITS as f64;

    let intra_p10 = intra_dists[intra_dists.len() / 10] as f64;
    let intra_p50 = intra_dists[intra_dists.len() / 2] as f64;
    let intra_p90 = intra_dists[intra_dists.len() * 9 / 10] as f64;
    let inter_p10 = inter_dists[inter_dists.len() / 10] as f64;
    let inter_p50 = inter_dists[inter_dists.len() / 2] as f64;

    println!("✓");
    println!("      Intra-class: mean={:.0} ({:.1}%), p10={:.0}, p50={:.0}, p90={:.0}",
             intra_mean, intra_mean / d * 100.0, intra_p10, intra_p50, intra_p90);
    println!("      Inter-class: mean={:.0} ({:.1}%), p10={:.0}, p50={:.0}",
             inter_mean, inter_mean / d * 100.0, inter_p10, inter_p50);
    println!("      Separation ratio: {:.2}× (mean), {:.2}× (p50)",
             inter_mean / intra_mean.max(1.0), inter_p50 / intra_p50.max(1.0));
}

/// Run 3-layer resonance cascade to discover features → parts → objects.
///
/// Uses exposure-based learning: unmatched inputs create new containers.
fn resonance_cascade(containers: &[Container]) -> (Vec<Cluster>, Vec<Cluster>, Vec<Cluster>) {
    let d = CONTAINER_BITS as f32;

    // Layer 1: Features — low threshold, many clusters
    let l1_threshold = (d * L1_THRESHOLD_FRAC) as u32;
    let l1_features = exposure_pass(containers, l1_threshold, "L1");

    // For L2: superpose co-activated features per image → part prototypes
    let l2_threshold = (d * L2_THRESHOLD_FRAC) as u32;
    let l2_input = superpose_activations(containers, &l1_features, l1_threshold);
    let l2_parts = exposure_pass(&l2_input, l2_threshold, "L2");

    // For L3: superpose co-activated parts → object prototypes
    let l3_threshold = (d * L3_THRESHOLD_FRAC) as u32;
    let l3_input = superpose_activations(&l2_input, &l2_parts, l2_threshold);
    let l3_objects = exposure_pass(&l3_input, l3_threshold, "L3");

    (l1_features, l2_parts, l3_objects)
}

/// Single exposure pass: sweep each input against existing prototypes.
/// If nothing resonates, the input becomes a new prototype.
fn exposure_pass(inputs: &[Container], threshold: u32, layer_name: &str) -> Vec<Cluster> {
    let mut clusters: Vec<Cluster> = Vec::new();
    let config = SweepConfig {
        threshold,
        safety: 1.5,
        hdr: HdrConfig::default(),
        limit: 1, // only need closest match
    };

    let n = inputs.len();
    let report_interval = std::cmp::max(n / 20, 1); // report every 5%

    // Keep a live prototype vector for efficient sweeping
    let mut prototypes: Vec<Container> = Vec::new();

    for (idx, input) in inputs.iter().enumerate() {
        if prototypes.is_empty() {
            prototypes.push(input.clone());
            clusters.push(Cluster {
                prototype: input.clone(),
                members: vec![idx],
            });
            continue;
        }

        let result = hdr_sweep(input, &prototypes, &config);

        if let Some(best) = result.matches.first() {
            clusters[best.index].members.push(idx);
        } else {
            prototypes.push(input.clone());
            clusters.push(Cluster {
                prototype: input.clone(),
                members: vec![idx],
            });
        }

        if n > 1000 && (idx + 1) % report_interval == 0 {
            eprint!("\r      {}: {}/{} ({:.0}%), {} clusters",
                    layer_name, idx + 1, n,
                    (idx + 1) as f64 / n as f64 * 100.0,
                    clusters.len());
        }
    }
    if n > 1000 {
        eprint!("\r                                                        \r");
    }

    clusters
}

/// Superpose top-K activated clusters for each input → new representation.
fn superpose_activations(
    inputs: &[Container],
    clusters: &[Cluster],
    threshold: u32,
) -> Vec<Container> {
    if clusters.is_empty() {
        return inputs.to_vec();
    }

    let prototypes: Vec<Container> = clusters.iter().map(|c| c.prototype.clone()).collect();
    let config = SweepConfig {
        threshold,
        safety: 1.5,
        hdr: HdrConfig::default(),
        limit: TOP_K,
    };

    inputs.iter().map(|input| {
        let result = hdr_sweep(input, &prototypes, &config);
        if result.matches.is_empty() {
            return input.clone();
        }

        let activated: Vec<&Container> = result.matches.iter()
            .take(TOP_K)
            .map(|m| &prototypes[m.index])
            .collect();

        if activated.len() == 1 {
            return activated[0].clone();
        }

        Container::bundle(&activated)
    }).collect()
}

/// Evaluate cluster purity against ground truth labels.
fn evaluate_clusters(
    containers: &[Container],
    labels: &[String],
    l3_objects: &[Cluster],
    timer_start: Instant,
) {
    if l3_objects.is_empty() {
        println!("⚠ no L3 objects discovered");
        return;
    }

    // Assign each image to nearest L3 object
    let d = CONTAINER_BITS as f32;
    let threshold = (d * L3_THRESHOLD_FRAC) as u32;
    let prototypes: Vec<Container> = l3_objects.iter().map(|c| c.prototype.clone()).collect();
    let config = SweepConfig {
        threshold,
        safety: 1.5,
        hdr: HdrConfig::default(),
        limit: 1,
    };

    let mut assignments: Vec<Option<usize>> = vec![None; containers.len()];
    for (i, c) in containers.iter().enumerate() {
        let result = hdr_sweep(c, &prototypes, &config);
        if let Some(best) = result.matches.first() {
            assignments[i] = Some(best.index);
        }
    }

    let assigned_count = assignments.iter().filter(|a| a.is_some()).count();

    // Count class distribution per cluster
    let num_clusters = l3_objects.len();
    let mut cluster_class_counts: Vec<HashMap<&str, usize>> = vec![HashMap::new(); num_clusters];
    for (i, assignment) in assignments.iter().enumerate() {
        if let Some(cluster_idx) = assignment {
            *cluster_class_counts[*cluster_idx]
                .entry(labels[i].as_str())
                .or_insert(0) += 1;
        }
    }

    // Compute purity: fraction of dominant class in each cluster
    let mut total_correct = 0usize;
    let mut total_in_clusters = 0usize;
    let mut perfect_clusters = 0usize;

    for counts in &cluster_class_counts {
        if counts.is_empty() { continue; }
        let total: usize = counts.values().sum();
        let max_count = *counts.values().max().unwrap_or(&0);
        total_correct += max_count;
        total_in_clusters += total;
        if counts.len() == 1 {
            perfect_clusters += 1;
        }
    }

    let purity = if total_in_clusters > 0 {
        total_correct as f64 / total_in_clusters as f64 * 100.0
    } else {
        0.0
    };

    // Unique classes
    let unique_classes: std::collections::HashSet<&str> = labels.iter().map(|l| l.as_str()).collect();
    let chance = if unique_classes.len() > 0 {
        100.0 / unique_classes.len() as f64
    } else {
        0.0
    };

    // Top-5: for each image, find top-5 nearest clusters, check if correct class is there
    let config5 = SweepConfig {
        threshold: (d * 0.49) as u32, // wider net for top-5
        safety: 1.5,
        hdr: HdrConfig::default(),
        limit: 5,
    };
    let mut top5_hits = 0usize;
    let mut top5_total = 0usize;

    // Sample for top-5 (every 10th image for speed)
    for (i, c) in containers.iter().enumerate().step_by(10) {
        let result = hdr_sweep(c, &prototypes, &config5);
        if result.matches.is_empty() { continue; }

        top5_total += 1;
        let img_class = &labels[i];

        // Check if any of top-5 clusters contains this class as dominant
        for m in &result.matches {
            if let Some(counts) = cluster_class_counts.get(m.index) {
                let dominant = counts.iter().max_by_key(|(_, v)| **v).map(|(k, _)| *k);
                if dominant == Some(img_class.as_str()) {
                    top5_hits += 1;
                    break;
                }
            }
        }
    }

    let top5_acc = if top5_total > 0 {
        top5_hits as f64 / top5_total as f64 * 100.0
    } else {
        0.0
    };

    println!("✓ ({:.1}s)", timer_start.elapsed().as_secs_f64());
    println!("      Cluster purity: {:.1}% (chance = {:.1}%)", purity, chance);
    println!("      Top-5 accuracy: {:.1}%", top5_acc);
    println!("      Images assigned: {}/{}", assigned_count, containers.len());
    println!("      L3 clusters: {} (perfect: {})", num_clusters, perfect_clusters);

    // Show top 10 clusters by size
    let mut cluster_sizes: Vec<(usize, usize)> = cluster_class_counts.iter()
        .enumerate()
        .map(|(i, counts)| (i, counts.values().sum::<usize>()))
        .filter(|(_, size)| *size > 0)
        .collect();
    cluster_sizes.sort_by(|a, b| b.1.cmp(&a.1));

    println!("\n      Top 10 clusters:");
    for (cluster_idx, size) in cluster_sizes.iter().take(10) {
        let counts = &cluster_class_counts[*cluster_idx];
        let (dominant_class, dominant_count) = counts.iter()
            .max_by_key(|(_, v)| **v)
            .map(|(k, v)| (*k, *v))
            .unwrap_or(("?", 0));
        let local_purity = dominant_count as f64 / *size as f64 * 100.0;
        println!("        #{}: {} images, dominant={} ({:.0}% pure)",
                 cluster_idx, size, dominant_class, local_purity);
    }
}

/// Benchmark HDR sweep vs full sweep.
fn sweep_benchmark(containers: &[Container]) {
    let n = containers.len();
    if n == 0 { return; }

    let query = &containers[0];
    // Use a tight threshold (15% of d) to demonstrate early exit speedup.
    // At 30%, nearly all CLIP containers match (too loose for meaningful benchmark).
    let threshold = (CONTAINER_BITS as f32 * 0.15) as u32;
    let config = SweepConfig {
        threshold,
        safety: 1.5,
        ..Default::default()
    };

    // Warm up
    let _ = hdr_sweep(query, containers, &config);

    // HDR sweep
    let t = Instant::now();
    let iterations = 5;
    let mut hdr_result = None;
    for _ in 0..iterations {
        hdr_result = Some(hdr_sweep(query, containers, &config));
    }
    let hdr_us = t.elapsed().as_micros() as f64 / iterations as f64;
    let hdr_r = hdr_result.unwrap();

    // Full sweep
    let t = Instant::now();
    let mut full_result = None;
    for _ in 0..iterations {
        full_result = Some(full_sweep(query, containers, threshold));
    }
    let full_us = t.elapsed().as_micros() as f64 / iterations as f64;
    let full_r = full_result.unwrap();

    // Verify zero false negatives
    let hdr_set: std::collections::HashSet<usize> = hdr_r.matches.iter().map(|m| m.index).collect();
    let false_negatives = full_r.iter().filter(|m| !hdr_set.contains(&m.index)).count();

    let speedup = full_us / hdr_us.max(1.0);

    println!("      Full sweep:  {:.0} μs/query ({} matches)", full_us, full_r.len());
    println!("      HDR sweep:   {:.0} μs/query ({} matches)", hdr_us, hdr_r.matches.len());
    println!("      Speedup: {:.1}×", speedup);
    println!("      Instructions: {} (HDR) vs {} (full)",
             hdr_r.instructions, n as u64 * CONTAINER_WORDS as u64);
    println!("      False negatives: {}", false_negatives);
}
