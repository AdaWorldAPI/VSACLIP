//! X-Trans Projection Experiment — Break the 1.22x Separation Barrier
//!
//! Compares four SimHash projection methods:
//! 1. Random (baseline): i.i.d. random ±1 hyperplanes (Bayer equivalent)
//! 2. Golden Angle: structured hyperplanes on golden spiral (X-Trans equivalent)
//! 3. Cross-Bind: XOR differential of golden-ratio-spaced pairs (edge detection)
//! 4. Holographic: permute-bind 3 views (multi-channel superposition)
//!
//! Run: cargo run --release --bin xtrans

use ladybug_contract::container::{Container, CONTAINER_BITS};
use vsaclip::ingest::{self, CLIP_SIMHASH_SEED};

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

/// Projection method descriptor
struct ProjectionMethod {
    name: &'static str,
    func: fn(&[f32], u64) -> Container,
}

fn main() {
    println!();
    println!("X-Trans Projection Experiment");
    println!("==============================");
    println!("Container: {} bits", CONTAINER_BITS);
    println!("Target: separation ratio > 1.5x (baseline: 1.22x)");
    println!();

    // =========================================================================
    // Load embeddings + labels
    // =========================================================================
    let data_dir = Path::new("data");
    let embeddings_path = data_dir.join("embeddings.bin");
    let labels_path = data_dir.join("labels.txt");

    print!("[1/4] Loading embeddings...                      ");
    let t = Instant::now();
    let (embeddings, dim) = ingest::load_embeddings(&embeddings_path)
        .expect("Failed to load embeddings.bin — run scripts/embed_hf.py first");
    let labels = load_labels(&labels_path);
    println!("ok {} x {} ({:.1}s)", embeddings.len(), dim, t.elapsed().as_secs_f64());

    assert_eq!(embeddings.len(), labels.len(), "embedding/label count mismatch");

    // =========================================================================
    // Define projection methods
    // =========================================================================
    let methods: Vec<ProjectionMethod> = vec![
        ProjectionMethod { name: "Random (baseline)", func: ingest::simhash },
        ProjectionMethod { name: "Golden Angle",      func: ingest::simhash_golden },
        ProjectionMethod { name: "Cross-Bind",        func: ingest::simhash_crossbind },
        ProjectionMethod { name: "Holographic",       func: ingest::simhash_holographic },
        ProjectionMethod { name: "Quad Crosshair",    func: ingest::simhash_quad_crosshair },
        ProjectionMethod { name: "45° Crosshair",     func: ingest::simhash_45deg_crosshair },
        ProjectionMethod { name: "XY Bind",           func: ingest::simhash_xy_bind },
    ];

    // =========================================================================
    // Project with each method and measure
    // =========================================================================
    println!();
    println!("[2/4] Projecting with all methods (rayon parallel)...");
    println!();

    let mut all_containers: Vec<(&str, Vec<Container>)> = Vec::new();

    for method in &methods {
        print!("  {:20} ... ", method.name);
        let t = Instant::now();
        let containers = ingest::batch_project(&embeddings, CLIP_SIMHASH_SEED, method.func);
        let elapsed = t.elapsed().as_secs_f64();
        println!("ok ({:.1}s)", elapsed);
        all_containers.push((method.name, containers));
    }

    // =========================================================================
    // Measure separation for each method
    // =========================================================================
    println!();
    println!("[3/4] Measuring separation ratios...");
    println!();
    println!("  {:<20} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10}",
             "Method", "Intra%", "Inter%", "Ratio", "IntraSD", "InterSD", "Overlap", "Popcount");
    println!("  {}", "-".repeat(94));

    let mut best_ratio = 0.0f64;
    let mut best_method = "";

    for (name, containers) in &all_containers {
        let stats = measure_separation(containers, &labels);

        let flag = if stats.ratio > best_ratio {
            best_ratio = stats.ratio;
            best_method = name;
            " <-- best"
        } else {
            ""
        };

        println!("  {:<20} {:>7.1}% {:>7.1}% {:>7.2}x {:>7.1}% {:>7.1}% {:>7.3}  {:>9.0}{}",
                 name,
                 stats.intra_mean_pct,
                 stats.inter_mean_pct,
                 stats.ratio,
                 stats.intra_stddev_pct,
                 stats.inter_stddev_pct,
                 stats.overlap,
                 stats.avg_popcount,
                 flag);
    }

    // =========================================================================
    // Distance distribution details for best method
    // =========================================================================
    println!();
    println!("[4/4] Distribution details for best method: {}", best_method);
    println!();

    for (name, containers) in &all_containers {
        if *name == best_method {
            print_distribution_details(containers, &labels, name);
        }
    }

    // Also print baseline for comparison
    if best_method != "Random (baseline)" {
        for (name, containers) in &all_containers {
            if *name == "Random (baseline)" {
                print_distribution_details(containers, &labels, name);
            }
        }
    }

    println!();
    println!("Winner: {} (ratio {:.2}x vs baseline 1.22x)", best_method, best_ratio);
    println!();
}

/// Separation statistics for a projection method
struct SeparationStats {
    intra_mean_pct: f64,
    inter_mean_pct: f64,
    ratio: f64,
    intra_stddev_pct: f64,
    inter_stddev_pct: f64,
    overlap: f64,       // approximate overlap integral
    avg_popcount: f64,
}

/// Measure intra-class vs inter-class Hamming distance separation.
fn measure_separation(containers: &[Container], labels: &[String]) -> SeparationStats {
    let d = CONTAINER_BITS as f64;

    // Build class → indices map
    let mut class_indices: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, label) in labels.iter().enumerate() {
        class_indices.entry(label.as_str()).or_default().push(i);
    }
    let classes: Vec<&str> = class_indices.keys().copied().collect();
    let num_classes = classes.len();

    // Sample intra-class distances (more samples for better statistics)
    let mut intra_dists: Vec<f64> = Vec::new();
    for cls in &classes {
        let indices = &class_indices[*cls];
        if indices.len() < 2 { continue; }
        // Sample up to 10 pairs per class
        for pair in 0..std::cmp::min(10, indices.len() - 1) {
            let i = indices[pair];
            let j = indices[pair + 1];
            intra_dists.push(containers[i].hamming(&containers[j]) as f64 / d * 100.0);
        }
    }

    // Sample inter-class distances
    let mut inter_dists: Vec<f64> = Vec::new();
    let mut rng_state = 0xDEADBEEFu64;
    for _ in 0..std::cmp::min(2000, num_classes * 10) {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let c1 = (rng_state as usize) % num_classes;
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let c2 = (rng_state as usize) % num_classes;
        if c1 == c2 { continue; }

        let i1 = &class_indices[classes[c1]];
        let i2 = &class_indices[classes[c2]];
        if i1.is_empty() || i2.is_empty() { continue; }

        let idx1 = i1[0];
        let idx2 = i2[0];
        inter_dists.push(containers[idx1].hamming(&containers[idx2]) as f64 / d * 100.0);
    }

    // Statistics
    let intra_mean = mean(&intra_dists);
    let inter_mean = mean(&inter_dists);
    let intra_std = stddev(&intra_dists);
    let inter_std = stddev(&inter_dists);

    // Approximate overlap: fraction of intra samples > inter_mean or inter samples < intra_mean
    let intra_above_inter = intra_dists.iter().filter(|&&d| d > inter_mean).count() as f64
        / intra_dists.len().max(1) as f64;
    let inter_below_intra = inter_dists.iter().filter(|&&d| d < intra_mean).count() as f64
        / inter_dists.len().max(1) as f64;
    let overlap = (intra_above_inter + inter_below_intra) / 2.0;

    // Average popcount (should be ~4096 for balanced projection)
    let avg_pc: f64 = containers.iter().take(1000)
        .map(|c| c.popcount() as f64)
        .sum::<f64>() / 1000.0f64.min(containers.len() as f64);

    SeparationStats {
        intra_mean_pct: intra_mean,
        inter_mean_pct: inter_mean,
        ratio: inter_mean / intra_mean.max(0.01),
        intra_stddev_pct: intra_std,
        inter_stddev_pct: inter_std,
        overlap,
        avg_popcount: avg_pc,
    }
}

/// Print detailed distance distribution for a projection method.
fn print_distribution_details(containers: &[Container], labels: &[String], name: &str) {
    let d = CONTAINER_BITS as f64;

    let mut class_indices: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, label) in labels.iter().enumerate() {
        class_indices.entry(label.as_str()).or_default().push(i);
    }
    let classes: Vec<&str> = class_indices.keys().copied().collect();

    // Collect intra-class distances
    let mut intra_dists: Vec<u32> = Vec::new();
    for cls in &classes {
        let indices = &class_indices[*cls];
        if indices.len() < 2 { continue; }
        for pair in 0..std::cmp::min(10, indices.len() - 1) {
            intra_dists.push(containers[indices[pair]].hamming(&containers[indices[pair + 1]]));
        }
    }
    intra_dists.sort();

    // Collect inter-class distances
    let mut inter_dists: Vec<u32> = Vec::new();
    let mut rng_state = 0xCAFEBABEu64;
    for _ in 0..2000 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let c1 = (rng_state as usize) % classes.len();
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let c2 = (rng_state as usize) % classes.len();
        if c1 == c2 { continue; }
        let i1 = &class_indices[classes[c1]];
        let i2 = &class_indices[classes[c2]];
        if i1.is_empty() || i2.is_empty() { continue; }
        inter_dists.push(containers[i1[0]].hamming(&containers[i2[0]]));
    }
    inter_dists.sort();

    let percentile = |dists: &[u32], p: usize| -> f64 {
        if dists.is_empty() { return 0.0; }
        let idx = (dists.len() * p / 100).min(dists.len() - 1);
        dists[idx] as f64 / d * 100.0
    };

    println!("  {} distribution:", name);
    println!("    Intra-class: p5={:.1}%, p25={:.1}%, p50={:.1}%, p75={:.1}%, p95={:.1}%",
             percentile(&intra_dists, 5),
             percentile(&intra_dists, 25),
             percentile(&intra_dists, 50),
             percentile(&intra_dists, 75),
             percentile(&intra_dists, 95));
    println!("    Inter-class: p5={:.1}%, p25={:.1}%, p50={:.1}%, p75={:.1}%, p95={:.1}%",
             percentile(&inter_dists, 5),
             percentile(&inter_dists, 25),
             percentile(&inter_dists, 50),
             percentile(&inter_dists, 75),
             percentile(&inter_dists, 95));

    // Suggest optimal threshold
    let intra_p75 = percentile(&intra_dists, 75);
    let inter_p25 = percentile(&inter_dists, 25);
    let gap = inter_p25 - intra_p75;
    println!("    Gap (inter_p25 - intra_p75): {:.1}%{}", gap,
             if gap > 0.0 { " (clean separation zone)" } else { " (distributions overlap!)" });
    if gap > 0.0 {
        let suggested_threshold = (intra_p75 + inter_p25) / 2.0;
        println!("    Suggested L1 threshold: {:.1}%", suggested_threshold);
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

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() { return 0.0; }
    v.iter().sum::<f64>() / v.len() as f64
}

fn stddev(v: &[f64]) -> f64 {
    if v.len() < 2 { return 0.0; }
    let m = mean(v);
    let variance = v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() - 1) as f64;
    variance.sqrt()
}
