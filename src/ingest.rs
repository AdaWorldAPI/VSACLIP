//! Ingest Pipeline — float embeddings → Hamming containers via SimHash.
//!
//! Two modes:
//! 1. `fastembed` feature: embed images directly via CLIP ViT-B/32 (ONNX)
//! 2. Pre-computed: load embeddings from binary files (Python fallback)

use ladybug_contract::container::{Container, CONTAINER_WORDS};
use rayon::prelude::*;
use std::io::{self, Read, Write, BufReader, BufWriter};
use std::path::Path;

/// SimHash seed — MUST be same everywhere for consistent projection.
pub const CLIP_SIMHASH_SEED: u64 = 0xADA0C11B_2025_0001;

/// CLIP ViT-B/32 embedding dimension.
pub const CLIP_DIM: usize = 512;

/// Fast SimHash: project float embedding into binary Container.
///
/// Each bit = sign(dot(embedding, random_hyperplane)).
/// Preserves cosine similarity as Hamming similarity.
pub fn simhash(embedding: &[f32], seed: u64) -> Container {
    let dim = embedding.len();
    let mut result = Container::zero();

    for word_idx in 0..CONTAINER_WORDS {
        let mut word = 0u64;

        for bit in 0..64u32 {
            let bit_idx = word_idx * 64 + bit as usize;
            let mut dot = 0.0f32;

            let mut state = seed
                .wrapping_add(bit_idx as u64)
                .wrapping_mul(0x9e3779b97f4a7c15);

            for d in 0..dim {
                state ^= state >> 30;
                state = state.wrapping_mul(0xbf58476d1ce4e5b9);
                state ^= state >> 27;
                state = state.wrapping_mul(0x94d049bb133111eb);
                state ^= state >> 31;

                let sign = if state & 1 == 0 { 1.0f32 } else { -1.0f32 };
                dot += embedding[d] * sign;
            }

            if dot >= 0.0 {
                word |= 1u64 << bit;
            }
        }

        result.words[word_idx] = word;
    }

    result
}

/// An image entry with its class label and file path.
#[derive(Debug, Clone)]
pub struct ImageEntry {
    pub path: String,
    pub class_id: String,
    pub class_name: String,
}

/// A container with its associated label (for evaluation only).
#[derive(Debug, Clone)]
pub struct LabeledContainer {
    pub container: Container,
    pub class_id: String,
    pub class_name: String,
    pub image_path: String,
}

/// Scan Tiny ImageNet train directory for image entries.
///
/// Returns entries sorted by class_id then image name.
pub fn scan_tiny_imagenet(data_dir: &Path) -> io::Result<Vec<ImageEntry>> {
    let train_dir = data_dir.join("tiny-imagenet-200").join("train");
    if !train_dir.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("Train directory not found: {}", train_dir.display()),
        ));
    }

    // Load words.txt for human-readable labels
    let words_path = data_dir.join("tiny-imagenet-200").join("words.txt");
    let words_map = load_words_file(&words_path).unwrap_or_default();

    let mut entries = Vec::new();
    let mut class_dirs: Vec<_> = std::fs::read_dir(&train_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|ft| ft.is_dir()).unwrap_or(false))
        .collect();
    class_dirs.sort_by_key(|e| e.file_name());

    for class_entry in class_dirs {
        let class_id = class_entry.file_name().to_string_lossy().to_string();
        let class_name = words_map
            .get(&class_id)
            .cloned()
            .unwrap_or_else(|| class_id.clone());
        let images_dir = class_entry.path().join("images");

        if !images_dir.exists() {
            continue;
        }

        let mut image_files: Vec<_> = std::fs::read_dir(&images_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext.eq_ignore_ascii_case("jpeg") || ext.eq_ignore_ascii_case("jpg"))
                    .unwrap_or(false)
            })
            .collect();
        image_files.sort_by_key(|e| e.file_name());

        for img_entry in image_files {
            entries.push(ImageEntry {
                path: img_entry.path().to_string_lossy().to_string(),
                class_id: class_id.clone(),
                class_name: class_name.clone(),
            });
        }
    }

    Ok(entries)
}

/// Load words.txt mapping synset IDs to human-readable names.
fn load_words_file(path: &Path) -> io::Result<std::collections::HashMap<String, String>> {
    let content = std::fs::read_to_string(path)?;
    let mut map = std::collections::HashMap::new();
    for line in content.lines() {
        if let Some((id, name)) = line.split_once('\t') {
            map.insert(id.to_string(), name.to_string());
        }
    }
    Ok(map)
}

/// Save pre-computed float32 embeddings to a binary file.
///
/// Format: [u32 count] [u32 dim] [f32 × count × dim]
/// All values little-endian.
pub fn save_embeddings(
    embeddings: &[Vec<f32>],
    path: &Path,
) -> io::Result<()> {
    let mut f = BufWriter::new(std::fs::File::create(path)?);
    let count = embeddings.len() as u32;
    let dim = embeddings.first().map(|e| e.len() as u32).unwrap_or(0);

    f.write_all(&count.to_le_bytes())?;
    f.write_all(&dim.to_le_bytes())?;

    for emb in embeddings {
        for &val in emb {
            f.write_all(&val.to_le_bytes())?;
        }
    }
    f.flush()?;
    Ok(())
}

/// Load pre-computed float32 embeddings from a binary file.
///
/// Returns (embeddings, dim).
pub fn load_embeddings(path: &Path) -> io::Result<(Vec<Vec<f32>>, usize)> {
    let mut f = BufReader::new(std::fs::File::open(path)?);
    let mut buf4 = [0u8; 4];

    f.read_exact(&mut buf4)?;
    let count = u32::from_le_bytes(buf4) as usize;
    f.read_exact(&mut buf4)?;
    let dim = u32::from_le_bytes(buf4) as usize;

    let mut embeddings = Vec::with_capacity(count);
    for _ in 0..count {
        let mut emb = Vec::with_capacity(dim);
        for _ in 0..dim {
            f.read_exact(&mut buf4)?;
            emb.push(f32::from_le_bytes(buf4));
        }
        embeddings.push(emb);
    }

    Ok((embeddings, dim))
}

/// Save containers to a binary file for fast reload.
///
/// Format: [u32 count] [u64 × CONTAINER_WORDS × count]
pub fn save_containers(containers: &[Container], path: &Path) -> io::Result<()> {
    let mut f = BufWriter::new(std::fs::File::create(path)?);
    let count = containers.len() as u32;
    f.write_all(&count.to_le_bytes())?;

    for c in containers {
        for &word in &c.words {
            f.write_all(&word.to_le_bytes())?;
        }
    }
    f.flush()?;
    Ok(())
}

/// Load containers from a binary file.
pub fn load_containers(path: &Path) -> io::Result<Vec<Container>> {
    let mut f = BufReader::new(std::fs::File::open(path)?);
    let mut buf4 = [0u8; 4];
    let mut buf8 = [0u8; 8];

    f.read_exact(&mut buf4)?;
    let count = u32::from_le_bytes(buf4) as usize;

    let mut containers = Vec::with_capacity(count);
    for _ in 0..count {
        let mut c = Container::zero();
        for i in 0..CONTAINER_WORDS {
            f.read_exact(&mut buf8)?;
            c.words[i] = u64::from_le_bytes(buf8);
        }
        containers.push(c);
    }

    Ok(containers)
}

/// Batch project embeddings → containers via SimHash (parallelized with Rayon).
pub fn batch_simhash(embeddings: &[Vec<f32>], seed: u64) -> Vec<Container> {
    embeddings.par_iter().map(|emb| simhash(emb, seed)).collect()
}

/// Batch project using any projection function (parallelized with Rayon).
pub fn batch_project<F>(embeddings: &[Vec<f32>], seed: u64, proj: F) -> Vec<Container>
where
    F: Fn(&[f32], u64) -> Container + Sync,
{
    embeddings.par_iter().map(|emb| proj(emb, seed)).collect()
}

// =============================================================================
// X-Trans Projections — break the 1.22x separation barrier
// =============================================================================

/// Golden ratio and golden angle constants.
const PHI: f64 = 1.618033988749895;
const GOLDEN_ANGLE: f64 = 2.399963229728653; // pi * (3 - sqrt(5))

/// Approach 1: Golden Angle Spiral Hyperplanes
///
/// Instead of random ±1 hyperplanes, generate structured hyperplanes
/// on a golden angle spiral. The golden angle (137.508°) produces
/// maximally non-repeating angular coverage — like sunflower seeds.
/// No two hyperplanes cluster in the same direction.
pub fn simhash_golden(embedding: &[f32], seed: u64) -> Container {
    let dim = embedding.len();
    let mut result = Container::zero();

    // Pre-seed for mixing with golden angle
    let seed_mix = seed.wrapping_mul(0x9e3779b97f4a7c15);

    for word_idx in 0..CONTAINER_WORDS {
        let mut word = 0u64;

        for bit in 0..64u32 {
            let bit_idx = word_idx * 64 + bit as usize;
            let mut dot = 0.0f64;

            for d in 0..dim {
                // Golden angle rotation in (bit_idx, d) space
                let angle = (bit_idx as f64 * GOLDEN_ANGLE)
                    + (d as f64 * PHI)
                    + (seed_mix as f64 * 1e-18);
                // Use cos for the hyperplane component
                let component = angle.cos();
                dot += embedding[d] as f64 * component;
            }

            if dot >= 0.0 {
                word |= 1u64 << bit;
            }
        }

        result.words[word_idx] = word;
    }

    result
}

/// Approach 2: Cross-Bind Projection (XOR Differential)
///
/// Each bit encodes the DIFFERENCE between two random projections:
///   bit_i = sign(dot(emb, h_i)) XOR sign(dot(emb, h_j))
/// where j = (i * PHI) mod CONTAINER_BITS.
///
/// This is like edge detection vs pixel sampling — edges carry more
/// discriminative information. The golden-ratio pairing ensures
/// no two crosses overlap.
pub fn simhash_crossbind(embedding: &[f32], seed: u64) -> Container {
    let dim = embedding.len();
    let total_bits = CONTAINER_WORDS * 64;
    let mut result = Container::zero();

    // Pre-compute all dot-product signs using the standard random hyperplane method
    let mut signs = vec![false; total_bits];
    for bit_idx in 0..total_bits {
        let mut dot = 0.0f32;
        let mut state = seed
            .wrapping_add(bit_idx as u64)
            .wrapping_mul(0x9e3779b97f4a7c15);

        for d in 0..dim {
            state ^= state >> 30;
            state = state.wrapping_mul(0xbf58476d1ce4e5b9);
            state ^= state >> 27;
            state = state.wrapping_mul(0x94d049bb133111eb);
            state ^= state >> 31;

            let sign = if state & 1 == 0 { 1.0f32 } else { -1.0f32 };
            dot += embedding[d] * sign;
        }

        signs[bit_idx] = dot >= 0.0;
    }

    // Cross-bind: XOR pairs at golden-ratio spacing
    for word_idx in 0..CONTAINER_WORDS {
        let mut word = 0u64;

        for bit in 0..64u32 {
            let i = word_idx * 64 + bit as usize;
            let j = ((i as f64 * PHI) as usize) % total_bits;

            if signs[i] ^ signs[j] {
                word |= 1u64 << bit;
            }
        }

        result.words[word_idx] = word;
    }

    result
}

/// Approach 3: Permute-Bind Multi-View (Holographic)
///
/// Takes K=3 different SimHash projections with different seeds,
/// permutes each by a different offset, then XOR-binds them.
/// The result encodes 3 simultaneous views holographically —
/// like X-Trans putting R,G,B in every row and column.
pub fn simhash_holographic(embedding: &[f32], seed: u64) -> Container {
    // Three views with different seeds
    let view1 = simhash(embedding, seed);
    let view2 = simhash(embedding, seed ^ 0xDEAD_BEEF_CAFE_0001);
    let view3 = simhash(embedding, seed ^ 0xBEEF_DEAD_F00D_0002);

    // Permute each by different offsets (in bits, not words)
    // Using golden-ratio-spaced offsets for maximal separation
    let p1 = view1; // identity
    let p2 = view2.permute(2731);  // ~8192/3 = aperiodic offset
    let p3 = view3.permute(5461);  // ~8192*2/3

    // XOR-bind all views: holographic superposition
    let mut result = Container::zero();
    for i in 0..CONTAINER_WORDS {
        result.words[i] = p1.words[i] ^ p2.words[i] ^ p3.words[i];
    }

    result
}

// =============================================================================
// Crosshair Projections — directional measurement patterns
// =============================================================================

/// Helper: compute dot-product sign for a single bit using the standard
/// xorshift random hyperplane. Returns true if dot >= 0.
#[inline]
fn random_sign(embedding: &[f32], bit_idx: usize, seed: u64) -> bool {
    let dim = embedding.len();
    let mut dot = 0.0f32;
    let mut state = seed
        .wrapping_add(bit_idx as u64)
        .wrapping_mul(0x9e3779b97f4a7c15);

    for d in 0..dim {
        state ^= state >> 30;
        state = state.wrapping_mul(0xbf58476d1ce4e5b9);
        state ^= state >> 27;
        state = state.wrapping_mul(0x94d049bb133111eb);
        state ^= state >> 31;

        let sign = if state & 1 == 0 { 1.0f32 } else { -1.0f32 };
        dot += embedding[d] * sign;
    }

    dot >= 0.0
}

/// Approach 4: Quad Crosshair (Cardinal Directions)
///
/// For each bit, take 4 measurements at golden-angle-spaced offsets
/// (cardinal directions in hyperplane index space) and majority-vote.
///
/// Like a camera spot meter with 4 sensors in a + pattern:
///   bit_i = majority(sign_i, sign_{i+Δ}, sign_{i+2Δ}, sign_{i+3Δ})
/// where Δ = CONTAINER_BITS/4 (quarter-rotation in bit-index space).
///
/// 4 measurements per bit → more robust decision boundary.
/// The "crosshair" eliminates noise from single random hyperplanes.
pub fn simhash_quad_crosshair(embedding: &[f32], seed: u64) -> Container {
    let total_bits = CONTAINER_WORDS * 64;
    // Spacing: quarter of total bits (cardinal: 0°, 90°, 180°, 270°)
    let delta = total_bits / 4;
    let mut result = Container::zero();

    // Pre-compute all raw signs (we need 4× the lookup per output bit)
    let mut signs = vec![false; total_bits];
    for bit_idx in 0..total_bits {
        signs[bit_idx] = random_sign(embedding, bit_idx, seed);
    }

    for word_idx in 0..CONTAINER_WORDS {
        let mut word = 0u64;

        for bit in 0..64u32 {
            let i = word_idx * 64 + bit as usize;

            // 4-point cardinal crosshair
            let s0 = signs[i];
            let s1 = signs[(i + delta) % total_bits];
            let s2 = signs[(i + 2 * delta) % total_bits];
            let s3 = signs[(i + 3 * delta) % total_bits];

            // Majority vote (3 out of 4, or 2-2 tie → use s0)
            let count = s0 as u8 + s1 as u8 + s2 as u8 + s3 as u8;
            if count > 2 || (count == 2 && s0) {
                word |= 1u64 << bit;
            }
        }

        result.words[word_idx] = word;
    }

    result
}

/// Approach 5: 45-Degree Crosshair (Diagonal Directions)
///
/// Same as quad crosshair but offset by 1/8 rotation (45°):
///   bit_i = majority(sign_{i+Δ/2}, sign_{i+3Δ/2}, sign_{i+5Δ/2}, sign_{i+7Δ/2})
/// where Δ = CONTAINER_BITS/4.
///
/// The diagonal cross × instead of the cardinal cross +.
/// Samples different directions than the quad crosshair.
pub fn simhash_45deg_crosshair(embedding: &[f32], seed: u64) -> Container {
    let total_bits = CONTAINER_WORDS * 64;
    let delta = total_bits / 4;
    let half_delta = delta / 2; // 45° offset = 1/8 rotation
    let mut result = Container::zero();

    let mut signs = vec![false; total_bits];
    for bit_idx in 0..total_bits {
        signs[bit_idx] = random_sign(embedding, bit_idx, seed);
    }

    for word_idx in 0..CONTAINER_WORDS {
        let mut word = 0u64;

        for bit in 0..64u32 {
            let i = word_idx * 64 + bit as usize;

            // 4-point diagonal crosshair (45° offset from cardinal)
            let s0 = signs[(i + half_delta) % total_bits];
            let s1 = signs[(i + half_delta + delta) % total_bits];
            let s2 = signs[(i + half_delta + 2 * delta) % total_bits];
            let s3 = signs[(i + half_delta + 3 * delta) % total_bits];

            let count = s0 as u8 + s1 as u8 + s2 as u8 + s3 as u8;
            if count > 2 || (count == 2 && s0) {
                word |= 1u64 << bit;
            }
        }

        result.words[word_idx] = word;
    }

    result
}

/// Approach 6: XY Hyperposition Bind
///
/// Split the embedding into two halves (X and Y subspaces).
/// Project each half independently into a Container via SimHash.
/// XOR-bind the two projections.
///
/// Like photographing a scene from two orthogonal viewpoints and
/// combining them into a single holographic plate.
///
/// The XOR binding encodes the RELATIONSHIP between X and Y subspaces.
/// Two images with similar X but different Y → different Container.
/// This should capture more structure than a single flat projection.
pub fn simhash_xy_bind(embedding: &[f32], seed: u64) -> Container {
    let dim = embedding.len();
    let half = dim / 2;

    // Split embedding into X (first half) and Y (second half)
    let x_emb = &embedding[..half];
    let y_emb = &embedding[half..];

    // Project each half with different seeds
    let x_container = simhash(x_emb, seed);
    let y_container = simhash(y_emb, seed ^ 0xADA0_0000_B10D_0001);

    // XOR-bind: the result encodes the relationship between X and Y
    x_container.xor(&y_container)
}

// =============================================================================
// fastembed integration (optional)
// =============================================================================

/// Embed images via CLIP ViT-B/32 using fastembed-rs.
///
/// Only available with `fastembed` feature.
#[cfg(feature = "fastembed")]
pub fn embed_images(image_paths: &[String]) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    use fastembed::{ImageEmbedding, ImageInitOptions, ImageEmbeddingModel};

    let model = ImageEmbedding::try_new(
        ImageInitOptions::new(ImageEmbeddingModel::ClipVitB32)
            .with_show_download_progress(true),
    )?;

    let paths: Vec<&str> = image_paths.iter().map(|s| s.as_str()).collect();
    let embeddings = model.embed(paths, None)?;
    Ok(embeddings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ladybug_contract::container::CONTAINER_BITS;

    #[test]
    fn test_deterministic() {
        let emb = vec![0.1, 0.2, 0.3, -0.1, 0.5];
        assert_eq!(simhash(&emb, 42).hamming(&simhash(&emb, 42)), 0);
    }

    #[test]
    fn test_preserves_similarity() {
        let a = vec![1.0, 0.0, 0.5, 0.3, -0.2];
        let b = vec![0.9, 0.1, 0.4, 0.3, -0.1];
        let c = vec![-1.0, 0.5, -0.3, 0.8, 0.9];

        let ca = simhash(&a, 42);
        let cb = simhash(&b, 42);
        let cc = simhash(&c, 42);

        assert!(ca.hamming(&cb) < ca.hamming(&cc));
    }

    #[test]
    fn test_balanced_popcount() {
        let emb: Vec<f32> = (0..512).map(|i| (i as f32 * 0.1).sin()).collect();
        let c = simhash(&emb, 42);
        let pc = c.popcount();
        let mid = CONTAINER_BITS as u32 / 2;
        assert!(pc.abs_diff(mid) < mid / 4);
    }

    #[test]
    fn test_embedding_round_trip() {
        let embeddings: Vec<Vec<f32>> = (0..10)
            .map(|i| (0..512).map(|j| ((i * 512 + j) as f32 * 0.01).sin()).collect())
            .collect();

        let tmp = std::env::temp_dir().join("vsaclip_test_embeddings.bin");
        save_embeddings(&embeddings, &tmp).unwrap();
        let (loaded, dim) = load_embeddings(&tmp).unwrap();
        std::fs::remove_file(&tmp).ok();

        assert_eq!(dim, 512);
        assert_eq!(loaded.len(), 10);
        for i in 0..10 {
            for j in 0..512 {
                assert!((embeddings[i][j] - loaded[i][j]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_container_round_trip() {
        let containers: Vec<Container> = (0..10u64).map(|s| Container::random(s)).collect();

        let tmp = std::env::temp_dir().join("vsaclip_test_containers.bin");
        save_containers(&containers, &tmp).unwrap();
        let loaded = load_containers(&tmp).unwrap();
        std::fs::remove_file(&tmp).ok();

        assert_eq!(loaded.len(), 10);
        for (a, b) in containers.iter().zip(loaded.iter()) {
            assert_eq!(a.hamming(b), 0);
        }
    }

    #[test]
    fn test_batch_simhash() {
        let embeddings: Vec<Vec<f32>> = (0..5)
            .map(|i| (0..512).map(|j| ((i * 512 + j) as f32 * 0.01).sin()).collect())
            .collect();

        let containers = batch_simhash(&embeddings, CLIP_SIMHASH_SEED);
        assert_eq!(containers.len(), 5);

        // Same embedding → same container
        let c0 = simhash(&embeddings[0], CLIP_SIMHASH_SEED);
        assert_eq!(containers[0].hamming(&c0), 0);
    }
}
