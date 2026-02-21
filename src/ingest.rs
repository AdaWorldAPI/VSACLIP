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
