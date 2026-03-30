# SESSION: VSACLIP × bgz17 Palette — Image Recognition via Archetype Lookup

## MISSION

Connect bgz17 palette compression to the VSACLIP image recognition pipeline.
Replace Hamming sweep (O(N×8192 popcount)) with palette lookup (O(N×1 byte compare)).

Prior results:
  Attempt 1 (8K Hamming):    1.22× separation ceiling (JL wall)
  Attempt 2 (organic learn): ∞ stored/unstored, 5% recognition (chance)

This attempt: palette archetypes as class-discriminative codebook.

## READ FIRST

```bash
# VSACLIP pipeline (Rust-only)
cat src/lib.rs                    # module structure
cat src/ingest.rs                 # CLIP embedding loading
cat src/container_ext.rs          # Container<128> operations
cat CLAUDE.md                     # project rules (Rust only, no Python)

# bgz17 palette system (in ndarray repo)
# These are the reference implementations — port to VSACLIP as needed
cat src/hpc/bgz17_bridge.rs       # Base17 type, L1 distance, golden-step encode
cat src/hpc/gguf_indexer.rs       # project_row_to_base17 (f32 row → Base17)
```

## KEY INSIGHT

project_row_to_base17 already works on ANY f32 row.
CLIP f32[512] is a valid f32 row. 512 elements = 30 octaves.

```rust
// This ALREADY WORKS in ndarray:
let clip_embedding: &[f32] = &image_embedding[..512];
let base17 = project_row_to_base17(clip_embedding);
// → i16[17], 34 bytes, captures the structural invariant of the embedding
```

No SimHash needed. No 8K-bit Container needed. Go directly from
CLIP floats to Base17 to palette. The binary intermediate is skipped.

## ARCHITECTURE

```
CLIP f32[512]
  │
  ├── OLD: SimHash → 8K-bit → Hamming sweep (O(N × 8192 popcount))
  │
  └── NEW: golden-step project → Base17 i16[17] → palette u8
           │                          │                │
           │                          │                └─ 1 byte per image
           │                          │                   256 archetypes
           │                          │                   distance = matrix[a][b]
           │                          │
           │                          └─ 34 bytes per image
           │                             L1 distance
           │                             ρ=0.992 rank correlation
           │
           └── 30 octaves of golden-step averaging
               CLIP dim 0..511 folded into 17 bins
               Each bin = mean of ~30 CLIP dimensions
               sampled at golden-step positions
```

## PHASE 1: Port Base17 to VSACLIP (standalone, no ndarray dependency)

The Base17 type is ~60 lines. Self-contained. Port it:

```rust
// vsaclip/src/base17.rs

const BASE_DIM: usize = 17;
const GOLDEN_STEP: usize = 11;
const FP_SCALE: f64 = 256.0;

const GOLDEN_POS: [u8; BASE_DIM] = {
    let mut t = [0u8; BASE_DIM];
    let mut i = 0;
    while i < BASE_DIM {
        t[i] = ((i * GOLDEN_STEP) % BASE_DIM) as u8;
        i += 1;
    }
    t
};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Base17 {
    pub dims: [i16; BASE_DIM],
}

impl Base17 {
    pub fn from_f32_row(row: &[f32]) -> Self {
        let d = row.len();
        let n_octaves = (d + BASE_DIM - 1) / BASE_DIM;
        let mut sum = [0.0f64; BASE_DIM];
        let mut count = [0u32; BASE_DIM];

        for octave in 0..n_octaves {
            for bi in 0..BASE_DIM {
                let dim = octave * BASE_DIM + GOLDEN_POS[bi] as usize;
                if dim < d {
                    sum[bi] += row[dim] as f64;
                    count[bi] += 1;
                }
            }
        }

        let mut dims = [0i16; BASE_DIM];
        for i in 0..BASE_DIM {
            if count[i] > 0 {
                let mean = sum[i] / count[i] as f64;
                dims[i] = (mean * FP_SCALE).round().clamp(-32768.0, 32767.0) as i16;
            }
        }
        Base17 { dims }
    }

    pub fn l1(&self, other: &Base17) -> u32 {
        let mut d = 0u32;
        for i in 0..BASE_DIM {
            d += (self.dims[i] as i32 - other.dims[i] as i32).unsigned_abs();
        }
        d
    }
}
```

## PHASE 2: Build Palette from Training Set

k-means on Base17 patterns of ALL training images.

```rust
// vsaclip/src/palette.rs

pub struct Palette {
    pub centroids: Vec<Base17>,  // k=256 max
}

impl Palette {
    /// Build palette from training embeddings via k-means.
    ///
    /// Golden-ratio initialization: seed centroids at positions
    /// i/φ mod 1 through the sorted training patterns. This prevents
    /// cluster collapse that random initialization risks.
    pub fn build(patterns: &[Base17], k: usize, max_iter: usize) -> Self {
        // Golden-ratio init: pick centroids at positions
        // floor(i × N / φ) for i in 0..k
        let n = patterns.len();
        let phi = 1.618033988749895;
        let mut centroids: Vec<Base17> = (0..k)
            .map(|i| {
                let idx = ((i as f64 * n as f64 / phi) as usize) % n;
                patterns[idx].clone()
            })
            .collect();

        // k-means iterations
        for _iter in 0..max_iter {
            // Assign each pattern to nearest centroid
            let mut assignments = vec![0u8; n];
            let mut changed = 0;
            for (j, pat) in patterns.iter().enumerate() {
                let mut best = 0u8;
                let mut best_dist = u32::MAX;
                for (c, centroid) in centroids.iter().enumerate() {
                    let d = pat.l1(centroid);
                    if d < best_dist {
                        best_dist = d;
                        best = c as u8;
                    }
                }
                if assignments[j] != best { changed += 1; }
                assignments[j] = best;
            }

            // Update centroids
            let mut sums = vec![[0i64; 17]; k];
            let mut counts = vec![0u32; k];
            for (j, pat) in patterns.iter().enumerate() {
                let c = assignments[j] as usize;
                counts[c] += 1;
                for d in 0..17 {
                    sums[c][d] += pat.dims[d] as i64;
                }
            }
            for c in 0..k {
                if counts[c] > 0 {
                    for d in 0..17 {
                        centroids[c].dims[d] = (sums[c][d] / counts[c] as i64) as i16;
                    }
                }
            }

            if changed == 0 { break; }
        }

        Palette { centroids }
    }

    /// Quantize a pattern to its nearest centroid index.
    pub fn quantize(&self, pattern: &Base17) -> u8 {
        let mut best = 0u8;
        let mut best_dist = u32::MAX;
        for (i, c) in self.centroids.iter().enumerate() {
            let d = pattern.l1(c);
            if d < best_dist {
                best_dist = d;
                best = i as u8;
            }
        }
        best
    }

    /// Build 256×256 distance matrix for O(1) lookup.
    pub fn build_distance_matrix(&self) -> [[u16; 256]; 256] {
        let mut matrix = [[0u16; 256]; 256];
        let k = self.centroids.len();
        for i in 0..k {
            for j in i..k {
                let d = self.centroids[i].l1(&self.centroids[j]) as u16;
                matrix[i][j] = d;
                matrix[j][i] = d;
            }
        }
        matrix
    }
}
```

## PHASE 3: Recognition via Palette Histogram

```rust
// Train: build class→palette histogram
let mut class_histograms: Vec<[u32; 256]> = vec![[0u32; 256]; n_classes];
for (embedding, label) in training_set {
    let b17 = Base17::from_f32_row(embedding);
    let pal_idx = palette.quantize(&b17);
    class_histograms[label][pal_idx as usize] += 1;
}

// Recognize: embed → palette → histogram match
fn recognize(embedding: &[f32], palette: &Palette, histograms: &[[u32; 256]]) -> usize {
    let b17 = Base17::from_f32_row(embedding);
    let pal_idx = palette.quantize(&b17);

    // Score each class by how often this palette region appears
    let mut best_class = 0;
    let mut best_score = 0;
    for (class, hist) in histograms.iter().enumerate() {
        // Soft match: weight by palette distance to this archetype
        let mut score = 0u64;
        for (other_idx, &count) in hist.iter().enumerate() {
            if count > 0 {
                let dist = distance_matrix[pal_idx as usize][other_idx];
                score += count as u64 * (65535 - dist as u64); // inverse distance weighting
            }
        }
        if score > best_score {
            best_score = score;
            best_class = class;
        }
    }
    best_class
}
```

## PHASE 4: Evaluate on Tiny ImageNet

```
200 classes, 500 train images each = 100K training images
10K validation images (50 per class)

Metrics:
  Top-1 accuracy: % of correct first predictions
  Top-5 accuracy: % where correct class is in top 5
  Separation ratio: distance to nearest wrong class / distance to correct class

Compare:
  1. CLIP cosine (float baseline, theoretical ceiling)
  2. 8K-bit Hamming sweep (prior attempt 1)
  3. Base17 L1 (34 bytes, direct comparison)
  4. Palette lookup (1 byte, distance matrix)
  5. Palette histogram (soft match, above)
```

## EXPECTED RESULTS

The bet: 48 bytes (ZeckBF17) or even 1 byte (palette) should match
or BEAT 8K-bit Hamming (1024 bytes) for class recognition because:

1. Base17 strips noise, keeps class-discriminative structure
2. Palette centroids ARE class prototypes (k-means finds them)
3. L1 on i16[17] is a BETTER metric than Hamming for CLIP embeddings
   (continuous distance vs binary threshold)
4. The golden-step octave folding acts as a matched filter tuned to
   the structural repetition in CLIP's representation space

If top-1 accuracy at 1 byte > top-1 at 1024 bytes, that's the headline:
**indexed color beats high-dimensional binary for image recognition.**

## RUN COMMAND

```bash
cargo test test_palette_recognition --release -- --nocapture
```

## CONSTRAINTS

1. RUST ONLY. No Python. (VSACLIP golden rule)
2. Port Base17 + Palette self-contained — no ndarray dependency
3. CLIP embeddings pre-computed as f32 binary files in data/
4. All evaluation deterministic (no random splits)
