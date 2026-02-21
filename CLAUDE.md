# VSACLIP — Claude Code Development Prompt

## Identity

You are building **VSACLIP**: a pure-Rust proof-of-concept that does image recognition 
through Hamming resonance — NO training, NO gradient descent, NO GPU, NO numpy.

Repository: `https://github.com/AdaWorldAPI/VSACLIP`  
Substrate: `https://github.com/AdaWorldAPI/ladybug-rs` (ladybug-contract: 8192-bit Containers)

## Golden Rule

**Rust ONLY. No numpy. No Python in any hot path. EVER.**

Python exists ONLY as the CLIP embedding translator (`scripts/embed_hf.py`).
This is a one-time pre-processing step that writes binary files.
ALL Hamming operations — SimHash, sweep, cascade, Belichtungsmesser — are pure Rust.
If a Rust crate is broken, fix it or find another Rust crate. Do NOT reach for numpy.
Everything else stays in Rust.

## CRITICAL LESSONS FROM FIRST RUN

### 1. Container size is 8192 bits (128 × u64), NOT 16384

ladybug-contract uses `Container { words: [u64; 128] }` = 8192 bits.
The VSACLIP code correctly uses this. Do NOT mix up with ladybug-rs Fingerprint (16384-bit).

```rust
use ladybug_contract::container::{Container, CONTAINER_BITS, CONTAINER_WORDS};
// CONTAINER_BITS = 8192, CONTAINER_WORDS = 128
```

### 2. SimHash MUST be parallelized with rayon

SimHash projection of 100K images took 25+ minutes single-threaded.
Each image projection is independent. Use rayon:

```rust
use rayon::prelude::*;

let containers: Vec<Container> = embeddings.par_iter()
    .map(|emb| simhash(emb))
    .collect();
```

This should reduce 25 min to 2-3 min on 8+ cores.

### 3. CLIP float embeddings are a pre-processing step

The pipeline is:
1. **Pre-process (once)**: CLIP ViT-B/32 generates float32[512] per image
   - Used Python fastembed + HuggingFace datasets to load Tiny ImageNet
   - Saved as binary: `[count: u32][dim: u32][f32 x count x dim]`
   - This is a ONE-TIME translator step, not part of the architecture
2. **Everything else is Rust**: SimHash, POPCNT sweep, cascade, evaluation
   - Rust loads the binary embeddings, projects via SimHash, sweeps via Hamming
   - The entire hot path is Rust + SIMD, zero Python

The CLIP model is just a TRANSLATOR from pixel-space to vector-space.
Once translated, the float vectors become binary Containers and never touch floats again.

### 4. File size limits — split large binaries

- `embeddings.bin` (100K x 512 x 4B = 195MB) exceeds GitHub 100MB limit
- `containers.bin` (100K x 128 x 8B = 97MB) just under limit
- Solution: only commit containers.bin + labels.bin, not embeddings.bin
- Embeddings can be regenerated from the Python script

### 5. The poc.rs binary was NOT pushed — commit it!

The POC binary (`src/bin/poc.rs`) with the full pipeline exists only locally.
It must be committed to GitHub. Also commit:
- `.gitignore`
- `scripts/embed_tiny_imagenet.py` (the Python embedding script)
- `data/containers.bin` (if under 100MB)
- `data/labels.bin` (small, ~400KB)

## Architecture in One Sentence

```
Image -> CLIP float32[512] (one-time pre-process) -> SimHash (Rust/rayon) -> Container[u64;128] -> HDR POPCNT Sweep (Rust/SIMD) -> Recognition
```

The CLIP embedding is a one-time translation step. ALL Hamming work is Rust.
No Python in the hot path. No Python in SimHash. No Python in sweep or cascade.

## Current Status — POC COMPLETE

Branch: `claude/vsaclip-hamming-recognition-y0b94` (14 files, 104K insertions)

### Actual Results (100K Tiny ImageNet, 200 classes, 8192-bit Container)

```
L1 Features:          4,971
L2 Parts:             1,812
L3 Objects:             481
Cluster purity:        7.7%  (15x above 0.5% chance)
Single-class clusters:  204
Images assigned:     99,607 / 100,000
HDR sweep speedup:      2x  (zero false negatives)
SimHash (rayon, 16c):  221s  (was 30+ min single-threaded)
All tests:            16/16 passing
```

### Distance Distribution (the bottleneck)

```
Intra-class mean:  20.3% of d  (~1664 bits)
Inter-class mean:  24.6% of d  (~2015 bits)
Separation ratio:  1.22x        <- THIS IS THE BOTTLENECK
```

### What is pushed to GitHub

- [x] `src/bin/poc.rs` — full pipeline binary
- [x] `src/bin/download_data.rs` — dataset download
- [x] `src/ingest.rs` — rayon parallel SimHash + I/O
- [x] `scripts/embed_hf.py` — Python CLIP fallback
- [x] `scripts/embed_images.py` — local image embedding
- [x] `data/containers.bin` — 98MB pre-computed database (100K containers)
- [x] `data/labels.txt` — 200 class labels
- [x] `benches/hdr_sweep.rs` — multi-scale benchmarks
- [x] `Cargo.toml` — rayon + binary definitions
- [x] 16/16 tests passing, all warnings cleaned

## NEXT PHASE: Break the 1.22x Barrier

The POC works. 7.7% purity = 15x above chance = the architecture is sound.
But 1.22x separation is the ceiling. Two levers to pull:

### Lever 1: Switch to 16,384-bit Fingerprint (ladybug-rs)

ladybug-rs main crate has `Fingerprint` (256 x u64 = 16,384 bits).
ladybug-contract has `Container` (128 x u64 = 8,192 bits).
Doubling the bits = doubling the SimHash hyperplanes = better fidelity.

```rust
// SWITCH FROM:
use ladybug_contract::container::{Container, CONTAINER_BITS, CONTAINER_WORDS};
// TO:
use ladybug::core::fingerprint::Fingerprint;
// Fingerprint is 16,384 bits (256 x u64)
```

This is the simplest change and should improve separation directly.

### Lever 2: X-Trans Structured Projection (see experiment section below)

Replace random hyperplanes with golden-angle / cross-bind / holographic projection.
Maximally independent bits instead of redundant random sampling.

### Lever 3: Use native ladybug HdrIndex

ladybug-rs `hdr_cascade.rs` already has:
- `belichtung_meter()` — 7-point at `[0, 37, 79, 127, 167, 211, 251]`
- `MexicanHat` — excitation/inhibition curves
- `HdrIndex` — multi-resolution sketch cascade
- `QualityTracker` — adaptive threshold from stddev history

Currently VSACLIP reimplements a simpler 3-stage version. Switch to native API.

### Lever 4: Better CLIP Model

Current: CLIP ViT-B/32 (512-dim, 2021). Alternatives:
- Jina CLIP v2 (1024-dim, 2024) — 2x embedding dimensions
- Unicom ViT-B-32 (512-dim, 2023) — better embedding structure
- Larger model = more information to project

### Priority Order

1. X-Trans projection (highest expected impact, same Container size)
2. 16,384-bit Fingerprint (double bits, simple change)
3. Both combined (X-Trans + 16K bits)
4. Native HdrIndex (better cascade, less code)
5. Better CLIP model (external dependency)

## Performance Numbers

```
SimHash 100K (rayon, 16 cores):  221 seconds
HDR sweep speedup:               2x at 100K containers
Early exit (proof, 10K):         34.9x speedup
All 16 unit tests:               PASS
Zero false negatives:            Confirmed at all scales
```

## Dataset — Tiny ImageNet 200

**Direct download (may have SSL issues):**
```bash
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
```

**Working approach — HuggingFace datasets (Python):**
```python
from datasets import load_dataset
ds = load_dataset("zh-plus/tiny-imagenet", split="train")
# ds[i]["image"] -> PIL Image, ds[i]["label"] -> int (0-199)
```

**Dataset facts:**
- 200 classes x 500 train images = 100,000 total
- 64x64 RGB JPEG
- 237MB zipped
- Labels: WordNet synset IDs (n01443537 = goldfish, etc.)

## Dependencies

```toml
[dependencies]
ladybug-contract = { git = "https://github.com/AdaWorldAPI/ladybug-rs.git", branch = "main" }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
anyhow = "1"
rand = "0.8"
rayon = "1"           # CRITICAL for SimHash parallelization

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
```

## The 5 RISC Operations (from cam_graph.rs)

```rust
// OP 1: BIND — XOR. Self-inverse: bind(bind(a,b), b) = a
// OP 2: BUNDLE — Majority vote. Superposition.
// OP 3: MATCH — Hamming distance via POPCNT.
// OP 4: PERMUTE — Bit rotation for roles.
// OP 5: RANDOM — Deterministic PRNG expansion.
```

## Belichtungsmesser — Early Exit

```
Stage 1 — Spot:          64 bits (1 u64)    -> eliminates ~80-95%
Stage 2 — Center:       512 bits (8 u64)    -> eliminates ~80-90% of survivors
Stage 3 — Matrix:      8192 bits (128 u64)  -> exact Hamming

threshold_stage = total_threshold x (stage_bits / 8192) x 1.5
Safety margin 1.5x = ZERO false negatives (proven, 16/16 tests pass)
```

## HDR Scoring (0-6)

```
hot  = 3 if dist < 10% x 8192 =  819
mid  = 2 if dist < 30% x 8192 = 2458
cold = 1 if dist < 49% x 8192 = 4014
total = hot + mid + cold
Anti-resonance: dist > 90% x 8192 = 7373 -> inhibition
```

## What Success Looks Like

```
$ cargo run --release --bin poc

VSACLIP Proof-of-Concept
========================

[1/6] Loading embeddings...                        ok 100,000 x 512
[2/6] SimHash projection (parallel)...             ok 100,000 containers (2.3s)
[3/6] Running resonance cascade...                 ok L1: ~3K, L2: ~400, L3: ~50
[4/6] Ground truth evaluation...
      Cluster purity: 40-70% (chance = 0.5%)
      Top-5 accuracy: 60-85%
[5/6] Sweep benchmark (N=100,000)...
      Full sweep:  4.3 ms/query
      HDR sweep:   68 us/query  (63x speedup)
      False negatives: 0
[6/6] Saving containers.bin...                     ok 97 MB

No training. No GPU. No numpy. Period.
```

## Anti-Patterns — DO NOT

- Do NOT use cosine similarity in the hot path
- Do NOT use numpy or Python for ANY new code — Rust only
- Do NOT run SimHash single-threaded on 100K+ images (use rayon)
- Do NOT try to push >100MB files to GitHub without LFS
- Do NOT use HashMap for fingerprint lookup — POPCNT scan only
- Do NOT store labels in containers — labels are for evaluation only
- Do NOT modify `proof/hdr_proof.py` or `reference/cam_graph.rs`
- Do NOT confuse Container (8192-bit) with Fingerprint (16384-bit)

## Philosophy

> Containers are the weights. POPCNT is the activation. XOR is learning.
> HDR stacking is attention. Early exit is the light meter.
> There is no training. There is only resonance.

---

## EXPERIMENT: X-Trans Projection (Break the 1.22x Barrier)

### The Problem

Current SimHash uses random hyperplanes — equivalent to a **Bayer filter** in camera sensors.
Every 2x2 block repeats. Massive redundancy. When two regular grids interact, you get moiré —
aliasing that destroys information. Our version of "moiré" is the collapsed distance distribution:
intra-class 20.3%, inter-class 24.6%, separation ratio only 1.22x. The bits are correlated,
measuring the same directions over and over.

### The Insight: Fujifilm X-Trans

Fujifilm solved moiré by replacing the 2x2 Bayer grid with a **6x6 non-periodic pattern**.
Every row and column sees all three colors. The aperiodicity eliminates interference without
needing a low-pass filter (which blurs = loses information). Result: sharper images from the
same sensor resolution.

**Our equivalence:**

```
Camera sensor    →  SimHash projection
Bayer 2x2 grid   →  i.i.d. random hyperplanes (correlated, redundant)
X-Trans 6x6      →  Structured non-periodic projection (maximally independent bits)
Moiré artifacts   →  Collapsed distance distribution (1.22x)
Low-pass filter   →  Not needed if projection has no aliasing
```

### Three Approaches to Try (in order of expected impact)

#### Approach 1: Golden Angle Spiral Hyperplanes

Instead of random hyperplanes, generate them on a **golden angle spiral** in embedding space.
The golden angle (137.508 degrees) produces maximally non-repeating angular coverage —
exactly like sunflower seed placement. No two hyperplanes cluster.

```rust
const PHI: f64 = 1.618033988749895; // golden ratio
const GOLDEN_ANGLE: f64 = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt()); // ~2.399963 rad

fn golden_hyperplane(bit_idx: usize, dim: usize) -> Vec<f32> {
    let mut plane = vec![0.0f32; dim];
    for d in 0..dim {
        // Golden angle rotation in (bit_idx, d) space
        let angle = (bit_idx as f64 * GOLDEN_ANGLE) + (d as f64 * PHI);
        plane[d] = angle.cos() as f32;
    }
    // Normalize
    let norm: f32 = plane.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in &mut plane { *x /= norm; }
    plane
}
```

Each bit measures a genuinely different direction. Expected: wider distance distribution,
better separation.

#### Approach 2: Cross-Bind Projection (XOR Differential)

Instead of `bit_i = sign(dot(emb, h_i))`, use **bound pairs**:

```rust
// Cross-bind: bit_i = sign(dot(emb, h_i)) XOR sign(dot(emb, h_j))
// where j = (i * PHI) mod CONTAINER_BITS
// Each bit encodes the DIFFERENCE between two projections
// This is a differential measurement — like edge detection
fn cross_bind_simhash(embedding: &[f32], seed: u64) -> Container {
    let mut result = Container::zero();
    for bit in 0..CONTAINER_BITS {
        let j = ((bit as f64 * PHI) as usize) % CONTAINER_BITS;
        let sign_i = dot_hyperplane(embedding, bit, seed) >= 0.0;
        let sign_j = dot_hyperplane(embedding, j, seed) >= 0.0;
        if sign_i ^ sign_j {  // XOR = differential
            result.words[bit / 64] |= 1u64 << (bit % 64);
        }
    }
    result
}
```

Each bit captures a **relationship between two directions**, not just one direction.
The golden-ratio pairing ensures no two crosses overlap. Information per bit is higher.
This is like edge detection vs pixel sampling — edges carry more discriminative information.

#### Approach 3: Permute-Bind Multi-View (Holographic)

Use ladybug's native PERMUTE + BIND to create multiple "views" and superpose them:

```rust
// Take K different projections of the same embedding
// Permute each by a different offset
// XOR-bind them together into one Container
// The result encodes K simultaneous views holographically
fn holographic_simhash(embedding: &[f32], seed: u64) -> Container {
    let view1 = simhash(embedding, seed);
    let view2 = simhash(embedding, seed ^ 0xDEAD);
    let view3 = simhash(embedding, seed ^ 0xBEEF);

    let p1 = view1;  // identity
    let p2 = Container::permute(&view2, 1);  // rotate by 1 word
    let p3 = Container::permute(&view3, 2);  // rotate by 2 words

    // XOR-bind all views: holographic superposition
    let mut result = Container::zero();
    for i in 0..CONTAINER_WORDS {
        result.words[i] = p1.words[i] ^ p2.words[i] ^ p3.words[i];
    }
    result
}
```

This is the full X-Trans approach: multiple color channels (views) at different
spatial offsets (permutations), bound into a single representation. Each word in the
Container contains information from all three views simultaneously — just like
X-Trans puts R, G, B in every row and column.

### Measurement

For each approach, measure:

```
1. Intra-class mean Hamming distance (currently 20.3%)
2. Inter-class mean Hamming distance (currently 24.6%)
3. Separation ratio (currently 1.22x) — TARGET: > 1.5x
4. Distance distribution width (stddev of intra vs inter)
5. Overlap integral between intra and inter distributions
6. Cluster purity after resonance cascade
```

### Why This Should Work (Information Theory)

Random hyperplanes: each bit has ~1 bit of information about 1 random direction.
With 8192 bits and 512 dims, you get ~16x oversampling of the same space.
Most bits are redundant. The effective dimensionality is << 8192.

Golden/cross/holographic: each bit captures information that is maximally
UNLIKE the information in other bits. The effective dimensionality approaches 8192.
More independent bits = wider distance distribution = better separation.

The Bayer-to-X-Trans improvement in cameras is roughly 15-25% more effective resolution
from the same pixel count. We should expect a similar jump in effective Hamming discrimination.

### Integration with Existing Code

The projection change is ISOLATED to `ingest.rs` and `simhash()`. Everything downstream
(sweep, cascade, evaluation) stays exactly the same. The Belichtungsmesser, early exit,
HDR scoring — all still work. Only the projection function changes.

```rust
// In ingest.rs — add these as alternatives:
pub fn simhash_golden(embedding: &[f32], seed: u64) -> Container;
pub fn simhash_crossbind(embedding: &[f32], seed: u64) -> Container;
pub fn simhash_holographic(embedding: &[f32], seed: u64) -> Container;

// In poc.rs — compare all four:
let containers_random = batch_simhash(embeddings, simhash);
let containers_golden = batch_simhash(embeddings, simhash_golden);
let containers_cross  = batch_simhash(embeddings, simhash_crossbind);
let containers_holo   = batch_simhash(embeddings, simhash_holographic);
// Measure separation ratio for each
```
