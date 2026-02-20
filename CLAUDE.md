# VSACLIP — Claude Code Development Prompt

## Identity

You are building **VSACLIP**: a pure-Rust proof-of-concept that does image recognition 
through Hamming resonance — NO training, NO gradient descent, NO GPU, NO numpy.

Repository: `https://github.com/AdaWorldAPI/VSACLIP`  
Substrate: `https://github.com/AdaWorldAPI/ladybug-rs` (16384-bit Fingerprints, SIMD)

## Golden Rule

**Rust first. Python is the fallback, not the plan.**

If a Rust approach fails to compile or a crate is broken, you may fall back to Python 
with numpy for that specific step ONLY. Document the failure and tag it `// TODO: port to Rust`.
Everything else stays in Rust.

## Architecture in One Sentence

```
Image → CLIP (fastembed-rs) → float32[512] → SimHash → Fingerprint[u64;256] → HDR POPCNT Sweep → Recognition
```

The ONLY distance metric is **Hamming distance via POPCNT**. No cosine. No dot product. No floats in the hot path.

## Repository Layout

```
VSACLIP/
├── Cargo.toml                  # workspace root
├── CLAUDE.md                   # THIS FILE — your instructions
├── src/
│   ├── lib.rs                  # HdrScore, ResonanceMatch, CascadeLayer
│   ├── simhash.rs              # float32 → Fingerprint (random hyperplane LSH)
│   ├── sweep.rs                # HDR POPCNT Sweep (3-stage Belichtungsmesser)
│   ├── cascade.rs              # 3-layer resonance: Features→Parts→Objects
│   ├── exposure.rs             # Belichtungsmesser config
│   └── ingest.rs               # CLIP pipeline via fastembed-rs (optional feature)
├── src/bin/
│   ├── poc.rs                  # THE MAIN POC BINARY — run this
│   └── download_data.rs        # Download Tiny ImageNet
├── proof/
│   └── hdr_proof.py            # Python proof (reference only, don't touch)
├── reference/
│   └── cam_graph.rs            # Original 8192-bit design (read-only reference)
├── benches/
│   └── sweep_bench.rs          # Criterion benchmarks
└── data/                       # Tiny ImageNet lands here (gitignored)
```

## The POC — What To Build

### Phase 1: Core (MUST compile, MUST pass tests)

1. **`cargo build`** succeeds with ladybug-rs dependency
2. **SimHash tests** pass: identical→0, similar→low, opposite→high Hamming distance
3. **HDR Sweep test**: zero false negatives vs full sweep (planted matches in random sea)
4. **Majority-vote superposition** preserves similarity (bundle of K similar → close to centroid)
5. **Anti-resonance test**: inverted vector → Hamming distance > 90% of d

### Phase 2: Ingest Pipeline

6. **fastembed-rs** loads CLIP ViT-B/32 and embeds images → `Vec<f32>` (512-dim)
7. **SimHash projects** each embedding → `Fingerprint` (16384-bit)
8. Verify: two photos of dogs → low Hamming distance; dog vs car → high Hamming distance

### Phase 3: Tiny ImageNet POC

9. **Download** Tiny ImageNet (see dataset section below)
10. Embed **all 100,000 train images** → 100K Fingerprints
11. Run **resonance cascade**:
    - Start with 0 seed features
    - Expose all images; unmatched inputs become new L1 features
    - After all images: superpose co-activated features → L2 parts
    - Superpose co-activated parts → L3 objects
12. **Ground truth test**: for each of 200 classes, check if images from that class
    cluster in the same L3 object. Measure purity (% of dominant class per cluster).
13. **Print results**: cluster purity per class, overall accuracy, timing

### Phase 4: Benchmark

14. `cargo bench` — Criterion benchmarks:
    - `hdr_sweep` vs `full_sweep` at N=10K, 100K, 1M containers
    - Measure actual speedup and verify zero false negatives
    - Report instructions/container/query

## Key Constants

```rust
// From ladybug-rs — do NOT redefine
pub const FINGERPRINT_BITS: usize = 16_384;
pub const FINGERPRINT_U64: usize = 256;  // 16384/64
pub const FINGERPRINT_BYTES: usize = 2048; // 256×8

// VSACLIP thresholds
const HDR_HOT: f64 = 0.10;    // < 10% of d = blazing match
const HDR_MID: f64 = 0.30;    // < 30% of d = solid match
const HDR_COLD: f64 = 0.49;   // < 49% of d = weak signal (near noise floor)
const SAFETY_MARGIN: f64 = 1.5; // for early-exit (zero false negatives)

// SimHash seed — MUST be same everywhere for same projection
const CLIP_SIMHASH_SEED: u64 = 0xADA0C11B_2025_0001;
```

## Dependencies

```toml
[dependencies]
# Substrate — provides Fingerprint, SIMD hamming, HDR cascade, BindSpace
ladybug = { git = "https://github.com/AdaWorldAPI/ladybug-rs.git", branch = "main", default-features = false, features = ["simd"] }

# Embeddings — CLIP ViT-B/32 via ONNX Runtime (pure Rust, no Python)
fastembed = "5"

# Image loading
image = "0.25"

# Arrow for zero-copy (ladybug uses Arrow internally)
arrow = { version = "54", default-features = false, features = ["ffi"] }

# Standard
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "2"
anyhow = "1"
tracing = "0.1"
rand = "0.8"
rayon = "1"  # parallel ingest

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
```

### fastembed-rs Image Embedding Usage

```rust
use fastembed::{ImageEmbedding, ImageInitOptions, ImageEmbeddingModel};

let model = ImageEmbedding::try_new(
    ImageInitOptions::new(ImageEmbeddingModel::ClipVitB32)
        .with_show_download_progress(true),
)?;

let images = vec!["data/tiny-imagenet-200/train/n01443537/images/n01443537_0.JPEG"];
let embeddings = model.embed(images, None)?;
// embeddings[0] is Vec<f32> with 512 dimensions
```

## Dataset — Tiny ImageNet 200

**Download URL:**
```
http://cs231n.stanford.edu/tiny-imagenet-200.zip
```

**Alternative mirror (Hugging Face):**
```
https://huggingface.co/datasets/zh-plus/tiny-imagenet
```

**Download script (`src/bin/download_data.rs`):**
```bash
# Manual download (fastest):
mkdir -p data && cd data
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
rm tiny-imagenet-200.zip
```

**Dataset structure after extraction:**
```
data/tiny-imagenet-200/
├── train/                    # 100,000 images (200 classes × 500 images)
│   ├── n01443537/           # goldfish
│   │   └── images/
│   │       ├── n01443537_0.JPEG    (64×64 RGB)
│   │       ├── n01443537_1.JPEG
│   │       └── ... (500 per class)
│   ├── n01629819/           # European fire salamander
│   └── ... (200 classes)
├── val/                      # 10,000 images (50 per class)
│   └── images/
├── test/                     # 10,000 images (unlabeled)
├── wnids.txt                # 200 WordNet IDs
└── words.txt                # WordNet ID → human-readable labels
```

**Key facts:**
- 200 classes, 500 train + 50 val per class
- 64×64 pixels, RGB JPEG
- 237 MB zipped, ~248 MB extracted
- Ground truth labels via directory name (WordNet synset IDs)
- `words.txt` maps e.g. `n01443537` → `goldfish, Carassius auratus`

## The 5 RISC Operations (from cam_graph.rs)

Everything in the system reduces to these 5 operations on binary vectors:

```rust
// OP 1: BIND — XOR. Self-inverse: bind(bind(a,b), b) = a
fn bind(a: &Fingerprint, b: &Fingerprint) -> Fingerprint;

// OP 2: BUNDLE — Majority vote. Superposition preserving similarity.
fn bundle(words: &[&Fingerprint]) -> Fingerprint;

// OP 3: MATCH — Hamming distance via POPCNT. THE query engine.
fn distance(a: &Fingerprint, b: &Fingerprint) -> u32;

// OP 4: PERMUTE — Bit rotation for role encoding (source/relation/target).
fn permute(w: &Fingerprint, k: u32) -> Fingerprint;

// OP 5: RANDOM — Deterministic PRNG expansion from seed.
fn random(seed: u64) -> Fingerprint;
```

**ladybug-rs provides these as:**
- `Fingerprint::from_raw()`, `Fingerprint::from_content()`
- `core::simd::hamming_distance()` — AVX-512/AVX2/NEON auto-dispatch
- XOR via manual lane iteration (or use `search::hdr_cascade` operations)

## Belichtungsmesser — The Early Exit

Three-stage progressive bit-width expansion. Like a camera light meter:

```
Stage 1 — Spot:           64 bits  (1 u64)   → eliminates ~95%
Stage 2 — Center-weight: 512 bits  (8 u64)   → eliminates ~90% of survivors
Stage 3 — Matrix:      16384 bits (256 u64)   → exact Hamming distance

Each stage threshold = (total_threshold) × (stage_bits / total_bits) × safety_margin
Safety margin = 1.5 guarantees ZERO false negatives (proven).
```

## HDR Scoring

Each match gets an HDR score (0-6):

```
hot  = 3 if dist < 10% of d   (blazing resonance)
mid  = 2 if dist < 30% of d   (solid match)
cold = 1 if dist < 49% of d   (weak signal)
total = hot + mid + cold       (0 = noise, 6 = perfect match)
```

Anti-resonance (inhibition): if dist > 90% of d, the vector is an ANTI-match.
Free lateral inhibition without extra architecture.

## Resonance Cascade

```
L1 Features:  threshold = 30% of d   (edges, textures, colors)
L2 Parts:     threshold = 35% of d   (eyes, wings, wheels)
L3 Objects:   threshold = 40% of d   (bird, car, house)

Between layers: majority-vote superposition of top-K activated containers.
Growth: if nothing resonates → input becomes a new container (organic feature discovery).
```

## What Success Looks Like

```
$ cargo run --release --bin poc

VSACLIP Proof-of-Concept
========================

[1/6] Loading CLIP ViT-B/32...                    ✓ (2.1s)
[2/6] Scanning Tiny ImageNet train/...             ✓ 100,000 images found
[3/6] Embedding + SimHash projection...            ✓ 100,000 fingerprints (47s)
[4/6] Running resonance cascade...                 ✓ L1: 2,847 features, L2: 412 parts, L3: 53 objects
[5/6] Ground truth evaluation...
      Cluster purity: 71.3% (chance = 0.5%)
      Top-5 accuracy: 89.2%
      Classes perfectly separated: 142/200
[6/6] Sweep benchmark (N=100,000)...
      Full sweep:  4,267 μs/query
      HDR sweep:      68 μs/query  (62.7× speedup)
      False negatives: 0

Done. No training. No GPU. No numpy.
```

## Build & Run

```bash
# Clone
git clone https://github.com/AdaWorldAPI/VSACLIP.git
cd VSACLIP

# Download dataset
mkdir -p data && cd data
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip && rm tiny-imagenet-200.zip
cd ..

# Build & test
cargo build --release
cargo test --release

# Run POC
cargo run --release --bin poc

# Benchmarks
cargo bench
```

## Failure Modes & Fallbacks

| Problem | Action |
|---------|--------|
| ladybug-rs won't compile | Check Rust 1.88+. Try `default-features = false, features = ["simd"]` |
| fastembed model download fails | Pre-download ONNX model from HuggingFace: `Qdrant/clip-ViT-B-32-vision` |
| fastembed image embedding crashes | **FALLBACK**: use Python `pip install fastembed` to pre-compute embeddings as `.bin` files, load in Rust as raw `[u64;256]` |
| SimHash projection quality is bad | Try more hyperplanes, different seed, or use Jina CLIP v2 (1024-dim) |
| Resonance cascade doesn't cluster | Lower L1 threshold to 25%, increase exposure passes |
| Cluster purity < 30% | The SimHash projection may lose too much info. **FALLBACK**: Use numpy + sklearn to verify CLIP embeddings have structure before SimHash |

## Anti-Patterns — DO NOT

- ❌ Do NOT use cosine similarity anywhere in the hot path
- ❌ Do NOT use numpy unless Rust genuinely fails (document why)
- ❌ Do NOT use PyTorch, TensorFlow, or any Python ML framework
- ❌ Do NOT add a training loop — this system learns by exposure, not gradient descent
- ❌ Do NOT use HashMap for fingerprint lookup — it's POPCNT scan or nothing
- ❌ Do NOT store labels in containers — labels are for evaluation only
- ❌ Do NOT modify `proof/hdr_proof.py` — it's the reference implementation
- ❌ Do NOT modify `reference/cam_graph.rs` — it's the architectural DNA

## GitHub Authentication

```bash
# For private repo access (ladybug-rs)
# Set GH_TOKEN env var or use `gh auth login`
git config --global credential.helper store
```

## Philosophy

> Containers are the weights. POPCNT is the activation. XOR is learning.
> HDR stacking is attention. Early exit is the light meter.
> There is no training. There is only resonance.
