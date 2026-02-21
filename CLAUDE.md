# VSACLIP — Claude Code Development Prompt

## Identity

You are building **VSACLIP**: a pure-Rust proof-of-concept that does image recognition 
through Hamming resonance — NO training, NO gradient descent, NO GPU, NO numpy.

Repository: `https://github.com/AdaWorldAPI/VSACLIP`  
Substrate: `https://github.com/AdaWorldAPI/ladybug-rs` (ladybug-contract: 8192-bit Containers)

## Golden Rule

**Rust first. Python is the fallback, not the plan.**

If a Rust approach fails to compile or a crate is broken, you may fall back to Python 
with numpy for that specific step ONLY. Document the failure and tag it `// TODO: port to Rust`.
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

## Current Status (from first run — all Rust)

- [x] 16/16 unit tests pass (`cargo test --release`)
- [x] Proof binary passes all tests
- [x] Benchmark: 34.9x speedup at 10K containers (early exit vs full sweep)
- [x] 100K CLIP float embeddings computed (195MB binary, one-time pre-process)
- [x] SimHash projection working in Rust (was single-threaded, rayon fix done locally)
- [x] AVX-512 confirmed (avx512f, avx512bw, avx512vpopcntdq)
- [x] POC ran end-to-end: intra=20.3%, inter=24.6%, ratio=1.22x
- [x] rayon added to Cargo.toml (LOCAL ONLY — not committed)
- [x] batch_simhash parallelized with par_iter (LOCAL ONLY — not committed)
- [ ] **poc.rs NOT committed** — must recreate and push
- [ ] **Thresholds need tuning** — 30% is too loose, need 22-25%
- [ ] **Cascade with tuned thresholds** — not yet run with correct settings
- [ ] **Ground truth purity** — not yet measured with correct thresholds
- [ ] **.gitignore NOT committed**

## IMMEDIATE TODO (resume from here)

1. **Add rayon to Cargo.toml** — `rayon = "1"` (was added locally but not committed)
2. **Parallelize batch_simhash** in `ingest.rs` with `par_iter()` (was done locally, not committed)
3. **Create/commit poc.rs** — the main binary that runs the full pipeline
4. **Tune thresholds** — see data distribution findings below
5. **Run cascade with tuned thresholds** — aim for the 22-25% sweet spot
6. **Commit everything** — poc.rs, .gitignore, Cargo.toml with rayon, updated ingest.rs

## CRITICAL DATA FROM FIRST FULL RUN

The POC ran end-to-end on 100K images. Key findings:

```
SimHash Quality (8192-bit Containers from 512-dim CLIP):
  Intra-class mean Hamming: 20.3% of d  (~1664 bits)
  Inter-class mean Hamming: 24.6% of d  (~2015 bits)
  Separation ratio: 1.22x
```

This means:
- The distributions OVERLAP significantly
- A 30% threshold captures almost everything (too loose) -> 64 L1 clusters, 1 L3 cluster
- Thresholds MUST be in the 20-25% range (between intra and inter means)
- The sweet spot for L1 is ~22%, for L2 ~23%, for L3 ~24%

**Recommended thresholds (from data):**
```rust
const L1_THRESHOLD_PCT: f64 = 0.22;  // tight: between intra(20.3%) and inter(24.6%)
const L2_THRESHOLD_PCT: f64 = 0.23;
const L3_THRESHOLD_PCT: f64 = 0.24;
```

**The cascade needs adaptive thresholding.** Consider computing percentiles of the
actual distance distribution and setting thresholds at p25-p50 of intra-class distances.

## Performance from first run

```
SimHash 100K images:
  Single-threaded: 30+ minutes (killed)
  Rayon parallel (16 cores): ~2 minutes (estimated, was about to run)
  
Proof benchmarks:
  HDR Sweep speedup: 34.9x at 10K containers
  All 16 tests: PASS
  
Containers cached: second run loads from containers.bin (instant)
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

No training. No GPU. No numpy in the hot path.
```

## Anti-Patterns — DO NOT

- Do NOT use cosine similarity in the hot path
- Do NOT use numpy unless Rust genuinely fails (document why)
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
