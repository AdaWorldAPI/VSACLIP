# VSACLIP — Development Guide

## What This Is
Resonance-based vision: CLIP embeddings → SimHash → 16384-bit Hamming fingerprints.
No training. HDR POPCNT Sweep for search. Powered by ladybug-rs.

## Architecture Lineage
- `reference/cam_graph.rs` — Original 8192-bit design (5 RISC ops: BIND, BUNDLE, MATCH, PERMUTE, ZERO)
- `ladybug-rs` — Production 16384-bit substrate with SIMD, HDR cascade, BindSpace
- `VSACLIP` — CLIP embedding pipeline + resonance cascade on top of ladybug-rs

## Key Files
- `src/simhash.rs` — Float→binary projection (random hyperplane LSH)
- `src/sweep.rs` — HDR POPCNT Sweep (3-stage Belichtungsmesser: 64→512→16384 bit)
- `src/cascade.rs` — Three-layer resonance (Features→Parts→Objects)
- `src/exposure.rs` — Exposure metering configuration
- `proof/hdr_proof.py` — Python proof-of-concept (ALL 7 TESTS PASS)

## Building
```bash
cargo build --release  # needs rust 1.88+, ladybug-rs fetched from git
cargo test
cargo bench            # Criterion benchmarks
```

## Constants
- `FINGERPRINT_BITS = 16384` (from ladybug-rs)
- `FINGERPRINT_U64 = 256` (256 × u64 words)
- Safety margin: 1.5× for zero false negatives
- HDR bands: hot<10%, mid<30%, cold<49% of d

## The Proof
```bash
cd proof && python3 hdr_proof.py
```
Tests: superposition, early-exit, HDR scoring, resonance cascade, 
three-layer pipeline, instruction count, anti-resonance.

## Next Steps
1. `cargo test` — verify Rust compiles against ladybug-rs
2. SimHash benchmark — verify cosine→Hamming preservation
3. Tiny ImageNet pipeline — real CLIP embeddings through cascade
4. Ground truth correlation — emergent clusters vs ImageNet labels
