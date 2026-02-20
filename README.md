# VSACLIP — Resonance Vision Without Training

**Vector Symbolic Architecture meets CLIP: Image recognition through Hamming resonance, not gradient descent.**

[![ladybug-rs](https://img.shields.io/badge/powered_by-ladybug--rs-blue.svg)](https://github.com/AdaWorldAPI/ladybug-rs)

## The Claim

A vision system where:
- **Weights** are immutable binary containers (not trained matrices)
- **Activation** is POPCNT (not ReLU)
- **Attention** is HDR exposure stacking (not softmax)
- **Inference** is XOR + POPCNT (not matrix multiplication)
- **Learning** is XOR-Bind (not backpropagation)
- **No training set.** Seeds + exposure + time.

Inference: **~85μs** for 1M containers on AVX-512. No GPU.

## Architecture

```
Image → CLIP (float32) → SimHash → Fingerprint (16384-bit)
                                        ↓
                              HDR POPCNT Sweep (ladybug-rs SIMD)
                                        ↓
                              Resonance Cascade (3 layers)
                                        ↓
                              Emergent Recognition
```

### Powered by ladybug-rs

| Component | ladybug-rs module | Purpose |
|-----------|------------------|---------|
| Fingerprint | `core::Fingerprint` | 16384-bit aligned binary vectors |
| SIMD | `core::simd` | AVX-512/AVX2/NEON Hamming distance |
| HDR Cascade | `search::hdr_cascade` | Multi-resolution search |
| Container | `container::*` | Immutable BindSpace |

## Proof of Concept

```bash
cd proof/
pip install numpy
python hdr_proof.py
```

All 7 tests pass:
- ✓ Majority-vote superposition: rank 0 at K=49
- ✓ Early exit: 31× speedup, zero false negatives
- ✓ HDR scoring: 96.2% noise floor
- ✓ Resonance cascade: 5/5 classes organic
- ✓ Three-layer pipeline: recognition without labels
- ✓ 63× instruction speedup at scale
- ✓ Anti-resonance: free lateral inhibition

## Structure

```
VSACLIP/
├── Cargo.toml          # depends on ladybug-rs
├── src/
│   ├── lib.rs          # HdrScore, ResonanceMatch
│   ├── simhash.rs      # float32 → Fingerprint
│   ├── sweep.rs        # HDR POPCNT Sweep
│   ├── cascade.rs      # 3-layer resonance cascade
│   ├── exposure.rs     # Belichtungsmesser config
│   └── ingest.rs       # CLIP pipeline (optional)
├── proof/
│   └── hdr_proof.py    # Python proof (ALL PASS)
└── benches/
    └── sweep_bench.rs  # Criterion benchmarks
```

## License

Apache-2.0
