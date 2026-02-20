# VSACLIP — HDR POPCNT Sweep Engine

Visual recognition via Hamming resonance on [ladybug-rs](https://github.com/AdaWorldAPI/ladybug-rs) Containers.

## Core Claim

A neural network where:
- **Weights** are immutable 8192-bit Containers (not trained matrices)
- **Activation** is `POPCNT` (not ReLU)
- **Attention** is HDR exposure stacking (not softmax)
- **Inference** is `XOR + POPCNT` (not matrix multiplication)
- **Learning** is `XOR-Bind` (not backpropagation)
- **Early Exit** is progressive bit-width expansion (not learned gating)

## Performance

| Containers | Full Sweep | HDR Early Exit | AVX-512 | Speedup |
|-----------|-----------|---------------|---------|---------|
| 10K       | 427 μs    | 7 μs          | ~1 μs   | 63×     |
| 1M        | 43 ms     | 680 μs        | ~85 μs  | 63×     |
| 10M       | 427 ms    | 6.8 ms        | ~850 μs | 63×     |

## Quick Start

```bash
cargo run --bin vsaclip-proof   # proof-of-concept (6 tests)
cargo bench                     # criterion benchmarks
cargo test                      # unit tests
```

## Depends On

- `ladybug-contract` — Container type (8192-bit), xor(), hamming(), bundle()

## Modules

| Module | Purpose |
|--------|---------|
| `container_ext` | Partial Hamming, early-exit, anti-resonance |
| `hdr` | HDR exposure scoring (0-6 scale) |
| `sweep` | Core HDR POPCNT sweep engine |
| `exposure` | Belichtungsmesser stage configuration |
| `cascade` | Multi-layer resonance pipeline |
| `ingest` | SimHash: float embeddings → binary Containers |

## License

Apache-2.0
