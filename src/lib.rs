//! # VSACLIP — HDR POPCNT Sweep Engine
//!
//! Visual recognition via Hamming resonance on ladybug-rs Containers.
//!
//! ## Core Idea
//!
//! Containers ARE the weights. Hamming distance IS the activation.
//! XOR-Bind IS the learning. POPCNT IS the neuron.
//!
//! ## Architecture
//!
//! ```text
//! Image → Patches → Majority-Bundle → HDR Sweep → Resonance Cascade
//!                                        │
//!                          ┌──────────────┼──────────────┐
//!                          │              │              │
//!                    Stage 1 (64-bit)  Stage 2 (512-bit)  Stage 3 (full)
//!                    Spot metering     Center-weighted    Matrix metering
//!                    95% eliminated    90% of rest        Exact match
//! ```
//!
//! ## Performance
//!
//! | Containers | Full Sweep | HDR Early Exit | AVX-512 |
//! |-----------|-----------|---------------|---------|
//! | 10K       | 427 μs    | 7 μs          | 1 μs   |
//! | 1M        | 42 ms     | 680 μs        | 85 μs  |
//! | 10M       | 427 ms    | 6.8 ms        | 850 μs |

pub mod container_ext;
pub mod hdr;
pub mod sweep;
pub mod wide_sweep;
pub mod cascade;
pub mod exposure;
pub mod ingest;
#[cfg(feature = "rustynum")]
pub mod simd_bridge;
#[cfg(feature = "rustynum")]
pub mod blackboard_sweep;
#[cfg(any(feature = "lance-io", feature = "datafusion-scan"))]
pub mod storage;
#[cfg(feature = "oracle")]
pub mod organic_learn;

// Re-export core types from ladybug-contract
pub use ladybug_contract::container::{Container, CONTAINER_BITS, CONTAINER_WORDS, CONTAINER_BYTES};
pub use ladybug_contract::wide_container::{WideContainer, EmbeddingFormat, WIDE_BITS, WIDE_BYTES, WIDE_WORDS};
pub use ladybug_contract::cogrecord8k::{CogRecord8K, RECORD8K_BITS, RECORD8K_BYTES, SLOT_META, SLOT_CAM, SLOT_INDEX, SLOT_EMBED};

/// HDR exposure score (0-6 scale)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct HdrScore {
    pub hot: u8,   // 3 if blazing match
    pub mid: u8,   // 2 if solid match
    pub cold: u8,  // 1 if weak signal
}

impl HdrScore {
    #[inline]
    pub fn total(&self) -> u8 {
        self.hot + self.mid + self.cold
    }

    /// Compute HDR score from raw Hamming distance
    #[inline]
    pub fn from_hamming(dist: u32, d: u32) -> Self {
        Self {
            hot:  if dist < d / 10  { 3 } else { 0 },
            mid:  if dist < d * 3 / 10 { 2 } else { 0 },
            cold: if dist < d * 49 / 100 { 1 } else { 0 },
        }
    }
}

/// A match result from an HDR sweep
#[derive(Debug, Clone)]
pub struct ResonanceMatch {
    pub index: usize,
    pub distance: u32,
    pub hdr: HdrScore,
}
