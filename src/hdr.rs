//! HDR Exposure Scoring — attention without softmax.
//!
//! Three exposure levels capture the full resonance spectrum:
//! - Hot  (3 points): dist < 10% of d → blazing match
//! - Mid  (2 points): dist < 30% of d → solid match  
//! - Cold (1 point):  dist < 49% of d → weak signal
//! - None (0 points): dist ≥ 49% of d → noise
//!
//! The HDR score (0-6) replaces softmax attention weights.

use crate::HdrScore;
use ladybug_contract::container::CONTAINER_BITS;

/// Default thresholds as fractions of d
pub const HDR_HOT_FRAC: f32 = 0.10;
pub const HDR_MID_FRAC: f32 = 0.30;
pub const HDR_COLD_FRAC: f32 = 0.49;

/// Precomputed thresholds for CONTAINER_BITS = 8192
pub const HDR_HOT_THRESHOLD: u32 = (CONTAINER_BITS as f32 * HDR_HOT_FRAC) as u32;   // 819
pub const HDR_MID_THRESHOLD: u32 = (CONTAINER_BITS as f32 * HDR_MID_FRAC) as u32;   // 2457
pub const HDR_COLD_THRESHOLD: u32 = (CONTAINER_BITS as f32 * HDR_COLD_FRAC) as u32; // 4014

/// Configurable HDR thresholds
#[derive(Debug, Clone, Copy)]
pub struct HdrConfig {
    pub hot: u32,
    pub mid: u32,
    pub cold: u32,
}

impl Default for HdrConfig {
    fn default() -> Self {
        Self {
            hot: HDR_HOT_THRESHOLD,
            mid: HDR_MID_THRESHOLD,
            cold: HDR_COLD_THRESHOLD,
        }
    }
}

impl HdrConfig {
    /// Create for a given bit-width using default fractions.
    pub fn for_bits(d: u32) -> Self {
        Self::from_fractions(HDR_HOT_FRAC, HDR_MID_FRAC, HDR_COLD_FRAC, d)
    }

    /// Create from fractional thresholds
    pub fn from_fractions(hot: f32, mid: f32, cold: f32, d: u32) -> Self {
        Self {
            hot: (d as f32 * hot) as u32,
            mid: (d as f32 * mid) as u32,
            cold: (d as f32 * cold) as u32,
        }
    }

    /// Score a single Hamming distance
    #[inline]
    pub fn score(&self, dist: u32) -> HdrScore {
        HdrScore {
            hot:  if dist < self.hot  { 3 } else { 0 },
            mid:  if dist < self.mid  { 2 } else { 0 },
            cold: if dist < self.cold { 1 } else { 0 },
        }
    }
}

/// Distribution of HDR scores across a sweep
#[derive(Debug, Default)]
pub struct HdrDistribution {
    pub counts: [usize; 7], // index = total score (0-6)
}

impl HdrDistribution {
    pub fn record(&mut self, score: &HdrScore) {
        let total = score.total() as usize;
        if total < 7 {
            self.counts[total] += 1;
        }
    }

    pub fn noise_fraction(&self) -> f64 {
        let total: usize = self.counts.iter().sum();
        if total == 0 { return 1.0; }
        self.counts[0] as f64 / total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hdr_scoring_bands() {
        let cfg = HdrConfig::default();

        // Blazing
        let s = cfg.score(100);
        assert_eq!(s.total(), 6);

        // Solid
        let s = cfg.score(1500);
        assert_eq!(s.total(), 3); // mid + cold

        // Weak
        let s = cfg.score(3500);
        assert_eq!(s.total(), 1); // cold only

        // Noise
        let s = cfg.score(5000);
        assert_eq!(s.total(), 0);
    }
}
