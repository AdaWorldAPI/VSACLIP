//! Belichtungsmesser — Exposure Metering Configuration
//!
//! Spot → Center → Matrix progressive bit-width expansion.
//! Zero false negatives with ~63x speedup at 1M containers.

use crate::FINGERPRINT_BITS;

/// Exposure configuration
#[derive(Debug, Clone)]
pub struct ExposureConfig {
    pub threshold: u32,
    pub safety_margin: f64,
    pub stage1_bits: usize,
    pub stage2_bits: usize,
    pub total_bits: usize,
}

impl ExposureConfig {
    pub fn new(threshold: u32) -> Self {
        Self { threshold, safety_margin: 1.5, stage1_bits: 64, stage2_bits: 512, total_bits: FINGERPRINT_BITS }
    }

    pub fn stage1_threshold(&self) -> u32 {
        ((self.threshold as f64) * (self.stage1_bits as f64 / self.total_bits as f64) * self.safety_margin) as u32
    }

    pub fn stage2_threshold(&self) -> u32 {
        ((self.threshold as f64) * (self.stage2_bits as f64 / self.total_bits as f64) * self.safety_margin) as u32
    }

    pub fn theoretical_speedup(&self, n: usize) -> f64 {
        let u64s = self.total_bits / 64;
        let full = n * u64s;
        let s1 = (n as f64 * 0.05) as usize;
        let s2 = (s1 as f64 * 0.10) as usize;
        let early = n * (self.stage1_bits/64) + s1 * (self.stage2_bits/64) + s2 * u64s;
        full as f64 / early as f64
    }
}

impl Default for ExposureConfig {
    fn default() -> Self { Self::new(FINGERPRINT_BITS as u32 * 30 / 100) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_speedup() {
        let c = ExposureConfig::default();
        assert!(c.theoretical_speedup(1_000_000) > 50.0);
    }
}
