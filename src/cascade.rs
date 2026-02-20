//! Three-Layer Resonance Cascade
//!
//! L1 Features → L2 Parts → L3 Objects
//! Majority-vote superposition between layers.

use crate::{Fingerprint, ResonanceMatch, FINGERPRINT_BITS, FINGERPRINT_U64};
use crate::sweep::hdr_sweep;
use crate::hamming_distance;

/// A layer in the resonance cascade
pub struct CascadeLevel {
    pub containers: Vec<Fingerprint>,
    pub labels: Vec<Option<String>>,
    pub threshold: u32,
    pub activations: Vec<u64>,
}

impl CascadeLevel {
    pub fn new(threshold_pct: f64) -> Self {
        Self {
            containers: Vec::new(), labels: Vec::new(),
            threshold: (FINGERPRINT_BITS as f64 * threshold_pct) as u32,
            activations: Vec::new(),
        }
    }

    pub fn add(&mut self, fp: Fingerprint, label: Option<String>) {
        self.containers.push(fp); self.labels.push(label); self.activations.push(0);
    }

    pub fn activate(&mut self, input: &Fingerprint) -> Vec<ResonanceMatch> {
        let matches = hdr_sweep(input, &self.containers, self.threshold);
        for m in &matches { self.activations[m.index] += 1; }
        matches
    }

    pub fn len(&self) -> usize { self.containers.len() }
}

/// Majority-vote superposition — robust for any K
pub fn majority_superpose(fps: &[&Fingerprint]) -> Fingerprint {
    let k = fps.len();
    if k <= 1 { return fps.first().map(|f| (*f).clone()).unwrap_or_else(|| Fingerprint::from_raw([0u64; FINGERPRINT_U64])); }
    let thresh = k / 2;
    let mut result = [0u64; FINGERPRINT_U64];
    for word in 0..FINGERPRINT_U64 {
        let mut out = 0u64;
        for bit in 0..64u32 {
            let mask = 1u64 << bit;
            let count = fps.iter().filter(|fp| fp.as_raw()[word] & mask != 0).count();
            if count > thresh { out |= mask; }
        }
        result[word] = out;
    }
    Fingerprint::from_raw(result)
}

/// Three-layer resonance cascade
pub struct Cascade {
    pub features: CascadeLevel,
    pub parts: CascadeLevel,
    pub objects: CascadeLevel,
}

impl Cascade {
    pub fn new() -> Self {
        Self {
            features: CascadeLevel::new(0.30),
            parts: CascadeLevel::new(0.35),
            objects: CascadeLevel::new(0.40),
        }
    }

    pub fn recognize(&mut self, input: &Fingerprint) -> Vec<ResonanceMatch> {
        let l1 = self.features.activate(input);
        if l1.is_empty() { return Vec::new(); }
        let refs1: Vec<&Fingerprint> = l1.iter().take(20).map(|m| &self.features.containers[m.index]).collect();
        let sup1 = majority_superpose(&refs1);
        let l2 = self.parts.activate(&sup1);
        if l2.is_empty() { return Vec::new(); }
        let refs2: Vec<&Fingerprint> = l2.iter().take(10).map(|m| &self.parts.containers[m.index]).collect();
        let sup2 = majority_superpose(&refs2);
        self.objects.activate(&sup2)
    }

    /// Exposure learning: unmatched inputs become new containers
    pub fn expose(&mut self, input: &Fingerprint) {
        let l1 = self.features.activate(input);
        if l1.is_empty() { self.features.add(input.clone(), None); }
    }
}
