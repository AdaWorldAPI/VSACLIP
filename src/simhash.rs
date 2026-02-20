//! SimHash Projection — float32 embedding → binary Fingerprint
//!
//! Projects CLIP embeddings into Hamming space preserving cosine similarity.
//! Uses random hyperplane LSH: P(sign(r·u) = sign(r·v)) = 1 - θ(u,v)/π

use crate::{Fingerprint, FINGERPRINT_BITS, FINGERPRINT_U64};
use rand::Rng;
use rand::SeedableRng;

/// SimHash projector: float32[embed_dim] → Fingerprint(16384-bit)
pub struct SimHashProjector {
    hyperplanes: Vec<f32>,
    embed_dim: usize,
}

impl SimHashProjector {
    /// Create with deterministic random hyperplanes (same seed = same projection)
    pub fn new(embed_dim: usize, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let total = FINGERPRINT_BITS * embed_dim;
        let mut hyperplanes = Vec::with_capacity(total);
        for _ in 0..total {
            let u1: f64 = rng.gen_range(0.0001f64..1.0);
            let u2: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
            hyperplanes.push((-2.0 * u1.ln()).sqrt() as f32 * u2.cos() as f32);
        }
        Self { hyperplanes, embed_dim }
    }

    /// Project float32 embedding to binary Fingerprint
    pub fn project(&self, embedding: &[f32]) -> Fingerprint {
        assert_eq!(embedding.len(), self.embed_dim);
        let mut data = [0u64; FINGERPRINT_U64];
        for bit in 0..FINGERPRINT_BITS {
            let offset = bit * self.embed_dim;
            let dot: f32 = (0..self.embed_dim)
                .map(|d| self.hyperplanes[offset + d] * embedding[d])
                .sum();
            if dot > 0.0 {
                data[bit / 64] |= 1u64 << (bit % 64);
            }
        }
        Fingerprint::from_raw(data)
    }

    pub fn embed_dim(&self) -> usize { self.embed_dim }
}

/// Standard CLIP ViT-B/32 projector (512-dim)
pub fn clip_projector() -> SimHashProjector {
    SimHashProjector::new(512, 0xADA0C11B_2025_0001)
}

/// Standard Jina CLIP v2 projector (1024-dim)
pub fn jina_projector() -> SimHashProjector {
    SimHashProjector::new(1024, 0xADA0_J1NA_2025)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hamming_distance;

    #[test]
    fn test_deterministic() {
        let proj = SimHashProjector::new(512, 42);
        let emb = vec![0.5f32; 512];
        assert_eq!(hamming_distance(&proj.project(&emb), &proj.project(&emb)), 0);
    }

    #[test]
    fn test_similar_close() {
        let proj = SimHashProjector::new(512, 42);
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let e1: Vec<f32> = (0..512).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        let e2: Vec<f32> = e1.iter().map(|x| x + rng.gen_range(-0.05f32..0.05)).collect();
        let dist = hamming_distance(&proj.project(&e1), &proj.project(&e2));
        assert!(dist < FINGERPRINT_BITS as u32 / 4, "dist={dist}");
    }

    #[test]
    fn test_opposite_far() {
        let proj = SimHashProjector::new(512, 42);
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let e1: Vec<f32> = (0..512).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        let e2: Vec<f32> = e1.iter().map(|x| -x).collect();
        let dist = hamming_distance(&proj.project(&e1), &proj.project(&e2));
        assert!(dist > FINGERPRINT_BITS as u32 * 3 / 4, "dist={dist}");
    }
}
