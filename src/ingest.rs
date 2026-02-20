//! Ingest Pipeline — float embeddings → Hamming containers via SimHash.

use ladybug_contract::container::{Container, CONTAINER_BITS, CONTAINER_WORDS};

/// Fast SimHash: project float embedding into binary Container.
///
/// Each bit = sign(dot(embedding, random_hyperplane)).
/// Preserves cosine similarity as Hamming similarity.
pub fn simhash(embedding: &[f32], seed: u64) -> Container {
    let dim = embedding.len();
    let mut result = Container::zero();

    for word_idx in 0..CONTAINER_WORDS {
        let mut word = 0u64;

        for bit in 0..64u32 {
            let bit_idx = word_idx * 64 + bit as usize;
            let mut dot = 0.0f32;

            let mut state = seed
                .wrapping_add(bit_idx as u64)
                .wrapping_mul(0x9e3779b97f4a7c15);

            for d in 0..dim {
                state ^= state >> 30;
                state = state.wrapping_mul(0xbf58476d1ce4e5b9);
                state ^= state >> 27;
                state = state.wrapping_mul(0x94d049bb133111eb);
                state ^= state >> 31;

                let sign = if state & 1 == 0 { 1.0f32 } else { -1.0f32 };
                dot += embedding[d] * sign;
            }

            if dot >= 0.0 {
                word |= 1u64 << bit;
            }
        }

        result.words[word_idx] = word;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic() {
        let emb = vec![0.1, 0.2, 0.3, -0.1, 0.5];
        assert_eq!(simhash(&emb, 42).hamming(&simhash(&emb, 42)), 0);
    }

    #[test]
    fn test_preserves_similarity() {
        let a = vec![1.0, 0.0, 0.5, 0.3, -0.2];
        let b = vec![0.9, 0.1, 0.4, 0.3, -0.1];
        let c = vec![-1.0, 0.5, -0.3, 0.8, 0.9];

        let ca = simhash(&a, 42);
        let cb = simhash(&b, 42);
        let cc = simhash(&c, 42);

        assert!(ca.hamming(&cb) < ca.hamming(&cc));
    }

    #[test]
    fn test_balanced_popcount() {
        let emb: Vec<f32> = (0..512).map(|i| (i as f32 * 0.1).sin()).collect();
        let c = simhash(&emb, 42);
        let pc = c.popcount();
        let mid = CONTAINER_BITS as u32 / 2;
        assert!(pc.abs_diff(mid) < mid / 4);
    }
}
