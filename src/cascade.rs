//! Resonance Cascade — multi-layer visual recognition pipeline.
//!
//! Three-layer architecture where each layer's output becomes
//! the next layer's input via majority-vote superposition:
//!
//! ```text
//! Layer 1 (Features):  patches → sweep → activated features
//! Layer 2 (Parts):     features → bundle → sweep → activated parts
//! Layer 3 (Objects):   parts → bundle → sweep → recognized objects
//! ```
//!
//! No training. No backpropagation. No loss function.
//! Learning happens via XOR-bind commitments during inference.

use ladybug_contract::container::Container;
use crate::sweep::{hdr_sweep, SweepConfig, SweepResult};
use crate::ResonanceMatch;

/// A layer in the resonance cascade
pub struct ResonanceLayer {
    /// Name for debugging
    pub name: String,
    /// The container corpus for this layer
    pub containers: Vec<Container>,
    /// Sweep configuration (threshold, HDR bands)
    pub config: SweepConfig,
    /// How many top activations to propagate to next layer
    pub top_k: usize,
}

impl ResonanceLayer {
    pub fn new(name: impl Into<String>, containers: Vec<Container>, threshold: u32) -> Self {
        Self {
            name: name.into(),
            containers,
            config: SweepConfig {
                threshold,
                ..Default::default()
            },
            top_k: 20,
        }
    }

    /// Run a sweep against this layer
    pub fn sweep(&self, query: &Container) -> SweepResult {
        hdr_sweep(query, &self.containers, &self.config)
    }

    /// Get the containers of the top-K activated matches
    pub fn activated_containers(&self, matches: &[ResonanceMatch]) -> Vec<&Container> {
        matches.iter()
            .take(self.top_k)
            .map(|m| &self.containers[m.index])
            .collect()
    }
}

/// Result of a full cascade (all layers)
#[derive(Debug)]
pub struct CascadeResult {
    pub layer_results: Vec<LayerResult>,
    pub recognized: Vec<ResonanceMatch>,
}

#[derive(Debug)]
pub struct LayerResult {
    pub name: String,
    pub matches: Vec<ResonanceMatch>,
    pub instructions: u64,
}

/// Run a full resonance cascade through multiple layers.
///
/// Input `query` is typically a majority-bundle of image patches.
/// Each layer sweeps its corpus, then bundles the top-K activations
/// into a new query for the next layer.
pub fn cascade(query: &Container, layers: &[ResonanceLayer]) -> CascadeResult {
    let mut current_query = query.clone();
    let mut layer_results = Vec::new();

    for layer in layers {
        let result = layer.sweep(&current_query);

        let lr = LayerResult {
            name: layer.name.clone(),
            matches: result.matches.clone(),
            instructions: result.instructions,
        };

        // Bundle top-K activations as query for next layer
        if !result.matches.is_empty() {
            let activated = layer.activated_containers(&result.matches);
            let refs: Vec<&Container> = activated.into_iter().collect();
            if !refs.is_empty() {
                current_query = Container::bundle(&refs);
            }
        }

        layer_results.push(lr);
    }

    let recognized = layer_results.last()
        .map(|lr| lr.matches.clone())
        .unwrap_or_default();

    CascadeResult {
        layer_results,
        recognized,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_near(reference: &Container, approx_dist: usize) -> Container {
        let mut c = reference.clone();
        let mut state = approx_dist as u64 ^ 0xCAFEBABE;
        for _ in 0..approx_dist {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let bit = (state as usize) % 8192;
            c.words[bit / 64] ^= 1u64 << (bit % 64);
        }
        c
    }

    #[test]
    fn test_two_layer_cascade() {
        // Layer 1: 50 features
        let features: Vec<Container> = (0..50u64)
            .map(|s| Container::random(s + 100))
            .collect();

        // Layer 2: 10 parts, each is bundle of 2-3 features
        let mut parts = Vec::new();
        for i in 0..10 {
            let refs: Vec<&Container> = (0..3).map(|j| &features[(i * 3 + j) % 50]).collect();
            parts.push(Container::bundle(&refs));
        }

        // Query: close to feature[0] and feature[1]
        let query_parts = vec![
            make_near(&features[0], 400),
            make_near(&features[1], 400),
        ];
        let refs: Vec<&Container> = query_parts.iter().collect();
        let query = Container::bundle(&refs);

        let layers = vec![
            ResonanceLayer::new("features", features, 3500),
            ResonanceLayer::new("parts", parts, 3800),
        ];

        let result = cascade(&query, &layers);

        assert_eq!(result.layer_results.len(), 2);
        println!("L1 matches: {}", result.layer_results[0].matches.len());
        println!("L2 matches: {}", result.layer_results[1].matches.len());
    }
}
