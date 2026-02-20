//! CLIP Ingest Pipeline — Image → Fingerprint
//!
//! Uses fastembed-rs (ONNX Runtime) to embed images via CLIP ViT-B/32,
//! then projects to binary Fingerprint via SimHash.
//!
//! This module is optional (`ingest` feature) because it pulls in ONNX.
//! The core library (sweep, cascade, exposure) works without it.
//!
//! # Usage
//!
//! ```rust,ignore
//! use vsaclip::ingest::ClipIngest;
//!
//! let ingest = ClipIngest::new()?;
//! let fp = ingest.image_to_fingerprint("photo.jpg")?;
//! let fps = ingest.batch_images(&["a.jpg", "b.jpg", "c.jpg"])?;
//! ```

// This module only compiles with the `ingest` feature
#![cfg(feature = "ingest")]

use crate::Fingerprint;
use crate::simhash::clip_projector;
use anyhow::Result;

/// CLIP-based image ingest pipeline.
///
/// Lifecycle:
/// 1. Load CLIP ViT-B/32 model (once, ~350MB download on first use)
/// 2. For each image: embed → float32[512] → SimHash → Fingerprint
pub struct ClipIngest {
    // fastembed model handle
    // model: fastembed::ImageEmbedding,
    projector: crate::simhash::SimHashProjector,
}

impl ClipIngest {
    /// Create ingest pipeline. Downloads model on first use.
    pub fn new() -> Result<Self> {
        // TODO: Initialize fastembed CLIP model
        // let model = fastembed::ImageEmbedding::try_new(
        //     fastembed::ImageInitOptions::new(
        //         fastembed::ImageEmbeddingModel::CLIPVitB32
        //     )
        // )?;

        Ok(Self {
            // model,
            projector: clip_projector(),
        })
    }

    /// Embed a single image file → Fingerprint
    pub fn image_to_fingerprint(&self, _path: &str) -> Result<Fingerprint> {
        // TODO: Read image, resize to 224×224, embed with CLIP
        // let embedding = self.model.embed(vec![path])?;
        // Ok(self.projector.project(&embedding[0]))
        todo!("fastembed CLIP integration")
    }

    /// Batch embed multiple images
    pub fn batch_images(&self, _paths: &[&str]) -> Result<Vec<Fingerprint>> {
        // TODO: Batch embed for efficiency
        todo!("fastembed batch integration")
    }

    /// Embed from raw float32 vector (for pre-computed embeddings)
    pub fn from_embedding(&self, embedding: &[f32]) -> Fingerprint {
        self.projector.project(embedding)
    }

    /// Embed from raw bytes (PNG/JPEG in memory)
    pub fn from_bytes(&self, _bytes: &[u8]) -> Result<Fingerprint> {
        todo!("fastembed from-bytes integration")
    }
}

/// Text-based CLIP embedding (for grounding labels to containers)
pub struct ClipTextIngest {
    projector: crate::simhash::SimHashProjector,
}

impl ClipTextIngest {
    pub fn new() -> Result<Self> {
        Ok(Self {
            projector: clip_projector(),
        })
    }

    /// Embed text label → Fingerprint (same space as images)
    pub fn text_to_fingerprint(&self, _text: &str) -> Result<Fingerprint> {
        // TODO: CLIP text encoder
        todo!("CLIP text encoding")
    }
}
