//! SIMD Bridge — connects ladybug-contract WideContainer to rustynum SIMD ops.
//!
//! This module provides zero-copy conversion between:
//! - `WideContainer` (16,384-bit, 256×u64) ↔ `NumArrayU8` (2,048 bytes)
//! - `CogRecord8K` (65,536-bit, 4×WideContainer) ↔ `NumArrayU8` (8,192 bytes)
//!
//! Once converted, all rustynum SIMD operations are available:
//! - **Hamming distance**: fused XOR + POPCNT (4× unrolled, AVX-512)
//! - **Int8 dot product**: VNNI-targetable VPDPBUSD
//! - **Bundle**: ripple-carry majority vote with explicit u64x8 SIMD
//! - **Permute**: circular bit-rotation
//! - **Bitwise**: XOR, AND, OR, NOT (AVX-512 u8x64)
//! - **Batch Hamming**: parallel sweep over multiple containers
//!
//! # Architecture
//!
//! The bridge is zero-allocation for reads: `as_bytes()` on WideContainer
//! gives a `&[u8; 2048]` that can be wrapped into a NumArrayU8 view.
//! For writes, we copy back from NumArrayU8 → WideContainer.

use ladybug_contract::wide_container::{WideContainer, WIDE_BYTES};
use ladybug_contract::cogrecord8k::CogRecord8K;
use rustynum_rs::NumArrayU8;

// =============================================================================
// WIDECONTAINER ↔ NUMARRAYU8
// =============================================================================

/// Convert a WideContainer to a NumArrayU8 (copy).
///
/// The resulting NumArrayU8 is 2,048 bytes — a single CogRecord8K container.
/// All rustynum SIMD operations are now available.
#[inline]
pub fn wide_to_array(c: &WideContainer) -> NumArrayU8 {
    NumArrayU8::new(c.as_bytes().to_vec())
}

/// Convert a NumArrayU8 back to a WideContainer (copy).
///
/// # Panics
/// Panics if the array is not exactly 2,048 bytes.
#[inline]
pub fn array_to_wide(arr: &NumArrayU8) -> WideContainer {
    let data = arr.get_data();
    assert_eq!(data.len(), WIDE_BYTES, "NumArrayU8 must be exactly {} bytes", WIDE_BYTES);
    let slice: &[u8; WIDE_BYTES] = data.as_slice().try_into().unwrap();
    WideContainer::from_bytes(slice)
}

// =============================================================================
// COGRECORD8K ↔ NUMARRAYU8
// =============================================================================

/// Convert a full CogRecord8K (8,192 bytes) to a single NumArrayU8.
///
/// Layout: META (0..2048) | CAM (2048..4096) | INDEX (4096..6144) | EMBED (6144..8192)
pub fn record8k_to_array(record: &CogRecord8K) -> NumArrayU8 {
    NumArrayU8::new(record.as_bytes().to_vec())
}

/// Convert 4 individual NumArrayU8 containers into a CogRecord8K.
pub fn arrays_to_record8k(
    meta: &NumArrayU8,
    cam: &NumArrayU8,
    index: &NumArrayU8,
    embed: &NumArrayU8,
) -> CogRecord8K {
    CogRecord8K {
        meta: array_to_wide(meta),
        cam: array_to_wide(cam),
        index: array_to_wide(index),
        embed: array_to_wide(embed),
    }
}

// =============================================================================
// SIMD-ACCELERATED OPERATIONS
// =============================================================================

/// SIMD Hamming distance between two WideContainers.
///
/// Uses rustynum's fused XOR + POPCNT with 4× unrolling.
/// This is the hardware path for Container 1 (CAM) queries.
#[inline]
pub fn simd_hamming(a: &WideContainer, b: &WideContainer) -> u64 {
    let arr_a = wide_to_array(a);
    let arr_b = wide_to_array(b);
    arr_a.hamming_distance(&arr_b)
}

/// SIMD popcount on a WideContainer.
#[inline]
pub fn simd_popcount(c: &WideContainer) -> u64 {
    wide_to_array(c).popcount()
}

/// SIMD int8 dot product on two WideContainers (embedding containers).
///
/// Compiler emits VPDPBUSD (AVX-512 VNNI) with `-C target-cpu=native`.
/// For Container 3 (EMBED) queries.
#[inline]
pub fn simd_int8_dot(a: &WideContainer, b: &WideContainer) -> i64 {
    let arr_a = wide_to_array(a);
    let arr_b = wide_to_array(b);
    arr_a.dot_i8(&arr_b)
}

/// SIMD int8 cosine similarity on two WideContainers.
#[inline]
pub fn simd_int8_cosine(a: &WideContainer, b: &WideContainer) -> f64 {
    let arr_a = wide_to_array(a);
    let arr_b = wide_to_array(b);
    arr_a.cosine_i8(&arr_b)
}

/// SIMD XOR binding (WideContainer level).
#[inline]
pub fn simd_xor(a: &WideContainer, b: &WideContainer) -> WideContainer {
    let arr_a = wide_to_array(a);
    let arr_b = wide_to_array(b);
    let result = &arr_a ^ &arr_b;
    array_to_wide(&result)
}

/// SIMD majority-vote bundle on WideContainers.
///
/// Uses rustynum's hybrid strategy:
/// - Small n (≤16): compiler auto-vectorized to AVX-512 byte ops
/// - Large n (>16): ripple-carry with explicit u64x8 SIMD
/// - Very large: blackboard parallelization
pub fn simd_bundle(items: &[&WideContainer]) -> WideContainer {
    let arrays: Vec<NumArrayU8> = items.iter().map(|c| wide_to_array(c)).collect();
    let refs: Vec<&NumArrayU8> = arrays.iter().collect();
    let result = NumArrayU8::bundle(&refs);
    array_to_wide(&result)
}

/// SIMD circular bit-rotation (permute).
pub fn simd_permute(c: &WideContainer, k: usize) -> WideContainer {
    let arr = wide_to_array(c);
    let result = arr.permute(k);
    array_to_wide(&result)
}

/// SIMD edge encoding: src ⊕ permute(rel, 1) ⊕ permute(tgt, 2).
///
/// Uses rustynum SIMD for all three operations.
pub fn simd_make_edge(
    src: &WideContainer,
    rel: &WideContainer,
    tgt: &WideContainer,
) -> WideContainer {
    let s = wide_to_array(src);
    let r = wide_to_array(rel).permute(1);
    let t = wide_to_array(tgt).permute(2);
    let edge = &(&s ^ &r) ^ &t;
    array_to_wide(&edge)
}

/// SIMD edge target recovery.
pub fn simd_recover_target(
    edge: &WideContainer,
    src: &WideContainer,
    rel: &WideContainer,
) -> WideContainer {
    let e = wide_to_array(edge);
    let s = wide_to_array(src);
    let r = wide_to_array(rel).permute(1);
    let recovered = &(&e ^ &s) ^ &r;
    let total_bits = WIDE_BYTES * 8;
    let result = recovered.permute(total_bits - 2);
    array_to_wide(&result)
}

// =============================================================================
// BATCH OPERATIONS
// =============================================================================

/// Batch Hamming distance: query against N WideContainers.
///
/// Uses rustynum's batch parallelization for large N.
/// Returns N distances.
pub fn simd_hamming_batch(query: &WideContainer, corpus: &[WideContainer]) -> Vec<u64> {
    if corpus.is_empty() {
        return Vec::new();
    }

    // Pack query repeated N times and corpus into contiguous arrays
    let n = corpus.len();
    let q_bytes = query.as_bytes();
    let mut a_data = Vec::with_capacity(WIDE_BYTES * n);
    let mut b_data = Vec::with_capacity(WIDE_BYTES * n);

    for c in corpus {
        a_data.extend_from_slice(q_bytes);
        b_data.extend_from_slice(c.as_bytes());
    }

    let arr_a = NumArrayU8::new(a_data);
    let arr_b = NumArrayU8::new(b_data);
    arr_a.hamming_distance_batch(&arr_b, WIDE_BYTES, n)
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wide_to_array_roundtrip() {
        let c = WideContainer::random(42);
        let arr = wide_to_array(&c);
        let back = array_to_wide(&arr);
        assert_eq!(c, back);
    }

    #[test]
    fn test_simd_hamming_matches_native() {
        let a = WideContainer::random(1);
        let b = WideContainer::random(2);
        let native = a.hamming(&b);
        let simd = simd_hamming(&a, &b);
        assert_eq!(native as u64, simd);
    }

    #[test]
    fn test_simd_popcount_matches_native() {
        let c = WideContainer::random(42);
        let native = c.popcount();
        let simd = simd_popcount(&c);
        assert_eq!(native as u64, simd);
    }

    #[test]
    fn test_simd_xor_matches_native() {
        let a = WideContainer::random(1);
        let b = WideContainer::random(2);
        let native = a.xor(&b);
        let simd = simd_xor(&a, &b);
        assert_eq!(native, simd);
    }

    #[test]
    fn test_simd_int8_dot() {
        let mut a = WideContainer::zero();
        let mut b = WideContainer::zero();
        let vals: Vec<i8> = (0..100).map(|i| (i % 127) as i8).collect();
        a.store_int8(&vals);
        b.store_int8(&vals);
        let dot = simd_int8_dot(&a, &b);
        // Should match native dot
        let native_dot = a.int8_dot(&b, 2048); // rustynum operates on full 2048 bytes
        assert_eq!(dot, native_dot as i64);
    }

    #[test]
    fn test_simd_edge_roundtrip() {
        let src = WideContainer::random(10);
        let rel = WideContainer::random(20);
        let tgt = WideContainer::random(30);

        let edge = simd_make_edge(&src, &rel, &tgt);
        let recovered = simd_recover_target(&edge, &src, &rel);
        assert_eq!(recovered, tgt);
    }

    #[test]
    fn test_simd_bundle() {
        let a = WideContainer::random(1);
        let b = WideContainer::random(2);
        let c = WideContainer::random(3);
        let bundled = simd_bundle(&[&a, &b, &c]);
        // Bundle should be closer to inputs than to random
        let random = WideContainer::random(999);
        let d_a = bundled.hamming(&a);
        let d_rand = bundled.hamming(&random);
        assert!(d_a < d_rand, "bundle should be closer to input a");
    }

    #[test]
    fn test_simd_hamming_batch() {
        let query = WideContainer::random(42);
        let corpus: Vec<WideContainer> = (0..10).map(|i| WideContainer::random(i + 100)).collect();

        let batch_results = simd_hamming_batch(&query, &corpus);
        assert_eq!(batch_results.len(), 10);

        // Verify each matches individual computation
        for (i, c) in corpus.iter().enumerate() {
            let individual = simd_hamming(&query, c);
            assert_eq!(batch_results[i], individual, "mismatch at index {}", i);
        }
    }

    #[test]
    fn test_record8k_roundtrip() {
        let mut r = CogRecord8K::new();
        r.cam = WideContainer::random(1);
        r.meta = WideContainer::random(2);
        r.index = WideContainer::random(3);
        r.embed = WideContainer::random(4);

        let arr = record8k_to_array(&r);
        assert_eq!(arr.get_data().len(), 8192);

        // Reconstruct from individual arrays
        let meta_arr = wide_to_array(&r.meta);
        let cam_arr = wide_to_array(&r.cam);
        let idx_arr = wide_to_array(&r.index);
        let emb_arr = wide_to_array(&r.embed);
        let r2 = arrays_to_record8k(&meta_arr, &cam_arr, &idx_arr, &emb_arr);
        assert_eq!(r.cam, r2.cam);
        assert_eq!(r.meta, r2.meta);
        assert_eq!(r.index, r2.index);
        assert_eq!(r.embed, r2.embed);
    }
}
