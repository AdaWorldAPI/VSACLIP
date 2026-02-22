//! Storage bridge — CogRecord8K ↔ Arrow/Lance/DataFusion via rustynum-arrow.
//!
//! Bridges VSACLIP's `CogRecord8K` (ladybug-contract) to rustynum-arrow's
//! persistence and query layers:
//!
//! ```text
//! CogRecord8K (ladybug)          rustynum CogRecord
//!     │                                │
//!     └──── record8k_to_cogrecord ─────┘
//!                    │
//!          cogrecords_to_record_batch
//!                    │
//!                    ▼
//!            Arrow RecordBatch
//!           ┌────────┴────────┐
//!           │                 │
//!      Lance Dataset    DataFusion Scan
//!      (on-disk)        (zero-copy cascade)
//! ```
//!
//! # Features
//!
//! - `lance-io`: Enable Lance dataset read/write
//! - `datafusion-scan`: Enable DataFusion cascade scan

use ladybug_contract::cogrecord8k::CogRecord8K;
use ladybug_contract::wide_container::WIDE_BYTES;
use rustynum_rs::{CogRecord, NumArrayU8};

use crate::simd_bridge::{wide_to_array, array_to_wide};

// Re-export rustynum-arrow types for convenience
#[cfg(feature = "lance-io")]
pub use rustynum_arrow::{
    cogrecord_schema, cogrecords_to_record_batch, record_batch_to_cogrecords,
    write_cogrecords, read_cogrecords, append_cogrecords,
};

#[cfg(all(feature = "datafusion-scan", not(feature = "lance-io")))]
pub use rustynum_arrow::{
    cogrecord_schema, cogrecords_to_record_batch, record_batch_to_cogrecords,
};

#[cfg(feature = "datafusion-scan")]
pub use rustynum_arrow::{
    arrow_to_flat_bytes, hamming_scan_column, cascade_scan_4ch,
};

// =============================================================================
// CogRecord8K ↔ rustynum CogRecord conversion
// =============================================================================

/// Convert a CogRecord8K (ladybug-contract) to a rustynum CogRecord.
///
/// Field mapping:
/// - `CogRecord8K.meta`  → `CogRecord.meta`
/// - `CogRecord8K.cam`   → `CogRecord.cam`
/// - `CogRecord8K.index` → `CogRecord.btree`
/// - `CogRecord8K.embed` → `CogRecord.embed`
pub fn record8k_to_cogrecord(record: &CogRecord8K) -> CogRecord {
    CogRecord::new(
        wide_to_array(&record.meta),
        wide_to_array(&record.cam),
        wide_to_array(&record.index),
        wide_to_array(&record.embed),
    )
}

/// Convert a rustynum CogRecord back to a CogRecord8K.
pub fn cogrecord_to_record8k(record: &CogRecord) -> CogRecord8K {
    CogRecord8K {
        meta: array_to_wide(&record.meta),
        cam: array_to_wide(&record.cam),
        index: array_to_wide(&record.btree),
        embed: array_to_wide(&record.embed),
    }
}

/// Batch-convert CogRecord8K records to rustynum CogRecords.
pub fn batch_to_cogrecords(records: &[CogRecord8K]) -> Vec<CogRecord> {
    records.iter().map(record8k_to_cogrecord).collect()
}

/// Batch-convert rustynum CogRecords back to CogRecord8K.
pub fn batch_to_record8k(records: &[CogRecord]) -> Vec<CogRecord8K> {
    records.iter().map(cogrecord_to_record8k).collect()
}

// =============================================================================
// Lance I/O helpers
// =============================================================================

/// Write CogRecord8K records to a Lance dataset.
///
/// Converts to rustynum CogRecords, then serializes to Arrow RecordBatch,
/// and writes to Lance. Each container is stored as FixedSizeBinary(2048).
#[cfg(feature = "lance-io")]
pub async fn write_records_8k(
    uri: &str,
    records: &[CogRecord8K],
) -> Result<(), Box<dyn std::error::Error>> {
    let cogrecords = batch_to_cogrecords(records);
    write_cogrecords(uri, &cogrecords).await?;
    Ok(())
}

/// Read CogRecord8K records from a Lance dataset.
#[cfg(feature = "lance-io")]
pub async fn read_records_8k(
    uri: &str,
) -> Result<Vec<CogRecord8K>, Box<dyn std::error::Error>> {
    let cogrecords = read_cogrecords(uri).await?;
    Ok(batch_to_record8k(&cogrecords))
}

/// Append CogRecord8K records to an existing Lance dataset.
#[cfg(feature = "lance-io")]
pub async fn append_records_8k(
    uri: &str,
    records: &[CogRecord8K],
) -> Result<(), Box<dyn std::error::Error>> {
    let cogrecords = batch_to_cogrecords(records);
    append_cogrecords(uri, &cogrecords).await?;
    Ok(())
}

// =============================================================================
// DataFusion cascade scan helpers
// =============================================================================

/// Run a 4-channel cascade scan on Arrow columns using a CogRecord8K query.
///
/// Zero-copy: operates directly on Arrow column buffers.
/// Uses compound early exit with per-container thresholds.
#[cfg(feature = "datafusion-scan")]
pub fn cascade_scan_8k(
    query: &CogRecord8K,
    meta_col: &arrow::array::FixedSizeBinaryArray,
    cam_col: &arrow::array::FixedSizeBinaryArray,
    btree_col: &arrow::array::FixedSizeBinaryArray,
    embed_col: &arrow::array::FixedSizeBinaryArray,
    thresholds: [u64; 4],
) -> Vec<(usize, [u64; 4])> {
    let query_cr = record8k_to_cogrecord(query);
    cascade_scan_4ch(&query_cr, meta_col, cam_col, btree_col, embed_col, thresholds)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ladybug_contract::wide_container::WideContainer;

    fn make_record8k(fill: u8) -> CogRecord8K {
        CogRecord8K {
            meta: WideContainer::from_bytes(&[fill; WIDE_BYTES]),
            cam: WideContainer::from_bytes(&[fill; WIDE_BYTES]),
            index: WideContainer::from_bytes(&[fill; WIDE_BYTES]),
            embed: WideContainer::from_bytes(&[fill; WIDE_BYTES]),
        }
    }

    #[test]
    fn test_record8k_cogrecord_roundtrip() {
        let r8k = CogRecord8K {
            meta: WideContainer::random(10),
            cam: WideContainer::random(20),
            index: WideContainer::random(30),
            embed: WideContainer::random(40),
        };

        let cr = record8k_to_cogrecord(&r8k);
        let back = cogrecord_to_record8k(&cr);

        assert_eq!(r8k.meta, back.meta);
        assert_eq!(r8k.cam, back.cam);
        assert_eq!(r8k.index, back.index);
        assert_eq!(r8k.embed, back.embed);
    }

    #[test]
    fn test_batch_conversion() {
        let records: Vec<CogRecord8K> = (0..10)
            .map(|i| make_record8k(i as u8))
            .collect();

        let cogrecords = batch_to_cogrecords(&records);
        assert_eq!(cogrecords.len(), 10);

        let back = batch_to_record8k(&cogrecords);
        assert_eq!(back.len(), 10);

        for (orig, round) in records.iter().zip(back.iter()) {
            assert_eq!(orig.meta, round.meta);
            assert_eq!(orig.cam, round.cam);
            assert_eq!(orig.index, round.index);
            assert_eq!(orig.embed, round.embed);
        }
    }

    #[test]
    fn test_cogrecord_container_sizes() {
        let r8k = make_record8k(0xAA);
        let cr = record8k_to_cogrecord(&r8k);

        assert_eq!(cr.meta.data_slice().len(), 2048);
        assert_eq!(cr.cam.data_slice().len(), 2048);
        assert_eq!(cr.btree.data_slice().len(), 2048);
        assert_eq!(cr.embed.data_slice().len(), 2048);
        assert_eq!(cr.to_bytes().len(), 8192);
    }
}
