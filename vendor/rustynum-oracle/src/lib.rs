//! # rustynum-oracle
//!
//! Three-temperature holographic oracle with exhaustive capacity sweep.
//!
//! This crate implements two things:
//!
//! 1. **Capacity sweep** — test every combination of D, base, signing, axes, K,
//!    and bind depth to find the sweet spot for holographic storage.
//!
//! 2. **Three-temperature oracle** — one oracle per entity with hot/warm/cold tiers,
//!    overexposure-triggered flush, and coefficient-as-canonical-storage.
//!
//! The oracle is the thinking/storage tier. The 8KB CogRecordV3 in `rustynum-holo`
//! remains as the fast-rejection search tier.

pub mod linalg;
pub mod sweep;
pub mod oracle;
pub mod organic;
pub mod ghost_discovery;

pub use linalg::{
    cholesky_solve, condition_number,
    upsample_to_f32, downsample_to_base,
};

pub use sweep::{
    Base, DIMS, BASES, AXES, BUNDLE_SIZES, BIND_DEPTHS,
    generate_template, generate_templates,
    bind, bind_deep, bundle,
    measure_recovery, measure_recovery_multiaxis,
    measure_bell_coefficient,
    run_sweep,
    RecoveryResult, AxisResult, MultiAxisResult,
};

pub use oracle::{
    Oracle, Temperature, FlushAction,
    TemplateLibrary,
    MaterializedHolograph,
};

pub use organic::{
    XTransPattern, MultiResPattern,
    receptivity, organic_write, organic_write_f32, organic_read,
    OrganicWAL, WriteResult,
    PlasticityTracker, AbsorptionTracker,
    FlushResult, organic_flush,
    OrganicResult, measure_recovery_organic, run_organic_sweep,
    organic_results_to_csv,
    ORGANIC_CHANNELS, ORGANIC_PLASTICITY,
};
pub use organic::FlushAction as OrganicFlushAction;
