//! Criterion benchmarks for HDR POPCNT Sweep
//!
//! Run: cargo bench

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_placeholder(c: &mut Criterion) {
    c.bench_function("placeholder", |b| {
        b.iter(|| {
            // TODO: benchmark hdr_sweep vs full_sweep at various N
            42
        });
    });
}

criterion_group!(benches, bench_placeholder);
criterion_main!(benches);
