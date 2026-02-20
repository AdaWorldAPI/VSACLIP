use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ladybug_contract::container::Container;
use vsaclip::sweep::{hdr_sweep, full_sweep, SweepConfig};

fn bench_sweep(c: &mut Criterion) {
    let query = Container::random(42);
    let corpus: Vec<Container> = (0..10_000u64)
        .map(|s| Container::random(s + 1000))
        .collect();

    let config = SweepConfig::default();

    c.bench_function("hdr_sweep_10k", |b| {
        b.iter(|| hdr_sweep(black_box(&query), black_box(&corpus), black_box(&config)))
    });

    c.bench_function("full_sweep_10k", |b| {
        b.iter(|| full_sweep(black_box(&query), black_box(&corpus), black_box(2500)))
    });
}

criterion_group!(benches, bench_sweep);
criterion_main!(benches);
