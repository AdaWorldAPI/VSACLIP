use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ladybug_contract::container::Container;
use vsaclip::sweep::{hdr_sweep, full_sweep, SweepConfig};

fn bench_sweep_10k(c: &mut Criterion) {
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

fn bench_sweep_scaling(c: &mut Criterion) {
    let query = Container::random(42);

    let mut group = c.benchmark_group("hdr_sweep_scaling");
    group.sample_size(10);

    for &n in &[1_000u64, 10_000, 100_000] {
        let corpus: Vec<Container> = (0..n)
            .map(|s| Container::random(s + 1000))
            .collect();

        let config = SweepConfig::default();

        group.bench_with_input(BenchmarkId::new("hdr", n), &corpus, |b, corpus| {
            b.iter(|| hdr_sweep(black_box(&query), black_box(corpus), black_box(&config)))
        });

        group.bench_with_input(BenchmarkId::new("full", n), &corpus, |b, corpus| {
            b.iter(|| full_sweep(black_box(&query), black_box(corpus), black_box(2500)))
        });
    }

    group.finish();
}

fn bench_simhash(c: &mut Criterion) {
    let embedding: Vec<f32> = (0..512).map(|i| (i as f32 * 0.1).sin()).collect();

    c.bench_function("simhash_512d", |b| {
        b.iter(|| vsaclip::ingest::simhash(black_box(&embedding), black_box(42)))
    });
}

fn bench_bundle(c: &mut Criterion) {
    let containers: Vec<Container> = (0..20u64)
        .map(|s| Container::random(s))
        .collect();
    let refs: Vec<&Container> = containers.iter().collect();

    c.bench_function("bundle_20", |b| {
        b.iter(|| Container::bundle(black_box(&refs)))
    });
}

criterion_group!(benches, bench_sweep_10k, bench_sweep_scaling, bench_simhash, bench_bundle);
criterion_main!(benches);
