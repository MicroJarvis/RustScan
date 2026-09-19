//! QuadricT Benchmark

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustmesh::{Point3, QuadricT};

fn bench_from_plane(c: &mut Criterion) {
    c.bench_function("quadric_from_plane", |b| {
        b.iter(|| {
            QuadricT::from_plane(
                black_box(1.0),
                black_box(0.0),
                black_box(0.0),
                black_box(0.0),
            )
        });
    });
}

fn bench_value(c: &mut Criterion) {
    let q = QuadricT::from_plane(0.0, 0.0, 1.0, 0.0);
    let v = Point3::new(1.0, 2.0, 3.0);

    c.bench_function("quadric_value", |b| b.iter(|| q.value(black_box(v))));
}

fn bench_add(c: &mut Criterion) {
    let q1 = QuadricT::from_plane(1.0, 0.0, 0.0, 0.0);
    let q2 = QuadricT::from_plane(0.0, 1.0, 0.0, 0.0);

    c.bench_function("quadric_add", |b| b.iter(|| black_box(q1) + black_box(q2)));
}

fn bench_optimize(c: &mut Criterion) {
    let q1 = QuadricT::from_plane(1.0, 0.0, 0.0, 0.0);
    let q2 = QuadricT::from_plane(0.0, 1.0, 0.0, 0.0);
    let q = q1 + q2;

    c.bench_function("quadric_optimize", |b| b.iter(|| q.optimize()));
}

fn bench_accumulate(c: &mut Criterion) {
    let faces: [QuadricT; 6] = [
        QuadricT::from_plane(1.0, 0.0, 0.0, 0.0),
        QuadricT::from_plane(0.0, 1.0, 0.0, 0.0),
        QuadricT::from_plane(0.0, 0.0, 1.0, 0.0),
        QuadricT::from_plane(0.5, 0.5, 0.0, 0.0),
        QuadricT::from_plane(0.0, 0.5, 0.5, 0.0),
        QuadricT::from_plane(0.5, 0.0, 0.5, 0.0),
    ];

    c.bench_function("quadric_accumulate_6", |b| {
        b.iter(|| {
            let mut q = QuadricT::zero();
            for face in &faces {
                q += *face;
            }
            q
        })
    });
}

criterion_group!(
    benches,
    bench_from_plane,
    bench_value,
    bench_add,
    bench_optimize,
    bench_accumulate
);
criterion_main!(benches);
