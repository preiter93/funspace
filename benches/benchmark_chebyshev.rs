use criterion::Criterion;
use criterion::{criterion_group, criterion_main};
use funspace::*;
use ndarray::Array1;

const SIZES: [usize; 4] = [128, 264, 512, 1024];

pub fn bench_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("Transform");
    group.significance_level(0.1).sample_size(10);
    for n in SIZES.iter() {
        let mut ch = cheb_dirichlet(*n);
        let mut arr = Array1::from_elem(*n, 1.);
        let name = format!("Size: {}", *n);
        group.bench_function(&name, |b| b.iter(|| ch.forward(&mut arr, 0)));
    }
    group.finish();
}

pub fn bench_to_ortho(c: &mut Criterion) {
    let mut group = c.benchmark_group("ToOrtho");
    group.significance_level(0.1).sample_size(10);
    for n in SIZES.iter() {
        let ch = cheb_dirichlet(*n);
        let mut arr = Array1::from_elem(ch.len_spec(), 1.);
        let name = format!("Size: {}", *n);
        group.bench_function(&name, |b| b.iter(|| ch.to_ortho(&mut arr, 0)));
    }
    group.finish();
}

pub fn bench_differentiate(c: &mut Criterion) {
    let mut group = c.benchmark_group("Differentiate");
    group.significance_level(0.1).sample_size(10);
    for n in SIZES.iter() {
        let ch = cheb_dirichlet(*n);
        let mut arr = Array1::from_elem(ch.len_spec(), 1.);
        let name = format!("Size: {}", *n);
        group.bench_function(&name, |b| b.iter(|| ch.differentiate(&mut arr, 2, 0)));
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_transform,
    bench_to_ortho,
    bench_differentiate
);
criterion_main!(benches);
