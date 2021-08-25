use criterion::Criterion;
use criterion::{criterion_group, criterion_main};
use funspace::*;
use ndarray::Array2;

const SIZES: [usize; 3] = [128, 264, 512];

pub fn bench_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("TransformChebDirichlet");
    group.significance_level(0.1).sample_size(10);
    for n in SIZES.iter() {
        let mut ch = cheb_dirichlet::<f64>(*n);
        let mut arr = Array2::from_elem((*n, *n), 1.);
        let name = format!("Size: {} x {}", *n, *n);
        group.bench_function(&name, |b| {
            b.iter(|| {
                let _: Array2<f64> = ch.forward(&mut arr, 0);
            })
        });
        let name = format!("Size: {} x {} (Par)", *n, *n);
        group.bench_function(&name, |b| {
            b.iter(|| {
                let _: Array2<f64> = ch.forward_par(&mut arr, 0);
            })
        });
    }
    group.finish();
}

pub fn bench_to_ortho(c: &mut Criterion) {
    let mut group = c.benchmark_group("ToOrthoChebDirichlet");
    group.significance_level(0.1).sample_size(10);
    for n in SIZES.iter() {
        let ch = cheb_dirichlet(*n);
        let ns = ch.len_spec();
        let mut arr = Array2::from_elem((ns, ns), 1.);
        let name = format!("Size: {} x {}", *n, *n);
        group.bench_function(&name, |b| b.iter(|| ch.to_ortho(&mut arr, 0)));
    }
    group.finish();
}

pub fn bench_from_ortho(c: &mut Criterion) {
    let mut group = c.benchmark_group("FromOrthoChebDirichlet");
    group.significance_level(0.1).sample_size(10);
    for n in SIZES.iter() {
        let ch = cheb_dirichlet(*n);
        let ns = ch.len_phys();
        let mut arr = Array2::from_elem((ns, ns), 1.);
        let name = format!("Size: {} x {}", *n, *n);
        group.bench_function(&name, |b| b.iter(|| ch.from_ortho(&mut arr, 0)));
    }
    group.finish();
}
pub fn bench_differentiate(c: &mut Criterion) {
    let mut group = c.benchmark_group("DifferentiateChebDirichlet");
    group.significance_level(0.1).sample_size(10);
    for n in SIZES.iter() {
        let ch = cheb_dirichlet(*n);
        let ns = ch.len_spec();
        let mut arr = Array2::from_elem((ns, ns), 1.);
        let name = format!("Size: {} x {}", *n, *n);
        group.bench_function(&name, |b| b.iter(|| ch.differentiate(&mut arr, 2, 0)));
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_transform,
    bench_to_ortho,
    bench_from_ortho,
    bench_differentiate
);
criterion_main!(benches);
