use criterion::Criterion;
use criterion::{criterion_group, criterion_main};
use funspace::*;
use ndarray::Array2;

const SIZES: [usize; 4] = [128, 264, 512, 1024];

pub fn bench_sb_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("SpaceBaseTransformChebDirichlet");
    group.significance_level(0.1).sample_size(10);
    for n in SIZES.iter() {
        let cd = cheb_dirichlet::<f64>(*n);
        let mut space = Space2::new(&cd.clone(), &cd.clone());
        let mut arr = Array2::from_elem((*n, *n), 1.);
        let name = format!("Size: {} x {} (Serial)", *n, *n);
        group.bench_function(&name, |b| {
            b.iter(|| {
                let _: Array2<f64> = space.forward(&mut arr);
            })
        });
        let name = format!("Size: {} x {} (Parallel)", *n, *n);
        group.bench_function(&name, |b| {
            b.iter(|| {
                let _: Array2<f64> = space.forward_par(&mut arr);
            })
        });
    }
    group.finish();
}

pub fn bench_sb_to_ortho(c: &mut Criterion) {
    let mut group = c.benchmark_group("SpaceBaseToOrthoChebDirichlet");
    group.significance_level(0.1).sample_size(10);
    for n in SIZES.iter() {
        let cd = cheb_dirichlet::<f64>(*n);
        let space = Space2::new(&cd.clone(), &cd.clone());
        let ns = cd.len_spec();
        let mut arr = Array2::from_elem((ns, ns), 1.);
        let name = format!("Size: {} x {} (Serial)", *n, *n);
        group.bench_function(&name, |b| b.iter(|| space.to_ortho(&mut arr)));
        let name = format!("Size: {} x {} (Parallel)", *n, *n);
        group.bench_function(&name, |b| b.iter(|| space.to_ortho_par(&mut arr)));
    }
    group.finish();
}

pub fn bench_sb_from_ortho(c: &mut Criterion) {
    let mut group = c.benchmark_group("SpaceBaseFromOrthoChebDirichlet");
    group.significance_level(0.1).sample_size(10);
    for n in SIZES.iter() {
        let cd = cheb_dirichlet::<f64>(*n);
        let space = Space2::new(&cd.clone(), &cd.clone());
        let ns = cd.len_phys();
        let mut arr = Array2::from_elem((ns, ns), 1.);
        let name = format!("Size: {} x {} (Serial)", *n, *n);
        group.bench_function(&name, |b| b.iter(|| space.from_ortho(&mut arr)));
        let name = format!("Size: {} x {} (Parallel)", *n, *n);
        group.bench_function(&name, |b| b.iter(|| space.from_ortho_par(&mut arr)));
    }
    group.finish();
}

pub fn bench_sb_differentiate(c: &mut Criterion) {
    let mut group = c.benchmark_group("SpaceBaseDifferentiateChebDirichlet");
    group.significance_level(0.1).sample_size(10);
    for n in SIZES.iter() {
        let cd = cheb_dirichlet::<f64>(*n);
        let space = Space2::new(&cd.clone(), &cd.clone());
        let ns = cd.len_spec();
        let mut arr = Array2::from_elem((ns, ns), 1.);
        let name = format!("Size: {} x {} (Serial)", *n, *n);
        group.bench_function(&name, |b| b.iter(|| space.gradient(&mut arr, [2, 0], None)));
        let name = format!("Size: {} x {} (Parallel)", *n, *n);
        group.bench_function(&name, |b| {
            b.iter(|| space.gradient_par(&mut arr, [2, 0], None))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_sb_transform,
    bench_sb_to_ortho,
    bench_sb_from_ortho,
    bench_sb_differentiate
);
criterion_main!(benches);
