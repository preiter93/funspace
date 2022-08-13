use criterion::Criterion;
use criterion::{criterion_group, criterion_main};
use funspace::chebyshev;
use funspace::space::{traits::HasShape, traits::SpaceTransform, Space2};
use ndarray::Array2;

const SIZES: [usize; 4] = [129, 265, 513, 1025];

pub fn bench_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("SpaceChebyshev");
    group.significance_level(0.1).sample_size(10);
    for n in SIZES.iter() {
        let ch = chebyshev::<f64>(*n);
        let space = Space2::new(&ch, &ch);
        let v = Array2::from_elem(space.shape_phys(), 1.);
        let mut vhat = Array2::from_elem(space.shape_spec(), 0.);
        let name = format!("Size: {} x {}", *n, *n);
        group.bench_function(&name, |b| {
            b.iter(|| {
                space.forward(&v, &mut vhat);
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_transform);
criterion_main!(benches);
