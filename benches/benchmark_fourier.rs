use criterion::Criterion;
use criterion::{criterion_group, criterion_main};
use funspace::space::{traits::HasShape, traits::SpaceTransform, Space2};
use funspace::{fourier_c2c, fourier_r2c};
use ndarray::Array2;
use num_complex::Complex;
use num_traits::Zero;

const SIZES: [usize; 4] = [128, 264, 512, 1024];

pub fn bench_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("SpaceFourier");
    group.significance_level(0.1).sample_size(10);
    for n in SIZES.iter() {
        let r2c = fourier_r2c::<f64>(*n);
        let c2c = fourier_c2c::<f64>(*n);
        let space = Space2::new(&c2c, &r2c);
        let v = Array2::from_elem(space.shape_phys(), 1.);
        let mut vhat = Array2::from_elem(space.shape_spec(), Complex::zero());
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
