use criterion::Criterion;
use criterion::{criterion_group, criterion_main};
use funspace::*;
use ndarray::{Array1, Array2};
use num_complex::Complex;
const SIZES: [usize; 4] = [128, 264, 512, 1024];

pub fn bench_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("TransformFourierR2c");
    group.significance_level(0.1).sample_size(10);
    for n in SIZES.iter() {
        let mut fo = fourier_r2c::<f64>(*n);
        let mut arr = Array2::from_elem((*n, *n), 1.);
        let name = format!("Size: {} x {}", *n, *n);
        group.bench_function(&name, |b| {
            b.iter(|| {
                let _: Array2<Complex<f64>> = fo.forward(&mut arr, 0);
            })
        });
        let name = format!("Size: {} x {} (Par)", *n, *n);
        group.bench_function(&name, |b| {
            b.iter(|| {
                let _: Array2<Complex<f64>> = fo.forward_par(&mut arr, 0);
            })
        });
    }
    group.finish();
}

pub fn bench_differentiate(c: &mut Criterion) {
    let mut group = c.benchmark_group("DifferentiateFourierR2c");
    group.significance_level(0.1).sample_size(10);
    for n in SIZES.iter() {
        let fo = fourier_r2c::<f64>(*n);
        let mut arr = Array1::from_elem(fo.len_spec(), Complex::new(1., 1.));
        let name = format!("Size: {}", *n);
        group.bench_function(&name, |b| b.iter(|| fo.differentiate(&mut arr, 2, 0)));
    }
    group.finish();
}

criterion_group!(benches, bench_transform, bench_differentiate);
criterion_main!(benches);
