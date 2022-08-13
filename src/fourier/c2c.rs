//! # Complex - to - complex fourier space
use crate::enums::{BaseKind, BaseType, TransformKind};
use crate::traits::{Differentiate, HasCoords, HasLength, HasType, ToOrtho, Transform};
use crate::types::{Real, Scalar};
use num_complex::Complex;
use num_traits::Zero;
use rustfft::{Fft, FftPlanner};
use std::f64::consts::PI;
use std::ops::Mul;
use std::sync::Arc;

/// # Container for fourier space (Complex-to-complex)
#[derive(Clone)]
pub struct FourierC2c<A> {
    /// Number of coefficients in physical space
    n: usize,
    /// Handles fourier transforms
    plan_fwd: Arc<dyn Fft<A>>,
    plan_bwd: Arc<dyn Fft<A>>,
}

impl<A: Real> FourierC2c<A> {
    /// Returns a new Fourier Basis for complex-to-complex transforms
    #[must_use]
    pub fn new(n: usize) -> Self {
        let mut planner = FftPlanner::<A>::new();
        Self {
            n,
            plan_fwd: Arc::clone(&planner.plan_fft_forward(n)),
            plan_bwd: Arc::clone(&planner.plan_fft_inverse(n)),
        }
    }

    /// Return complex wavenumber vector(0, 1, 2, -3, -2, -1)
    ///
    /// # Panics
    /// Float conversion fails
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn k(&self) -> Vec<Complex<A>> {
        let n = self.n;
        let n2 = (n - 1) / 2 + 1;
        let mut k: Vec<Complex<A>> = vec![Complex::<A>::zero(); n];
        for (i, ki) in k.iter_mut().take(n2).enumerate() {
            ki.im = A::from(i).unwrap();
        }
        for (i, ki) in k.iter_mut().rev().take(n / 2).enumerate() {
            ki.im = A::from(-1. * (i + 1) as f64).unwrap();
        }
        k
    }
}

impl<A: Real> HasLength for FourierC2c<A> {
    fn len_phys(&self) -> usize {
        self.n
    }

    fn len_spec(&self) -> usize {
        self.n
    }

    fn len_ortho(&self) -> usize {
        self.n
    }
}

impl<A: Real> HasCoords<A> for FourierC2c<A> {
    fn coords(&self) -> Vec<A> {
        let step = 2. * PI / self.n as f64;
        (0..self.n)
            .map(|x| A::from(x as f64 * step).unwrap())
            .collect()
    }
}

impl<A: Real> HasType for FourierC2c<A> {
    fn base_kind(&self) -> BaseKind {
        BaseKind::FourierC2c
    }

    fn base_type(&self) -> BaseType {
        BaseType::Orthogonal
    }

    fn transform_kind(&self) -> TransformKind {
        TransformKind::R2c
    }
}

impl<A, T> Differentiate<T> for FourierC2c<A>
where
    A: Real,
    T: Scalar + Mul<Complex<A>, Output = T>,
{
    fn diff(&self, v: &[T], dv: &mut [T], order: usize) {
        dv.clone_from_slice(v);
        self.diff_inplace(dv, order);
    }

    fn diff_inplace(&self, v: &mut [T], order: usize) {
        for _ in 0..order {
            for (vi, ki) in v.iter_mut().zip(self.k().iter()) {
                *vi = *vi * *ki;
            }
        }
    }
}

impl<A: Real> Transform for FourierC2c<A> {
    type Physical = Complex<A>;

    type Spectral = Complex<A>;

    fn forward(&self, phys: &[Self::Physical], spec: &mut [Self::Spectral]) {
        spec.clone_from_slice(phys);
        self.plan_fwd.process(spec);
    }

    fn backward(&self, spec: &[Self::Spectral], phys: &mut [Self::Physical]) {
        // Copy and normalize
        let norm = A::one() / A::from_usize(self.len_phys()).unwrap();
        for (a, b) in phys.iter_mut().zip(spec.iter()) {
            a.re = norm * b.re;
            a.im = norm * b.im;
        }
        self.plan_bwd.process(phys);
    }
}

impl<A, T> ToOrtho<T> for FourierC2c<A>
where
    A: Real,
    T: Scalar,
{
    fn to_ortho(&self, comp: &[T], ortho: &mut [T]) {
        ortho.clone_from_slice(comp);
    }

    fn from_ortho(&self, ortho: &[T], comp: &mut [T]) {
        comp.clone_from_slice(ortho);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::utils::approx_eq_complex;

    #[test]
    fn test_fo_c2c_diff() {
        let n = 5;
        let fo = FourierC2c::<f64>::new(n);
        let mut dv = vec![Complex::<f64>::new(1., 0.); n];
        let exp: Vec<Complex<f64>> = fo.k();
        fo.diff_inplace(&mut dv, 1);
        approx_eq_complex(&dv, &exp);
    }

    #[test]
    fn test_fo_c2c_transform() {
        let n = 5;
        let fo = FourierC2c::<f64>::new(n);
        // cos(2*pi*x)
        let v = fo
            .coords()
            .iter()
            .map(|x| Complex::new(x.cos(), 0.))
            .collect::<Vec<Complex<f64>>>();
        let mut vhat = vec![Complex::<f64>::zero(); fo.len_spec()];
        fo.forward(&v, &mut vhat);
        let mut exp: Vec<Complex<f64>> = vec![Complex::<f64>::new(0., 0.); fo.len_spec()];
        exp[1] = Complex::new(1., 0.);
        approx_eq_complex(&vhat, &exp);
    }
}
