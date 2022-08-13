//! # Real - to - complex fourier space
use crate::enums::{BaseKind, BaseType, TransformKind};
use crate::traits::{Differentiate, HasCoords, HasLength, HasType, ToOrtho, Transform};
use crate::types::{Real, Scalar};
use num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::f64::consts::PI;
use std::ops::Mul;
use std::sync::Arc;

/// # Container for fourier space (Real-to-complex)
#[derive(Clone)]
pub struct FourierR2c<A> {
    /// Number of coefficients in physical space
    n: usize,
    /// Number of coefficients in spectral space
    m: usize,
    /// Handles fourier transforms
    plan_fwd: Arc<dyn RealToComplex<A>>,
    plan_bwd: Arc<dyn ComplexToReal<A>>,
}

impl<A: Real> FourierR2c<A> {
    /// Returns a new Fourier Basis for real-to-complex transforms
    #[must_use]
    pub fn new(n: usize) -> Self {
        let mut planner = RealFftPlanner::<A>::new();
        Self {
            n,
            m: n / 2 + 1,
            plan_fwd: Arc::clone(&planner.plan_fft_forward(n)),
            plan_bwd: Arc::clone(&planner.plan_fft_inverse(n)),
        }
    }

    /// Return complex wavenumber vector for r2c transform (0, 1, 2, 3)
    ///
    /// # Panics
    /// Float conversion fails
    #[must_use]
    pub fn k(&self) -> Vec<Complex<A>> {
        (0..=self.n / 2)
            .map(|x| Complex::new(A::zero(), A::from(x).unwrap()))
            .collect::<Vec<Complex<A>>>()
    }
}

impl<A: Real> HasLength for FourierR2c<A> {
    fn len_phys(&self) -> usize {
        self.n
    }

    fn len_spec(&self) -> usize {
        self.m
    }

    fn len_ortho(&self) -> usize {
        self.n
    }
}

impl<A: Real> HasCoords<A> for FourierR2c<A> {
    fn coords(&self) -> Vec<A> {
        let step = 2. * PI / self.n as f64;
        (0..self.n)
            .map(|x| A::from(x as f64 * step).unwrap())
            .collect()
    }
}

impl<A: Real> HasType for FourierR2c<A> {
    fn base_kind(&self) -> BaseKind {
        BaseKind::FourierR2c
    }

    fn base_type(&self) -> BaseType {
        BaseType::Orthogonal
    }

    fn transform_kind(&self) -> TransformKind {
        TransformKind::C2c
    }
}

impl<A, T> Differentiate<T> for FourierR2c<A>
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

impl<A: Real> Transform for FourierR2c<A> {
    type Physical = A;

    type Spectral = Complex<A>;

    fn forward(&self, phys: &[Self::Physical], spec: &mut [Self::Spectral]) {
        assert!(phys.len() == self.len_phys());
        assert!(spec.len() == self.len_spec());
        // Copy - indata must be mutable for realfft
        let mut phys_mut = phys.to_vec();
        self.plan_fwd.process(&mut phys_mut, spec).unwrap();
    }

    fn backward(&self, spec: &[Self::Spectral], phys: &mut [Self::Physical]) {
        // Copy and normalize
        let norm = A::one() / A::from_usize(self.len_phys()).unwrap();
        let mut spec_mut = spec.to_vec();
        for a in spec_mut.iter_mut() {
            a.re = norm * a.re;
            a.im = norm * a.im;
        }
        self.plan_bwd.process(&mut spec_mut, phys).unwrap();
    }
}

impl<A, T> ToOrtho<T> for FourierR2c<A>
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
    use num_traits::Zero;

    #[test]
    fn test_fo_r2c_diff() {
        let n = 5;
        let fo = FourierR2c::<f64>::new(n);
        let mut dv = vec![Complex::<f64>::new(1., 0.); n];
        let exp: Vec<Complex<f64>> = fo.k();
        fo.diff_inplace(&mut dv, 1);
        approx_eq_complex(&dv, &exp);
    }

    #[test]
    fn test_fo_r2c_transform() {
        let n = 5;
        let fo = FourierR2c::<f64>::new(n);
        // cos(2*pi*x)
        let v = fo.coords().iter().map(|x| x.cos()).collect::<Vec<f64>>();
        let mut vhat = vec![Complex::<f64>::zero(); fo.len_spec()];
        fo.forward(&v, &mut vhat);
        let mut exp: Vec<Complex<f64>> = vec![Complex::<f64>::new(0., 0.); fo.len_spec()];
        exp[1] = Complex::new(1., 0.);
        approx_eq_complex(&vhat, &exp);
    }
}
