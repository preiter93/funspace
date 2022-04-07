//! # Complex - to - complex fourier space
use crate::enums::{BaseKind, TransformKind};
use crate::traits::{
    BaseElements, BaseFromOrtho, BaseGradient, BaseMatOpDiffmat, BaseMatOpLaplacian,
    BaseMatOpStencil, BaseSize, BaseTransform,
};
use crate::types::{FloatNum, ScalarNum};
use ndarray::{s, Array2};
use num_complex::Complex;
use num_traits::{One, Zero};
use rustfft::{Fft, FftPlanner};
use std::convert::TryInto;
use std::f64::consts::PI;
use std::ops::{Add, Div, Mul, Sub};
use std::sync::Arc;

/// # Container for fourier space (Complex-to-complex)
#[derive(Clone)]
pub struct FourierC2c<A> {
    /// Number of coefficients in physical space
    n: usize,
    /// Number of coefficients in spectral space
    m: usize,
    /// Handles fourier transforms
    plan_fwd: Arc<dyn Fft<A>>,
    plan_bwd: Arc<dyn Fft<A>>,
}

impl<A: FloatNum> FourierC2c<A> {
    /// Returns a new Fourier Basis for complex-to-complex transforms
    #[must_use]
    pub fn new(n: usize) -> Self {
        let mut planner = FftPlanner::<A>::new();
        Self {
            n,
            m: n,
            plan_fwd: Arc::clone(&planner.plan_fft_forward(n)),
            plan_bwd: Arc::clone(&planner.plan_fft_inverse(n)),
        }
    }

    /// Equispaced points on intervall [0, 2pi[
    ///
    /// ## Panics
    /// Float conversion fails
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn nodes(n: usize) -> Vec<A> {
        let n64 = n as f64;
        ndarray::Array1::range(0., 2. * PI, 2. * PI / n64)
            .mapv(|elem| A::from_f64(elem).unwrap())
            .to_vec()
    }

    /// Return complex wavenumber vector(0, 1, 2, -3, -2, -1)
    ///
    /// # Panics
    /// Float conversion fails
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn wavenumber(n: usize) -> Vec<Complex<A>> {
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

impl<A: FloatNum> BaseSize for FourierC2c<A> {
    /// Size in physical space
    fn len_phys(&self) -> usize {
        self.n
    }
    /// Size in spectral space
    fn len_spec(&self) -> usize {
        self.m
    }
    /// Size of orthogonal space
    fn len_orth(&self) -> usize {
        self.m
    }
}

impl<A: FloatNum> BaseElements for FourierC2c<A> {
    /// Real valued scalar type
    type RealNum = A;

    /// Return kind of base
    fn base_kind(&self) -> BaseKind {
        BaseKind::FourierC2c
    }

    /// Return kind of transform
    fn transform_kind(&self) -> TransformKind {
        TransformKind::C2c
    }

    /// Coordinates in physical space
    fn coords(&self) -> Vec<A> {
        Self::nodes(self.len_phys())
    }
}

impl<A: FloatNum> BaseMatOpDiffmat for FourierC2c<A> {
    /// Scalar type of matrix
    type NumType = Complex<A>;

    /// Explicit differential operator $ D $
    ///
    /// Matrix-based version of [`BaseGradient::gradient()`]
    ///
    /// # Panics
    /// Type conversion fails
    fn diffmat(&self, deriv: usize) -> Array2<Self::NumType> {
        assert!(deriv > 0);
        let mut mat = Array2::<Self::NumType>::zeros((self.len_spec(), self.len_spec()));
        let wavenum = Self::wavenumber(self.len_phys());
        for (l, k) in mat.diag_mut().iter_mut().zip(wavenum.iter()) {
            *l = k.powi(deriv.try_into().unwrap());
        }
        mat
    }

    /// Explicit inverse of differential operator $ D^* $
    ///
    /// Returns ``(D_pinv, I_pinv)``, where `D_pinv` is the pseudoinverse
    /// and ``I_pinv`` the corresponding pseudoidentity matrix, such
    /// that
    ///
    /// ```text
    /// D_pinv @ D = I_pinv
    /// ```
    ///
    /// Can be used as a preconditioner.
    fn diffmat_pinv(&self, deriv: usize) -> (Array2<Self::NumType>, Array2<Self::NumType>) {
        assert!(deriv > 0);
        let peye: Array2<Self::NumType> = Array2::<Self::NumType>::eye(self.m)
            .slice(s![1.., ..])
            .to_owned();
        let mut pinv = self.diffmat(deriv);
        for p in pinv.slice_mut(ndarray::s![1.., 1..]).diag_mut().iter_mut() {
            *p = Self::NumType::one() / *p;
        }
        (pinv, peye)
    }
}

impl<A: FloatNum> BaseMatOpStencil for FourierC2c<A> {
    /// Scalar type of matrix
    type NumType = A;

    /// Transformation stencil composite -> orthogonal space
    fn stencil(&self) -> Array2<Self::NumType> {
        Array2::<A>::eye(self.len_spec())
    }

    /// Inverse of transformation stencil
    fn stencil_inv(&self) -> Array2<Self::NumType> {
        Array2::<A>::eye(self.len_spec())
    }
}

impl<A: FloatNum> BaseMatOpLaplacian for FourierC2c<A> {
    /// Scalar type of laplacian matrix
    type NumType = A;

    /// Laplacian $ L $
    fn laplacian(&self) -> Array2<Self::NumType> {
        let wavenum = Self::wavenumber(self.len_phys());
        let mut lap = Array2::<A>::zeros((self.m, self.m));
        for (l, k) in lap.diag_mut().iter_mut().zip(wavenum.iter()) {
            *l = -k.im * k.im;
        }
        lap
    }

    /// Pseudoinverse matrix of Laplacian $ L^{-1} $
    ///
    /// Returns pseudoinverse and pseudoidentity,i.e
    /// ``(D_pinv, I_pinv)``
    ///
    /// ```text
    /// D_pinv @ D = I_pinv
    /// ``
    fn laplacian_pinv(&self) -> (Array2<Self::NumType>, Array2<Self::NumType>) {
        let mut d_pinv = self.laplacian();
        for p in d_pinv.slice_mut(s![1.., 1..]).diag_mut().iter_mut() {
            *p = A::one() / *p;
        }
        let i_pinv = Array2::<A>::eye(self.m);
        (d_pinv, i_pinv.slice(s![1.., ..]).to_owned())
    }
}

impl<A, T> BaseGradient<T> for FourierC2c<A>
where
    A: FloatNum,
    T: ScalarNum
        + Add<Complex<A>, Output = T>
        + Mul<Complex<A>, Output = T>
        + Div<Complex<A>, Output = T>
        + Sub<Complex<A>, Output = T>,
{
    /// Differentiate along slice
    /// ```
    /// use funspace::traits::BaseGradient;
    /// use funspace::fourier::FourierC2c;
    /// use funspace::utils::approx_eq_complex;
    /// use num_complex::Complex;
    /// use num_traits::Zero;
    /// let fo = FourierC2c::<f64>::new(5);
    /// let mut k = FourierC2c::<f64>::wavenumber(5);
    /// let expected: Vec<Complex<f64>> = k.iter().map(|x| x.powi(2)).collect();
    /// let mut outdata = vec![Complex::<f64>::zero(); 5];
    /// fo.gradient_slice(&k, &mut outdata, 1);
    /// approx_eq_complex(&outdata, &expected);
    /// ```
    ///
    /// # Panics
    /// When type conversion fails ( safe )
    fn gradient_slice(&self, indata: &[T], outdata: &mut [T], n_times: usize) {
        assert!(outdata.len() == indata.len());
        assert!(outdata.len() == self.len_spec());
        // Copy over
        for (y, x) in outdata.iter_mut().zip(indata.iter()) {
            *y = *x;
        }
        let n = self.len_spec();
        let n2 = (n - 1) / 2 + 1;
        for _ in 0..n_times {
            for (k, y) in outdata.iter_mut().take(n2).enumerate() {
                let ki: Complex<A> = Complex::<A>::new(A::zero(), A::from_usize(k).unwrap());
                *y = *y * ki;
            }
            for (k, y) in outdata.iter_mut().rev().take(n / 2).enumerate() {
                let ki: Complex<A> = Complex::<A>::new(A::zero(), A::from_usize(k).unwrap());
                *y = *y * ki;
            }
        }
    }
}

impl<A, T> BaseFromOrtho<T> for FourierC2c<A>
where
    A: FloatNum,
    T: ScalarNum,
{
    /// Composite space coefficients -> Orthogonal space coefficients
    fn to_ortho_slice(&self, indata: &[T], outdata: &mut [T]) {
        for (y, x) in outdata.iter_mut().zip(indata.iter()) {
            *y = *x;
        }
    }

    /// Orthogonal space coefficients -> Composite space coefficients
    fn from_ortho_slice(&self, indata: &[T], outdata: &mut [T]) {
        for (y, x) in outdata.iter_mut().zip(indata.iter()) {
            *y = *x;
        }
    }
}

impl<A: FloatNum> BaseTransform for FourierC2c<A> {
    type Physical = Complex<A>;

    type Spectral = Complex<A>;

    /// # Example
    /// Forward transform
    /// ```
    /// use funspace::traits::BaseTransform;
    /// use funspace::fourier::FourierC2c;
    /// use funspace::utils::approx_eq_complex;
    /// use num_complex::Complex;
    /// use num_traits::Zero;
    /// let mut fo = FourierC2c::<f64>::new(4);
    /// let indata = vec![
    ///     Complex::new(1., 1.),
    ///     Complex::new(2., 2.),
    ///     Complex::new(3., 3.),
    ///     Complex::new(4., 4.)
    /// ];
    /// let expected = vec![
    ///     Complex::new(10., 10.),
    ///     Complex::new(-4., 0.),
    ///     Complex::new(-2., -2.),
    ///     Complex::new(0., -4.)
    /// ];
    /// let mut outdata = vec![Complex::<f64>::zero(); 4];
    /// fo.forward_slice(&indata, &mut outdata);
    /// approx_eq_complex(&outdata, &expected);
    /// ```
    fn forward_slice(&self, indata: &[Self::Physical], outdata: &mut [Self::Spectral]) {
        // Check input
        assert!(indata.len() == self.len_phys());
        assert!(outdata.len() == self.len_spec());
        // Copy
        for (a, b) in outdata.iter_mut().zip(indata.iter()) {
            *a = *b;
        }
        self.plan_fwd.process(outdata);
    }

    /// # Example
    /// Backward transform
    /// ```
    /// use funspace::traits::BaseTransform;
    /// use funspace::fourier::FourierC2c;
    /// use funspace::utils::approx_eq_complex;
    /// use num_complex::Complex;
    /// use num_traits::Zero;
    /// let mut fo = FourierC2c::<f64>::new(4);
    /// let indata = vec![
    ///     Complex::new(10., 10.),
    ///     Complex::new(-4., 0.),
    ///     Complex::new(-2., -2.),
    ///     Complex::new(0., -4.)
    /// ];
    /// let expected = vec![
    ///     Complex::new(1., 1.),
    ///     Complex::new(2., 2.),
    ///     Complex::new(3., 3.),
    ///     Complex::new(4., 4.)
    /// ];
    /// let mut outdata = vec![Complex::<f64>::zero(); 4];
    /// fo.backward_slice(&indata, &mut outdata);
    /// approx_eq_complex(&outdata, &expected);
    /// ```
    fn backward_slice(&self, indata: &[Self::Spectral], outdata: &mut [Self::Physical]) {
        // Check input
        assert!(indata.len() == self.len_spec());
        assert!(outdata.len() == self.len_phys());
        // Copy and correct fft
        let cor = A::one() / A::from_usize(self.len_phys()).unwrap();
        for (a, b) in outdata.iter_mut().zip(indata.iter()) {
            a.re = cor * b.re;
            a.im = cor * b.im;
        }
        self.plan_bwd.process(outdata);
    }
}
