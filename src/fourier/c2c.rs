//! # Complex - to - complex fourier space
use crate::enums::BaseKind;
use crate::traits::{FunspaceElemental, FunspaceExtended, FunspaceSize};
use crate::types::{FloatNum, ScalarNum};
use ndarray::{s, Array2};
use num_complex::Complex;
use num_traits::Zero;
use rustfft::{Fft, FftPlanner};
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

impl<A: FloatNum> FunspaceExtended for FourierC2c<A> {
    type Real = A;

    type Spectral = Complex<A>;

    /// Return kind of base
    fn base_kind(&self) -> BaseKind {
        BaseKind::FourierC2c
    }

    /// Coordinates in physical space
    fn get_nodes(&self) -> Vec<A> {
        Self::nodes(self.len_phys())
    }

    /// Mass matrix
    fn mass(&self) -> Array2<A> {
        Array2::<A>::eye(self.len_spec())
    }

    /// Explicit differential operator
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    fn diffmat(&self, deriv: usize) -> Array2<Self::Spectral> {
        let mut diff_mat = Array2::<Complex<A>>::zeros((self.len_spec(), self.len_spec()));
        let wavenum = Self::wavenumber(self.len_phys());
        for (l, k) in diff_mat.diag_mut().iter_mut().zip(wavenum.iter()) {
            *l = Complex::new(A::zero(), k.im).powi(deriv as i32);
        }
        diff_mat
    }

    /// Laplacian $ L $
    fn laplace(&self) -> Array2<A> {
        let mut lap = Array2::<A>::zeros((self.len_spec(), self.len_spec()));
        let wavenum = Self::wavenumber(self.len_phys());
        for (l, k) in lap.diag_mut().iter_mut().zip(wavenum.iter()) {
            *l = -k.im * k.im;
        }
        lap
    }

    /// Pseudoinverse mtrix of Laplacian $ L^{-1} $
    fn laplace_inv(&self) -> Array2<A> {
        let mut pinv = self.laplace();
        for p in pinv.slice_mut(ndarray::s![1.., 1..]).diag_mut().iter_mut() {
            *p = A::one() / *p;
        }
        pinv
    }

    /// Pseudoidentity matrix of laplacian $ L^{-1} L $
    fn laplace_inv_eye(&self) -> Array2<A> {
        let eye = Array2::<A>::eye(self.m);
        eye.slice(s![1.., ..]).to_owned()
    }
}

impl<A: FloatNum> FunspaceSize for FourierC2c<A> {
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

impl<A: FloatNum> FunspaceElemental for FourierC2c<A> {
    type Physical = Complex<A>;

    type Spectral = Complex<A>;

    /// Differentiate along slice
    /// ```
    /// use funspace::traits::FunspaceElemental;
    /// use funspace::fourier::FourierC2c;
    /// use funspace::utils::approx_eq_complex;
    /// use num_complex::Complex;
    /// use num_traits::Zero;
    /// let fo = FourierC2c::<f64>::new(5);
    /// let mut k = FourierC2c::<f64>::wavenumber(5);
    /// let expected: Vec<Complex<f64>> = k.iter().map(|x| x.powi(2)).collect();
    /// let mut outdata = vec![Complex::<f64>::zero(); 5];
    /// fo.differentiate_slice(&k, &mut outdata, 1);
    /// approx_eq_complex(&outdata, &expected);
    /// ```
    ///
    /// # Panics
    /// When type conversion fails ( safe )
    fn differentiate_slice<T>(&self, indata: &[T], outdata: &mut [T], n_times: usize)
    where
        T: ScalarNum
            + Add<Self::Spectral, Output = T>
            + Mul<Self::Spectral, Output = T>
            + Div<Self::Spectral, Output = T>
            + Sub<Self::Spectral, Output = T>,
    {
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

    /// # Example
    /// Forward transform
    /// ```
    /// use funspace::traits::FunspaceElemental;
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
    /// use funspace::traits::FunspaceElemental;
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

    /// Composite space coefficients -> Orthogonal space coefficients
    fn to_ortho_slice<T: Copy>(&self, indata: &[T], outdata: &mut [T]) {
        // panic!("Function space Chebyshev is already orthogonal");
        for (y, x) in outdata.iter_mut().zip(indata.iter()) {
            *y = *x;
        }
    }

    /// Orthogonal space coefficients -> Composite space coefficients
    fn from_ortho_slice<T: Copy>(&self, indata: &[T], outdata: &mut [T]) {
        // panic!("Function space Chebyshev is already orthogonal");
        for (y, x) in outdata.iter_mut().zip(indata.iter()) {
            *y = *x;
        }
    }
}
