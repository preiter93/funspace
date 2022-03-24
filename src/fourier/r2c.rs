//! # Real - to - complex fourier space
use crate::enums::BaseKind;
use crate::traits::{FunspaceElemental, FunspaceExtended, FunspaceSize};
use crate::types::{FloatNum, ScalarNum};
use ndarray::{s, Array2};
use num_complex::Complex;
use num_traits::Zero;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::f64::consts::PI;
use std::ops::{Add, Div, Mul, Sub};
use std::sync::Arc;
//use crate::utils::check_array_axis;

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

impl<A: FloatNum> FourierR2c<A> {
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

    /// Return complex wavenumber vector for r2c transform (0, 1, 2, 3)
    ///
    /// # Panics
    /// Float conversion fails
    #[must_use]
    pub fn wavenumber(n: usize) -> Vec<Complex<A>> {
        (0..=n / 2)
            .map(|x| Complex::new(A::zero(), A::from(x).unwrap()))
            .collect::<Vec<Complex<A>>>()
    }
}

impl<A: FloatNum> FunspaceSize for FourierR2c<A> {
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

impl<A: FloatNum> FunspaceExtended for FourierR2c<A> {
    type Real = A;

    type Spectral = Complex<A>;

    /// Return kind of base
    fn base_kind(&self) -> BaseKind {
        BaseKind::FourierR2c
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

impl<A: FloatNum> FunspaceElemental for FourierR2c<A> {
    type Physical = A;

    type Spectral = Complex<A>;

    /// Differentiate along slice
    /// ```
    /// use funspace::traits::FunspaceElemental;
    /// use funspace::fourier::FourierR2c;
    /// use funspace::utils::approx_eq_complex;
    /// use num_complex::Complex;
    /// use num_traits::Zero;
    /// let fo = FourierR2c::<f64>::new(5);
    /// let mut k = FourierR2c::<f64>::wavenumber(5);
    /// let expected: Vec<Complex<f64>> = k.iter().map(|x| x.powi(2)).collect();
    /// let mut outdata = vec![Complex::<f64>::zero(); 3];
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
        for _ in 0..n_times {
            for (k, y) in outdata.iter_mut().enumerate() {
                let ki: Complex<A> = Complex::<A>::new(A::zero(), A::from_usize(k).unwrap());
                *y = *y * ki;
            }
        }
    }

    /// # Example
    /// Forward transform
    /// ```
    /// use funspace::traits::FunspaceElemental;
    /// use funspace::fourier::FourierR2c;
    /// use funspace::utils::approx_eq_complex;
    /// use num_complex::Complex;
    /// use num_traits::Zero;
    /// let mut fo = FourierR2c::<f64>::new(4);
    /// let indata = vec![1., 2., 3., 4.];
    /// let expected = vec![
    ///     Complex::new(10., 0.),
    ///     Complex::new(-2., 2.),
    ///     Complex::new(-2., 0.)
    /// ];
    /// let mut outdata = vec![Complex::<f64>::zero(); 3];
    /// fo.forward_slice(&indata, &mut outdata);
    /// approx_eq_complex(&outdata, &expected);
    /// ```
    fn forward_slice(&self, indata: &[Self::Physical], outdata: &mut [Self::Spectral]) {
        // Check input
        assert!(indata.len() == self.len_phys());
        assert!(outdata.len() == self.len_spec());
        // Copy - indata must be mutable for realfft
        let mut indata_mut = vec![Self::Physical::zero(); indata.len()];
        for (a, b) in indata_mut.iter_mut().zip(indata.iter()) {
            *a = *b;
        }
        self.plan_fwd.process(&mut indata_mut, outdata).unwrap();
    }

    /// # Example
    /// Backward transform
    /// ```
    /// use funspace::traits::FunspaceElemental;
    /// use funspace::fourier::FourierR2c;
    /// use funspace::utils::approx_eq;
    /// use num_complex::Complex;
    /// use num_traits::Zero;
    /// let mut fo = FourierR2c::<f64>::new(4);
    /// let indata = vec![
    ///     Complex::new(10., 0.),
    ///     Complex::new(-2., 2.),
    ///     Complex::new(-2., 0.)
    /// ];
    /// let expected = vec![1., 2., 3., 4.];
    /// let mut outdata = vec![0.; 4];
    /// fo.backward_slice(&indata, &mut outdata);
    /// approx_eq(&outdata, &expected);
    /// ```
    fn backward_slice(&self, indata: &[Self::Spectral], outdata: &mut [Self::Physical]) {
        // Check input
        assert!(indata.len() == self.len_spec());
        assert!(outdata.len() == self.len_phys());
        // Copy and correct fft
        let mut indata_mut = vec![Self::Spectral::zero(); indata.len()];
        let cor = A::one() / A::from_usize(self.len_phys()).unwrap();
        for (a, b) in indata_mut.iter_mut().zip(indata.iter()) {
            a.re = cor * b.re;
            a.im = cor * b.im;
        }
        // First element must be real
        indata_mut[0].im = A::zero();
        // If size is even, last element must be real too
        if self.n % 2 == 0 {
            indata_mut[self.m - 1].im = A::zero();
        }
        self.plan_bwd.process(&mut indata_mut, outdata).unwrap();
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
