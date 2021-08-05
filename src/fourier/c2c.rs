//! # Complex - to - complex fourier space
//!
//! # Example
//! Initialize new fourier basis
//! ```
//! use funspace::fourier::Fourier;
//! let fo = Fourier::<f64>::new(4);
//! ```
//#![allow(unused_imports)]
use super::Complex;
use super::FloatNum;
use crate::Differentiate;
use crate::FromOrtho;
use crate::LaplacianInverse;
use crate::Mass;
use crate::Scalar;
use crate::Size;
use crate::Transform;
use crate::TransformPar;
use core::f64::consts::PI;
use ndarray::prelude::*;
use ndrustfft::FftHandler;

/// # Container for fourier space (Complex-to-complex)
#[derive(Clone)]
pub struct Fourier<A> {
    /// Number of coefficients in physical space
    pub n: usize,
    /// Number of coefficients in spectral space
    pub m: usize,
    /// Grid coordinates of fourier nodes
    pub x: Array1<A>,
    /// Complex wavenumber vector
    pub k: Array1<Complex<A>>,
    /// Handles discrete cosine transform
    pub fft_handler: FftHandler<A>,
}

impl<A: FloatNum> Fourier<A> {
    /// Returns a new Fourier Basis
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            n,
            m: n,
            x: Self::nodes(n),
            k: Self::wavenumber(n),
            fft_handler: FftHandler::new(n),
        }
    }

    /// Return equispaced points on intervall [0, 2pi[
    ///
    /// ## Panics
    /// **Panics** when conversion f64 to generic A fails
    #[allow(clippy::must_use_candidate)]
    pub fn nodes(n: usize) -> Array1<A> {
        let n64 = n as f64;
        Array1::range(0., 2. * PI, 2. * PI / n64).mapv(|elem| A::from_f64(elem).unwrap())
    }

    /// Return complex wavenumber vector(0, 1, 2, 3, -2, -1)
    #[allow(clippy::missing_panics_doc)]
    fn wavenumber(n: usize) -> Array1<Complex<A>> {
        let mut k: Array1<A> = Array1::zeros(n);
        let n2 = (n - 1) / 2 + 1;
        for (i, ki) in Array1::range(0., n2 as f64, 1.)
            .iter()
            .zip(k.slice_mut(s![..n2]))
        {
            *ki = A::from_f64(*i as f64).unwrap();
        }
        for (i, ki) in Array1::range(-1. * (n2 / 2 + 1) as f64, 0., 1.)
            .iter()
            .zip(k.slice_mut(s![n2..]))
        {
            *ki = A::from_f64(*i as f64).unwrap();
        }
        k.mapv(|x| Complex::new(A::zero(), x))
    }

    /// Differentiate 1d Array *n_times*
    /// # Example
    /// Differentiate along lane
    /// ```
    /// use funspace::fourier::Fourier;
    /// use funspace::utils::approx_eq_complex;
    /// use ndarray::prelude::*;
    /// let fo = Fourier::<f64>::new(5);
    /// let mut k = fo.k.clone();
    /// let expected = k.mapv(|x| x.powf(2.));
    /// fo.differentiate_lane(&mut k, 1);
    /// approx_eq_complex(&k, &expected);
    /// ```
    ///
    /// # Panics
    /// When type conversion fails ( safe )
    pub fn differentiate_lane<S, T2>(&self, data: &mut ArrayBase<S, Ix1>, n_times: usize)
    where
        S: ndarray::Data<Elem = T2> + ndarray::DataMut,
        T2: Scalar + From<Complex<A>>,
    {
        let deriv = A::from_f64(n_times as f64).unwrap();
        let kpow = self.k.mapv(|x| x.powf(deriv));
        for (d, k) in data.iter_mut().zip(kpow.iter()) {
            *d = *d * T2::from(*k);
        }
    }
}

impl<A: FloatNum> Mass<A> for Fourier<A> {
    /// Return mass matrix (= eye)
    fn mass(&self) -> Array2<A> {
        Array2::<A>::eye(self.n)
    }
    /// Coordinates in physical space
    fn coords(&self) -> &Array1<A> {
        &self.x
    }
}

impl<A: FloatNum> Size for Fourier<A> {
    /// Size in physical space
    fn len_phys(&self) -> usize {
        self.n
    }
    /// Size in spectral space
    fn len_spec(&self) -> usize {
        self.m
    }
}

/// Perform differentiation in spectral space
impl<A: FloatNum> Differentiate<Complex<A>> for Fourier<A> {
    fn differentiate<S, D>(
        &self,
        data: &ArrayBase<S, D>,
        n_times: usize,
        axis: usize,
    ) -> Array<Complex<A>, D>
    where
        S: ndarray::Data<Elem = Complex<A>>,
        D: Dimension,
    {
        let mut output = data.to_owned();
        self.differentiate_inplace(&mut output, n_times, axis);
        output
    }

    fn differentiate_inplace<S, D>(&self, data: &mut ArrayBase<S, D>, n_times: usize, axis: usize)
    where
        S: ndarray::Data<Elem = Complex<A>> + ndarray::DataMut,
        D: Dimension,
    {
        use crate::utils::check_array_axis;
        check_array_axis(data, self.m, axis, Some("fourier differentiate"));
        ndarray::Zip::from(data.lanes_mut(Axis(axis))).for_each(|mut lane| {
            self.differentiate_lane(&mut lane, n_times);
        });
    }
}

impl<A: FloatNum> Transform<Complex<A>, Complex<A>> for Fourier<A> {
    type Physical = Complex<A>;
    type Spectral = Complex<A>;

    /// # Example
    /// Forward transform along first axis
    /// ```
    /// use funspace::fourier::Fourier;
    /// use funspace::Transform;
    /// use funspace::utils::approx_eq_complex;
    /// use num_complex::Complex;
    /// use ndarray::prelude::*;
    /// let mut fo = Fourier::new(4);
    /// let mut input = array![
    ///     Complex::new(1., 1.),
    ///     Complex::new(2., 2.),
    ///     Complex::new(3., 3.),
    ///     Complex::new(4., 4.)
    /// ];
    /// let expected = array![
    ///     Complex::new(10., 10.),
    ///     Complex::new(-4., 0.),
    ///     Complex::new(-2., -2.),
    ///     Complex::new(0., -4.)
    /// ];
    /// let output = fo.forward(&mut input, 0);
    /// approx_eq_complex(&output, &expected);
    /// ```
    fn forward<S, D>(
        &mut self,
        input: &mut ArrayBase<S, D>,
        axis: usize,
    ) -> Array<Self::Spectral, D>
    where
        S: ndarray::Data<Elem = Self::Physical>,
        D: Dimension + ndarray::RemoveAxis,
    {
        use crate::utils::array_resized_axis;
        let mut output = array_resized_axis(input, self.m, axis);
        self.forward_inplace(input, &mut output, axis);
        output
    }

    /// See [`Fourier::forward`]
    fn forward_inplace<S1, S2, D>(
        &mut self,
        input: &mut ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = Self::Physical>,
        S2: ndarray::Data<Elem = Self::Spectral> + ndarray::DataMut,
        D: Dimension + ndarray::RemoveAxis,
    {
        use crate::utils::check_array_axis;
        use ndrustfft::ndfft;
        check_array_axis(input, self.n, axis, Some("fourier forward"));
        check_array_axis(output, self.m, axis, Some("fourier forward"));
        ndfft(input, output, &mut self.fft_handler, axis);
    }

    /// # Example
    /// Backward transform along first axis
    /// ```
    /// use funspace::fourier::Fourier;
    /// use funspace::Transform;
    /// use funspace::utils::approx_eq_complex;
    /// use num_complex::Complex;
    /// use ndarray::prelude::*;
    /// let mut fo = Fourier::new(4);
    /// let mut input = array![
    ///     Complex::new(10., 10.),
    ///     Complex::new(-4., 0.),
    ///     Complex::new(-2., -2.),
    ///     Complex::new(0., -4.)
    /// ];
    /// let expected = array![
    ///     Complex::new(1., 1.),
    ///     Complex::new(2., 2.),
    ///     Complex::new(3., 3.),
    ///     Complex::new(4., 4.)
    /// ];
    /// let output = fo.backward(&mut input, 0);
    /// approx_eq_complex(&output, &expected);
    /// ```
    fn backward<S, D>(
        &mut self,
        input: &mut ArrayBase<S, D>,
        axis: usize,
    ) -> Array<Self::Physical, D>
    where
        S: ndarray::Data<Elem = Self::Spectral>,
        D: Dimension + ndarray::RemoveAxis,
    {
        use crate::utils::array_resized_axis;
        let mut output = array_resized_axis(input, self.m, axis);
        self.backward_inplace(input, &mut output, axis);
        output
    }

    /// See [`Fourier::backward`]
    ///
    /// # Panics
    /// Panics when input type cannot be cast from f64.
    #[allow(clippy::used_underscore_binding)]
    fn backward_inplace<S1, S2, D>(
        &mut self,
        input: &mut ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = Self::Spectral>,
        S2: ndarray::Data<Elem = Self::Physical> + ndarray::DataMut,
        D: Dimension + ndarray::RemoveAxis,
    {
        use crate::utils::check_array_axis;
        use ndrustfft::ndifft;
        check_array_axis(input, self.m, axis, Some("fourier backward"));
        check_array_axis(output, self.n, axis, Some("fourier backward"));
        ndifft(input, output, &mut self.fft_handler, axis);
    }
}

impl<A: FloatNum> TransformPar<Complex<A>, Complex<A>> for Fourier<A> {
    type Physical = Complex<A>;
    type Spectral = Complex<A>;

    /// # Example
    /// Forward transform along first axis
    /// ```
    /// use funspace::fourier::Fourier;
    /// use funspace::TransformPar;
    /// use funspace::utils::approx_eq_complex;
    /// use num_complex::Complex;
    /// use ndarray::prelude::*;
    /// let mut fo = Fourier::new(4);
    /// let mut input = array![
    ///     Complex::new(1., 1.),
    ///     Complex::new(2., 2.),
    ///     Complex::new(3., 3.),
    ///     Complex::new(4., 4.)
    /// ];
    /// let expected = array![
    ///     Complex::new(10., 10.),
    ///     Complex::new(-4., 0.),
    ///     Complex::new(-2., -2.),
    ///     Complex::new(0., -4.)
    /// ];
    /// let output = fo.forward_par(&mut input, 0);
    /// approx_eq_complex(&output, &expected);
    /// ```
    fn forward_par<S, D>(
        &mut self,
        input: &mut ArrayBase<S, D>,
        axis: usize,
    ) -> Array<Self::Spectral, D>
    where
        S: ndarray::Data<Elem = Self::Physical>,
        D: Dimension + ndarray::RemoveAxis,
    {
        use crate::utils::array_resized_axis;
        let mut output = array_resized_axis(input, self.m, axis);
        self.forward_inplace(input, &mut output, axis);
        output
    }

    /// See [`Fourier::forward_par`]
    fn forward_inplace_par<S1, S2, D>(
        &mut self,
        input: &mut ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = Self::Physical>,
        S2: ndarray::Data<Elem = Self::Spectral> + ndarray::DataMut,
        D: Dimension + ndarray::RemoveAxis,
    {
        use crate::utils::check_array_axis;
        use ndrustfft::ndfft_par;
        check_array_axis(input, self.n, axis, Some("fourier forward"));
        check_array_axis(output, self.m, axis, Some("fourier forward"));
        ndfft_par(input, output, &mut self.fft_handler, axis);
    }

    /// # Example
    /// Backward transform along first axis
    /// ```
    /// use funspace::fourier::Fourier;
    /// use funspace::TransformPar;
    /// use funspace::utils::approx_eq_complex;
    /// use num_complex::Complex;
    /// use ndarray::prelude::*;
    /// let mut fo = Fourier::new(4);
    /// let mut input = array![
    ///     Complex::new(10., 10.),
    ///     Complex::new(-4., 0.),
    ///     Complex::new(-2., -2.),
    ///     Complex::new(0., -4.)
    /// ];
    /// let expected = array![
    ///     Complex::new(1., 1.),
    ///     Complex::new(2., 2.),
    ///     Complex::new(3., 3.),
    ///     Complex::new(4., 4.)
    /// ];
    /// let output = fo.backward_par(&mut input, 0);
    /// approx_eq_complex(&output, &expected);
    /// ```
    fn backward_par<S, D>(
        &mut self,
        input: &mut ArrayBase<S, D>,
        axis: usize,
    ) -> Array<Self::Physical, D>
    where
        S: ndarray::Data<Elem = Self::Spectral>,
        D: Dimension + ndarray::RemoveAxis,
    {
        use crate::utils::array_resized_axis;
        let mut output = array_resized_axis(input, self.m, axis);
        self.backward_inplace(input, &mut output, axis);
        output
    }

    /// See [`Fourier::backward_par`]
    ///
    /// # Panics
    /// Panics when input type cannot be cast from f64.
    #[allow(clippy::used_underscore_binding)]
    fn backward_inplace_par<S1, S2, D>(
        &mut self,
        input: &mut ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = Self::Spectral>,
        S2: ndarray::Data<Elem = Self::Physical> + ndarray::DataMut,
        D: Dimension + ndarray::RemoveAxis,
    {
        use crate::utils::check_array_axis;
        use ndrustfft::ndifft_par;
        check_array_axis(input, self.m, axis, Some("fourier backward"));
        check_array_axis(output, self.n, axis, Some("fourier backward"));
        ndifft_par(input, output, &mut self.fft_handler, axis);
    }
}

impl<A: FloatNum> LaplacianInverse<A> for Fourier<A> {
    /// Laplacian ( = |k^2| ) diagonal matrix
    fn laplace(&self) -> Array2<A> {
        let mut lap = Array2::<A>::zeros((self.m, self.m));
        for (l, k) in lap.diag_mut().iter_mut().zip(self.k.iter()) {
            *l = k.im * k.im;
        }
        lap
    }

    /// Pseudoinverse Laplacian for `Fourier` basis
    /// ( = 1 / |k^2| ) diagonal matrix
    ///
    /// ```
    /// use funspace::fourier::Fourier;
    /// use funspace::LaplacianInverse;
    /// use ndarray::prelude::*;
    /// use funspace::utils::approx_eq;
    /// let fo = Fourier::<f64>::new(4);
    /// let mut laplacian = fo.laplace();
    /// let result = fo.laplace_inv().dot(&laplacian);
    /// approx_eq(
    ///     &fo.laplace_inv_eye(), &result.slice(s![1..,..]).to_owned()
    /// );
    /// ```
    fn laplace_inv(&self) -> Array2<A> {
        let mut pinv = self.laplace();
        for p in pinv.slice_mut(s![1.., 1..]).diag_mut().iter_mut() {
            *p = A::one() / *p;
        }
        pinv
    }

    /// Pseudoidentity matrix (= eye matrix with removed
    /// first row for `Fourier`)
    fn laplace_inv_eye(&self) -> Array2<A> {
        let eye = Array2::<A>::eye(self.m);
        eye.slice(s![1.., ..]).to_owned()
    }
}

impl<A: FloatNum> FromOrtho<Complex<A>> for Fourier<A> {
    /// Return itself
    fn to_ortho<S, D>(&self, input: &ArrayBase<S, D>, _axis: usize) -> Array<Complex<A>, D>
    where
        S: ndarray::Data<Elem = Complex<A>>,
        D: Dimension,
    {
        input.to_owned()
    }

    /// Return itself
    fn to_ortho_inplace<S1, S2, D>(
        &self,
        input: &ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        _axis: usize,
    ) where
        S1: ndarray::Data<Elem = Complex<A>>,
        S2: ndarray::Data<Elem = Complex<A>> + ndarray::DataMut,
        D: Dimension,
    {
        output.assign(input);
    }

    /// Return itself
    fn from_ortho<S, D>(&self, input: &ArrayBase<S, D>, _axis: usize) -> Array<Complex<A>, D>
    where
        S: ndarray::Data<Elem = Complex<A>>,
        D: Dimension,
    {
        input.to_owned()
    }

    /// Return itself
    fn from_ortho_inplace<S1, S2, D>(
        &self,
        input: &ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        _axis: usize,
    ) where
        S1: ndarray::Data<Elem = Complex<A>>,
        S2: ndarray::Data<Elem = Complex<A>> + ndarray::DataMut,
        D: Dimension,
    {
        output.assign(input);
    }
}