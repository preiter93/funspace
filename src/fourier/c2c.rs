//! # Complex - to - complex fourier space
//!
//! # Example
//! Initialize new fourier basis
//! ```
//! use funspace::fourier::FourierC2c;
//! let fo = FourierC2c::<f64>::new(4);
//! ```
//#![allow(unused_imports)]
#![allow(clippy::module_name_repetitions)]
use crate::traits::Basics;
use crate::traits::Differentiate;
use crate::traits::DifferentiatePar;
use crate::traits::FromOrtho;
use crate::traits::FromOrthoPar;
use crate::traits::LaplacianInverse;
use crate::traits::Transform;
use crate::traits::TransformKind;
use crate::traits::TransformPar;
use crate::types::FloatNum;
use crate::types::Scalar;
use core::f64::consts::PI;
use ndarray::prelude::*;
use ndrustfft::FftHandler;
use num_complex::Complex;

/// # Container for fourier space (Complex-to-complex)
#[derive(Clone)]
pub struct FourierC2c<A> {
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
    /// Transform kind (complex-to-complex)
    transform_kind: TransformKind,
}

impl<A: FloatNum> FourierC2c<A> {
    /// Returns a new Fourier Basis
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            n,
            m: n,
            x: Self::nodes(n),
            k: Self::wavenumber(n),
            fft_handler: FftHandler::new(n),
            transform_kind: TransformKind::ComplexToComplex,
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
    /// use funspace::fourier::FourierC2c;
    /// use funspace::utils::approx_eq_complex;
    /// use ndarray::prelude::*;
    /// let fo = FourierC2c::<f64>::new(5);
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
        let k = self.k.mapv(T2::from);
        for _ in 0..n_times {
            for (d, ki) in data.iter_mut().zip(k.iter()) {
                *d = *d * *ki;
            }
        }
    }
}

impl<A: FloatNum> Basics<A> for FourierC2c<A> {
    /// Size in physical space
    fn len_phys(&self) -> usize {
        self.n
    }
    /// Size in spectral space
    fn len_spec(&self) -> usize {
        self.m
    }
    /// Coordinates in physical space
    fn coords(&self) -> &Array1<A> {
        &self.x
    }
    /// Return mass matrix (= eye)
    fn mass(&self) -> Array2<A> {
        Array2::<A>::eye(self.n)
    }
    /// Return transform kind
    fn get_transform_kind(&self) -> &TransformKind {
        &self.transform_kind
    }
}

/// Perform differentiation in spectral space
impl<A: FloatNum> Differentiate<Complex<A>> for FourierC2c<A> {
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

/// Perform differentiation in spectral space
impl<A: FloatNum> DifferentiatePar<Complex<A>> for FourierC2c<A> {
    fn differentiate_par<S, D>(
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
        self.differentiate_inplace_par(&mut output, n_times, axis);
        output
    }

    fn differentiate_inplace_par<S, D>(
        &self,
        data: &mut ArrayBase<S, D>,
        n_times: usize,
        axis: usize,
    ) where
        S: ndarray::Data<Elem = Complex<A>> + ndarray::DataMut,
        D: Dimension,
    {
        use crate::utils::check_array_axis;
        check_array_axis(data, self.m, axis, Some("fourier differentiate"));
        ndarray::Zip::from(data.lanes_mut(Axis(axis))).par_for_each(|mut lane| {
            self.differentiate_lane(&mut lane, n_times);
        });
    }
}

impl<A: FloatNum> Transform for FourierC2c<A> {
    type Physical = Complex<A>;
    type Spectral = Complex<A>;

    /// # Example
    /// Forward transform along first axis
    /// ```
    /// use funspace::fourier::FourierC2c;
    /// use funspace::Transform;
    /// use funspace::utils::approx_eq_complex;
    /// use num_complex::Complex;
    /// use ndarray::prelude::*;
    /// let mut fo = FourierC2c::new(4);
    /// let input = array![
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
    /// let output = fo.forward(&input, 0);
    /// approx_eq_complex(&output, &expected);
    /// ```
    fn forward<S, D>(&mut self, input: &ArrayBase<S, D>, axis: usize) -> Array<Self::Spectral, D>
    where
        S: ndarray::Data<Elem = Self::Physical>,
        D: Dimension,
    {
        use crate::utils::array_resized_axis;
        let mut output = array_resized_axis(input, self.m, axis);
        self.forward_inplace(input, &mut output, axis);
        output
    }

    /// See [`FourierC2c::forward`]
    fn forward_inplace<S1, S2, D>(
        &mut self,
        input: &ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = Self::Physical>,
        S2: ndarray::Data<Elem = Self::Spectral> + ndarray::DataMut,
        D: Dimension,
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
    /// use funspace::fourier::FourierC2c;
    /// use funspace::Transform;
    /// use funspace::utils::approx_eq_complex;
    /// use num_complex::Complex;
    /// use ndarray::prelude::*;
    /// let mut fo = FourierC2c::new(4);
    /// let input = array![
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
    /// let output = fo.backward(&input, 0);
    /// approx_eq_complex(&output, &expected);
    /// ```
    fn backward<S, D>(&mut self, input: &ArrayBase<S, D>, axis: usize) -> Array<Self::Physical, D>
    where
        S: ndarray::Data<Elem = Self::Spectral>,
        D: Dimension,
    {
        use crate::utils::array_resized_axis;
        let mut output = array_resized_axis(input, self.n, axis);
        self.backward_inplace(input, &mut output, axis);
        output
    }

    /// See [`FourierC2c::backward`]
    ///
    /// # Panics
    /// Panics when input type cannot be cast from f64.
    #[allow(clippy::used_underscore_binding)]
    fn backward_inplace<S1, S2, D>(
        &mut self,
        input: &ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = Self::Spectral>,
        S2: ndarray::Data<Elem = Self::Physical> + ndarray::DataMut,
        D: Dimension,
    {
        use crate::utils::check_array_axis;
        use ndrustfft::ndifft;
        check_array_axis(input, self.m, axis, Some("fourier backward"));
        check_array_axis(output, self.n, axis, Some("fourier backward"));
        ndifft(input, output, &mut self.fft_handler, axis);
    }
}

impl<A: FloatNum> TransformPar for FourierC2c<A> {
    type Physical = Complex<A>;
    type Spectral = Complex<A>;

    /// Parallel version. See [`FourierC2c::forward`]
    fn forward_par<S, D>(
        &mut self,
        input: &ArrayBase<S, D>,
        axis: usize,
    ) -> Array<Self::Spectral, D>
    where
        S: ndarray::Data<Elem = Self::Physical>,
        D: Dimension,
    {
        use crate::utils::array_resized_axis;
        let mut output = array_resized_axis(input, self.m, axis);
        self.forward_inplace_par(input, &mut output, axis);
        output
    }

    /// Parallel version. See [`FourierC2c::forward_inplace`]
    fn forward_inplace_par<S1, S2, D>(
        &mut self,
        input: &ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = Self::Physical>,
        S2: ndarray::Data<Elem = Self::Spectral> + ndarray::DataMut,
        D: Dimension,
    {
        use crate::utils::check_array_axis;
        use ndrustfft::ndfft_par;
        check_array_axis(input, self.n, axis, Some("fourier forward"));
        check_array_axis(output, self.m, axis, Some("fourier forward"));
        ndfft_par(input, output, &mut self.fft_handler, axis);
    }

    /// Parallel version. See [`FourierC2c::backward`]
    fn backward_par<S, D>(
        &mut self,
        input: &ArrayBase<S, D>,
        axis: usize,
    ) -> Array<Self::Physical, D>
    where
        S: ndarray::Data<Elem = Self::Spectral>,
        D: Dimension,
    {
        use crate::utils::array_resized_axis;
        let mut output = array_resized_axis(input, self.n, axis);
        self.backward_inplace_par(input, &mut output, axis);
        output
    }

    /// Parallel version. See [`FourierC2c::backward_inplace`]
    ///
    /// # Panics
    /// Panics when input type cannot be cast from f64.
    #[allow(clippy::used_underscore_binding)]
    fn backward_inplace_par<S1, S2, D>(
        &mut self,
        input: &ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = Self::Spectral>,
        S2: ndarray::Data<Elem = Self::Physical> + ndarray::DataMut,
        D: Dimension,
    {
        use crate::utils::check_array_axis;
        use ndrustfft::ndifft_par;
        check_array_axis(input, self.m, axis, Some("fourier backward"));
        check_array_axis(output, self.n, axis, Some("fourier backward"));
        ndifft_par(input, output, &mut self.fft_handler, axis);
    }
}

impl<A: FloatNum> LaplacianInverse<A> for FourierC2c<A> {
    /// Laplacian ( = |k^2| ) diagonal matrix
    fn laplace(&self) -> Array2<A> {
        let mut lap = Array2::<A>::zeros((self.m, self.m));
        for (l, k) in lap.diag_mut().iter_mut().zip(self.k.iter()) {
            *l = -k.im * k.im;
        }
        lap
    }

    /// Pseudoinverse Laplacian for `FourierC2c` basis
    /// ( = 1 / |k^2| ) diagonal matrix
    ///
    /// ```
    /// use funspace::fourier::FourierC2c;
    /// use funspace::LaplacianInverse;
    /// use ndarray::prelude::*;
    /// use funspace::utils::approx_eq;
    /// let fo = FourierC2c::<f64>::new(4);
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
    /// first row for `FourierC2c`)
    fn laplace_inv_eye(&self) -> Array2<A> {
        let eye = Array2::<A>::eye(self.m);
        eye.slice(s![1.., ..]).to_owned()
    }
}

impl<A: FloatNum> FromOrtho<Complex<A>> for FourierC2c<A> {
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

impl<A: FloatNum> FromOrthoPar<Complex<A>> for FourierC2c<A> {
    /// Return itself
    fn to_ortho_par<S, D>(&self, input: &ArrayBase<S, D>, _axis: usize) -> Array<Complex<A>, D>
    where
        S: ndarray::Data<Elem = Complex<A>>,
        D: Dimension,
    {
        input.to_owned()
    }

    /// Return itself
    fn to_ortho_inplace_par<S1, S2, D>(
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
    fn from_ortho_par<S, D>(&self, input: &ArrayBase<S, D>, _axis: usize) -> Array<Complex<A>, D>
    where
        S: ndarray::Data<Elem = Complex<A>>,
        D: Dimension,
    {
        input.to_owned()
    }

    /// Return itself
    fn from_ortho_inplace_par<S1, S2, D>(
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
