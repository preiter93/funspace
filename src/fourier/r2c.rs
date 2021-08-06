//! # Real - to - complex fourier space
//! # Example
//! Initialize new fourier basis
//! ```
//! use funspace::fourier::Fourier;
//! let fo = Fourier::<f64>::new_r2c(4);
//! ```
use super::FloatNum;
use super::Fourier;
use crate::Transform;
use crate::TransformPar;
use ndarray::prelude::*;
use ndrustfft::FftHandler;
use num_complex::Complex;

impl<A: FloatNum> Fourier<A> {
    /// Returns a new Fourier Basis for real-to-complex transforms
    #[must_use]
    pub fn new_r2c(n: usize) -> Self {
        Self {
            n,
            m: n / 2 + 1,
            x: Self::nodes(n),
            k: Self::wavenumber_half(n),
            fft_handler: FftHandler::new(n),
        }
    }

    /// Return complex wavenumber vector for r2c transform (0, 1, 2, 3)
    #[allow(clippy::missing_panics_doc)]
    fn wavenumber_half(n: usize) -> Array1<Complex<A>> {
        let n2 = (n - 1) / 2 + 1;
        let mut k: Array1<A> = Array1::zeros(n2);

        for (i, ki) in Array1::range(0., n2 as f64, 1.)
            .iter()
            .zip(k.slice_mut(s![..n2]))
        {
            *ki = A::from_f64(*i as f64).unwrap();
        }

        k.mapv(|x| Complex::new(A::zero(), x))
    }
}

/// Copied from c2c
impl<A: FloatNum> Transform<A, Complex<A>> for Fourier<A> {
    type Physical = A;
    type Spectral = Complex<A>;

    /// # Example
    /// Forward transform along first axis
    /// ```
    /// use funspace::fourier::Fourier;
    /// use funspace::Transform;
    /// use funspace::utils::approx_eq_complex;
    /// use num_complex::Complex;
    /// use ndarray::prelude::*;
    /// let mut fo = Fourier::new_r2c(4);
    /// let mut input = array![1., 2., 3., 4.];
    /// let expected = array![
    ///     Complex::new(10., 0.),
    ///     Complex::new(-2., 2.),
    ///     Complex::new(-2., 0.)
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
        D: Dimension,
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
        D: Dimension,
    {
        use crate::utils::check_array_axis;
        use ndrustfft::ndfft_r2c;
        check_array_axis(input, self.n, axis, Some("fourier forward"));
        check_array_axis(output, self.m, axis, Some("fourier forward"));
        ndfft_r2c(input, output, &mut self.fft_handler, axis);
    }

    /// # Example
    /// Backward transform along first axis
    /// ```
    /// use funspace::fourier::Fourier;
    /// use funspace::Transform;
    /// use funspace::utils::approx_eq;
    /// use num_complex::Complex;
    /// use ndarray::prelude::*;
    /// let mut fo = Fourier::new_r2c(4);
    /// let mut input = array![
    ///     Complex::new(10., 0.),
    ///     Complex::new(-2., 2.),
    ///     Complex::new(-2., 0.)
    /// ];
    /// let expected = array![1., 2., 3., 4.];
    /// let output = fo.backward(&mut input, 0);
    /// approx_eq(&output, &expected);
    /// ```
    fn backward<S, D>(
        &mut self,
        input: &mut ArrayBase<S, D>,
        axis: usize,
    ) -> Array<Self::Physical, D>
    where
        S: ndarray::Data<Elem = Self::Spectral>,
        D: Dimension,
    {
        use crate::utils::array_resized_axis;
        let mut output = array_resized_axis(input, self.n, axis);
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
        D: Dimension,
    {
        use crate::utils::check_array_axis;
        use ndrustfft::ndifft_r2c;
        check_array_axis(input, self.m, axis, Some("fourier backward"));
        check_array_axis(output, self.n, axis, Some("fourier backward"));
        ndifft_r2c(input, output, &mut self.fft_handler, axis);
    }
}

impl<A: FloatNum> TransformPar<A, Complex<A>> for Fourier<A> {
    type Physical = A;
    type Spectral = Complex<A>;

    /// # Example
    /// Forward transform along first axis
    /// ```
    /// use funspace::fourier::Fourier;
    /// use funspace::TransformPar;
    /// use funspace::utils::approx_eq_complex;
    /// use num_complex::Complex;
    /// use ndarray::prelude::*;
    /// let mut fo = Fourier::new_r2c(4);
    /// let mut input = array![1., 2., 3., 4.];
    /// let expected = array![
    ///     Complex::new(10., 0.),
    ///     Complex::new(-2., 2.),
    ///     Complex::new(-2., 0.)
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
        D: Dimension,
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
        D: Dimension,
    {
        use crate::utils::check_array_axis;
        use ndrustfft::ndfft_r2c_par;
        check_array_axis(input, self.n, axis, Some("fourier forward"));
        check_array_axis(output, self.m, axis, Some("fourier forward"));
        ndfft_r2c_par(input, output, &mut self.fft_handler, axis);
    }

    /// # Example
    /// Backward transform along first axis
    /// ```
    /// use funspace::fourier::Fourier;
    /// use funspace::TransformPar;
    /// use funspace::utils::approx_eq;
    /// use num_complex::Complex;
    /// use ndarray::prelude::*;
    /// let mut fo = Fourier::new_r2c(4);
    /// let mut input = array![
    ///     Complex::new(10., 0.),
    ///     Complex::new(-2., 2.),
    ///     Complex::new(-2., 0.)
    /// ];
    /// let expected = array![1., 2., 3., 4.];
    /// let output = fo.backward_par(&mut input, 0);
    /// approx_eq(&output, &expected);
    /// ```
    fn backward_par<S, D>(
        &mut self,
        input: &mut ArrayBase<S, D>,
        axis: usize,
    ) -> Array<Self::Physical, D>
    where
        S: ndarray::Data<Elem = Self::Spectral>,
        D: Dimension,
    {
        use crate::utils::array_resized_axis;
        let mut output = array_resized_axis(input, self.n, axis);
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
        D: Dimension,
    {
        use crate::utils::check_array_axis;
        use ndrustfft::ndifft_r2c_par;
        check_array_axis(input, self.m, axis, Some("fourier backward"));
        check_array_axis(output, self.n, axis, Some("fourier backward"));
        ndifft_r2c_par(input, output, &mut self.fft_handler, axis);
    }
}
