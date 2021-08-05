//! # Real - to - complex fourier space
//! # Example
//! Initialize new fourier basis
//! ```
//! use funspace::fourier::Fourier;
//! let fo = Fourier::<f64>::new_r2c(4);
//! ```
use super::FloatNum;
use super::Fourier;
use ndarray::prelude::*;
use ndrustfft::FftHandler;
use num_complex::Complex;

impl<A: FloatNum> Fourier<A> {
    /// Returns a new Fourier Basis for real-to-complex transforms
    #[must_use]
    pub fn new_r2c(n: usize) -> Self {
        Self {
            n,
            m: (n - 1) / 2 + 1,
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
