//! # Complex fourier space
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
use crate::Scalar;

use crate::Mass;
use crate::Size;

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
    fft_handler: FftHandler<A>,
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
    fn nodes(n: usize) -> Array1<A> {
        let n64 = n as f64;
        Array1::range(0., 2. * PI, 2. * PI / n64).mapv(|elem| A::from_f64(elem).unwrap())
    }

    /// Return complex wavenumber vector
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

// pub trait Differentiate2<T> {
//     /// Differentiate on input array
//     fn differentiate_inplace<S, D, T2>(
//         &self,
//         data: &mut ArrayBase<S, D>,
//         n_times: usize,
//         axis: usize,
//     ) where
//         S: ndarray::Data<Elem = T2> + ndarray::DataMut,
//         D: Dimension;
// }

// // impl<A: FloatNum> Differentiate2<A> for Fourier<A> {

// //     fn differentiate_inplace<S, D, T2>(&self, data: &mut ArrayBase<S, D>, n_times: usize, axis: usize)
// //     where
// //         S: ndarray::Data<Elem = T2> + ndarray::DataMut,
// //         D: Dimension,
// //         //T: Scalar,
// //     {
// //         let n_a = A::from_f64(n_times as f64).unwrap();
// //         let k: Array1<Complex<A>> = self.k.mapv(|x| Complex::new(A::zero(), x).powf(n_a));
// //         for mut v in data.lanes_mut(Axis(axis)) {
// //             if n_times % 2 == 0 {
// //                 for (vi,ki) in v.iter_mut().zip(k.iter()) {
// //                     vi.re = vi.re * ki.re;
// //                 }
// //             } else {
// //                 for (vi,ki) in v.iter_mut().zip(k.iter()) {
// //                     vi.im = vi.im * ki.im;
// //                 }
// //             }
// //         }
// //     }

// // }

// /// Perform differentiation in spectral space
// impl<A: FloatNum> Differentiate<Complex<A>> for Fourier<A> {
//     fn differentiate<S, D>(
//         &self,
//         data: &ArrayBase<S, D>,
//         n_times: usize,
//         axis: usize,
//     ) -> Array<Complex<A>, D>
//     where
//         S: ndarray::Data<Elem = Complex<A>>,
//         D: Dimension,
//     {
//         let mut output = data.to_owned();
//         self.differentiate_inplace(&mut output, n_times, axis);
//         output
//     }

//     fn differentiate_inplace<S, D>(&self, data: &mut ArrayBase<S, D>, n_times: usize, axis: usize)
//     where
//         S: ndarray::Data<Elem = Complex<A>> + ndarray::DataMut,
//         D: Dimension,
//     {
//         use crate::utils::check_array_axis;
//         check_array_axis(data, self.m, axis, Some("fourier differentiate"));
//         ndarray::Zip::from(data.lanes_mut(Axis(axis))).for_each(|mut lane| {
//             self.differentiate_lane(&mut lane, n_times);
//         });
//     }
// }
