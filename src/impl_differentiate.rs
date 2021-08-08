//! Implement Differentiate trait.
use crate::Base;
use crate::Differentiate;
use crate::FloatNum;
use ndarray::prelude::*;
use num_complex::Complex;

/// Implement Differentiate for Float
impl<A: FloatNum> Differentiate<A> for Base<A> {
    fn differentiate<S, D>(
        &self,
        data: &ArrayBase<S, D>,
        n_times: usize,
        axis: usize,
    ) -> Array<A, D>
    where
        S: ndarray::Data<Elem = A>,
        D: Dimension,
    {
        match self {
            Self::Chebyshev(ref b) => b.differentiate(data, n_times, axis),
            Self::CompositeChebyshev(ref b) => b.differentiate(data, n_times, axis),
            Self::FourierC2c(_) | Self::FourierR2c(_) => {
                panic!("Expect complex array for Fourier, but got real!");
            }
        }
    }

    fn differentiate_inplace<S, D>(&self, data: &mut ArrayBase<S, D>, n_times: usize, axis: usize)
    where
        S: ndarray::Data<Elem = A> + ndarray::DataMut,
        D: Dimension,
    {
        match self {
            Self::Chebyshev(ref b) => b.differentiate_inplace(data, n_times, axis),
            Self::CompositeChebyshev(ref b) => b.differentiate_inplace(data, n_times, axis),
            Self::FourierC2c(_) | Self::FourierR2c(_) => {
                panic!("Expect complex array for Fourier, but got real!");
            }
        }
    }
}

/// Implement Differentiate for Complex
impl<A: FloatNum> Differentiate<Complex<A>> for Base<A> {
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
        match self {
            Self::Chebyshev(ref b) => b.differentiate(data, n_times, axis),
            Self::CompositeChebyshev(ref b) => b.differentiate(data, n_times, axis),
            Self::FourierC2c(ref b) => b.differentiate(data, n_times, axis),
            Self::FourierR2c(ref b) => b.differentiate(data, n_times, axis),
        }
    }

    fn differentiate_inplace<S, D>(&self, data: &mut ArrayBase<S, D>, n_times: usize, axis: usize)
    where
        S: ndarray::Data<Elem = Complex<A>> + ndarray::DataMut,
        D: Dimension,
    {
        match self {
            Self::Chebyshev(ref b) => b.differentiate_inplace(data, n_times, axis),
            Self::CompositeChebyshev(ref b) => b.differentiate_inplace(data, n_times, axis),
            Self::FourierC2c(ref b) => b.differentiate_inplace(data, n_times, axis),
            Self::FourierR2c(ref b) => b.differentiate_inplace(data, n_times, axis),
        }
    }
}
