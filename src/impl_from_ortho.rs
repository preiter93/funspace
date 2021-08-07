//! Implement FromOrtho trait.
use crate::BaseKind;
use crate::FloatNum;
use crate::FromOrtho;
use ndarray::prelude::*;
use num_complex::Complex;

/// Implement FromOrtho for Float
impl<T: FloatNum> FromOrtho<T> for BaseKind<T> {
    fn to_ortho<S, D>(&self, input: &ArrayBase<S, D>, axis: usize) -> Array<T, D>
    where
        S: ndarray::Data<Elem = T>,
        D: Dimension,
    {
        match self {
            Self::Chebyshev(ref b) => b.to_ortho(input, axis),
            Self::CompositeChebyshev(ref b) => b.to_ortho(input, axis),
            Self::FourierC2c(_) | Self::FourierR2c(_) => {
                panic!("Expect complex array for Fourier, but got real!");
            }
        }
    }

    fn to_ortho_inplace<S1, S2, D>(
        &self,
        input: &ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = T>,
        S2: ndarray::Data<Elem = T> + ndarray::DataMut,
        D: Dimension,
    {
        match self {
            Self::Chebyshev(ref b) => b.to_ortho_inplace(input, output, axis),
            Self::CompositeChebyshev(ref b) => b.to_ortho_inplace(input, output, axis),
            Self::FourierC2c(_) | Self::FourierR2c(_) => {
                panic!("Expect complex array for Fourier, but got real!");
            }
        }
    }

    fn from_ortho<S, D>(&self, input: &ArrayBase<S, D>, axis: usize) -> Array<T, D>
    where
        S: ndarray::Data<Elem = T>,
        D: Dimension,
    {
        match self {
            Self::Chebyshev(ref b) => b.from_ortho(input, axis),
            Self::CompositeChebyshev(ref b) => b.from_ortho(input, axis),
            Self::FourierC2c(_) | Self::FourierR2c(_) => {
                panic!("Expect complex array for Fourier, but got real!");
            }
        }
    }

    fn from_ortho_inplace<S1, S2, D>(
        &self,
        input: &ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = T>,
        S2: ndarray::Data<Elem = T> + ndarray::DataMut,
        D: Dimension,
    {
        match self {
            Self::Chebyshev(ref b) => b.from_ortho_inplace(input, output, axis),
            Self::CompositeChebyshev(ref b) => b.from_ortho_inplace(input, output, axis),
            Self::FourierC2c(_) | Self::FourierR2c(_) => {
                panic!("Expect complex array for Fourier, but got real!");
            }
        }
    }
}

/// Implement FromOrtho for Complex
impl<T: FloatNum> FromOrtho<Complex<T>> for BaseKind<T> {
    fn to_ortho<S, D>(&self, input: &ArrayBase<S, D>, axis: usize) -> Array<Complex<T>, D>
    where
        S: ndarray::Data<Elem = Complex<T>>,
        D: Dimension,
    {
        match self {
            Self::Chebyshev(ref b) => b.to_ortho(input, axis),
            Self::CompositeChebyshev(ref b) => b.to_ortho(input, axis),
            Self::FourierC2c(ref b) => b.to_ortho(input, axis),
            Self::FourierR2c(ref b) => b.to_ortho(input, axis),
        }
    }

    fn to_ortho_inplace<S1, S2, D>(
        &self,
        input: &ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = Complex<T>>,
        S2: ndarray::Data<Elem = Complex<T>> + ndarray::DataMut,
        D: Dimension,
    {
        match self {
            Self::Chebyshev(ref b) => b.to_ortho_inplace(input, output, axis),
            Self::CompositeChebyshev(ref b) => b.to_ortho_inplace(input, output, axis),
            Self::FourierC2c(ref b) => b.to_ortho_inplace(input, output, axis),
            Self::FourierR2c(ref b) => b.to_ortho_inplace(input, output, axis),
        }
    }

    fn from_ortho<S, D>(&self, input: &ArrayBase<S, D>, axis: usize) -> Array<Complex<T>, D>
    where
        S: ndarray::Data<Elem = Complex<T>>,
        D: Dimension,
    {
        match self {
            Self::Chebyshev(ref b) => b.from_ortho(input, axis),
            Self::CompositeChebyshev(ref b) => b.from_ortho(input, axis),
            Self::FourierC2c(ref b) => b.from_ortho(input, axis),
            Self::FourierR2c(ref b) => b.from_ortho(input, axis),
        }
    }

    fn from_ortho_inplace<S1, S2, D>(
        &self,
        input: &ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = Complex<T>>,
        S2: ndarray::Data<Elem = Complex<T>> + ndarray::DataMut,
        D: Dimension,
    {
        match self {
            Self::Chebyshev(ref b) => b.from_ortho_inplace(input, output, axis),
            Self::CompositeChebyshev(ref b) => b.from_ortho_inplace(input, output, axis),
            Self::FourierC2c(ref b) => b.from_ortho_inplace(input, output, axis),
            Self::FourierR2c(ref b) => b.from_ortho_inplace(input, output, axis),
        }
    }
}
