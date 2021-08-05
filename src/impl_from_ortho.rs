//! Implement FromOrtho trait.
use crate::Base;
use crate::FloatNum;
use crate::FromOrtho;
use ndarray::prelude::*;
use num_complex::Complex;

/// Implement Differentiate for Float
impl<T: FloatNum> FromOrtho<T> for Base<T> {
    fn to_ortho<S, D>(&self, input: &ArrayBase<S, D>, axis: usize) -> Array<T, D>
    where
        S: ndarray::Data<Elem = T>,
        D: Dimension,
    {
        match self {
            Self::Chebyshev(ref b) => b.to_ortho(input, axis),
            Self::CompositeChebyshev(ref b) => b.to_ortho(input, axis),
            Self::Fourier(_) => {
                panic!("Expect complex array for Fourier, but got real!")
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
            Self::Fourier(_) => {
                panic!("Expect complex array for Fourier, but got real!")
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
            Self::Fourier(_) => {
                panic!("Expect complex array for Fourier, but got real!")
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
            Self::Fourier(_) => {
                panic!("Expect complex array for Fourier, but got real!")
            }
        }
    }
}

/// Implement Differentiate for Complex
impl<T: FloatNum> FromOrtho<Complex<T>> for Base<T> {
    fn to_ortho<S, D>(&self, input: &ArrayBase<S, D>, axis: usize) -> Array<Complex<T>, D>
    where
        S: ndarray::Data<Elem = Complex<T>>,
        D: Dimension,
    {
        match self {
            Self::Chebyshev(ref b) => b.to_ortho(input, axis),
            Self::CompositeChebyshev(ref b) => b.to_ortho(input, axis),
            Self::Fourier(ref b) => b.to_ortho(input, axis),
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
            Self::Fourier(ref b) => b.to_ortho_inplace(input, output, axis),
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
            Self::Fourier(ref b) => b.from_ortho(input, axis),
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
            Self::Fourier(ref b) => b.from_ortho_inplace(input, output, axis),
        }
    }
}

// /// Implement Differentiate for Complex
// impl<A: FloatNum> Differentiate<Complex<A>> for Base<A> {
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
//         match self {
//             Self::Chebyshev(ref b) => b.differentiate(data, n_times, axis),
//             Self::CompositeChebyshev(ref b) => b.differentiate(data, n_times, axis),
//             Self::Fourier(ref b) => b.differentiate(data, n_times, axis),
//         }
//     }

//     fn differentiate_inplace<S, D>(&self, data: &mut ArrayBase<S, D>, n_times: usize, axis: usize)
//     where
//         S: ndarray::Data<Elem = Complex<A>> + ndarray::DataMut,
//         D: Dimension,
//     {
//         match self {
//             Self::Chebyshev(ref b) => b.differentiate_inplace(data, n_times, axis),
//             Self::CompositeChebyshev(ref b) => b.differentiate_inplace(data, n_times, axis),
//             Self::Fourier(ref b) => b.differentiate_inplace(data, n_times, axis),
//         }
//     }
// }
