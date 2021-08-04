//! Implement Differentiate trait.
use crate::Base;
use crate::Differentiate;
use crate::FloatNum;
use crate::Scalar;
use ndarray::prelude::*;

/// Implement Differentiate for Float
impl<A: FloatNum> Differentiate<A> for Base<A> {
    fn differentiate<S, D, T2>(
        &self,
        data: &ArrayBase<S, D>,
        n_times: usize,
        axis: usize,
    ) -> Array<T2, D>
    where
        S: ndarray::Data<Elem = T2>,
        D: Dimension,
        T2: Scalar + From<A>,
    {
        match self {
            Self::Chebyshev(ref b) => b.differentiate(data, n_times, axis),
            Self::CompositeChebyshev(ref b) => b.differentiate(data, n_times, axis),
        }
    }

    fn differentiate_inplace<S, D, T2>(
        &self,
        data: &mut ArrayBase<S, D>,
        n_times: usize,
        axis: usize,
    ) where
        S: ndarray::Data<Elem = T2> + ndarray::DataMut,
        D: Dimension,
        T2: Scalar + From<A>,
    {
        match self {
            Self::Chebyshev(ref b) => b.differentiate(data, n_times, axis),
            Self::CompositeChebyshev(ref b) => b.differentiate(data, n_times, axis),
        };
    }
}

// /// Implement Differentiate for Complex
// impl<A: FloatNum> Differentiate<Complex<A>> for Base<A> {
//     fn differentiate<S, D, T2>(
//         &self,
//         data: &ArrayBase<S, D>,
//         n_times: usize,
//         axis: usize,
//     ) -> Array<T2, D>
//     where
//         S: ndarray::Data<Elem = T2>,
//         D: Dimension,
//         T2: Scalar + From<Complex<A>>,
//     {
//         match self {
//             Self::Chebyshev(ref b) => b.differentiate(data, n_times, axis),
//             Self::CompositeChebyshev(ref b) => b.differentiate(data, n_times, axis),
//         }
//     }

//     fn differentiate_inplace<S, D, T2>(
//         &self,
//         data: &mut ArrayBase<S, D>,
//         n_times: usize,
//         axis: usize,
//     ) where
//         S: ndarray::Data<Elem = T2> + ndarray::DataMut,
//         D: Dimension,
//         T2: Scalar + From<Complex<A>>,
//     {
//         match self {
//             Self::Chebyshev(ref b) => b.differentiate(data, n_times, axis),
//             Self::CompositeChebyshev(ref b) => b.differentiate(data, n_times, axis),
//         };
//     }
// }
