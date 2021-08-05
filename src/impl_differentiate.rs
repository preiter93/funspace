//! Implement Differentiate trait.
use crate::Base;
use crate::Differentiate;
use crate::FloatNum;
use ndarray::prelude::*;
use num_complex::Complex;

/// Implement Differentiate for Base
macro_rules! impl_differentiate_base {
    ($a: ty) => {
        impl<A: FloatNum> Differentiate<$a> for Base<A> {
            fn differentiate<S, D>(
                &self,
                data: &ArrayBase<S, D>,
                n_times: usize,
                axis: usize,
            ) -> Array<$a, D>
            where
                S: ndarray::Data<Elem = $a>,
                D: Dimension,
            {
                match self {
                    Self::Chebyshev(ref b) => b.differentiate(data, n_times, axis),
                    Self::CompositeChebyshev(ref b) => b.differentiate(data, n_times, axis),
                }
            }

            fn differentiate_inplace<S, D>(&self, data: &mut ArrayBase<S, D>, n_times: usize, axis: usize)
            where
                S: ndarray::Data<Elem = $a> + ndarray::DataMut,
                D: Dimension,
            {
                match self {
                    Self::Chebyshev(ref b) => b.differentiate_inplace(data, n_times, axis),
                    Self::CompositeChebyshev(ref b) => b.differentiate_inplace(data, n_times, axis),
                }
            }
        }
    };
}

// Implement Differentiate for Float
impl_differentiate_base!(A);
// Implement Differentiate for Complex
impl_differentiate_base!(Complex<A>);
