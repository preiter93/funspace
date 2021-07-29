//! Implement transform trait.
use crate::Base;
use crate::FloatNum;
use crate::Transform;
use crate::TransformPar;
use ndarray::prelude::*;

/// Transform trait is implemented "per hand", can't be enum_dispatched
/// because of associated types.
impl<A: FloatNum + std::ops::MulAssign> Transform for Base<A> {
    type Physical = A;
    type Spectral = A;

    fn forward<S, D>(
        &mut self,
        input: &mut ArrayBase<S, D>,
        axis: usize,
    ) -> Array<Self::Spectral, D>
    where
        S: ndarray::Data<Elem = Self::Physical>,
        D: Dimension + ndarray::RemoveAxis,
    {
        match self {
            Self::Chebyshev(ref mut b) => b.forward(input, axis),
            Self::CompositeChebyshev(ref mut b) => b.forward(input, axis),
        }
    }

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
        match self {
            Self::Chebyshev(ref mut b) => b.forward_inplace(input, output, axis),
            Self::CompositeChebyshev(ref mut b) => b.forward_inplace(input, output, axis),
        }
    }

    fn backward<S, D>(
        &mut self,
        input: &mut ArrayBase<S, D>,
        axis: usize,
    ) -> Array<Self::Physical, D>
    where
        S: ndarray::Data<Elem = Self::Spectral>,
        D: Dimension + ndarray::RemoveAxis,
    {
        match self {
            Self::Chebyshev(ref mut b) => b.backward(input, axis),
            Self::CompositeChebyshev(ref mut b) => b.backward(input, axis),
        }
    }

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
        match self {
            Self::Chebyshev(ref mut b) => b.backward_inplace(input, output, axis),
            Self::CompositeChebyshev(ref mut b) => b.backward_inplace(input, output, axis),
        }
    }
}

/// Transform trait is implemented "per hand", can't be enum_dispatched
/// because of associated types.
impl<A: FloatNum + std::ops::MulAssign> TransformPar for Base<A> {
    type Physical = A;
    type Spectral = A;

    fn forward_par<S, D>(
        &mut self,
        input: &mut ArrayBase<S, D>,
        axis: usize,
    ) -> Array<Self::Spectral, D>
    where
        S: ndarray::Data<Elem = Self::Physical>,
        D: Dimension + ndarray::RemoveAxis,
    {
        match self {
            Self::Chebyshev(ref mut b) => b.forward_par(input, axis),
            Self::CompositeChebyshev(ref mut b) => b.forward_par(input, axis),
        }
    }

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
        match self {
            Self::Chebyshev(ref mut b) => b.forward_inplace_par(input, output, axis),
            Self::CompositeChebyshev(ref mut b) => b.forward_inplace_par(input, output, axis),
        }
    }

    fn backward_par<S, D>(
        &mut self,
        input: &mut ArrayBase<S, D>,
        axis: usize,
    ) -> Array<Self::Physical, D>
    where
        S: ndarray::Data<Elem = Self::Spectral>,
        D: Dimension + ndarray::RemoveAxis,
    {
        match self {
            Self::Chebyshev(ref mut b) => b.backward_par(input, axis),
            Self::CompositeChebyshev(ref mut b) => b.backward_par(input, axis),
        }
    }

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
        match self {
            Self::Chebyshev(ref mut b) => b.backward_inplace_par(input, output, axis),
            Self::CompositeChebyshev(ref mut b) => b.backward_inplace_par(input, output, axis),
        }
    }
}
