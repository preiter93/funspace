//! # SpaceBase
//! Multidimensional space, where each dimension
//! is spanned by a `Base`.
//!
//! Implements `Transform`, `Differentiate` and `FromOrtho` trait as defined
//! for `Base` itself. Other traits, who do not act along an axis, are defined
//! on the `SpaceBase` struct itself, without a respective trait.
#![allow(clippy::module_name_repetitions, clippy::must_use_candidate)]
use crate::Base;
use crate::{Differentiate, FromOrtho, LaplacianInverse, Mass, Size, Transform, TransformPar};
use crate::{FloatNum, Scalar};
use ndarray::prelude::*;
use num_complex::Complex;
use std::convert::TryInto;

/// One dimensional Space
pub type Space1 = SpaceBase<f64, 1>;
/// Two dimensional Space
pub type Space2 = SpaceBase<f64, 2>;

/// Create multidimensional space
///
/// First create a space, then
/// initialize field with it.
#[derive(Clone)]
pub struct SpaceBase<T: FloatNum, const N: usize> {
    pub bases: [Base<T>; N],
}

impl<T, const N: usize> SpaceBase<T, N>
where
    T: FloatNum,
    [usize; N]: ndarray::Dimension,
{
    /// Return new space
    #[must_use]
    pub fn new(bases: [Base<T>; N]) -> Self {
        Self { bases }
    }

    /// Shape in physical space
    ///
    /// ## Panics
    /// When vector of size *N* cannot be
    /// cast to array
    pub fn shape_phys(&self) -> [usize; N] {
        self.bases
            .iter()
            .map(|x| x.len_phys())
            .collect::<Vec<usize>>()
            .try_into()
            .unwrap()
    }
    /// Shape in spectral space
    ///
    /// ## Panics
    /// When vector of size *N* cannot be
    /// cast to array
    pub fn shape_spec(&self) -> [usize; N] {
        self.bases
            .iter()
            .map(|x| x.len_spec())
            .collect::<Vec<usize>>()
            .try_into()
            .unwrap()
    }

    /// Return ndarray with shape of physical space
    pub fn ndarray_physical<A: Scalar>(&self) -> Array<A, [usize; N]> {
        Self::ndarray_from_shape(self.shape_phys())
    }

    /// Return ndarray with shape of spectral space
    pub fn ndarray_spectral<A: Scalar>(&self) -> Array<A, [usize; N]> {
        Self::ndarray_from_shape(self.shape_spec())
    }

    fn ndarray_from_shape<A, S>(shape: S) -> Array<A, S>
    where
        A: Scalar,
        S: ndarray::Dimension,
    {
        Array::zeros(shape)
    }

    /// Laplacian $ L $
    /// (square matrix)
    pub fn laplace(&self, axis: usize) -> Array2<T> {
        self.bases[axis].laplace()
    }

    /// Pseudoinverse mtrix of Laplacian $ L^{-1} $
    /// (square matrix)
    pub fn laplace_inv(&self, axis: usize) -> Array2<T> {
        self.bases[axis].laplace_inv()
    }

    /// Pseudoidentity matrix of laplacian $ L^{-1} L $
    /// (can return a non-square matric)
    pub fn laplace_inv_eye(&self, axis: usize) -> Array2<T> {
        self.bases[axis].laplace_inv_eye()
    }

    /// Return mass matrix
    pub fn mass(&self, axis: usize) -> Array2<T> {
        self.bases[axis].mass()
    }

    /// Coordinates in physical space along axis
    pub fn coords_axis(&self, axis: usize) -> Array1<T> {
        self.bases[axis].coords().clone()
    }

    /// Array of coordinates in physical space
    ///
    /// ## Panics
    /// When vector of size *N* cannot be
    /// cast to static array
    pub fn coords(&self) -> [Array1<T>; N] {
        self.bases
            .iter()
            .map(|x| x.coords().clone())
            .collect::<Vec<Array1<T>>>()
            .try_into()
            .unwrap()
    }

    /// Reference to coordinates in physical space along axis
    pub fn coords_ref_axis(&self, axis: usize) -> &Array1<T> {
        self.bases[axis].coords()
    }

    /// Array of references to coordinates in physical space
    ///
    /// ## Panics
    /// When vector of size *N* cannot be
    /// cast to static array
    pub fn coords_ref(&self) -> [&Array1<T>; N] {
        self.bases
            .iter()
            .map(|x| x.coords())
            .collect::<Vec<&Array1<T>>>()
            .try_into()
            .unwrap()
    }
}

macro_rules! impl_transform {
    ($a: ty) => {
        impl<A: FloatNum, const N: usize> Transform<A, $a> for SpaceBase<A, N> {
            type Physical = A;
            type Spectral = $a;

            fn forward<S, D>(
                &mut self,
                input: &mut ArrayBase<S, D>,
                axis: usize,
            ) -> Array<Self::Spectral, D>
            where
                S: ndarray::Data<Elem = Self::Physical>,
                D: Dimension + ndarray::RemoveAxis,
            {
                self.bases[axis].forward_par(input, axis)
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
                self.bases[axis].forward_inplace_par(input, output, axis)
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
                self.bases[axis].backward_par(input, axis)
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
                self.bases[axis].backward_inplace_par(input, output, axis)
            }
        }
    };
}

// Real to real
impl_transform!(A);
// Real to complex
impl_transform!(Complex<A>);

macro_rules! impl_differentiate {
    ($a: ty) => {
        impl<A: FloatNum, const N: usize> Differentiate<$a> for SpaceBase<A, N> {
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
                self.bases[axis].differentiate(data, n_times, axis)
            }

            fn differentiate_inplace<S, D>(
                &self,
                data: &mut ArrayBase<S, D>,
                n_times: usize,
                axis: usize,
            ) where
                S: ndarray::Data<Elem = $a> + ndarray::DataMut,
                D: Dimension,
            {
                self.bases[axis].differentiate_inplace(data, n_times, axis)
            }
        }
    };
}

// Real
impl_differentiate!(A);
// Complex
impl_differentiate!(Complex<A>);

/// Implement FromOrtho
macro_rules! impl_from_ortho {
    ($a: ty) => {
        impl<A: FloatNum, const N: usize> FromOrtho<$a> for SpaceBase<A, N> {
            fn to_ortho<S, D>(&self, input: &ArrayBase<S, D>, axis: usize) -> Array<$a, D>
            where
                S: ndarray::Data<Elem = $a>,
                D: Dimension,
            {
                self.bases[axis].to_ortho(input, axis)
            }

            fn to_ortho_inplace<S1, S2, D>(
                &self,
                input: &ArrayBase<S1, D>,
                output: &mut ArrayBase<S2, D>,
                axis: usize,
            ) where
                S1: ndarray::Data<Elem = $a>,
                S2: ndarray::Data<Elem = $a> + ndarray::DataMut,
                D: Dimension,
            {
                self.bases[axis].to_ortho_inplace(input, output, axis)
            }

            fn from_ortho<S, D>(&self, input: &ArrayBase<S, D>, axis: usize) -> Array<$a, D>
            where
                S: ndarray::Data<Elem = $a>,
                D: Dimension,
            {
                self.bases[axis].from_ortho(input, axis)
            }

            fn from_ortho_inplace<S1, S2, D>(
                &self,
                input: &ArrayBase<S1, D>,
                output: &mut ArrayBase<S2, D>,
                axis: usize,
            ) where
                S1: ndarray::Data<Elem = $a>,
                S2: ndarray::Data<Elem = $a> + ndarray::DataMut,
                D: Dimension,
            {
                self.bases[axis].from_ortho_inplace(input, output, axis)
            }
        }
    };
}

// Real
impl_from_ortho!(A);
// Complex
impl_from_ortho!(Complex<A>);