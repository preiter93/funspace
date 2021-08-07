//! # SpaceBase
//! Multidimensional space, where each dimension
//! is spanned by a `Base`.
//!
//! Implements `Transform`, `Differentiate` and `FromOrtho` trait as defined
//! for `Base` itself. Other traits, who do not act along an axis, are defined
//! on the `SpaceBase` struct itself, without a respective trait.
//!
//! # Example
//! ```
//! use funspace::{fourier_r2c, cheb_dirichlet, Space2, Transform};
//! use ndarray::prelude::*;
//! use num_complex::Complex;
//! let mut space = Space2::new(&[fourier_r2c(5), cheb_dirichlet(5)]);
//! let mut data: Array2<f64> = space.ndarray_physical();
//! let mut vhat: Array2<Complex<f64>> = space.ndarray_spectral();
//! data += 1.;
//! let mut inter: Array2<f64> = space.forward(&mut data, 1);
//! space.forward_inplace(&mut inter, &mut vhat, 0);
//! ```
#![allow(clippy::module_name_repetitions, clippy::must_use_candidate)]
use crate::traits::BaseBasics;
use crate::traits::Differentiate;
use crate::traits::FromOrtho;
use crate::traits::LaplacianInverse;
use crate::traits::Transform;
use crate::traits::TransformPar;
use crate::types::FloatNum;
use crate::types::Scalar;
use crate::BaseKind;
use ndarray::prelude::*;
use ndarray::{IntoDimension, Ix};
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
    pub bases: [BaseKind<T>; N],
}

impl<T, const N: usize> SpaceBase<T, N>
where
    T: FloatNum,
    [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
    Dim<[Ix; N]>: Dimension,
{
    /// Return new space
    #[must_use]
    pub fn new(bases: &[BaseKind<T>; N]) -> Self {
        Self {
            bases: bases.clone(),
        }
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
    pub fn ndarray_physical<A>(&self) -> Array<A, Dim<[usize; N]>>
    where
        A: Scalar,
    {
        Array::zeros(self.shape_phys())
    }

    /// Return ndarray with shape of spectral space
    pub fn ndarray_spectral<A>(&self) -> Array<A, Dim<[usize; N]>>
    where
        A: Scalar,
    {
        Array::zeros(self.shape_spec())
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
                D: Dimension,
            {
                self.bases[axis].forward(input, axis)
            }

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
                self.bases[axis].forward_inplace(input, output, axis)
            }

            fn backward<S, D>(
                &mut self,
                input: &mut ArrayBase<S, D>,
                axis: usize,
            ) -> Array<Self::Physical, D>
            where
                S: ndarray::Data<Elem = Self::Spectral>,
                D: Dimension,
            {
                self.bases[axis].backward(input, axis)
            }

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
                self.bases[axis].backward_inplace(input, output, axis)
            }
        }
    };
}

// Real to real
impl_transform!(A);
// Real to complex
impl_transform!(Complex<A>);

macro_rules! impl_transform_par {
    ($a: ty) => {
        impl<A: FloatNum, const N: usize> TransformPar<A, $a> for SpaceBase<A, N> {
            type Physical = A;
            type Spectral = $a;

            fn forward_par<S, D>(
                &mut self,
                input: &mut ArrayBase<S, D>,
                axis: usize,
            ) -> Array<Self::Spectral, D>
            where
                S: ndarray::Data<Elem = Self::Physical>,
                D: Dimension,
            {
                self.bases[axis].forward_par(input, axis)
            }

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
                self.bases[axis].forward_inplace_par(input, output, axis)
            }

            fn backward_par<S, D>(
                &mut self,
                input: &mut ArrayBase<S, D>,
                axis: usize,
            ) -> Array<Self::Physical, D>
            where
                S: ndarray::Data<Elem = Self::Spectral>,
                D: Dimension,
            {
                self.bases[axis].backward_par(input, axis)
            }

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
                self.bases[axis].backward_inplace_par(input, output, axis)
            }
        }
    };
}

// Real to real
impl_transform_par!(A);
// Real to complex
impl_transform_par!(Complex<A>);

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
