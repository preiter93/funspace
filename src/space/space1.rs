//! # One-dmensional space
//!
//! # Example
//! Transform to chebyshev - dirichlet space
//! ```
//! use funspace::{cheb_dirichlet, Space1};
//! use funspace::space::traits::BaseSpaceTransform;
//! use ndarray::prelude::*;
//! let mut space = Space1::new(&cheb_dirichlet::<f64>(5));
//! let mut v: Array1<f64> = space.ndarray_physical();
//! v += 1.;
//! let vhat = space.forward(&mut v);
//! println!("{:?}", vhat);
//! // Not how the cheb dirichlet base imposes dirichlet conditions on
//! // the array: the first and last point are now zero,
//! let v = space.backward(&vhat);
//! println!("{:?}", v);
//! ```
#![allow(clippy::module_name_repetitions)]
use crate::enums::{BaseKind, TransformKind};
use crate::space::traits::{
    BaseSpaceElements, BaseSpaceFromOrtho, BaseSpaceGradient, BaseSpaceMatOpStencil,
    BaseSpaceMatOpLaplacian, BaseSpaceSize, BaseSpaceTransform,
};
use crate::traits::{
    BaseElements, BaseFromOrtho, BaseGradient, BaseMatOpStencil, BaseMatOpLaplacian, BaseSize,
    BaseTransform,
};
use crate::{BaseC2c, BaseR2c, BaseR2r, FloatNum, ScalarNum};
use ndarray::{prelude::*, Data, DataMut};
use num_complex::Complex;
use num_traits::Zero;
use std::ops::{Add, Div, Mul, Sub};

/// Create two-dimensional space
#[derive(Clone)]
pub struct Space1<B0> {
    // Physical -> Spectral
    pub base0: B0,
}

impl<B0> Space1<B0>
where
    B0: Clone,
{
    /// Create a new space
    pub fn new(base0: &B0) -> Self {
        Self {
            base0: base0.clone(),
        }
    }
}

macro_rules! impl_space1 {
    ($base0: ident, $p: ty, $s: ty) => {
        impl<A> BaseSpaceSize<1> for Space1<$base0<A>>
        where
            A: FloatNum,
        {
            fn shape_physical(&self) -> [usize; 1] {
                [self.base0.len_phys()]
            }

            fn shape_spectral(&self) -> [usize; 1] {
                [self.base0.len_spec()]
            }

            fn shape_spectral_ortho(&self) -> [usize; 1] {
                [self.base0.len_orth()]
            }

            fn ndarray_from_shape<T: Clone + Zero>(&self, shape: [usize; 1]) -> Array1<T> {
                Array1::zeros(shape)
            }
        }

        impl<A> BaseSpaceElements<1> for Space1<$base0<A>>
        where
            A: FloatNum,
        {
            type RealNum = A;

            /// Array of coordinates
            fn coords(&self) -> [Array1<Self::RealNum>; 1] {
                [self.coords_axis(0)]
            }

            /// Coordinates of grid points (in physical space)
            ///
            /// # Arguments
            ///
            /// * `axis` - usize
            fn coords_axis(&self, _axis: usize) -> Array1<Self::RealNum> {
                self.base0.coords().into()
            }

            /// Return base key
            fn base_kind(&self, _axis: usize) -> BaseKind {
                self.base0.base_kind()
            }

            /// Return transform kind
            fn transform_kind(&self, _axis: usize) -> TransformKind {
                self.base0.transform_kind()
            }
        }

        impl<A> BaseSpaceMatOpStencil for Space1<$base0<A>>
        where
            A: FloatNum,
        {
            /// Scalar type of laplacian matrix
            type NumType = A;

            /// Transformation stencil
            ///
            /// Multiplication of this matrix with a coefficient vector has
            /// the same effect as  [`BaseSpaceFromOrtho::to_ortho()`],
            /// but is less efficient.
            ///
            /// Returns identity matrix for orthogonal bases
            ///
            /// # Arguments
            ///
            /// * `axis` - usize
            fn stencil(&self, _axis: usize) -> Array2<A> {
                self.base0.stencil()
            }

            /// Inverse of transformation stencil
            ///
            /// Multiplication of this matrix with a coefficient vector has
            /// the same effect as  [`BaseSpaceFromOrtho::from_ortho()`],
            /// but is less efficient.
            ///
            /// Returns identity matrix for orthogonal bases
            ///
            /// # Arguments
            ///
            /// * `axis` - usize
            fn stencil_inv(&self, _axis: usize) -> Array2<A> {
                self.base0.stencil_inv()
            }
        }

        impl<A> BaseSpaceMatOpLaplacian for Space1<$base0<A>>
        where
            A: FloatNum,
        {
            /// Scalar type of laplacian matrix
            type NumType = A;
            /// Laplacian `L`
            ///
            /// ```text
            /// L_pinv @ L = I_pinv
            /// ```
            ///
            /// # Arguments
            ///
            /// * `axis` - usize
            fn laplacian(&self, _axis: usize) -> Array2<A> {
                self.base0.laplacian()
            }

            /// Pseudoinverse matrix `L_pinv` of Laplacian
            ///
            /// Returns (`L_pinv`, `I_pinv`)
            ///
            /// ```text
            /// L_pinv @ L = I_pinv
            /// ```
            ///
            /// # Arguments
            ///
            /// * `axis` - usize
            fn laplacian_pinv(&self, _axis: usize) -> (Array2<A>, Array2<A>) {
                self.base0.laplacian_pinv()
            }
        }

        impl<A, T> BaseSpaceGradient<A, T, 1> for Space1<$base0<A>>
        where
            A: FloatNum + ScalarNum,
            T: ScalarNum
                + From<A>
                + Add<A, Output = T>
                + Mul<A, Output = T>
                + Div<A, Output = T>
                + Sub<A, Output = T>
                + Add<$s, Output = T>
                + Mul<$s, Output = T>
                + Div<$s, Output = T>
                + Sub<$s, Output = T>,
        {
            /// Take gradient. Optional: Rescale result by a constant.
            ///
            /// # Arguments
            ///
            /// * `input` - *ndarray* with num type of spectral space
            /// * `deriv` - [usize; N], derivative along each axis
            /// * `scale` - [float; N], scaling factor along each axis (default [1.;n])
            fn gradient<S>(
                &self,
                input: &ArrayBase<S, Dim<[usize; 1]>>,
                deriv: [usize; 1],
                scale: Option<[A; 1]>,
            ) -> Array<T, Dim<[usize; 1]>>
            where
                S: Data<Elem = T>,
            {
                let mut output = self.base0.gradient(input, deriv[0], 0);
                if let Some(s) = scale {
                    let sc: T = (s[0].powi(deriv[0] as i32)).into();
                    for x in output.iter_mut() {
                        *x /= sc;
                    }
                }
                output
            }
            /// Take gradient. Optional: Rescale result by a constant.
            ///
            /// # Arguments
            ///
            /// * `input` - *ndarray* with num type of spectral space
            /// * `deriv` - [usize; N], derivative along each axis
            /// * `scale` - [float; N], scaling factor along each axis (default [1.;n])
            fn gradient_par<S>(
                &self,
                input: &ArrayBase<S, Dim<[usize; 1]>>,
                deriv: [usize; 1],
                scale: Option<[A; 1]>,
            ) -> Array<T, Dim<[usize; 1]>>
            where
                S: Data<Elem = T>,
            {
                let mut output = self.base0.gradient_par(input, deriv[0], 0);
                if let Some(s) = scale {
                    let sc: T = (s[0].powi(deriv[0] as i32)).into();
                    for x in output.iter_mut() {
                        *x /= sc;
                    }
                }
                output
            }
        }
        impl<A, T> BaseSpaceFromOrtho<A, T, 1> for Space1<$base0<A>>
        where
            A: FloatNum + ScalarNum,
            T: ScalarNum
                + From<A>
                + Add<A, Output = T>
                + Mul<A, Output = T>
                + Div<A, Output = T>
                + Sub<A, Output = T>
                + Add<$s, Output = T>
                + Mul<$s, Output = T>
                + Div<$s, Output = T>
                + Sub<$s, Output = T>,
        {
            /// Transformation from composite and to orthonormal space (inplace).
            ///
            /// # Arguments
            ///
            /// * `input` - *ndarray* with num type of spectral space
            /// * `output` - *ndarray* with num type of spectral space
            fn to_ortho_inplace<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 1]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 1]>>,
            ) where
                S1: Data<Elem = T>,
                S2: Data<Elem = T> + DataMut,
            {
                self.base0.to_ortho_inplace(input, output, 0);
            }

            /// Transformation from orthonormal and to composite space (inplace).
            ///
            /// # Arguments
            ///
            /// * `input` - *ndarray* with num type of spectral space
            /// * `output` - *ndarray* with num type of spectral space
            fn from_ortho_inplace<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 1]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 1]>>,
            ) where
                S1: Data<Elem = T>,
                S2: Data<Elem = T> + DataMut,
            {
                self.base0.from_ortho_inplace(input, output, 0);
            }

            /// Transformation from composite and to orthonormal space (inplace).
            ///
            /// # Arguments
            ///
            /// * `input` - *ndarray* with num type of spectral space
            /// * `output` - *ndarray* with num type of spectral space
            fn to_ortho_inplace_par<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 1]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 1]>>,
            ) where
                S1: Data<Elem = T>,
                S2: Data<Elem = T> + DataMut,
            {
                self.base0.to_ortho_inplace_par(input, output, 0);
            }

            /// Transformation from orthonormal and to composite space (inplace).
            ///
            /// # Arguments
            ///
            /// * `input` - *ndarray* with num type of spectral space
            /// * `output` - *ndarray* with num type of spectral space
            fn from_ortho_inplace_par<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 1]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 1]>>,
            ) where
                S1: Data<Elem = T>,
                S2: Data<Elem = T> + DataMut,
            {
                self.base0.from_ortho_inplace_par(input, output, 0);
            }
        }

        impl<A> BaseSpaceTransform<A, 1> for Space1<$base0<A>>
        where
            A: FloatNum + ScalarNum,
            Complex<A>: ScalarNum,
        {
            type Physical = $p;

            type Spectral = $s;

            /// Transform physical -> spectral space (inplace)
            ///
            /// # Arguments
            ///
            /// * `input` - *ndarray* with num type of physical space
            /// * `output` - *ndarray* with num type of spectral space
            fn forward_inplace<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 1]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 1]>>,
            ) where
                S1: Data<Elem = Self::Physical>,
                S2: Data<Elem = Self::Spectral> + DataMut,
            {
                self.base0.forward_inplace(input, output, 0);
            }

            /// Transform spectral -> physical space (inplace)
            ///
            /// # Arguments
            ///
            /// * `input` - *ndarray* with num type of spectral space
            /// * `output` - *ndarray* with num type of physical space
            fn backward_inplace<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 1]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 1]>>,
            ) where
                S1: Data<Elem = Self::Spectral>,
                S2: Data<Elem = Self::Physical> + DataMut,
            {
                self.base0.backward_inplace(input, output, 0);
            }

            /// Transform physical -> spectral space (inplace)
            ///
            /// # Arguments
            ///
            /// * `input` - *ndarray* with num type of physical space
            /// * `output` - *ndarray* with num type of spectral space
            fn forward_inplace_par<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 1]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 1]>>,
            ) where
                S1: Data<Elem = Self::Physical>,
                S2: Data<Elem = Self::Spectral> + DataMut,
            {
                self.base0.forward_inplace_par(input, output, 0);
            }

            /// Transform spectral -> physical space (inplace)
            ///
            /// # Arguments
            ///
            /// * `input` - *ndarray* with num type of spectral space
            /// * `output` - *ndarray* with num type of physical space
            fn backward_inplace_par<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 1]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 1]>>,
            ) where
                S1: Data<Elem = Self::Spectral>,
                S2: Data<Elem = Self::Physical> + DataMut,
            {
                self.base0.backward_inplace_par(input, output, 0);
            }
        }
    };
}
impl_space1!(BaseR2r, A, A);
impl_space1!(BaseR2c, A, Complex<A>);
impl_space1!(BaseC2c, Complex<A>, Complex<A>);
