//! # Three-dmensional space
//!
//! # Example
//! Space3: Real - Real then Real - real then real - complex transforms
//! ```
//! use funspace::{cheb_dirichlet, fourier_r2c , Space3};
//! use funspace::space::traits::BaseSpaceTransform;
//! use ndarray::prelude::*;
//! let mut space = Space3::new(
//!    &fourier_r2c::<f64>(5), &cheb_dirichlet::<f64>(5), &cheb_dirichlet::<f64>(5)
//! );
//! let mut v: Array3<f64> = space.ndarray_physical();
//! v += 1.;
//! let vhat = space.forward(&v);
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
pub struct Space3<B0, B1, B2> {
    // IntermediateB -> Spectral
    pub base0: B0,
    // IntermediateA -> IntermediateB
    pub base1: B1,
    // Phsical -> IntermediateA
    pub base2: B2,
}

impl<B0, B1, B2> Space3<B0, B1, B2>
where
    B0: Clone,
    B1: Clone,
    B2: Clone,
{
    /// Create a new space
    pub fn new(base0: &B0, base1: &B1, base2: &B2) -> Self {
        Self {
            base0: base0.clone(),
            base1: base1.clone(),
            base2: base2.clone(),
        }
    }
}

macro_rules! impl_space3 {
    ($space: ident, $base0: ident, $base1: ident, $base2: ident, $p: ty, $s: ty) => {
        impl<A> BaseSpaceSize<3> for $space<$base0<A>, $base1<A>, $base2<A>>
        where
            A: FloatNum,
        {
            fn shape_physical(&self) -> [usize; 3] {
                [
                    self.base0.len_phys(),
                    self.base1.len_phys(),
                    self.base2.len_phys(),
                ]
            }

            fn shape_spectral(&self) -> [usize; 3] {
                [
                    self.base0.len_spec(),
                    self.base1.len_spec(),
                    self.base2.len_spec(),
                ]
            }

            fn shape_spectral_ortho(&self) -> [usize; 3] {
                [
                    self.base0.len_orth(),
                    self.base1.len_orth(),
                    self.base2.len_orth(),
                ]
            }

            fn ndarray_from_shape<T: Clone + Zero>(&self, shape: [usize; 3]) -> Array3<T> {
                Array3::zeros(shape)
            }
        }

        impl<A> BaseSpaceElements<3> for $space<$base0<A>, $base1<A>, $base2<A>>
        where
            A: FloatNum,
        {
            type RealNum = A;

            /// Array of coordinates
            fn coords(&self) -> [Array1<A>; 3] {
                [
                    self.coords_axis(0),
                    self.coords_axis(1),
                    self.coords_axis(2),
                ]
            }

            /// Coordinates of grid points (in physical space)
            ///
            /// # Arguments
            ///
            /// * `axis` - usize
            fn coords_axis(&self, axis: usize) -> Array1<A> {
                assert!(axis <= 2);
                if axis == 0 {
                    self.base0.coords().into()
                } else if axis == 1 {
                    self.base1.coords().into()
                } else {
                    self.base2.coords().into()
                }
            }

            /// Return base key
            fn base_kind(&self, axis: usize) -> BaseKind {
                assert!(axis <= 2);
                if axis == 0 {
                    self.base0.base_kind()
                } else if axis == 1 {
                    self.base1.base_kind()
                } else {
                    self.base2.base_kind()
                }
            }

            /// Return transform kind
            fn transform_kind(&self, axis: usize) -> TransformKind {
                assert!(axis <= 2);
                if axis == 0 {
                    self.base0.transform_kind()
                } else if axis == 1 {
                    self.base1.transform_kind()
                } else {
                    self.base2.transform_kind()
                }
            }
        }

        impl<A> BaseSpaceMatOpStencil for $space<$base0<A>, $base1<A>, $base2<A>>
        where
            A: FloatNum,
        {
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
            fn stencil(&self, axis: usize) -> Array2<A> {
                assert!(axis <= 2);
                if axis == 0 {
                    self.base0.stencil()
                } else if axis == 1 {
                    self.base1.stencil()
                } else {
                    self.base2.stencil()
                }
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
            fn stencil_inv(&self, axis: usize) -> Array2<A> {
                assert!(axis <= 2);
                if axis == 0 {
                    self.base0.stencil_inv()
                } else if axis == 1 {
                    self.base1.stencil_inv()
                } else {
                    self.base2.stencil_inv()
                }
            }
        }

        impl<A> BaseSpaceMatOpLaplacian for $space<$base0<A>, $base1<A>, $base2<A>>
        where
            A: FloatNum,
        {
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
            fn laplacian(&self, axis: usize) -> Array2<A> {
                assert!(axis <= 2);
                if axis == 0 {
                    self.base0.laplacian()
                } else if axis == 1 {
                    self.base1.laplacian()
                } else {
                    self.base2.laplacian()
                }
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
            fn laplacian_pinv(&self, axis: usize) -> (Array2<A>, Array2<A>) {
                assert!(axis <= 2);
                if axis == 0 {
                    self.base0.laplacian_pinv()
                } else if axis == 1 {
                    self.base1.laplacian_pinv()
                } else {
                    self.base2.laplacian_pinv()
                }
            }
        }

        impl<A, T> BaseSpaceGradient<A, T, 3> for $space<$base0<A>, $base1<A>, $base2<A>>
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
                input: &ArrayBase<S, Dim<[usize; 3]>>,
                deriv: [usize; 3],
                scale: Option<[A; 3]>,
            ) -> Array<T, Dim<[usize; 3]>>
            where
                S: Data<Elem = T>,
            {
                let buffer1 = self.base0.gradient(input, deriv[0], 0);
                let buffer2 = self.base1.gradient(&buffer1, deriv[1], 1);
                let mut output = self.base1.gradient(&buffer2, deriv[2], 2);
                if let Some(s) = scale {
                    let sc: T = (s[0].powi(deriv[0] as i32)
                        * s[1].powi(deriv[1] as i32)
                        * s[2].powi(deriv[2] as i32))
                    .into();
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
                input: &ArrayBase<S, Dim<[usize; 3]>>,
                deriv: [usize; 3],
                scale: Option<[A; 3]>,
            ) -> Array<T, Dim<[usize; 3]>>
            where
                S: Data<Elem = T>,
            {
                let buffer1 = self.base0.gradient_par(input, deriv[0], 0);
                let buffer2 = self.base1.gradient_par(&buffer1, deriv[1], 1);
                let mut output = self.base1.gradient_par(&buffer2, deriv[2], 2);
                if let Some(s) = scale {
                    let sc: T = (s[0].powi(deriv[0] as i32)
                        * s[1].powi(deriv[1] as i32)
                        * s[2].powi(deriv[2] as i32))
                    .into();
                    for x in output.iter_mut() {
                        *x /= sc;
                    }
                }
                output
            }
        }
        impl<A, T> BaseSpaceFromOrtho<A, T, 3> for $space<$base0<A>, $base1<A>, $base2<A>>
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
                input: &ArrayBase<S1, Dim<[usize; 3]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 3]>>,
            ) where
                S1: Data<Elem = T>,
                S2: Data<Elem = T> + DataMut,
            {
                let buffer1 = self.base0.to_ortho(input, 0);
                let buffer2 = self.base1.to_ortho(&buffer1, 1);
                self.base2.to_ortho_inplace(&buffer2, output, 2);
            }

            /// Transformation from orthonormal and to composite space (inplace).
            ///
            /// # Arguments
            ///
            /// * `input` - *ndarray* with num type of spectral space
            /// * `output` - *ndarray* with num type of spectral space
            fn from_ortho_inplace<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 3]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 3]>>,
            ) where
                S1: Data<Elem = T>,
                S2: Data<Elem = T> + DataMut,
            {
                let buffer1 = self.base0.from_ortho(input, 0);
                let buffer2 = self.base1.from_ortho(&buffer1, 1);
                self.base2.from_ortho_inplace(&buffer2, output, 2);
            }

            /// Transformation from composite and to orthonormal space (inplace).
            ///
            /// # Arguments
            ///
            /// * `input` - *ndarray* with num type of spectral space
            /// * `output` - *ndarray* with num type of spectral space
            fn to_ortho_inplace_par<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 3]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 3]>>,
            ) where
                S1: Data<Elem = T>,
                S2: Data<Elem = T> + DataMut,
            {
                let buffer1 = self.base0.to_ortho_par(input, 0);
                let buffer2 = self.base1.to_ortho_par(&buffer1, 1);
                self.base2.to_ortho_inplace_par(&buffer2, output, 2);
            }

            /// Transformation from orthonormal and to composite space (inplace).
            ///
            /// # Arguments
            ///
            /// * `input` - *ndarray* with num type of spectral space
            /// * `output` - *ndarray* with num type of spectral space
            fn from_ortho_inplace_par<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 3]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 3]>>,
            ) where
                S1: Data<Elem = T>,
                S2: Data<Elem = T> + DataMut,
            {
                let buffer1 = self.base0.from_ortho_par(input, 0);
                let buffer2 = self.base1.from_ortho_par(&buffer1, 1);
                self.base2.from_ortho_inplace_par(&buffer2, output, 2);
            }
        }

        impl<A> BaseSpaceTransform<A, 3> for $space<$base0<A>, $base1<A>, $base2<A>>
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
                input: &ArrayBase<S1, Dim<[usize; 3]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 3]>>,
            ) where
                S1: Data<Elem = Self::Physical>,
                S2: Data<Elem = Self::Spectral> + DataMut,
            {
                let buffer1 = self.base2.forward(input, 2);
                let buffer2 = self.base1.forward(&buffer1, 1);
                self.base0.forward_inplace(&buffer2, output, 0);
            }

            /// Transform spectral -> physical space (inplace)
            ///
            /// # Arguments
            ///
            /// * `input` - *ndarray* with num type of spectral space
            /// * `output` - *ndarray* with num type of physical space
            fn backward_inplace<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 3]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 3]>>,
            ) where
                S1: Data<Elem = Self::Spectral>,
                S2: Data<Elem = Self::Physical> + DataMut,
            {
                let buffer1 = self.base0.backward(input, 0);
                let buffer2 = self.base1.backward(&buffer1, 1);
                self.base2.backward_inplace(&buffer2, output, 2);
            }

            /// Transform physical -> spectral space (inplace)
            ///
            /// # Arguments
            ///
            /// * `input` - *ndarray* with num type of physical space
            /// * `output` - *ndarray* with num type of spectral space
            fn forward_inplace_par<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 3]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 3]>>,
            ) where
                S1: Data<Elem = Self::Physical>,
                S2: Data<Elem = Self::Spectral> + DataMut,
            {
                let buffer1 = self.base2.forward_par(input, 2);
                let buffer2 = self.base1.forward_par(&buffer1, 1);
                self.base0.forward_inplace_par(&buffer2, output, 0);
            }

            /// Transform spectral -> physical space (inplace)
            ///
            /// # Arguments
            ///
            /// * `input` - *ndarray* with num type of spectral space
            /// * `output` - *ndarray* with num type of physical space
            fn backward_inplace_par<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 3]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 3]>>,
            ) where
                S1: Data<Elem = Self::Spectral>,
                S2: Data<Elem = Self::Physical> + DataMut,
            {
                let buffer1 = self.base0.backward_par(input, 0);
                let buffer2 = self.base1.backward_par(&buffer1, 1);
                self.base2.backward_inplace_par(&buffer2, output, 2);
            }
        }
    };
}
impl_space3!(Space3, BaseR2r, BaseR2r, BaseR2r, A, A);
impl_space3!(Space3, BaseR2c, BaseR2r, BaseR2r, A, Complex<A>);
impl_space3!(Space3, BaseC2c, BaseR2c, BaseR2r, A, Complex<A>);
impl_space3!(Space3, BaseC2c, BaseC2c, BaseR2c, A, Complex<A>);
impl_space3!(Space3, BaseC2c, BaseC2c, BaseC2c, Complex<A>, Complex<A>);
