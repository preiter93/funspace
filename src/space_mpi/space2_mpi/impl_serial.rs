//! # Implement traits for serial function space
//!
//! Basically copy/paste of ``Space2`` implementation with an
//! additional lifetime parameter
use super::Space2Mpi;
use crate::enums::{BaseKind, TransformKind};
use crate::space::traits::{
    BaseSpaceElements, BaseSpaceFromOrtho, BaseSpaceGradient, BaseSpaceMatOpLaplacian,
    BaseSpaceMatOpStencil, BaseSpaceSize, BaseSpaceTransform,
};
use crate::traits::{
    BaseElements, BaseFromOrtho, BaseGradient, BaseMatOpLaplacian, BaseMatOpStencil, BaseSize,
    BaseTransform,
};
use crate::{BaseC2c, BaseR2c, BaseR2r, FloatNum, ScalarNum};
use ndarray::{Array, Array1, Array2, ArrayBase, Data, DataMut, Dim};
use num_complex::Complex;
use num_traits::Zero;
use std::ops::{Add, Div, Mul, Sub};

macro_rules! impl_space2 {
    ($space: ident, $base0: ident, $base1: ident, $p: ty, $s: ty) => {
        impl<A> BaseSpaceSize<2> for $space<'_, $base0<A>, $base1<A>>
        where
            A: FloatNum,
        {
            fn shape_physical(&self) -> [usize; 2] {
                [self.base0.len_phys(), self.base1.len_phys()]
            }

            fn shape_spectral(&self) -> [usize; 2] {
                [self.base0.len_spec(), self.base1.len_spec()]
            }

            fn shape_spectral_ortho(&self) -> [usize; 2] {
                [self.base0.len_orth(), self.base1.len_orth()]
            }

            fn ndarray_from_shape<T: Clone + Zero>(&self, shape: [usize; 2]) -> Array2<T> {
                Array2::zeros(shape)
            }
        }

        impl<A> BaseSpaceElements<2> for $space<'_, $base0<A>, $base1<A>>
        where
            A: FloatNum,
        {
            type RealNum = A;

            /// Array of coordinates
            fn coords(&self) -> [Array1<Self::RealNum>; 2] {
                [self.coords_axis(0), self.coords_axis(1)]
            }

            /// Coordinates of grid points (in physical space)
            ///
            /// # Arguments
            ///
            /// * `axis` - usize
            fn coords_axis(&self, axis: usize) -> Array1<A> {
                if axis == 0 {
                    self.base0.coords().into()
                } else {
                    self.base1.coords().into()
                }
            }

            /// Return base key
            fn base_kind(&self, axis: usize) -> BaseKind {
                if axis == 0 {
                    self.base0.base_kind()
                } else {
                    self.base1.base_kind()
                }
            }

            /// Return transform kind
            fn transform_kind(&self, axis: usize) -> TransformKind {
                if axis == 0 {
                    self.base0.transform_kind()
                } else {
                    self.base1.transform_kind()
                }
            }
        }

        impl<A> BaseSpaceMatOpStencil for $space<'_, $base0<A>, $base1<A>>
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
                if axis == 0 {
                    self.base0.stencil()
                } else {
                    self.base1.stencil()
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
                if axis == 0 {
                    self.base0.stencil_inv()
                } else {
                    self.base1.stencil_inv()
                }
            }
        }

        impl<A> BaseSpaceMatOpLaplacian for $space<'_, $base0<A>, $base1<A>>
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
                if axis == 0 {
                    self.base0.laplacian()
                } else {
                    self.base1.laplacian()
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
                if axis == 0 {
                    self.base0.laplacian_pinv()
                } else {
                    self.base1.laplacian_pinv()
                }
            }
        }

        impl<A, T> BaseSpaceGradient<A, T, 2> for $space<'_, $base0<A>, $base1<A>>
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
                input: &ArrayBase<S, Dim<[usize; 2]>>,
                deriv: [usize; 2],
                scale: Option<[A; 2]>,
            ) -> Array<T, Dim<[usize; 2]>>
            where
                S: Data<Elem = T>,
            {
                let buffer = self.base0.gradient(input, deriv[0], 0);
                let mut output = self.base1.gradient(&buffer, deriv[1], 1);
                if let Some(s) = scale {
                    let sc: T = (s[0].powi(deriv[0] as i32) * s[1].powi(deriv[1] as i32)).into();
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
                input: &ArrayBase<S, Dim<[usize; 2]>>,
                deriv: [usize; 2],
                scale: Option<[A; 2]>,
            ) -> Array<T, Dim<[usize; 2]>>
            where
                S: Data<Elem = T>,
            {
                let buffer = self.base0.gradient_par(input, deriv[0], 0);
                let mut output = self.base1.gradient_par(&buffer, deriv[1], 1);
                if let Some(s) = scale {
                    let sc: T = (s[0].powi(deriv[0] as i32) * s[1].powi(deriv[1] as i32)).into();
                    for x in output.iter_mut() {
                        *x /= sc;
                    }
                }
                output
            }
        }
        impl<A, T> BaseSpaceFromOrtho<A, T, 2> for $space<'_, $base0<A>, $base1<A>>
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
                input: &ArrayBase<S1, Dim<[usize; 2]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = T>,
                S2: Data<Elem = T> + DataMut,
            {
                let buffer = self.base0.to_ortho(input, 0);
                self.base1.to_ortho_inplace(&buffer, output, 1);
            }

            /// Transformation from orthonormal and to composite space (inplace).
            ///
            /// # Arguments
            ///
            /// * `input` - *ndarray* with num type of spectral space
            /// * `output` - *ndarray* with num type of spectral space
            fn from_ortho_inplace<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 2]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = T>,
                S2: Data<Elem = T> + DataMut,
            {
                let buffer = self.base0.from_ortho(input, 0);
                self.base1.from_ortho_inplace(&buffer, output, 1);
            }

            /// Transformation from composite and to orthonormal space (inplace).
            ///
            /// # Arguments
            ///
            /// * `input` - *ndarray* with num type of spectral space
            /// * `output` - *ndarray* with num type of spectral space
            fn to_ortho_inplace_par<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 2]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = T>,
                S2: Data<Elem = T> + DataMut,
            {
                let buffer = self.base0.to_ortho_par(input, 0);
                self.base1.to_ortho_inplace_par(&buffer, output, 1);
            }

            /// Transformation from orthonormal and to composite space (inplace).
            ///
            /// # Arguments
            ///
            /// * `input` - *ndarray* with num type of spectral space
            /// * `output` - *ndarray* with num type of spectral space
            fn from_ortho_inplace_par<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 2]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = T>,
                S2: Data<Elem = T> + DataMut,
            {
                let buffer = self.base0.from_ortho_par(input, 0);
                self.base1.from_ortho_inplace_par(&buffer, output, 1);
            }
        }

        impl<A> BaseSpaceTransform<A, 2> for $space<'_, $base0<A>, $base1<A>>
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
                input: &ArrayBase<S1, Dim<[usize; 2]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = Self::Physical>,
                S2: Data<Elem = Self::Spectral> + DataMut,
            {
                let buffer = self.base1.forward(input, 1);
                self.base0.forward_inplace(&buffer, output, 0);
            }

            /// Transform spectral -> physical space (inplace)
            ///
            /// # Arguments
            ///
            /// * `input` - *ndarray* with num type of spectral space
            /// * `output` - *ndarray* with num type of physical space
            fn backward_inplace<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 2]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = Self::Spectral>,
                S2: Data<Elem = Self::Physical> + DataMut,
            {
                let buffer = self.base0.backward(input, 0);
                self.base1.backward_inplace(&buffer, output, 1);
            }

            /// Transform physical -> spectral space (inplace)
            ///
            /// # Arguments
            ///
            /// * `input` - *ndarray* with num type of physical space
            /// * `output` - *ndarray* with num type of spectral space
            fn forward_inplace_par<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 2]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = Self::Physical>,
                S2: Data<Elem = Self::Spectral> + DataMut,
            {
                let buffer = self.base1.forward_par(input, 1);
                self.base0.forward_inplace_par(&buffer, output, 0);
            }

            /// Transform spectral -> physical space (inplace)
            ///
            /// # Arguments
            ///
            /// * `input` - *ndarray* with num type of spectral space
            /// * `output` - *ndarray* with num type of physical space
            fn backward_inplace_par<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 2]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = Self::Spectral>,
                S2: Data<Elem = Self::Physical> + DataMut,
            {
                let buffer = self.base0.backward_par(input, 0);
                self.base1.backward_inplace_par(&buffer, output, 1);
            }
        }
    };
}
impl_space2!(Space2Mpi, BaseR2r, BaseR2r, A, A);
impl_space2!(Space2Mpi, BaseR2c, BaseR2r, A, Complex<A>);
impl_space2!(Space2Mpi, BaseC2c, BaseR2c, A, Complex<A>);
impl_space2!(Space2Mpi, BaseC2c, BaseC2c, Complex<A>, Complex<A>);
