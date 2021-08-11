//! # Three-dmensional space
//!
//! # Example
//! Space3: Real - Real then Real - real then real - complex transforms
//! ```
//! use funspace::{cheb_dirichlet, fourier_r2c , Space3, BaseSpace};
//! use ndarray::prelude::*;
//! let mut space = Space3::new(
//!    &fourier_r2c::<f64>(5), &cheb_dirichlet::<f64>(5), &cheb_dirichlet::<f64>(5)
//! );
//! let mut v: Array3<f64> = space.ndarray_physical();
//! v += 1.;
//! let mut vhat = space.forward(&mut v);
//! println!("{:?}", vhat);
//! // Not how the cheb dirichlet base imposes dirichlet conditions on
//! // the array: the first and last point are now zero,
//! let v = space.backward(&mut vhat);
//! println!("{:?}", v);
//! ```
#![allow(clippy::module_name_repetitions)]
use crate::space_traits::BaseSpace;
use crate::traits::Basics;
use crate::traits::Differentiate;
use crate::traits::FromOrtho;
use crate::traits::FromOrthoPar;
use crate::traits::LaplacianInverse;
use crate::traits::Transform;
use crate::traits::TransformPar;
use crate::BaseAll;
use crate::{BaseC2c, BaseR2c, BaseR2r, FloatNum};
use ndarray::{prelude::*, Data, DataMut, ScalarOperand};
use num_complex::Complex;

/// Create three-dimensional space
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
    ($base0: ident, $base1: ident, $base2: ident, $p: ty, $s: ty) => {
        impl<A> BaseSpace<A, 3> for Space3<$base0<A>, $base1<A>, $base2<A>>
        where
            A: FloatNum,
            Complex<A>: ScalarOperand,
        {
            type Physical = $p;

            type Spectral = $s;

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

            fn ndarray_physical(&self) -> Array3<Self::Physical> {
                Array3::zeros(self.shape_physical())
            }

            fn ndarray_spectral(&self) -> Array3<Self::Spectral> {
                Array3::zeros(self.shape_spectral())
            }

            fn laplace(&self, axis: usize) -> Array2<A> {
                if axis == 0 {
                    self.base0.laplace()
                } else if axis == 1 {
                    self.base1.laplace()
                } else {
                    self.base2.laplace()
                }
            }

            fn laplace_inv(&self, axis: usize) -> Array2<A> {
                if axis == 0 {
                    self.base0.laplace_inv()
                } else if axis == 1 {
                    self.base1.laplace_inv()
                } else {
                    self.base2.laplace_inv()
                }
            }

            fn laplace_inv_eye(&self, axis: usize) -> Array2<A> {
                if axis == 0 {
                    self.base0.laplace_inv_eye()
                } else if axis == 1 {
                    self.base1.laplace_inv_eye()
                } else {
                    self.base2.laplace_inv_eye()
                }
            }

            /// Return mass matrix
            fn mass(&self, axis: usize) -> Array2<A> {
                if axis == 0 {
                    self.base0.mass()
                } else if axis == 1 {
                    self.base1.mass()
                } else {
                    self.base2.mass()
                }
            }

            fn coords(&self) -> [Array1<A>; 3] {
                [
                    self.coords_axis(0),
                    self.coords_axis(1),
                    self.coords_axis(2),
                ]
            }

            fn coords_axis(&self, axis: usize) -> Array1<A> {
                if axis == 0 {
                    self.base0.coords().clone()
                } else if axis == 1 {
                    self.base1.coords().clone()
                } else {
                    self.base2.coords().clone()
                }
            }

            fn base_all(&self) -> [BaseAll<A>; 3] {
                [
                    BaseAll::<A>::from(self.base0.clone()),
                    BaseAll::<A>::from(self.base1.clone()),
                    BaseAll::<A>::from(self.base2.clone()),
                ]
            }

            fn to_ortho<S>(
                &self,
                input: &ArrayBase<S, Dim<[usize; 3]>>,
            ) -> Array<Self::Spectral, Dim<[usize; 3]>>
            where
                S: Data<Elem = Self::Spectral>,
            {
                let buffer1 = self.base0.to_ortho(input, 0);
                let buffer2 = self.base1.to_ortho(&buffer1, 1);
                self.base2.to_ortho(&buffer2, 2)
            }

            fn to_ortho_inplace<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 3]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 3]>>,
            ) where
                S1: Data<Elem = Self::Spectral>,
                S2: Data<Elem = Self::Spectral> + DataMut,
            {
                let buffer1 = self.base0.to_ortho(input, 0);
                let buffer2 = self.base1.to_ortho(&buffer1, 1);
                self.base2.to_ortho_inplace(&buffer2, output, 2);
            }

            fn from_ortho<S>(
                &self,
                input: &ArrayBase<S, Dim<[usize; 3]>>,
            ) -> Array<Self::Spectral, Dim<[usize; 3]>>
            where
                S: Data<Elem = Self::Spectral>,
            {
                let buffer1 = self.base0.from_ortho(input, 0);
                let buffer2 = self.base1.from_ortho(&buffer1, 1);
                self.base2.from_ortho(&buffer2, 2)
            }

            fn from_ortho_inplace<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 3]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 3]>>,
            ) where
                S1: Data<Elem = Self::Spectral>,
                S2: Data<Elem = Self::Spectral> + DataMut,
            {
                let buffer1 = self.base0.from_ortho(input, 0);
                let buffer2 = self.base1.from_ortho(&buffer1, 1);
                self.base2.from_ortho_inplace(&buffer2, output, 2);
            }

            fn to_ortho_par<S>(
                &self,
                input: &ArrayBase<S, Dim<[usize; 3]>>,
            ) -> Array<Self::Spectral, Dim<[usize; 3]>>
            where
                S: Data<Elem = Self::Spectral>,
            {
                let buffer1 = self.base0.to_ortho_par(input, 0);
                let buffer2 = self.base1.to_ortho_par(&buffer1, 1);
                self.base2.to_ortho_par(&buffer2, 2)
            }

            fn to_ortho_inplace_par<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 3]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 3]>>,
            ) where
                S1: Data<Elem = Self::Spectral>,
                S2: Data<Elem = Self::Spectral> + DataMut,
            {
                let buffer1 = self.base0.to_ortho_par(input, 0);
                let buffer2 = self.base1.to_ortho_par(&buffer1, 1);
                self.base2.to_ortho_inplace_par(&buffer2, output, 2);
            }

            fn from_ortho_par<S>(
                &self,
                input: &ArrayBase<S, Dim<[usize; 3]>>,
            ) -> Array<Self::Spectral, Dim<[usize; 3]>>
            where
                S: Data<Elem = Self::Spectral>,
            {
                let buffer1 = self.base0.from_ortho_par(input, 0);
                let buffer2 = self.base1.from_ortho_par(&buffer1, 1);
                self.base2.from_ortho_par(&buffer2, 2)
            }

            fn from_ortho_inplace_par<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 3]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 3]>>,
            ) where
                S1: Data<Elem = Self::Spectral>,
                S2: Data<Elem = Self::Spectral> + DataMut,
            {
                let buffer1 = self.base0.from_ortho_par(input, 0);
                let buffer2 = self.base1.from_ortho_par(&buffer1, 1);
                self.base2.from_ortho_inplace_par(&buffer2, output, 2);
            }

            fn gradient<S>(
                &self,
                input: &ArrayBase<S, Dim<[usize; 3]>>,
                deriv: [usize; 3],
                scale: Option<[A; 3]>,
            ) -> Array<Self::Spectral, Dim<[usize; 3]>>
            where
                S: Data<Elem = Self::Spectral>,
            {
                let buffer1 = self.base0.differentiate(input, deriv[0], 0);
                let buffer2 = self.base1.differentiate(&buffer1, deriv[1], 1);
                let mut output = self.base1.differentiate(&buffer2, deriv[2], 2);
                if let Some(s) = scale {
                    let sc: Self::Spectral = (s[0].powi(deriv[0] as i32)
                        + s[1].powi(deriv[1] as i32)
                        + s[2].powi(deriv[2] as i32))
                    .into();
                    output = output / sc;
                }
                output
            }

            fn forward<S>(
                &mut self,
                input: &mut ArrayBase<S, Dim<[usize; 3]>>,
            ) -> Array<Self::Spectral, Dim<[usize; 3]>>
            where
                S: Data<Elem = Self::Physical>,
            {
                let mut buffer1 = self.base2.forward(input, 2);
                let mut buffer2 = self.base1.forward(&mut buffer1, 1);
                self.base0.forward(&mut buffer2, 0)
            }

            fn forward_inplace<S1, S2>(
                &mut self,
                input: &mut ArrayBase<S1, Dim<[usize; 3]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 3]>>,
            ) where
                S1: Data<Elem = Self::Physical>,
                S2: Data<Elem = Self::Spectral> + DataMut,
            {
                let mut buffer1 = self.base2.forward(input, 2);
                let mut buffer2 = self.base1.forward(&mut buffer1, 1);
                self.base0.forward_inplace(&mut buffer2, output, 0);
            }

            fn backward<S>(
                &mut self,
                input: &mut ArrayBase<S, Dim<[usize; 3]>>,
            ) -> Array<Self::Physical, Dim<[usize; 3]>>
            where
                S: Data<Elem = Self::Spectral>,
            {
                let mut buffer1 = self.base0.backward(input, 0);
                let mut buffer2 = self.base1.backward(&mut buffer1, 1);
                self.base2.backward(&mut buffer2, 2)
            }

            fn backward_inplace<S1, S2>(
                &mut self,
                input: &mut ArrayBase<S1, Dim<[usize; 3]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 3]>>,
            ) where
                S1: Data<Elem = Self::Spectral>,
                S2: Data<Elem = Self::Physical> + DataMut,
            {
                let mut buffer1 = self.base0.backward(input, 0);
                let mut buffer2 = self.base1.backward(&mut buffer1, 1);
                self.base2.backward_inplace(&mut buffer2, output, 2);
            }

            fn forward_par<S>(
                &mut self,
                input: &mut ArrayBase<S, Dim<[usize; 3]>>,
            ) -> Array<Self::Spectral, Dim<[usize; 3]>>
            where
                S: Data<Elem = Self::Physical>,
            {
                let mut buffer1 = self.base2.forward_par(input, 2);
                let mut buffer2 = self.base1.forward_par(&mut buffer1, 1);
                self.base0.forward_par(&mut buffer2, 0)
            }

            fn forward_inplace_par<S1, S2>(
                &mut self,
                input: &mut ArrayBase<S1, Dim<[usize; 3]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 3]>>,
            ) where
                S1: Data<Elem = Self::Physical>,
                S2: Data<Elem = Self::Spectral> + DataMut,
            {
                let mut buffer1 = self.base2.forward_par(input, 2);
                let mut buffer2 = self.base1.forward_par(&mut buffer1, 1);
                self.base0.forward_inplace_par(&mut buffer2, output, 0);
            }

            fn backward_par<S>(
                &mut self,
                input: &mut ArrayBase<S, Dim<[usize; 3]>>,
            ) -> Array<Self::Physical, Dim<[usize; 3]>>
            where
                S: Data<Elem = Self::Spectral>,
            {
                let mut buffer1 = self.base0.backward_par(input, 0);
                let mut buffer2 = self.base1.backward_par(&mut buffer1, 1);
                self.base2.backward_par(&mut buffer2, 2)
            }

            fn backward_inplace_par<S1, S2>(
                &mut self,
                input: &mut ArrayBase<S1, Dim<[usize; 3]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 3]>>,
            ) where
                S1: Data<Elem = Self::Spectral>,
                S2: Data<Elem = Self::Physical> + DataMut,
            {
                let mut buffer1 = self.base0.backward_par(input, 0);
                let mut buffer2 = self.base1.backward_par(&mut buffer1, 1);
                self.base2.backward_inplace_par(&mut buffer2, output, 2);
            }
        }
    };
}
impl_space3!(BaseR2r, BaseR2r, BaseR2r, A, A);
impl_space3!(BaseR2c, BaseR2r, BaseR2r, A, Complex<A>);
impl_space3!(BaseC2c, BaseR2c, BaseR2r, A, Complex<A>);
impl_space3!(BaseC2c, BaseC2c, BaseR2c, A, Complex<A>);
impl_space3!(BaseC2c, BaseC2c, BaseC2c, Complex<A>, Complex<A>);
