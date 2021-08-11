//! # Two-dmensional space
//!
//! # Example
//! Space2: Real - Real then Real - Complex transforms
//! ```
//! use funspace::{cheb_dirichlet, fourier_r2c , Space2, BaseSpace};
//! use ndarray::prelude::*;
//! let mut space = Space2::new(&fourier_r2c::<f64>(5), &cheb_dirichlet::<f64>(5));
//! let mut v: Array2<f64> = space.ndarray_physical();
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
use ndarray::{prelude::*, ScalarOperand};
use num_complex::Complex;

/// Create two-dimensional space
#[derive(Clone)]
pub struct Space2<B0, B1> {
    // Intermediate -> Spectral
    pub base0: B0,
    // Phsical -> Intermediate
    pub base1: B1,
}

impl<B0, B1> Space2<B0, B1>
where
    B0: Clone,
    B1: Clone,
{
    /// Create a new space
    pub fn new(base0: &B0, base1: &B1) -> Self {
        Self {
            base0: base0.clone(),
            base1: base1.clone(),
        }
    }
}

macro_rules! impl_space2 {
    ($base0: ident, $base1: ident, $p: ty, $s: ty) => {
        impl<A> BaseSpace<A, 2> for Space2<$base0<A>, $base1<A>>
        where
            A: FloatNum,
            Complex<A>: ScalarOperand,
        {
            type Physical = $p;

            type Spectral = $s;

            fn shape_physical(&self) -> [usize; 2] {
                [self.base0.len_phys(), self.base1.len_phys()]
            }

            fn shape_spectral(&self) -> [usize; 2] {
                [self.base0.len_spec(), self.base1.len_spec()]
            }

            fn ndarray_physical(&self) -> Array2<Self::Physical> {
                let shape = [self.base0.len_phys(), self.base1.len_phys()];
                Array2::zeros(shape)
            }

            fn ndarray_spectral(&self) -> Array2<Self::Spectral> {
                let shape = [self.base0.len_spec(), self.base1.len_spec()];
                Array2::zeros(shape)
            }

            fn laplace(&self, axis: usize) -> Array2<A> {
                if axis == 0 {
                    self.base0.laplace()
                } else {
                    self.base1.laplace()
                }
            }

            fn laplace_inv(&self, axis: usize) -> Array2<A> {
                if axis == 0 {
                    self.base0.laplace_inv()
                } else {
                    self.base1.laplace_inv()
                }
            }

            fn laplace_inv_eye(&self, axis: usize) -> Array2<A> {
                if axis == 0 {
                    self.base0.laplace_inv_eye()
                } else {
                    self.base1.laplace_inv_eye()
                }
            }

            fn mass(&self, axis: usize) -> Array2<A> {
                if axis == 0 {
                    self.base0.mass()
                } else {
                    self.base1.mass()
                }
            }

            fn coords(&self) -> [Array1<A>; 2] {
                [self.base0.coords().clone(), self.base1.coords().clone()]
            }

            fn coords_axis(&self, axis: usize) -> Array1<A> {
                if axis == 0 {
                    self.base0.coords().clone()
                } else {
                    self.base1.coords().clone()
                }
            }

            fn base_all(&self) -> [BaseAll<A>; 2] {
                [
                    BaseAll::<A>::from(self.base0.clone()),
                    BaseAll::<A>::from(self.base1.clone()),
                ]
            }

            fn to_ortho(&self, input: &Array2<Self::Spectral>) -> Array2<Self::Spectral> {
                let buffer = self.base0.to_ortho(input, 0);
                self.base1.to_ortho(&buffer, 1)
            }

            fn to_ortho_inplace(
                &self,
                input: &Array2<Self::Spectral>,
                output: &mut Array2<Self::Spectral>,
            ) {
                let buffer = self.base0.to_ortho(input, 0);
                self.base1.to_ortho_inplace(&buffer, output, 1);
            }

            fn from_ortho(&self, input: &Array2<Self::Spectral>) -> Array2<Self::Spectral> {
                let buffer = self.base0.from_ortho(input, 0);
                self.base1.from_ortho(&buffer, 1)
            }

            fn from_ortho_inplace(
                &self,
                input: &Array2<Self::Spectral>,
                output: &mut Array2<Self::Spectral>,
            ) {
                let buffer = self.base0.from_ortho(input, 0);
                self.base1.from_ortho_inplace(&buffer, output, 1);
            }

            fn to_ortho_par(&self, input: &Array2<Self::Spectral>) -> Array2<Self::Spectral> {
                let buffer = self.base0.to_ortho_par(input, 0);
                self.base1.to_ortho_par(&buffer, 1)
            }

            fn to_ortho_inplace_par(
                &self,
                input: &Array2<Self::Spectral>,
                output: &mut Array2<Self::Spectral>,
            ) {
                let buffer = self.base0.to_ortho_par(input, 0);
                self.base1.to_ortho_inplace_par(&buffer, output, 1);
            }

            fn from_ortho_par(&self, input: &Array2<Self::Spectral>) -> Array2<Self::Spectral> {
                let buffer = self.base0.from_ortho_par(input, 0);
                self.base1.from_ortho_par(&buffer, 1)
            }

            fn from_ortho_inplace_par(
                &self,
                input: &Array2<Self::Spectral>,
                output: &mut Array2<Self::Spectral>,
            ) {
                let buffer = self.base0.from_ortho_par(input, 0);
                self.base1.from_ortho_inplace_par(&buffer, output, 1);
            }

            fn gradient(
                &self,
                input: &Array2<Self::Spectral>,
                deriv: [usize; 2],
                scale: Option<[A; 2]>,
            ) -> Array2<Self::Spectral> {
                let buffer = self.base0.differentiate(input, deriv[0], 0);
                let mut output = self.base1.differentiate(&buffer, deriv[1], 1);
                if let Some(s) = scale {
                    let sc: Self::Spectral =
                        (s[0].powi(deriv[0] as i32) + s[1].powi(deriv[1] as i32)).into();
                    output = output / sc;
                }
                output
            }

            fn forward<S>(
                &mut self,
                input: &mut ArrayBase<S, Dim<[usize; 2]>>,
            ) -> Array<Self::Spectral, Dim<[usize; 2]>>
            where
                S: ndarray::Data<Elem = Self::Physical>,
            {
                let mut buffer = self.base1.forward(input, 1);
                self.base0.forward(&mut buffer, 0)
            }

            fn forward_inplace<S1, S2>(
                &mut self,
                input: &mut ArrayBase<S1, Dim<[usize; 2]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: ndarray::Data<Elem = Self::Physical>,
                S2: ndarray::Data<Elem = Self::Spectral> + ndarray::DataMut,
            {
                let mut buffer = self.base1.forward(input, 1);
                self.base0.forward_inplace(&mut buffer, output, 0);
            }

            fn backward<S>(
                &mut self,
                input: &mut ArrayBase<S, Dim<[usize; 2]>>,
            ) -> Array<Self::Physical, Dim<[usize; 2]>>
            where
                S: ndarray::Data<Elem = Self::Spectral>,
            {
                let mut buffer = self.base0.backward(input, 0);
                self.base1.backward(&mut buffer, 1)
            }

            fn backward_inplace<S1, S2>(
                &mut self,
                input: &mut ArrayBase<S1, Dim<[usize; 2]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: ndarray::Data<Elem = Self::Spectral>,
                S2: ndarray::Data<Elem = Self::Physical> + ndarray::DataMut,
            {
                let mut buffer = self.base0.backward(input, 0);
                self.base1.backward_inplace(&mut buffer, output, 1);
            }

            fn forward_par<S>(
                &mut self,
                input: &mut ArrayBase<S, Dim<[usize; 2]>>,
            ) -> Array<Self::Spectral, Dim<[usize; 2]>>
            where
                S: ndarray::Data<Elem = Self::Physical>,
            {
                let mut buffer = self.base1.forward_par(input, 1);
                self.base0.forward_par(&mut buffer, 0)
            }

            fn forward_inplace_par<S1, S2>(
                &mut self,
                input: &mut ArrayBase<S1, Dim<[usize; 2]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: ndarray::Data<Elem = Self::Physical>,
                S2: ndarray::Data<Elem = Self::Spectral> + ndarray::DataMut,
            {
                let mut buffer = self.base1.forward_par(input, 1);
                self.base0.forward_inplace_par(&mut buffer, output, 0);
            }

            fn backward_par<S>(
                &mut self,
                input: &mut ArrayBase<S, Dim<[usize; 2]>>,
            ) -> Array<Self::Physical, Dim<[usize; 2]>>
            where
                S: ndarray::Data<Elem = Self::Spectral>,
            {
                let mut buffer = self.base0.backward_par(input, 0);
                self.base1.backward_par(&mut buffer, 1)
            }

            fn backward_inplace_par<S1, S2>(
                &mut self,
                input: &mut ArrayBase<S1, Dim<[usize; 2]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: ndarray::Data<Elem = Self::Spectral>,
                S2: ndarray::Data<Elem = Self::Physical> + ndarray::DataMut,
            {
                let mut buffer = self.base0.backward_par(input, 0);
                self.base1.backward_inplace_par(&mut buffer, output, 1);
            }
        }
    };
}
impl_space2!(BaseR2r, BaseR2r, A, A);
impl_space2!(BaseR2c, BaseR2r, A, Complex<A>);
impl_space2!(BaseC2c, BaseR2c, A, Complex<A>);
impl_space2!(BaseC2c, BaseC2c, Complex<A>, Complex<A>);
