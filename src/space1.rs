//! # One-dmensional space
//!
//! # Example
//! Transform to chebyshev - dirichlet space
//! ```
//! use funspace::{cheb_dirichlet, Space1, BaseSpace};
//! use ndarray::prelude::*;
//! let mut space = Space1::new(&cheb_dirichlet::<f64>(5));
//! let mut v: Array1<f64> = space.ndarray_physical();
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
use crate::traits::DifferentiatePar;
use crate::traits::FromOrtho;
use crate::traits::FromOrthoPar;
use crate::traits::LaplacianInverse;
use crate::traits::Transform;
use crate::traits::TransformPar;
use crate::BaseAll;
use crate::{BaseC2c, BaseR2c, BaseR2r, FloatNum};
use ndarray::{prelude::*, Data, DataMut, ScalarOperand};
use num_complex::Complex;
use std::convert::From;

/// Create one-dimensional space
#[derive(Clone)]
pub struct Space1<B> {
    pub base0: B,
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
    ($base: ident, $p: ty, $s: ty) => {
        impl<A> BaseSpace<A, 1> for Space1<$base<A>>
        where
            A: FloatNum,
            Complex<A>: ScalarOperand,
        {
            type Physical = $p;

            type Spectral = $s;

            fn shape_physical(&self) -> [usize; 1] {
                [self.base0.len_phys()]
            }

            fn shape_spectral(&self) -> [usize; 1] {
                [self.base0.len_spec()]
            }

            fn ndarray_physical(&self) -> Array1<Self::Physical> {
                Array1::zeros([self.base0.len_phys()])
            }

            fn ndarray_spectral(&self) -> Array1<Self::Spectral> {
                Array1::zeros([self.base0.len_spec()])
            }

            fn laplace(&self, _axis: usize) -> Array2<A> {
                self.base0.laplace()
            }

            fn laplace_inv(&self, _axis: usize) -> Array2<A> {
                self.base0.laplace_inv()
            }

            fn laplace_inv_eye(&self, _axis: usize) -> Array2<A> {
                self.base0.laplace_inv_eye()
            }

            fn mass(&self, _axis: usize) -> Array2<A> {
                self.base0.mass()
            }

            fn coords(&self) -> [Array1<A>; 1] {
                [self.base0.coords().clone()]
            }

            fn coords_axis(&self, _axis: usize) -> Array1<A> {
                self.base0.coords().clone()
            }

            fn base_all(&self) -> [BaseAll<A>; 1] {
                [BaseAll::<A>::from(self.base0.clone())]
            }

            fn to_ortho<S>(
                &self,
                input: &ArrayBase<S, Dim<[usize; 1]>>,
            ) -> Array<Self::Spectral, Dim<[usize; 1]>>
            where
                S: Data<Elem = Self::Spectral>,
            {
                self.base0.to_ortho(input, 0)
            }

            fn to_ortho_inplace<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 1]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 1]>>,
            ) where
                S1: Data<Elem = Self::Spectral>,
                S2: Data<Elem = Self::Spectral> + DataMut,
            {
                self.base0.to_ortho_inplace(input, output, 0)
            }

            fn from_ortho<S>(
                &self,
                input: &ArrayBase<S, Dim<[usize; 1]>>,
            ) -> Array<Self::Spectral, Dim<[usize; 1]>>
            where
                S: Data<Elem = Self::Spectral>,
            {
                self.base0.from_ortho(input, 0)
            }

            fn from_ortho_inplace<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 1]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 1]>>,
            ) where
                S1: Data<Elem = Self::Spectral>,
                S2: Data<Elem = Self::Spectral> + DataMut,
            {
                self.base0.from_ortho_inplace(input, output, 0)
            }

            fn to_ortho_par<S>(
                &self,
                input: &ArrayBase<S, Dim<[usize; 1]>>,
            ) -> Array<Self::Spectral, Dim<[usize; 1]>>
            where
                S: Data<Elem = Self::Spectral>,
            {
                self.base0.to_ortho_par(input, 0)
            }

            fn to_ortho_inplace_par<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 1]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 1]>>,
            ) where
                S1: Data<Elem = Self::Spectral>,
                S2: Data<Elem = Self::Spectral> + DataMut,
            {
                self.base0.to_ortho_inplace_par(input, output, 0)
            }

            fn from_ortho_par<S>(
                &self,
                input: &ArrayBase<S, Dim<[usize; 1]>>,
            ) -> Array<Self::Spectral, Dim<[usize; 1]>>
            where
                S: Data<Elem = Self::Spectral>,
            {
                self.base0.from_ortho_par(input, 0)
            }

            fn from_ortho_inplace_par<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 1]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 1]>>,
            ) where
                S1: Data<Elem = Self::Spectral>,
                S2: Data<Elem = Self::Spectral> + DataMut,
            {
                self.base0.from_ortho_inplace_par(input, output, 0)
            }

            fn gradient<S>(
                &self,
                input: &ArrayBase<S, Dim<[usize; 1]>>,
                deriv: [usize; 1],
                scale: Option<[A; 1]>,
            ) -> Array<Self::Spectral, Dim<[usize; 1]>>
            where
                S: Data<Elem = Self::Spectral>,
            {
                let mut output = self.base0.differentiate(input, deriv[0], 0);
                if let Some(s) = scale {
                    let sc: Self::Spectral = s[0].powi(deriv[0] as i32).into();
                    output = output / sc;
                }
                output
            }

            fn gradient_par<S>(
                &self,
                input: &ArrayBase<S, Dim<[usize; 1]>>,
                deriv: [usize; 1],
                scale: Option<[A; 1]>,
            ) -> Array<Self::Spectral, Dim<[usize; 1]>>
            where
                S: Data<Elem = Self::Spectral>,
            {
                let mut output = self.base0.differentiate_par(input, deriv[0], 0);
                if let Some(s) = scale {
                    let sc: Self::Spectral = s[0].powi(deriv[0] as i32).into();
                    output = output / sc;
                }
                output
            }

            fn forward<S>(
                &mut self,
                input: &mut ArrayBase<S, Dim<[usize; 1]>>,
            ) -> Array<Self::Spectral, Dim<[usize; 1]>>
            where
                S: Data<Elem = Self::Physical>,
            {
                self.base0.forward(input, 0)
            }

            fn forward_inplace<S1, S2>(
                &mut self,
                input: &mut ArrayBase<S1, Dim<[usize; 1]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 1]>>,
            ) where
                S1: Data<Elem = Self::Physical>,
                S2: Data<Elem = Self::Spectral> + DataMut,
            {
                self.base0.forward_inplace(input, output, 0)
            }

            fn backward<S>(
                &mut self,
                input: &mut ArrayBase<S, Dim<[usize; 1]>>,
            ) -> Array<Self::Physical, Dim<[usize; 1]>>
            where
                S: Data<Elem = Self::Spectral>,
            {
                self.base0.backward(input, 0)
            }

            fn backward_inplace<S1, S2>(
                &mut self,
                input: &mut ArrayBase<S1, Dim<[usize; 1]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 1]>>,
            ) where
                S1: Data<Elem = Self::Spectral>,
                S2: Data<Elem = Self::Physical> + DataMut,
            {
                self.base0.backward_inplace(input, output, 0)
            }

            fn forward_par<S>(
                &mut self,
                input: &mut ArrayBase<S, Dim<[usize; 1]>>,
            ) -> Array<Self::Spectral, Dim<[usize; 1]>>
            where
                S: Data<Elem = Self::Physical>,
            {
                self.base0.forward_par(input, 0)
            }

            fn forward_inplace_par<S1, S2>(
                &mut self,
                input: &mut ArrayBase<S1, Dim<[usize; 1]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 1]>>,
            ) where
                S1: Data<Elem = Self::Physical>,
                S2: Data<Elem = Self::Spectral> + DataMut,
            {
                self.base0.forward_inplace_par(input, output, 0)
            }

            fn backward_par<S>(
                &mut self,
                input: &mut ArrayBase<S, Dim<[usize; 1]>>,
            ) -> Array<Self::Physical, Dim<[usize; 1]>>
            where
                S: Data<Elem = Self::Spectral>,
            {
                self.base0.backward_par(input, 0)
            }

            fn backward_inplace_par<S1, S2>(
                &mut self,
                input: &mut ArrayBase<S1, Dim<[usize; 1]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 1]>>,
            ) where
                S1: Data<Elem = Self::Spectral>,
                S2: Data<Elem = Self::Physical> + DataMut,
            {
                self.base0.backward_inplace_par(input, output, 0)
            }
        }
    };
}
impl_space1!(BaseR2r, A, A);
impl_space1!(BaseR2c, A, Complex<A>);
impl_space1!(BaseC2c, Complex<A>, Complex<A>);
