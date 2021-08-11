//! # One-dmensional space
//!
//! # Example
//! Transform to chebyshev - dirichlet space
//! ```
//! use funspace::{cheb_dirichlet, Space1, Space1Transform, SpaceCommon};
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
use crate::space_common::SpaceCommon;
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
use std::convert::From;

/// Create one-dimensional space
#[derive(Clone)]
pub struct Space1<B> {
    pub base0: B,
}

pub trait Space1Transform<A, B>: SpaceCommon<A, 1, Output = Self::Spectral>
where
    A: FloatNum,
    Complex<A>: ScalarOperand,
    B: Transform<Physical = Self::Physical, Spectral = Self::Spectral>,
{
    /// Scalar type in physical space (before transform)
    type Physical;
    /// Scalar type in spectral space (after transfrom)
    type Spectral;

    /// Create a new space
    fn new(base: &B) -> Self;

    /// Return array where size and type matches physical field
    fn ndarray_physical(&self) -> Array1<Self::Physical>;

    /// Return array where size and type matches spectral field
    fn ndarray_spectral(&self) -> Array1<Self::Spectral>;

    /// Transform physical -> spectral space
    fn forward(&mut self, input: &mut Array1<Self::Physical>) -> Array1<Self::Spectral>;

    /// Transform physical -> spectral space (inplace)
    fn forward_inplace(
        &mut self,
        input: &mut Array1<Self::Physical>,
        output: &mut Array1<Self::Spectral>,
    );

    /// Transform spectral -> physical space
    fn backward(&mut self, input: &mut Array1<Self::Spectral>) -> Array1<Self::Physical>;

    /// Transform spectral -> physical space (inplace)
    fn backward_inplace(
        &mut self,
        input: &mut Array1<Self::Spectral>,
        output: &mut Array1<Self::Physical>,
    );

    /// Transform physical -> spectral space
    fn forward_par(&mut self, input: &mut Array1<Self::Physical>) -> Array1<Self::Spectral>;

    /// Transform physical -> spectral space (inplace)
    fn forward_inplace_par(
        &mut self,
        input: &mut Array1<Self::Physical>,
        output: &mut Array1<Self::Spectral>,
    );

    /// Transform spectral -> physical space
    fn backward_par(&mut self, input: &mut Array1<Self::Spectral>) -> Array1<Self::Physical>;

    /// Transform spectral -> physical space (inplace)
    fn backward_inplace_par(
        &mut self,
        input: &mut Array1<Self::Spectral>,
        output: &mut Array1<Self::Physical>,
    );
}

macro_rules! impl_spacecommon_space1 {
    ($base: ident, $s: ty) => {
        impl<A> SpaceCommon<A, 1> for Space1<$base<A>>
        where
            A: FloatNum,
            Complex<A>: ScalarOperand,
        {
            type Output = $s;

            fn to_ortho(&self, input: &Array1<Self::Output>) -> Array1<Self::Output> {
                self.base0.to_ortho(input, 0)
            }

            fn to_ortho_inplace(
                &self,
                input: &Array1<Self::Output>,
                output: &mut Array1<Self::Output>,
            ) {
                self.base0.to_ortho_inplace(input, output, 0)
            }

            fn from_ortho(&self, input: &Array1<Self::Output>) -> Array1<Self::Output> {
                self.base0.from_ortho(input, 0)
            }

            fn from_ortho_inplace(
                &self,
                input: &Array1<Self::Output>,
                output: &mut Array1<Self::Output>,
            ) {
                self.base0.from_ortho_inplace(input, output, 0)
            }

            fn to_ortho_par(&self, input: &Array1<Self::Output>) -> Array1<Self::Output> {
                self.base0.to_ortho_par(input, 0)
            }

            fn to_ortho_inplace_par(
                &self,
                input: &Array1<Self::Output>,
                output: &mut Array1<Self::Output>,
            ) {
                self.base0.to_ortho_inplace_par(input, output, 0)
            }

            fn from_ortho_par(&self, input: &Array1<Self::Output>) -> Array1<Self::Output> {
                self.base0.from_ortho_par(input, 0)
            }

            fn from_ortho_inplace_par(
                &self,
                input: &Array1<Self::Output>,
                output: &mut Array1<Self::Output>,
            ) {
                self.base0.from_ortho_inplace_par(input, output, 0)
            }

            fn gradient(
                &self,
                input: &Array1<Self::Output>,
                deriv: [usize; 1],
                scale: Option<[A; 1]>,
            ) -> Array1<Self::Output> {
                let mut output = self.base0.differentiate(input, deriv[0], 0);
                if let Some(s) = scale {
                    let sc: Self::Output = s[0].powi(deriv[0] as i32).into();
                    output = output / sc;
                }
                output
            }

            fn shape_physical(&self) -> [usize; 1] {
                [self.base0.len_phys()]
            }

            fn shape_spectral(&self) -> [usize; 1] {
                [self.base0.len_spec()]
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
        }
    };
}
impl_spacecommon_space1!(BaseR2r, A);
impl_spacecommon_space1!(BaseR2c, Complex<A>);
impl_spacecommon_space1!(BaseC2c, Complex<A>);

macro_rules! impl_space1transform {
    ($base: ident, $r: ty, $s: ty) => {
        impl<A> Space1Transform<A, $base<A>> for Space1<$base<A>>
        where
            A: FloatNum,
            Complex<A>: ScalarOperand,
        {
            type Physical = $r;

            type Spectral = $s;

            fn new(base: &$base<A>) -> Self {
                Self {
                    base0: base.clone(),
                }
            }

            fn ndarray_physical(&self) -> Array1<Self::Physical> {
                Array1::zeros([self.base0.len_phys()])
            }

            fn ndarray_spectral(&self) -> Array1<Self::Spectral> {
                Array1::zeros([self.base0.len_spec()])
            }

            fn forward(&mut self, input: &mut Array1<Self::Physical>) -> Array1<Self::Spectral> {
                self.base0.forward(input, 0)
            }

            fn forward_inplace(
                &mut self,
                input: &mut Array1<Self::Physical>,
                output: &mut Array1<Self::Spectral>,
            ) {
                self.base0.forward_inplace(input, output, 0)
            }

            fn backward(&mut self, input: &mut Array1<Self::Spectral>) -> Array1<Self::Physical> {
                self.base0.backward(input, 0)
            }

            fn backward_inplace(
                &mut self,
                input: &mut Array1<Self::Spectral>,
                output: &mut Array1<Self::Physical>,
            ) {
                self.base0.backward_inplace(input, output, 0)
            }

            fn forward_par(
                &mut self,
                input: &mut Array1<Self::Physical>,
            ) -> Array1<Self::Spectral> {
                self.base0.forward_par(input, 0)
            }

            fn forward_inplace_par(
                &mut self,
                input: &mut Array1<Self::Physical>,
                output: &mut Array1<Self::Spectral>,
            ) {
                self.base0.forward_inplace_par(input, output, 0)
            }

            fn backward_par(
                &mut self,
                input: &mut Array1<Self::Spectral>,
            ) -> Array1<Self::Physical> {
                self.base0.backward_par(input, 0)
            }

            fn backward_inplace_par(
                &mut self,
                input: &mut Array1<Self::Spectral>,
                output: &mut Array1<Self::Physical>,
            ) {
                self.base0.backward_inplace_par(input, output, 0)
            }
        }
    };
}
impl_space1transform!(BaseR2r, A, A);
impl_space1transform!(BaseR2c, A, Complex<A>);
impl_space1transform!(BaseC2c, Complex<A>, Complex<A>);
