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
//use crate::traits::{BaseBasics, Differentiate, FromOrtho, LaplacianInverse};
use crate::space_common::SpaceCommon;
use crate::traits::BaseBasics;
use crate::traits::Differentiate;
use crate::traits::FromOrtho;
use crate::traits::LaplacianInverse;
use crate::traits::Transform;
use crate::BaseAll;
use crate::{BaseC2c, BaseR2c, BaseR2r, FloatNum};
use ndarray::{prelude::*, ScalarOperand};
use num_complex::Complex;
use std::convert::From;

/// Create one-dimensional space
#[derive(Clone)]
pub struct Space1<B> {
    pub base: B,
}

// impl<B> Space1<B> where B: Into<BaseAll<T>> {}

pub trait Space1Transform<B, F, T1, T2>: SpaceCommon<F, T1, T2, 1>
where
    B: Transform<T1, T2, Physical = T1, Spectral = T2>,
    F: FloatNum + Into<T2>,
    Complex<F>: ScalarOperand,
{
    /// Scalar type in physical space (before transform)
    //type Physical;
    /// Scalar type in spectral space (after transfrom)
    //type Spectral;
    /// Create a new space
    fn new(base: &B) -> Self;
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
}

macro_rules! impl_spacecommon_space1 {
    ($base: ident, $r: ty,$s: ty) => {
        impl<A> SpaceCommon<A, $r, $s, 1> for Space1<$base<A>>
        where
            A: FloatNum,
            Complex<A>: ScalarOperand,
        {
            type Physical = $r;

            type Spectral = $s;

            fn to_ortho(&self, input: &Array1<Self::Spectral>) -> Array1<Self::Spectral> {
                self.base.to_ortho(input, 0)
            }

            fn to_ortho_inplace(
                &self,
                input: &Array1<Self::Spectral>,
                output: &mut Array1<Self::Spectral>,
            ) {
                self.base.to_ortho_inplace(input, output, 0)
            }

            fn from_ortho(&self, input: &Array1<Self::Spectral>) -> Array1<Self::Spectral> {
                self.base.from_ortho(input, 0)
            }

            fn from_ortho_inplace(
                &self,
                input: &Array1<Self::Spectral>,
                output: &mut Array1<Self::Spectral>,
            ) {
                self.base.from_ortho_inplace(input, output, 0)
            }

            fn gradient(
                &self,
                input: &Array1<Self::Spectral>,
                deriv: [usize; 1],
                scale: Option<[A; 1]>,
            ) -> Array1<Self::Spectral> {
                let mut output = self.base.differentiate(input, deriv[0], 0);
                if let Some(s) = scale {
                    let sc: Self::Spectral = s[0].powi(deriv[0] as i32).into();
                    output = output / sc;
                }
                output
            }

            fn shape_physical(&self) -> [usize; 1] {
                [self.base.len_phys()]
            }

            fn shape_spectral(&self) -> [usize; 1] {
                [self.base.len_spec()]
            }

            fn ndarray_physical(&self) -> Array1<Self::Physical> {
                Array1::zeros([self.base.len_phys()])
            }

            fn ndarray_spectral(&self) -> Array1<Self::Spectral> {
                Array1::zeros([self.base.len_spec()])
            }

            fn laplace(&self, _axis: usize) -> Array2<A> {
                self.base.laplace()
            }

            fn laplace_inv(&self, _axis: usize) -> Array2<A> {
                self.base.laplace_inv()
            }

            fn laplace_inv_eye(&self, _axis: usize) -> Array2<A> {
                self.base.laplace_inv_eye()
            }

            fn mass(&self, _axis: usize) -> Array2<A> {
                self.base.mass()
            }

            fn coords(&self) -> [Array1<A>; 1] {
                [self.base.coords().clone()]
            }

            fn coords_axis(&self, _axis: usize) -> Array1<A> {
                self.base.coords().clone()
            }

            fn base_all(&self) -> [BaseAll<A>; 1] {
                [BaseAll::<A>::from(self.base.clone())]
            }
        }
    };
}
impl_spacecommon_space1!(BaseR2r, A, A);
impl_spacecommon_space1!(BaseR2c, A, Complex<A>);
impl_spacecommon_space1!(BaseC2c, Complex<A>, Complex<A>);

macro_rules! impl_space1transform {
    ($base: ident, $r: ty,$s: ty) => {
        impl<A> Space1Transform<$base<A>, A, $r, $s> for Space1<$base<A>>
        where
            A: FloatNum,
            Complex<A>: ScalarOperand,
        {
            //type Physical = $r;

            //type Spectral = $s;

            fn new(base: &$base<A>) -> Self {
                Self { base: base.clone() }
            }

            fn forward(&mut self, input: &mut Array1<Self::Physical>) -> Array1<Self::Spectral> {
                self.base.forward(input, 0)
            }

            fn forward_inplace(
                &mut self,
                input: &mut Array1<Self::Physical>,
                output: &mut Array1<Self::Spectral>,
            ) {
                self.base.forward_inplace(input, output, 0)
            }

            fn backward(&mut self, input: &mut Array1<Self::Spectral>) -> Array1<Self::Physical> {
                self.base.backward(input, 0)
            }

            fn backward_inplace(
                &mut self,
                input: &mut Array1<Self::Spectral>,
                output: &mut Array1<Self::Physical>,
            ) {
                self.base.backward_inplace(input, output, 0)
            }
        }
    };
}
impl_space1transform!(BaseR2r, A, A);
impl_space1transform!(BaseR2c, A, Complex<A>);
impl_space1transform!(BaseC2c, Complex<A>, Complex<A>);
