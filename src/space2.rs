//! # Two-dmensional space
//!
//! # Example
//! Space2: Real - Real then Real - Complex transforms
//! ```
//! use funspace::{cheb_dirichlet, fourier_r2c , Space2, Space2Transform, SpaceCommon};
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

/// Create two-dimensional space
#[derive(Clone)]
pub struct Space2<B1, B2> {
    // Intermediate -> Spectral
    pub base0: B1,
    // Phsical -> Intermediate
    pub base1: B2,
}

pub trait Space2Transform<B1, B2, F, T1, T2, I>: SpaceCommon<F, T1, T2, 2>
where
    B1: Transform<I, T2, Physical = I, Spectral = T2>,
    B2: Transform<T1, I, Physical = T1, Spectral = I>,
    F: FloatNum + Into<T2>,
    Complex<F>: ScalarOperand,
{
    /// Scalar type in physical space (before transform)
    //type Physical;
    /// Intermediate type phsical -> intermediate -> spectral
    type Intermediate;
    /// Scalar type in spectral space (after transfrom)
    //type Spectral;
    /// Create a new space
    fn new(base0: &B1, base1: &B2) -> Self;
    /// Transform physical -> spectral space
    fn forward(&mut self, input: &mut Array2<Self::Physical>) -> Array2<Self::Spectral>;
    /// Transform physical -> spectral space (inplace)
    fn forward_inplace(
        &mut self,
        input: &mut Array2<Self::Physical>,
        output: &mut Array2<Self::Spectral>,
    );
    /// Transform spectral -> physical space
    fn backward(&mut self, input: &mut Array2<Self::Spectral>) -> Array2<Self::Physical>;
    /// Transform spectral -> physical space (inplace)
    fn backward_inplace(
        &mut self,
        input: &mut Array2<Self::Spectral>,
        output: &mut Array2<Self::Physical>,
    );
}

macro_rules! impl_spacecommon_space2 {
    ($base1: ident, $base2: ident, $r: ty,$s: ty) => {
        impl<A> SpaceCommon<A, $r, $s, 2> for Space2<$base1<A>, $base2<A>>
        where
            A: FloatNum,
            Complex<A>: ScalarOperand,
        {
            type Physical = $r;

            type Spectral = $s;

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
        }
    };
}
impl_spacecommon_space2!(BaseR2r, BaseR2r, A, A);
impl_spacecommon_space2!(BaseR2c, BaseR2r, A, Complex<A>);
impl_spacecommon_space2!(BaseC2c, BaseR2c, A, Complex<A>);
impl_spacecommon_space2!(BaseC2c, BaseC2c, Complex<A>, Complex<A>);

macro_rules! impl_space2transform {
    ($base0: ident, $base1: ident, $r: ty, $i: ty, $s: ty) => {
        impl<A> Space2Transform<$base0<A>, $base1<A>, A, $r, $s, $i>
            for Space2<$base0<A>, $base1<A>>
        where
            A: FloatNum,
            Complex<A>: ScalarOperand,
        {
            //type Physical = $r;

            type Intermediate = $i;

            //type Spectral = $s;

            fn new(base0: &$base0<A>, base1: &$base1<A>) -> Self {
                Self {
                    base0: base0.clone(),
                    base1: base1.clone(),
                }
            }

            fn forward(&mut self, input: &mut Array2<Self::Physical>) -> Array2<Self::Spectral> {
                let mut buffer = self.base1.forward(input, 1);
                self.base0.forward(&mut buffer, 0)
            }

            fn forward_inplace(
                &mut self,
                input: &mut Array2<Self::Physical>,
                output: &mut Array2<Self::Spectral>,
            ) {
                let mut buffer = self.base1.forward(input, 1);
                self.base0.forward_inplace(&mut buffer, output, 0);
            }

            fn backward(&mut self, input: &mut Array2<Self::Spectral>) -> Array2<Self::Physical> {
                let mut buffer = self.base0.backward(input, 0);
                self.base1.backward(&mut buffer, 1)
            }

            fn backward_inplace(
                &mut self,
                input: &mut Array2<Self::Spectral>,
                output: &mut Array2<Self::Physical>,
            ) {
                let mut buffer = self.base0.backward(input, 0);
                self.base1.backward_inplace(&mut buffer, output, 1);
            }
        }
    };
}
impl_space2transform!(BaseR2r, BaseR2r, A, A, A);
impl_space2transform!(BaseR2c, BaseR2r, A, A, Complex<A>);
impl_space2transform!(BaseC2c, BaseR2c, A, Complex<A>, Complex<A>);
impl_space2transform!(BaseC2c, BaseC2c, Complex<A>, Complex<A>, Complex<A>);
