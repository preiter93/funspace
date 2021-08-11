//! # Three-dmensional space
//!
//! # Example
//! Space3: Real - Real then Real - real then real - complex transforms
//! ```
//! use funspace::{cheb_dirichlet, fourier_r2c , Space3, Space3Transform, SpaceCommon};
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
use crate::space_common::SpaceCommon;
use crate::traits::{BaseBasics, Differentiate, FromOrtho, LaplacianInverse};
use crate::BaseAll;
use crate::{BaseC2c, BaseR2c, BaseR2r, FloatNum, SuperBase, Transform};
use ndarray::{prelude::*, ScalarOperand};
use num_complex::Complex;

/// Create three-dimensional space
#[derive(Clone)]
pub struct Space3<B1, B2, B3> {
    // IntermediateB -> Spectral
    pub base0: B1,
    // IntermediateA -> IntermediateB
    pub base1: B2,
    // Phsical -> IntermediateA
    pub base2: B3,
}

pub trait Space3Transform<B1, B2, B3, F, T1, T2, IA, IB>: SpaceCommon<F, T1, T2, 3>
where
    B1: SuperBase<F, IB, T2> + Transform<IB, T2, Physical = IB, Spectral = T2>,
    B2: SuperBase<F, IA, IB> + Transform<IA, IB, Physical = IA, Spectral = IB>,
    B3: SuperBase<F, T1, IA> + Transform<T1, IA, Physical = T1, Spectral = IA>,
    F: FloatNum + Into<T2>,
    Complex<F>: ScalarOperand,
{
    /// Scalar type in physical space (before transform)
    //type Physical;
    /// Intermediate type phsical -> intermediatea -> intermediateb -> spectral
    type IntermediateA;
    /// Intermediate type phsical -> intermediatea -> intermediateb -> spectral
    type IntermediateB;
    /// Scalar type in spectral space (after transfrom)
    //type Spectral;
    /// Create a new space
    fn new(base0: &B1, base1: &B2, base2: &B3) -> Self;
    /// Transform physical -> spectral space
    fn forward(&mut self, input: &mut Array3<Self::Physical>) -> Array3<Self::Spectral>;
    /// Transform physical -> spectral space (inplace)
    fn forward_inplace(
        &mut self,
        input: &mut Array3<Self::Physical>,
        output: &mut Array3<Self::Spectral>,
    );
    /// Transform spectral -> physical space
    fn backward(&mut self, input: &mut Array3<Self::Spectral>) -> Array3<Self::Physical>;
    /// Transform spectral -> physical space (inplace)
    fn backward_inplace(
        &mut self,
        input: &mut Array3<Self::Spectral>,
        output: &mut Array3<Self::Physical>,
    );
}

macro_rules! impl_spacecommon_space3 {
    ($base1: ident, $base2: ident, $base3: ident, $r: ty,$s: ty) => {
        impl<A> SpaceCommon<A, $r, $s, 3> for Space3<$base1<A>, $base2<A>, $base3<A>>
        where
            A: FloatNum,
            Complex<A>: ScalarOperand,
        {
            type Physical = $r;

            type Spectral = $s;

            fn to_ortho(&self, input: &Array3<Self::Spectral>) -> Array3<Self::Spectral> {
                let buffer1 = self.base0.to_ortho(input, 0);
                let buffer2 = self.base1.to_ortho(&buffer1, 1);
                self.base2.to_ortho(&buffer2, 2)
            }

            fn to_ortho_inplace(
                &self,
                input: &Array3<Self::Spectral>,
                output: &mut Array3<Self::Spectral>,
            ) {
                let buffer1 = self.base0.to_ortho(input, 0);
                let buffer2 = self.base1.to_ortho(&buffer1, 1);
                self.base2.to_ortho_inplace(&buffer2, output, 2);
            }

            fn from_ortho(&self, input: &Array3<Self::Spectral>) -> Array3<Self::Spectral> {
                let buffer1 = self.base0.from_ortho(input, 0);
                let buffer2 = self.base1.from_ortho(&buffer1, 1);
                self.base2.from_ortho(&buffer2, 2)
            }

            fn from_ortho_inplace(
                &self,
                input: &Array3<Self::Spectral>,
                output: &mut Array3<Self::Spectral>,
            ) {
                let buffer1 = self.base0.from_ortho(input, 0);
                let buffer2 = self.base0.from_ortho(&buffer1, 1);
                self.base2.from_ortho_inplace(&buffer2, output, 2);
            }

            fn gradient(
                &self,
                input: &Array3<Self::Spectral>,
                deriv: [usize; 3],
                scale: Option<[A; 3]>,
            ) -> Array3<Self::Spectral> {
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
        }
    };
}
impl_spacecommon_space3!(BaseR2r, BaseR2r, BaseR2r, A, A);
impl_spacecommon_space3!(BaseR2c, BaseR2r, BaseR2r, A, Complex<A>);
impl_spacecommon_space3!(BaseC2c, BaseR2c, BaseR2r, A, Complex<A>);
impl_spacecommon_space3!(BaseC2c, BaseC2c, BaseR2c, A, Complex<A>);
impl_spacecommon_space3!(BaseC2c, BaseC2c, BaseC2c, Complex<A>, Complex<A>);

macro_rules! impl_space3transform {
    ($base0: ident, $base1: ident, $base2: ident, $r: ty, $ia: ty, $ib: ty, $s: ty) => {
        impl<A> Space3Transform<$base0<A>, $base1<A>, $base2<A>, A, $r, $s, $ia, $ib>
            for Space3<$base0<A>, $base1<A>, $base2<A>>
        where
            A: FloatNum,
            Complex<A>: ScalarOperand,
        {
            //type Physical = $r;

            type IntermediateA = $ia;

            type IntermediateB = $ib;

            //type Spectral = $s;

            fn new(base0: &$base0<A>, base1: &$base1<A>, base2: &$base2<A>) -> Self {
                Self {
                    base0: base0.clone(),
                    base1: base1.clone(),
                    base2: base2.clone(),
                }
            }

            fn forward(&mut self, input: &mut Array3<Self::Physical>) -> Array3<Self::Spectral> {
                let mut buffer1 = self.base2.forward(input, 2);
                let mut buffer2 = self.base1.forward(&mut buffer1, 1);
                self.base0.forward(&mut buffer2, 0)
            }

            fn forward_inplace(
                &mut self,
                input: &mut Array3<Self::Physical>,
                output: &mut Array3<Self::Spectral>,
            ) {
                let mut buffer1 = self.base2.forward(input, 2);
                let mut buffer2 = self.base1.forward(&mut buffer1, 1);
                self.base0.forward_inplace(&mut buffer2, output, 0);
            }

            fn backward(&mut self, input: &mut Array3<Self::Spectral>) -> Array3<Self::Physical> {
                let mut buffer1 = self.base0.backward(input, 0);
                let mut buffer2 = self.base1.backward(&mut buffer1, 1);
                self.base2.backward(&mut buffer2, 2)
            }

            fn backward_inplace(
                &mut self,
                input: &mut Array3<Self::Spectral>,
                output: &mut Array3<Self::Physical>,
            ) {
                let mut buffer1 = self.base0.backward(input, 0);
                let mut buffer2 = self.base1.backward(&mut buffer1, 1);
                self.base2.backward_inplace(&mut buffer2, output, 2);
            }
        }
    };
}
impl_space3transform!(BaseR2r, BaseR2r, BaseR2r, A, A, A, A);
impl_space3transform!(BaseR2c, BaseR2r, BaseR2r, A, A, A, Complex<A>);
impl_space3transform!(BaseC2c, BaseR2c, BaseR2r, A, A, Complex<A>, Complex<A>);
impl_space3transform!(
    BaseC2c,
    BaseC2c,
    BaseR2c,
    A,
    Complex<A>,
    Complex<A>,
    Complex<A>
);
impl_space3transform!(
    BaseC2c,
    BaseC2c,
    BaseC2c,
    Complex<A>,
    Complex<A>,
    Complex<A>,
    Complex<A>
);
