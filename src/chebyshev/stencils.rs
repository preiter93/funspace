//! # Collection of Chebyshev-composite-stencils
//!
//! A stencil $S$ transforms from orthogonal space `u` to composite space `v`, i.e.
//! $$$
//! u = S v
//! $$&
//!
//! The stencil matrix is usually sparse, so we can define efficient methods
//! to get `v` from `u` and vice versa.
use crate::types::FloatNum;
use ndarray::Array2;
use num_traits::Zero;
use std::clone::Clone;
use std::ops::{Add, Div, Mul, Sub};
mod helpers;
use helpers::{HelperStencil2Diag, HelperStencil3Diag, HelperStencil3Diag2};
mod traits;
pub(crate) use traits::StencilOperations;

#[enum_dispatch(StencilOperations<A>)]
#[derive(Clone)]
#[allow(clippy::module_name_repetitions)]
pub enum ChebyshevStencils<A: FloatNum> {
    Dirichlet(Dirichlet<A>),
    Neumann(Neumann<A>),
    DirichletNeumann(DirichletNeumann<A>),
    BiHarmonicA(BiHarmonicA<A>),
    BiHarmonicB(BiHarmonicB<A>),
}

/// Container for Chebyshev Stencil with Dirichlet boundary conditions
#[derive(Clone)]
pub struct Dirichlet<A> {
    /// Helper
    helper: HelperStencil2Diag<A>,
}

/// Container for Chebyshev Stencil with Neumann boundary conditions
#[derive(Clone)]
pub struct Neumann<A> {
    /// Helper
    helper: HelperStencil2Diag<A>,
}

/// Container for Chebyshev Stencil with
/// with Dirichlet boundary conditions at x=-1
/// and Neumann boundary conditions at x=1
#[derive(Clone)]
pub struct DirichletNeumann<A> {
    /// Helper
    helper: HelperStencil3Diag<A>,
}

/// Container for Chebyshev Stencil with
/// with biharmonic (i.e dirichlet + neumann)
/// boundary conditions at x=-1, 1
#[derive(Clone)]
pub struct BiHarmonicA<A> {
    /// Helper
    helper: HelperStencil3Diag2<A>,
}

/// Container for Chebyshev Stencil with
/// with biharmonic (i.e dirichlet + second derivative = zero)
/// boundary conditions at x=-1, 1
#[derive(Clone)]
pub struct BiHarmonicB<A> {
    /// Helper
    helper: HelperStencil3Diag2<A>,
}

impl<A: FloatNum> Dirichlet<A> {
    /// Return stencil of chebyshev dirichlet space
    /// ```text
    ///  \phi_k = T_k - T_{k+2}
    /// ```
    ///
    /// Reference:
    /// J. Shen: Effcient Spectral-Galerkin Method II.
    pub fn new(n: usize) -> Self {
        let m = Self::get_m(n);
        let diag = vec![A::one(); m];
        let low2 = vec![-A::one(); m];
        let helper = HelperStencil2Diag::<A>::new(diag, low2);
        Self { helper }
    }

    /// Composite spaces is 2 elements smaller than orthogonal space
    pub fn get_m(n: usize) -> usize {
        n - 2
    }
}

impl<A: FloatNum> StencilOperations<A> for Dirichlet<A> {
    /// Multiply stencil with a 1d vector
    fn dot_inplace<T>(&self, u: &[T], v: &mut [T])
    where
        T: Mul<Output = T>
            + Sub<Output = T>
            + Div<Output = T>
            + Add<Output = T>
            + Add<A, Output = T>
            + Mul<A, Output = T>
            + Div<A, Output = T>
            + Sub<A, Output = T>
            + Zero
            + Clone
            + Copy,
    {
        self.helper.dot_inplace(u, v);
    }

    /// Solve linear system $S v = u$
    fn solve_inplace<T>(&self, v: &[T], u: &mut [T])
    where
        T: Mul<Output = T>
            + Sub<Output = T>
            + Div<Output = T>
            + Add<Output = T>
            + Add<A, Output = T>
            + Mul<A, Output = T>
            + Div<A, Output = T>
            + Sub<A, Output = T>
            + Zero
            + Clone
            + Copy,
    {
        self.helper.solve_inplace(v, u);
    }

    /// Returns transform stencil as2d ndarray
    fn to_array(&self) -> Array2<A> {
        self.helper.to_array()
    }

    /// Returns inverse of transform stencil as 2d ndarray
    fn pinv(&self) -> Array2<A> {
        self.helper.pinv()
    }
}

impl<A: FloatNum> Neumann<A> {
    /// Return stencil of chebyshev neumann space
    /// ```text
    ///  \phi_k = T_k - k^{2} \/ (k+2)^2 T_{k+2}
    /// ```
    ///
    /// Reference:
    /// J. Shen: Effcient Spectral-Galerkin Method II.
    #[allow(clippy::cast_precision_loss)]
    pub fn new(n: usize) -> Self {
        let m = Self::get_m(n);
        let diag = vec![A::one(); m];
        let mut low2 = vec![A::zero(); m];
        for (k, v) in low2.iter_mut().enumerate() {
            let k_ = A::from_f64(k.pow(2) as f64).unwrap();
            let k2_ = A::from_f64((k + 2).pow(2) as f64).unwrap();
            *v = -A::one() * k_ / k2_;
        }
        let helper = HelperStencil2Diag::<A>::new(diag, low2);
        Self { helper }
    }

    /// Composite spaces is 2 elements smaller than orthogonal space
    pub fn get_m(n: usize) -> usize {
        n - 2
    }
}

impl<A: FloatNum> StencilOperations<A> for Neumann<A> {
    /// Multiply stencil with a 1d vector
    fn dot_inplace<T>(&self, u: &[T], v: &mut [T])
    where
        T: Mul<Output = T>
            + Sub<Output = T>
            + Div<Output = T>
            + Add<Output = T>
            + Add<A, Output = T>
            + Mul<A, Output = T>
            + Div<A, Output = T>
            + Sub<A, Output = T>
            + Zero
            + Clone
            + Copy,
    {
        self.helper.dot_inplace(u, v);
    }

    /// Solve linear system $S v = u$
    fn solve_inplace<T>(&self, v: &[T], u: &mut [T])
    where
        T: Mul<Output = T>
            + Sub<Output = T>
            + Div<Output = T>
            + Add<Output = T>
            + Add<A, Output = T>
            + Mul<A, Output = T>
            + Div<A, Output = T>
            + Sub<A, Output = T>
            + Zero
            + Clone
            + Copy,
    {
        self.helper.solve_inplace(v, u);
    }

    /// Returns transform stencil as2d ndarray
    fn to_array(&self) -> Array2<A> {
        self.helper.to_array()
    }

    /// Returns inverse of transform stencil as 2d ndarray
    fn pinv(&self) -> Array2<A> {
        self.helper.pinv()
    }
}

impl<A: FloatNum> DirichletNeumann<A> {
    /// Return stencil of chebyshev neumann space
    /// ```text
    ///  \phi_k = T_k - k^{2} \/ (k+2)^2 T_{k+2}
    /// ```
    ///
    /// Reference:
    /// J. Shen: Effcient Spectral-Galerkin Method II.
    #[allow(clippy::cast_precision_loss)]
    pub fn new(n: usize) -> Self {
        let m = Self::get_m(n);
        let diag = vec![A::one(); m];
        let mut low1 = vec![A::zero(); m];
        let mut low2 = vec![A::zero(); m];
        for (k, (v1, v2)) in low1.iter_mut().zip(low2.iter_mut()).enumerate() {
            let kf64 = k as f64;
            *v1 = A::from_f64(
                (-1. * kf64.powi(2) + (kf64 + 2.).powi(2))
                    / ((kf64 + 1.).powi(2) + (kf64 + 2.).powi(2)),
            )
            .unwrap();
            *v2 = A::from_f64(
                (-1. * kf64.powi(2) - (kf64 + 1.).powi(2))
                    / ((kf64 + 1.).powi(2) + (kf64 + 2.).powi(2)),
            )
            .unwrap();
        }
        let helper = HelperStencil3Diag::<A>::new(diag, low1, low2);
        Self { helper }
    }

    /// Composite spaces is 2 elements smaller than orthogonal space
    pub fn get_m(n: usize) -> usize {
        n - 2
    }
}

impl<A: FloatNum> StencilOperations<A> for DirichletNeumann<A> {
    /// Multiply stencil with a 1d vector
    fn dot_inplace<T>(&self, u: &[T], v: &mut [T])
    where
        T: Mul<Output = T>
            + Sub<Output = T>
            + Div<Output = T>
            + Add<Output = T>
            + Add<A, Output = T>
            + Mul<A, Output = T>
            + Div<A, Output = T>
            + Sub<A, Output = T>
            + Zero
            + Clone
            + Copy,
    {
        self.helper.dot_inplace(u, v);
    }

    /// Solve linear system $S v = u$
    fn solve_inplace<T>(&self, v: &[T], u: &mut [T])
    where
        T: Mul<Output = T>
            + Sub<Output = T>
            + Div<Output = T>
            + Add<Output = T>
            + Add<A, Output = T>
            + Mul<A, Output = T>
            + Div<A, Output = T>
            + Sub<A, Output = T>
            + Zero
            + Clone
            + Copy,
    {
        self.helper.solve_inplace(v, u);
    }

    /// Returns transform stencil as2d ndarray
    fn to_array(&self) -> Array2<A> {
        self.helper.to_array()
    }

    /// Returns inverse of transform stencil as 2d ndarray
    fn pinv(&self) -> Array2<A> {
        self.helper.pinv()
    }
}

impl<A: FloatNum> BiHarmonicA<A> {
    /// Return stencil of biarmonic space
    ///
    /// Reference:
    /// ```text
    /// F. Liu: doi: 10.4208/nmtma.2011.42s.5
    /// ```
    #[allow(clippy::cast_precision_loss)]
    pub fn new(n: usize) -> Self {
        let m = Self::get_m(n);

        let diag = vec![A::one(); m];
        let mut low2 = vec![A::zero(); m];
        let mut low4 = vec![A::zero(); m];
        for (ki, (v2, v4)) in low2.iter_mut().zip(low4.iter_mut()).enumerate() {
            let k = ki as f64;
            let d2 = -2. * (k + 2.) / (k + 3.);
            let d4 = (k + 1.) / (k + 3.);
            *v2 = A::from_f64(d2).unwrap();
            *v4 = A::from_f64(d4).unwrap();
        }
        let helper = HelperStencil3Diag2::<A>::new(diag, low2, low4);
        Self { helper }
    }

    /// Composite spaces is 4 elements smaller than orthogonal space
    pub fn get_m(n: usize) -> usize {
        n - 4
    }
}

impl<A: FloatNum> StencilOperations<A> for BiHarmonicA<A> {
    /// Multiply stencil with a 1d vector
    fn dot_inplace<T>(&self, v: &[T], u: &mut [T])
    where
        T: Mul<Output = T>
            + Sub<Output = T>
            + Div<Output = T>
            + Add<Output = T>
            + Add<A, Output = T>
            + Mul<A, Output = T>
            + Div<A, Output = T>
            + Sub<A, Output = T>
            + Zero
            + Clone
            + Copy,
    {
        self.helper.dot_inplace(v, u);
    }

    /// Solve linear system $S v = u$
    fn solve_inplace<T>(&self, u: &[T], v: &mut [T])
    where
        T: Mul<Output = T>
            + Sub<Output = T>
            + Div<Output = T>
            + Add<Output = T>
            + Add<A, Output = T>
            + Mul<A, Output = T>
            + Div<A, Output = T>
            + Sub<A, Output = T>
            + Zero
            + Clone
            + Copy,
    {
        self.helper.solve_inplace(u, v);
    }

    /// Returns transform stencil as2d ndarray
    fn to_array(&self) -> Array2<A> {
        self.helper.to_array()
    }

    /// Returns inverse of transform stencil as 2d ndarray
    fn pinv(&self) -> Array2<A> {
        self.helper.pinv()
    }
}

impl<A: FloatNum> BiHarmonicB<A> {
    /// Return stencil of biarmonic space
    ///
    /// Reference:
    /// ```text
    /// F. Liu: doi: 10.4208/nmtma.2011.42s.5
    /// ```
    #[allow(clippy::cast_precision_loss)]
    pub fn new(n: usize) -> Self {
        let m = Self::get_m(n);

        let diag = vec![A::one(); m];
        let mut low2 = vec![A::zero(); m];
        let mut low4 = vec![A::zero(); m];
        for (ki, (v2, v4)) in low2.iter_mut().zip(low4.iter_mut()).enumerate() {
            let k = ki as f64;
            let d2 = -1. * (2. * (k + 2.) * (15. + 2. * k * (k + 4.)))
                / ((k + 3.) * (19. + 2. * k * (6. + k)));
            let d4 = ((k + 1.) * (3. + 2. * k * (k + 2.))) / ((k + 3.) * (19. + 2. * k * (6. + k)));
            *v2 = A::from_f64(d2).unwrap();
            *v4 = A::from_f64(d4).unwrap();
        }
        let helper = HelperStencil3Diag2::<A>::new(diag, low2, low4);
        Self { helper }
    }

    /// Composite spaces is 4 elements smaller than orthogonal space
    pub fn get_m(n: usize) -> usize {
        n - 4
    }
}

impl<A: FloatNum> StencilOperations<A> for BiHarmonicB<A> {
    /// Multiply stencil with a 1d vector
    fn dot_inplace<T>(&self, v: &[T], u: &mut [T])
    where
        T: Mul<Output = T>
            + Sub<Output = T>
            + Div<Output = T>
            + Add<Output = T>
            + Add<A, Output = T>
            + Mul<A, Output = T>
            + Div<A, Output = T>
            + Sub<A, Output = T>
            + Zero
            + Clone
            + Copy,
    {
        self.helper.dot_inplace(v, u);
    }

    /// Solve linear system $S v = u$
    fn solve_inplace<T>(&self, u: &[T], v: &mut [T])
    where
        T: Mul<Output = T>
            + Sub<Output = T>
            + Div<Output = T>
            + Add<Output = T>
            + Add<A, Output = T>
            + Mul<A, Output = T>
            + Div<A, Output = T>
            + Sub<A, Output = T>
            + Zero
            + Clone
            + Copy,
    {
        self.helper.solve_inplace(u, v);
    }

    /// Returns transform stencil as2d ndarray
    fn to_array(&self) -> Array2<A> {
        self.helper.to_array()
    }

    /// Returns inverse of transform stencil as 2d ndarray
    fn pinv(&self) -> Array2<A> {
        self.helper.pinv()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    /// Sinv @ S = I
    fn test_stencil_pinv_neumann() {
        let n = 42;
        let stencil = Neumann::<f64>::new(n);
        let pinv = stencil.helper.pinv();
        let mass = stencil.to_array();
        let should_be_eye = pinv.dot(&mass);
        assert!(should_be_eye.is_square());
        for i in 0..should_be_eye.shape()[0] {
            for j in 0..should_be_eye.shape()[1] {
                if i == j {
                    assert!((1. - should_be_eye[[i, j]]).abs() < 1e-6);
                } else {
                    assert!(should_be_eye[[i, j]].abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    /// Sinv @ S = I
    fn test_stencil_pinv_dirichlet_neumann() {
        let n = 57;
        let stencil = DirichletNeumann::<f64>::new(n);
        let mass = stencil.to_array();
        let pinv = stencil.helper.pinv();
        let should_be_eye = pinv.dot(&mass);
        assert!(should_be_eye.is_square());
        for i in 0..should_be_eye.shape()[0] {
            for j in 0..should_be_eye.shape()[1] {
                if i == j {
                    assert!((1. - should_be_eye[[i, j]]).abs() < 1e-6);
                } else {
                    assert!(should_be_eye[[i, j]].abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    /// Sinv @ S = I
    fn test_stencil_pinv_biharmonic_a() {
        let n = 28;
        let stencil = BiHarmonicA::<f64>::new(n);
        let mass = stencil.to_array();
        let pinv = stencil.helper.pinv();
        let should_be_eye = pinv.dot(&mass);
        assert!(should_be_eye.is_square());
        for i in 0..should_be_eye.shape()[0] {
            for j in 0..should_be_eye.shape()[1] {
                if i == j {
                    assert!((1. - should_be_eye[[i, j]]).abs() < 1e-6);
                } else {
                    assert!(should_be_eye[[i, j]].abs() < 1e-6);
                }
            }
        }
    }
}
