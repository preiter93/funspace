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
use ndarray::{Array2, ShapeBuilder};
use num_traits::Zero;
use std::clone::Clone;
use std::ops::{Add, Div, Mul, Sub};

#[enum_dispatch(StencilOperations<A>)]
#[derive(Clone)]
#[allow(clippy::module_name_repetitions)]
pub enum ChebyshevStencils<A: FloatNum> {
    Dirichlet(Dirichlet<A>),
    Neumann(Neumann<A>),
    DirichletNeumann(DirichletNeumann<A>),
    BiHarmonic(BiHarmonic<A>),
}

/// Elementary methods for stencils
#[enum_dispatch]
pub trait StencilOperations<A> {
    // /// Multiply stencil with a 1d array
    // fn multiply_vec<S, T>(&self, composite_coeff: &ArrayBase<S, Ix1>) -> Array1<T>
    // where
    //     S: ndarray::Data<Elem = T>,
    //     T: ScalarNum;

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
            + Copy;

    // /// Solve linear system $A c = p$, where stencil is matrix $A$.
    // fn solve_vec<S, T>(&self, orthonorm_coeff: &ArrayBase<S, Ix1>) -> Array1<T>
    // where
    //     S: ndarray::Data<Elem = T>,
    //     T: Scalar
    //         + Add<A, Output = T>
    //         + Mul<A, Output = T>
    //         + Div<A, Output = T>
    //         + Sub<A, Output = T>;

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
            + Copy;

    /// Return stencil as 2d array
    fn to_array(&self) -> Array2<A>;
}

/// Container for Chebyshev Stencil with two diagonals
/// with offsets 0 and -2.
///
/// This struct is used in [`Dirichlet`] and [`Neumann`] stencils
#[derive(Clone)]
struct HelperStencil2Diag<A> {
    // /// Number of coefficients in orthonormal space
    // n: usize,
    // /// Number of coefficients in composite space
    // m: usize,
    /// Main diagonal
    diag: Vec<A>,
    /// Subdiagonal offset -2
    low2: Vec<A>,
    /// For tdma (diagonal)
    tdma_diag: Vec<A>,
    /// For tdma (off-diagonal)
    tdma_off2: Vec<A>,
}

impl<A: FloatNum> HelperStencil2Diag<A> {
    /// Retrieve diagonals which are later used to obtain `u` from `v`
    fn get_tdma_diagonals(diag: &[A], low2: &[A]) -> (Vec<A>, Vec<A>) {
        let m = diag.len();
        let mut tdma_diag = vec![A::zero(); m];
        let mut tdma_off2 = vec![A::zero(); m - 2];
        for (i, v) in tdma_diag.iter_mut().enumerate() {
            *v = diag[i] * diag[i] + low2[i] * low2[i];
        }
        for (i, v) in tdma_off2.iter_mut().enumerate() {
            *v = diag[i + 2] * low2[i];
        }
        (tdma_diag, tdma_off2)
    }

    /// Returns transform stencil as2d ndarray
    fn to_array(&self) -> Array2<A> {
        let m = self.diag.len();
        let mut mat = Array2::<A>::zeros((m + 2, m).f());
        for (i, (d, l)) in self.diag.iter().zip(self.low2.iter()).enumerate() {
            mat[[i, i]] = *d;
            mat[[i + 2, i]] = *l;
        }
        mat
    }

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
        assert!(v.len() == self.diag.len());
        let n = u.len();
        unsafe {
            *u.get_unchecked_mut(0) = *v.get_unchecked(0) * *self.diag.get_unchecked(0);
            *u.get_unchecked_mut(1) = *v.get_unchecked(1) * *self.diag.get_unchecked(1);
            for i in 2..n - 2 {
                *u.get_unchecked_mut(i) = *v.get_unchecked(i) * *self.diag.get_unchecked(i)
                    + *v.get_unchecked(i - 2) * *self.low2.get_unchecked(i - 2);
            }
            *u.get_unchecked_mut(n - 2) = *v.get_unchecked(n - 4) * *self.low2.get_unchecked(n - 4);
            *u.get_unchecked_mut(n - 1) = *v.get_unchecked(n - 3) * *self.low2.get_unchecked(n - 3);
        }
    }

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
        use crate::chebyshev::linalg::tdma;
        // assert!(Self::get_m(u.len()) == u.len());
        // Multiply right hand side
        unsafe {
            for i in 0..v.len() {
                *v.get_unchecked_mut(i) = *u.get_unchecked(i) * *self.diag.get_unchecked(i)
                    + *u.get_unchecked(i + 2) * *self.low2.get_unchecked(i);
            }
        }

        // Solve tridiagonal system
        tdma(&self.tdma_off2, &self.tdma_diag, &self.tdma_off2, v);
    }
}

/// Container for Chebyshev Stencil with Dirichlet boundary conditions
#[derive(Clone)]
pub struct Dirichlet<A> {
    // /// Number of coefficients in orthonormal space
    // n: usize,
    // /// Number of coefficients in composite space
    // m: usize,
    /// Helper
    helper: HelperStencil2Diag<A>,
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
        let (tdma_diag, tdma_off2) = HelperStencil2Diag::<A>::get_tdma_diagonals(&diag, &low2);
        let helper = HelperStencil2Diag {
            diag,
            low2,
            tdma_diag,
            tdma_off2,
        };
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
}

/// Container for Chebyshev Stencil with Neumann boundary conditions
#[derive(Clone)]
pub struct Neumann<A> {
    // /// Number of coefficients in orthonormal space
    // n: usize,
    // /// Number of coefficients in composite space
    // m: usize,
    /// Helper
    helper: HelperStencil2Diag<A>,
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
        let (tdma_diag, tdma_off2) = HelperStencil2Diag::<A>::get_tdma_diagonals(&diag, &low2);
        let helper = HelperStencil2Diag {
            diag,
            low2,
            tdma_diag,
            tdma_off2,
        };
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
}

/// Container for Chebyshev Stencil with two diagonals
/// with offsets 0, -1, -2.
///
/// This struct is used in [`DirichletNeumann`]
#[derive(Clone)]
struct HelperStencil3Diag<A> {
    // /// Number of coefficients in orthonormal space
    // n: usize,
    // /// Number of coefficients in composite space
    // m: usize,
    /// Main diagonal
    diag: Vec<A>,
    /// Subdiagonal offset -1
    low1: Vec<A>,
    /// Subdiagonal offset -2
    low2: Vec<A>,
    /// For tdma (diagonal)
    fdma_diag: Vec<A>,
    /// For tdma (off-diagonal 1
    fdma_off1: Vec<A>,
    /// For tdma (off-diagonal 2)
    fdma_off2: Vec<A>,
}

impl<A: FloatNum> HelperStencil3Diag<A> {
    /// Retrieve diagonals which are later used to obtain `u` from `v`
    fn get_tdma_diagonals(diag: &[A], low1: &[A], low2: &[A]) -> (Vec<A>, Vec<A>, Vec<A>) {
        let m = diag.len();
        let mut main = vec![A::zero(); m];
        let mut off1 = vec![A::zero(); m - 1];
        let mut off2 = vec![A::zero(); m - 2];
        for (i, v) in main.iter_mut().enumerate() {
            *v = diag[i] * diag[i] + low1[i] * low1[i] + low2[i] * low2[i];
        }
        for (i, v) in off1.iter_mut().enumerate() {
            *v = diag[i + 1] * low1[i] + low1[i + 1] * low2[i];
        }
        for (i, v) in off2.iter_mut().enumerate() {
            *v = diag[i + 2] * low2[i];
        }
        (main, off1, off2)
    }

    // /// Composite spaces is 2 elements smaller than orthogonal space
    // fn get_m(n: usize) -> usize {
    //     n - 2
    // }

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
        assert!(v.len() == self.diag.len());
        // assert!(Self::get_m(u.len()) == u.len());
        let n = u.len();
        unsafe {
            *u.get_unchecked_mut(0) = *v.get_unchecked(0) * *self.diag.get_unchecked(0);
            *u.get_unchecked_mut(1) = *v.get_unchecked(1) * *self.diag.get_unchecked(1)
                + *v.get_unchecked(0) * *self.low1.get_unchecked(0);
            for i in 2..n - 2 {
                *u.get_unchecked_mut(i) = *v.get_unchecked(i) * *self.diag.get_unchecked(i)
                    + *v.get_unchecked(i - 1) * *self.low1.get_unchecked(i - 1)
                    + *v.get_unchecked(i - 2) * *self.low2.get_unchecked(i - 2);
            }
            *u.get_unchecked_mut(n - 2) = *v.get_unchecked(n - 3) * *self.low1.get_unchecked(n - 3)
                + *v.get_unchecked(n - 4) * *self.low2.get_unchecked(n - 4);
            *u.get_unchecked_mut(n - 1) = *v.get_unchecked(n - 3) * *self.low2.get_unchecked(n - 3);
        }
    }

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
        use crate::chebyshev::linalg::pdma;
        // assert!(Self::get_m(u.len()) == u.len());
        // Multiply right hand side
        unsafe {
            for i in 0..v.len() {
                *v.get_unchecked_mut(i) = *u.get_unchecked(i) * *self.diag.get_unchecked(i)
                    + *u.get_unchecked(i + 1) * *self.low1.get_unchecked(i)
                    + *u.get_unchecked(i + 2) * *self.low2.get_unchecked(i);
            }
        }

        // Solve tridiagonal system
        pdma(
            &self.fdma_off2,
            &self.fdma_off1,
            &self.fdma_diag,
            &self.fdma_off1,
            &self.fdma_off2,
            v,
        );
    }

    /// Returns transform stencil as2d ndarray
    fn to_array(&self) -> Array2<A> {
        let m = self.diag.len();
        let mut mat = Array2::<A>::zeros((m + 2, m).f());
        for (i, ((d, l1), l2)) in self
            .diag
            .iter()
            .zip(self.low1.iter())
            .zip(self.low2.iter())
            .enumerate()
        {
            mat[[i, i]] = *d;
            mat[[i + 1, i]] = *l1;
            mat[[i + 2, i]] = *l2;
        }
        mat
    }
}

/// Container for Chebyshev Stencil with
/// with Dirichlet boundary conditions at x=-1
/// and Neumann boundary conditions at x=1
#[derive(Clone)]
pub struct DirichletNeumann<A> {
    // /// Number of coefficients in orthonormal space
    // n: usize,
    // /// Number of coefficients in composite space
    // m: usize,
    /// Helper
    helper: HelperStencil3Diag<A>,
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
        let (fdma_diag, fdma_off1, fdma_off2) =
            HelperStencil3Diag::<A>::get_tdma_diagonals(&diag, &low1, &low2);
        let helper = HelperStencil3Diag {
            diag,
            low1,
            low2,
            fdma_diag,
            fdma_off1,
            fdma_off2,
        };
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
}
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

/// Container for Chebyshev Stencil with two diagonals
/// with offsets 0, -2, -4.
///
/// This struct is used in [`BiHarmonic`]
#[derive(Clone)]
struct HelperStencil3Diag2<A> {
    // /// Number of coefficients in orthonormal space
    // n: usize,
    // /// Number of coefficients in composite space
    // m: usize,
    /// Main diagonal
    diag: Vec<A>,
    /// Subdiagonal offset -1
    low2: Vec<A>,
    /// Subdiagonal offset -2
    low4: Vec<A>,
    /// For tdma (diagonal)
    fdma_diag: Vec<A>,
    /// For tdma (off-diagonal 2
    fdma_off2: Vec<A>,
    /// For tdma (off-diagonal 4)
    fdma_off4: Vec<A>,
}

impl<A: FloatNum> HelperStencil3Diag2<A> {
    /// Retrieve diagonals which are later used to obtain `u` from `v`
    fn get_tdma_diagonals(diag: &[A], low2: &[A], low4: &[A]) -> (Vec<A>, Vec<A>, Vec<A>) {
        let m = diag.len();
        let mut main = vec![A::zero(); m];
        let mut off2 = vec![A::zero(); m - 2];
        let mut off4 = vec![A::zero(); m - 4];
        for (i, v) in main.iter_mut().enumerate() {
            *v = diag[i] * diag[i] + low2[i] * low2[i] + low4[i] * low4[i];
        }
        for (i, v) in off2.iter_mut().enumerate() {
            *v = diag[i + 2] * low2[i] + low2[i + 2] * low4[i];
        }
        for (i, v) in off4.iter_mut().enumerate() {
            *v = diag[i + 4] * low4[i];
        }
        (main, off2, off4)
    }

    // /// Composite spaces is 2 elements smaller than orthogonal space
    // fn get_m(n: usize) -> usize {
    //     n - 2
    // }

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
        assert!(v.len() == self.diag.len());
        let n = u.len();
        unsafe {
            *u.get_unchecked_mut(0) = *v.get_unchecked(0) * *self.diag.get_unchecked(0);
            *u.get_unchecked_mut(1) = *v.get_unchecked(1) * *self.diag.get_unchecked(1);
            *u.get_unchecked_mut(2) = *v.get_unchecked(2) * *self.diag.get_unchecked(2)
                + *v.get_unchecked(0) * *self.low2.get_unchecked(0);
            *u.get_unchecked_mut(3) = *v.get_unchecked(3) * *self.diag.get_unchecked(3)
                + *v.get_unchecked(1) * *self.low2.get_unchecked(1);
            for i in 4..n - 4 {
                *u.get_unchecked_mut(i) = *v.get_unchecked(i) * *self.diag.get_unchecked(i)
                    + *v.get_unchecked(i - 2) * *self.low2.get_unchecked(i - 2)
                    + *v.get_unchecked(i - 4) * *self.low4.get_unchecked(i - 4);
            }
            *u.get_unchecked_mut(n - 4) = *v.get_unchecked(n - 6) * *self.low2.get_unchecked(n - 6)
                + *v.get_unchecked(n - 8) * *self.low4.get_unchecked(n - 8);
            *u.get_unchecked_mut(n - 3) = *v.get_unchecked(n - 5) * *self.low2.get_unchecked(n - 6)
                + *v.get_unchecked(n - 7) * *self.low4.get_unchecked(n - 7);
            *u.get_unchecked_mut(n - 2) = *v.get_unchecked(n - 4) * *self.low2.get_unchecked(n - 4)
                + *v.get_unchecked(n - 6) * *self.low4.get_unchecked(n - 6);
            *u.get_unchecked_mut(n - 1) = *v.get_unchecked(n - 3) * *self.low2.get_unchecked(n - 3)
                + *v.get_unchecked(n - 5) * *self.low4.get_unchecked(n - 5);
        }
    }

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
        use crate::chebyshev::linalg::pdma2;
        // assert!(Self::get_m(u.len()) == u.len());
        // Multiply right hand side
        unsafe {
            for i in 0..v.len() {
                *v.get_unchecked_mut(i) = *u.get_unchecked(i) * *self.diag.get_unchecked(i)
                    + *u.get_unchecked(i + 2) * *self.low2.get_unchecked(i)
                    + *u.get_unchecked(i + 4) * *self.low4.get_unchecked(i);
            }
        }

        // Solve tridiagonal system
        pdma2(
            &self.fdma_off4,
            &self.fdma_off2,
            &self.fdma_diag,
            &self.fdma_off2,
            &self.fdma_off4,
            v,
        );
    }

    /// Returns transform stencil as2d ndarray
    fn to_array(&self) -> Array2<A> {
        let m = self.diag.len();
        let mut mat = Array2::<A>::zeros((m + 4, m).f());
        for (i, ((d, l1), l2)) in self
            .diag
            .iter()
            .zip(self.low2.iter())
            .zip(self.low4.iter())
            .enumerate()
        {
            mat[[i, i]] = *d;
            mat[[i + 2, i]] = *l1;
            mat[[i + 4, i]] = *l2;
        }
        mat
    }
}

/// Container for Chebyshev Stencil with
/// with biharmonic (i.e dirichlet + neumann)
/// boundary conditions at x=-1, 1
#[derive(Clone)]
pub struct BiHarmonic<A> {
    /// Helper
    helper: HelperStencil3Diag2<A>,
}

impl<A: FloatNum> BiHarmonic<A> {
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
        let mut low4 = vec![A::zero(); m];
        for (ki, (v2, v4)) in low2.iter_mut().zip(low4.iter_mut()).enumerate() {
            let k = ki as f64;
            let d2 = -2. * (k + 2.) / (k + 3.);
            let d4 = (k + 1.) / (k + 3.);
            *v2 = A::from_f64(d2).unwrap();
            *v4 = A::from_f64(d4).unwrap();
        }
        let (fdma_diag, fdma_off2, fdma_off4) =
            HelperStencil3Diag2::<A>::get_tdma_diagonals(&diag, &low2, &low4);
        let helper = HelperStencil3Diag2 {
            diag,
            low2,
            low4,
            fdma_diag,
            fdma_off2,
            fdma_off4,
        };

        Self { helper }
    }

    /// Composite spaces is 4 elements smaller than orthogonal space
    pub fn get_m(n: usize) -> usize {
        n - 4
    }
}

impl<A: FloatNum> StencilOperations<A> for BiHarmonic<A> {
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
}
