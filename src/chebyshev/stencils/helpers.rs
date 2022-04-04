//! Different stencils have the same structure/bandedness,
//! helpers define routines depending on the stencil structure.
use crate::types::FloatNum;
use ndarray::{Array2, ShapeBuilder};
use num_traits::Zero;
use std::clone::Clone;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Clone)]
/// Container for Chebyshev Stencil with two diagonals
/// with offsets 0 and -2.
///
/// This struct is used in [`super::Dirichlet`] and [`super::Neumann`] stencils
pub(super) struct HelperStencil2Diag<A> {
    /// Main diagonal
    pub(super) diag: Vec<A>,
    /// Subdiagonal offset -2
    pub(super) low2: Vec<A>,
    /// For tdma (diagonal)
    pub(super) tdma_diag: Vec<A>,
    /// For tdma (off-diagonal)
    pub(super) tdma_off2: Vec<A>,
}

/// Container for Chebyshev Stencil with two diagonals
/// with offsets 0, -1, -2.
///
/// This struct is used in [`super::DirichletNeumann`]
#[derive(Clone)]
pub(super) struct HelperStencil3Diag<A> {
    /// Main diagonal
    pub(super) diag: Vec<A>,
    /// Subdiagonal offset -1
    pub(super) low1: Vec<A>,
    /// Subdiagonal offset -2
    pub(super) low2: Vec<A>,
    /// For tdma (diagonal)
    pub(super) fdma_diag: Vec<A>,
    /// For tdma (off-diagonal 1
    pub(super) fdma_off1: Vec<A>,
    /// For tdma (off-diagonal 2)
    pub(super) fdma_off2: Vec<A>,
}

/// Container for Chebyshev Stencil with two diagonals
/// with offsets 0, -2, -4.
///
/// This struct is used in [`super::BiHarmonic`]
#[derive(Clone)]
pub(super) struct HelperStencil3Diag2<A> {
    /// Main diagonal
    pub(super) diag: Vec<A>,
    /// Subdiagonal offset -1
    pub(super) low2: Vec<A>,
    /// Subdiagonal offset -2
    pub(super) low4: Vec<A>,
    /// For tdma (diagonal)
    pub(super) fdma_diag: Vec<A>,
    /// For tdma (off-diagonal 2
    pub(super) fdma_off2: Vec<A>,
    /// For tdma (off-diagonal 4)
    pub(super) fdma_off4: Vec<A>,
}

impl<A: FloatNum> HelperStencil2Diag<A> {
    /// Constructor
    pub(super) fn new(diag: Vec<A>, low2: Vec<A>) -> Self {
        let (tdma_diag, tdma_off2) = Self::get_tdma_diagonals(&diag, &low2);
        Self {
            diag,
            low2,
            tdma_diag,
            tdma_off2,
        }
    }

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

    /// Returns stencil as 2d ndarray
    pub(super) fn to_array(&self) -> Array2<A> {
        let m = self.diag.len();
        let mut mat = Array2::<A>::zeros((m + 2, m).f());
        for (i, (d, l)) in self.diag.iter().zip(self.low2.iter()).enumerate() {
            mat[[i, i]] = *d;
            mat[[i + 2, i]] = *l;
        }
        mat
    }

    /// Return Pseudoinverse of stencil
    /// ```text
    /// Sinv @ S = I
    /// ```
    /// Hase a lower triangular structure
    pub(super) fn pinv(&self) -> Array2<A> {
        let m = self.diag.len();
        let n = m + 2;
        let mut pinv = Array2::<A>::zeros((m, n));
        for i in 0..m {
            // Diagonal
            pinv[[i, i]] = A::one() / self.diag[i];
            // Lower triangular part
            if i >= 2 {
                for j in (0..i - 1).rev().step_by(2) {
                    pinv[[i, j]] = -A::one() * (pinv[[i, j + 2]] * self.low2[j]) / self.diag[j];
                }
            }
        }
        pinv
    }

    /// S v = u
    ///
    /// v -> u
    pub(super) fn dot_inplace<T>(&self, v: &[T], u: &mut [T])
    where
        T: Mul<Output = T>
            + Add<Output = T>
            + Add<A, Output = T>
            + Mul<A, Output = T>
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

    /// S v = u
    ///
    /// u -> v
    pub(super) fn solve_inplace<T>(&self, u: &[T], v: &mut [T])
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

impl<A: FloatNum> HelperStencil3Diag<A> {
    /// Constructor
    pub(super) fn new(diag: Vec<A>, low1: Vec<A>, low2: Vec<A>) -> Self {
        let (fdma_diag, fdma_off1, fdma_off2) = Self::get_fdma_diagonals(&diag, &low1, &low2);
        Self {
            diag,
            low1,
            low2,
            fdma_diag,
            fdma_off1,
            fdma_off2,
        }
    }
    /// Retrieve diagonals which are later used to obtain `u` from `v`
    fn get_fdma_diagonals(diag: &[A], low1: &[A], low2: &[A]) -> (Vec<A>, Vec<A>, Vec<A>) {
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

    /// S v = u
    ///
    /// v -> u
    pub(super) fn dot_inplace<T>(&self, v: &[T], u: &mut [T])
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

    /// S v = u
    ///
    /// u -> v
    pub(super) fn solve_inplace<T>(&self, u: &[T], v: &mut [T])
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
    pub(super) fn to_array(&self) -> Array2<A> {
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

    /// Return Pseudoinverse of stencil
    /// ```text
    /// Sinv @ S = I
    /// ```
    /// Has a lower triangular structure
    #[allow(clippy::unused_self)]
    pub(super) fn pinv(&self) -> Array2<A> {
        let m = self.diag.len();
        let n = m + 2;
        let mut pinv = Array2::<A>::zeros((m, n));
        for i in 0..m {
            // Diagonal
            pinv[[i, i]] = A::one() / self.diag[i];
            // Lower triangular part
            if i >= 1 {
                let j = i - 1;
                pinv[[i, j]] = -A::one() * (pinv[[i, j + 1]] * self.low1[j]) / self.diag[j];
            }
            if i >= 2 {
                for j in (0..i - 1).rev() {
                    pinv[[i, j]] = -A::one()
                        * (pinv[[i, j + 2]] * self.low2[j] + pinv[[i, j + 1]] * self.low1[j])
                        / self.diag[j];
                }
            }
        }
        pinv
    }
}

impl<A: FloatNum> HelperStencil3Diag2<A> {
    /// Constructor
    pub(super) fn new(diag: Vec<A>, low2: Vec<A>, low4: Vec<A>) -> Self {
        let (fdma_diag, fdma_off2, fdma_off4) = Self::get_fdma_diagonals(&diag, &low2, &low4);
        Self {
            diag,
            low2,
            low4,
            fdma_diag,
            fdma_off2,
            fdma_off4,
        }
    }

    /// Retrieve diagonals which are later used to obtain `u` from `v`
    fn get_fdma_diagonals(diag: &[A], low2: &[A], low4: &[A]) -> (Vec<A>, Vec<A>, Vec<A>) {
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

    /// S v = u
    ///
    /// v -> u
    pub(super) fn dot_inplace<T>(&self, v: &[T], u: &mut [T])
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

    /// S v = u
    ///
    /// u -> v
    pub(super) fn solve_inplace<T>(&self, u: &[T], v: &mut [T])
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
    pub(super) fn to_array(&self) -> Array2<A> {
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

    /// Return Pseudoinverse of stencil
    /// ```text
    /// Sinv @ S = I
    /// ```
    /// Has a lower triangular structure
    #[allow(clippy::unused_self)]
    pub(super) fn pinv(&self) -> Array2<A> {
        let m = self.diag.len();
        let n = m + 4;
        let mut pinv = Array2::<A>::zeros((m, n));
        for i in 0..m {
            // Diagonal
            pinv[[i, i]] = A::one() / self.diag[i];
            // Lower triangular part
            if i >= 2 {
                let j = i - 2;
                pinv[[i, j]] = -A::one() * (pinv[[i, j + 2]] * self.low2[j]) / self.diag[j];
            }
            if i >= 4 {
                for j in (0..i - 1).rev().step_by(2) {
                    pinv[[i, j]] = -A::one()
                        * (pinv[[i, j + 4]] * self.low4[j] + pinv[[i, j + 2]] * self.low2[j])
                        / self.diag[j];
                }
            }
        }
        pinv
    }
}
