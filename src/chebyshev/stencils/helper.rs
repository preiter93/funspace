use super::traits::StencilOperand;
use crate::chebyshev::linalg::{pdma, pdma2, tdma};
use crate::types::{Real, ScalarOperand};

#[enum_dispatch(StencilOperand<A>)]
#[derive(Clone)]
pub(super) enum Helper<A: Real> {
    HelperDiag02(HelperDiag02<A>),
    HelperDiag012(HelperDiag012<A>),
    HelperDiag024(HelperDiag024<A>),
}

/// Lower banded matrix with bands at 0, 2
/// mtm denote bands after `m.t().dot(m)`
#[derive(Clone)]
pub(crate) struct HelperDiag02<A> {
    rows: usize,
    cols: usize,
    diag: Vec<A>,
    low2: Vec<A>,
    mtm_diag: Vec<A>,
    mtm_off2: Vec<A>,
}

impl<A: Real> HelperDiag02<A> {
    /// Constructor
    pub(crate) fn new(rows: usize, cols: usize, diag: Vec<A>, low2: Vec<A>) -> Self {
        let mut mtm_diag = vec![A::zero(); diag.len()];
        let mut mtm_off2 = vec![A::zero(); diag.len() - 2];
        for (i, v) in mtm_diag.iter_mut().enumerate() {
            *v = diag[i] * diag[i] + low2[i] * low2[i];
        }
        for (i, v) in mtm_off2.iter_mut().enumerate() {
            *v = diag[i + 2] * low2[i];
        }
        Self {
            rows,
            cols,
            diag,
            low2,
            mtm_diag,
            mtm_off2,
        }
    }
}

impl<A> StencilOperand<A> for HelperDiag02<A>
where
    A: Real,
{
    fn matvec<T: ScalarOperand<A>>(&self, x: &[T], b: &mut [T]) {
        debug_assert!(b.len() >= self.rows);
        debug_assert!(x.len() >= self.cols);
        // b[..self.cols].clone_from_slice(&x[..self.cols]);
        // self.matvec_inplace(b);
        let n = self.rows;
        unsafe {
            *b.get_unchecked_mut(n - 1) = *x.get_unchecked(n - 3) * *self.low2.get_unchecked(n - 3);
            *b.get_unchecked_mut(n - 2) = *x.get_unchecked(n - 4) * *self.low2.get_unchecked(n - 4);
            for i in (2..n - 2).rev() {
                *b.get_unchecked_mut(i) = *x.get_unchecked(i) * *self.diag.get_unchecked(i)
                    + *x.get_unchecked(i - 2) * *self.low2.get_unchecked(i - 2);
            }
            *b.get_unchecked_mut(1) = *x.get_unchecked(1) * *self.diag.get_unchecked(1);
            *b.get_unchecked_mut(0) = *x.get_unchecked(0) * *self.diag.get_unchecked(0);
        }
    }

    // fn matvec_inplace<T: ScalarOperand<A>>(&self, x: &mut [T]) {
    //     debug_assert!(x.len() >= self.rows);
    //     let n = self.rows;
    //     unsafe {
    //         *x.get_unchecked_mut(n - 1) = *x.get_unchecked(n - 3) * *self.low2.get_unchecked(n - 3);
    //         *x.get_unchecked_mut(n - 2) = *x.get_unchecked(n - 4) * *self.low2.get_unchecked(n - 4);
    //         for i in (2..n - 2).rev() {
    //             *x.get_unchecked_mut(i) = *x.get_unchecked(i) * *self.diag.get_unchecked(i)
    //                 + *x.get_unchecked(i - 2) * *self.low2.get_unchecked(i - 2);
    //         }
    //         *x.get_unchecked_mut(1) = *x.get_unchecked(1) * *self.diag.get_unchecked(1);
    //         *x.get_unchecked_mut(0) = *x.get_unchecked(0) * *self.diag.get_unchecked(0);
    //     }
    // }

    fn solve<T: ScalarOperand<A>>(&self, b: &[T], x: &mut [T]) {
        // Multiply rhs
        unsafe {
            for i in 0..self.cols {
                *x.get_unchecked_mut(i) = *b.get_unchecked(i) * *self.diag.get_unchecked(i)
                    + *b.get_unchecked(i + 2) * *self.low2.get_unchecked(i);
            }
        }
        // Solve tridiagonal system
        tdma(&self.mtm_off2, &self.mtm_diag, &self.mtm_off2, x);
    }

    // fn solve_inplace<T: ScalarOperand<A>>(&self, x: &mut [T]) {
    //     debug_assert!(x.len() >= self.rows);
    //     // Multiply rhs
    //     unsafe {
    //         for i in 0..self.cols {
    //             *x.get_unchecked_mut(i) = *x.get_unchecked(i) * *self.diag.get_unchecked(i)
    //                 + *x.get_unchecked(i + 2) * *self.low2.get_unchecked(i);
    //         }
    //     }

    //     // Solve tridiagonal system
    //     tdma(
    //         &self.mtm_off2,
    //         &self.mtm_diag,
    //         &self.mtm_off2,
    //         &mut x[..self.cols],
    //     );
    // }
}

/// Lower banded matrix with bands at 0, 1, 2
/// mtm denote bands after `m.t().dot(m)`
#[derive(Clone)]
pub(crate) struct HelperDiag012<A> {
    rows: usize,
    cols: usize,
    diag: Vec<A>,
    low1: Vec<A>,
    low2: Vec<A>,
    mtm_diag: Vec<A>,
    mtm_off1: Vec<A>,
    mtm_off2: Vec<A>,
}

impl<A: Real> HelperDiag012<A> {
    /// Constructor
    pub(crate) fn new(rows: usize, cols: usize, diag: Vec<A>, low1: Vec<A>, low2: Vec<A>) -> Self {
        let mut mtm_diag = vec![A::zero(); diag.len()];
        let mut mtm_off1 = vec![A::zero(); diag.len() - 1];
        let mut mtm_off2 = vec![A::zero(); diag.len() - 2];
        for (i, v) in mtm_diag.iter_mut().enumerate() {
            *v = diag[i] * diag[i] + low1[i] * low1[i] + low2[i] * low2[i];
        }
        for (i, v) in mtm_off1.iter_mut().enumerate() {
            *v = diag[i + 1] * low1[i] + low1[i + 1] * low2[i];
        }
        for (i, v) in mtm_off2.iter_mut().enumerate() {
            *v = diag[i + 2] * low2[i];
        }
        Self {
            rows,
            cols,
            diag,
            low1,
            low2,
            mtm_diag,
            mtm_off1,
            mtm_off2,
        }
    }
}

impl<A> StencilOperand<A> for HelperDiag012<A>
where
    A: Real,
{
    fn matvec<T: ScalarOperand<A>>(&self, x: &[T], b: &mut [T]) {
        debug_assert!(b.len() >= self.rows);
        debug_assert!(x.len() >= self.cols);
        let n = self.rows;
        unsafe {
            *b.get_unchecked_mut(0) = *x.get_unchecked(0) * *self.diag.get_unchecked(0);
            *b.get_unchecked_mut(1) = *x.get_unchecked(1) * *self.diag.get_unchecked(1)
                + *x.get_unchecked(0) * *self.low1.get_unchecked(0);
            for i in 2..n - 2 {
                *b.get_unchecked_mut(i) = *x.get_unchecked(i) * *self.diag.get_unchecked(i)
                    + *x.get_unchecked(i - 1) * *self.low1.get_unchecked(i - 1)
                    + *x.get_unchecked(i - 2) * *self.low2.get_unchecked(i - 2);
            }
            *b.get_unchecked_mut(n - 2) = *x.get_unchecked(n - 3) * *self.low1.get_unchecked(n - 3)
                + *x.get_unchecked(n - 4) * *self.low2.get_unchecked(n - 4);
            *b.get_unchecked_mut(n - 1) = *x.get_unchecked(n - 3) * *self.low2.get_unchecked(n - 3);
        }
    }

    fn solve<T: ScalarOperand<A>>(&self, b: &[T], x: &mut [T]) {
        // Multiply rhs
        unsafe {
            for i in 0..self.cols {
                *x.get_unchecked_mut(i) = *b.get_unchecked(i) * *self.diag.get_unchecked(i)
                    + *b.get_unchecked(i + 1) * *self.low1.get_unchecked(i)
                    + *b.get_unchecked(i + 2) * *self.low2.get_unchecked(i);
            }
        }
        // Solve tridiagonal system
        pdma(
            &self.mtm_off2,
            &self.mtm_off1,
            &self.mtm_diag,
            &self.mtm_off1,
            &self.mtm_off2,
            x,
        );
    }
}

/// Lower banded matrix with bands at 0, 2, 4
/// mtm denote bands after `m.t().dot(m)`
#[derive(Clone)]
pub(crate) struct HelperDiag024<A> {
    rows: usize,
    cols: usize,
    diag: Vec<A>,
    low2: Vec<A>,
    low4: Vec<A>,
    mtm_diag: Vec<A>,
    mtm_off2: Vec<A>,
    mtm_off4: Vec<A>,
}

impl<A: Real> HelperDiag024<A> {
    /// Constructor
    pub(crate) fn new(rows: usize, cols: usize, diag: Vec<A>, low2: Vec<A>, low4: Vec<A>) -> Self {
        let mut mtm_diag = vec![A::zero(); diag.len()];
        let mut mtm_off2 = vec![A::zero(); diag.len() - 2];
        let mut mtm_off4 = vec![A::zero(); diag.len() - 4];
        for (i, v) in mtm_diag.iter_mut().enumerate() {
            *v = diag[i] * diag[i] + low2[i] * low2[i] + low4[i] * low4[i];
        }
        for (i, v) in mtm_off2.iter_mut().enumerate() {
            *v = diag[i + 2] * low2[i] + low2[i + 2] * low4[i];
        }
        for (i, v) in mtm_off4.iter_mut().enumerate() {
            *v = diag[i + 4] * low4[i];
        }
        Self {
            rows,
            cols,
            diag,
            low2,
            low4,
            mtm_diag,
            mtm_off2,
            mtm_off4,
        }
    }
}

impl<A> StencilOperand<A> for HelperDiag024<A>
where
    A: Real,
{
    fn matvec<T: ScalarOperand<A>>(&self, x: &[T], b: &mut [T]) {
        debug_assert!(b.len() >= self.rows);
        debug_assert!(x.len() >= self.cols);
        let n = self.rows;
        unsafe {
            *b.get_unchecked_mut(0) = *x.get_unchecked(0) * *self.diag.get_unchecked(0);
            *b.get_unchecked_mut(1) = *x.get_unchecked(1) * *self.diag.get_unchecked(1);
            *b.get_unchecked_mut(2) = *x.get_unchecked(2) * *self.diag.get_unchecked(2)
                + *x.get_unchecked(0) * *self.low2.get_unchecked(0);
            *b.get_unchecked_mut(3) = *x.get_unchecked(3) * *self.diag.get_unchecked(3)
                + *x.get_unchecked(1) * *self.low2.get_unchecked(1);
            for i in 4..n - 4 {
                *b.get_unchecked_mut(i) = *x.get_unchecked(i) * *self.diag.get_unchecked(i)
                    + *x.get_unchecked(i - 2) * *self.low2.get_unchecked(i - 2)
                    + *x.get_unchecked(i - 4) * *self.low4.get_unchecked(i - 4);
            }
            *b.get_unchecked_mut(n - 4) = *x.get_unchecked(n - 6) * *self.low2.get_unchecked(n - 6)
                + *x.get_unchecked(n - 8) * *self.low4.get_unchecked(n - 8);
            *b.get_unchecked_mut(n - 3) = *x.get_unchecked(n - 5) * *self.low2.get_unchecked(n - 6)
                + *x.get_unchecked(n - 7) * *self.low4.get_unchecked(n - 7);
            *b.get_unchecked_mut(n - 2) = *x.get_unchecked(n - 4) * *self.low2.get_unchecked(n - 4)
                + *x.get_unchecked(n - 6) * *self.low4.get_unchecked(n - 6);
            *b.get_unchecked_mut(n - 1) = *x.get_unchecked(n - 3) * *self.low2.get_unchecked(n - 3)
                + *x.get_unchecked(n - 5) * *self.low4.get_unchecked(n - 5);
        }
    }

    fn solve<T: ScalarOperand<A>>(&self, b: &[T], x: &mut [T]) {
        // Multiply rhs
        unsafe {
            for i in 0..self.cols {
                *x.get_unchecked_mut(i) = *b.get_unchecked(i) * *self.diag.get_unchecked(i)
                    + *b.get_unchecked(i + 2) * *self.low2.get_unchecked(i)
                    + *b.get_unchecked(i + 4) * *self.low4.get_unchecked(i);
            }
        }
        // Solve tridiagonal system
        pdma2(
            &self.mtm_off4,
            &self.mtm_off2,
            &self.mtm_diag,
            &self.mtm_off2,
            &self.mtm_off4,
            x,
        );
    }
}
