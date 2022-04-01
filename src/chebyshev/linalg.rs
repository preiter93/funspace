//! # Linalg functions for chebyshev space
use num_traits::Zero;
use std::clone::Clone;
use std::ops::{Add, Div, Mul, Sub};

/// Tridiagonal matrix solver
///     Ax = d
/// where A is banded with diagonals in offsets -2, 0, 2
///
/// a: sub-diagonal (-2)
/// b: main-diagonal
/// c: sub-diagonal (+2)
#[allow(clippy::many_single_char_names)]
#[inline]
pub fn tdma<T1, T2>(a: &[T1], b: &[T1], c: &[T1], d: &mut [T2])
where
    T1: Mul<Output = T1>
        + Sub<Output = T1>
        + Div<Output = T1>
        + Add<Output = T1>
        + Zero
        + Clone
        + Copy,
    T2: Mul<Output = T2>
        + Sub<Output = T2>
        + Div<Output = T2>
        + Add<Output = T2>
        + Add<T1, Output = T2>
        + Mul<T1, Output = T2>
        + Div<T1, Output = T2>
        + Sub<T1, Output = T2>
        + Zero
        + Clone
        + Copy,
{
    let n = d.len();
    assert!(n > 2, "Error in tdma: Size must be > 2, try a larger base!");
    let mut w = vec![T1::zero(); n - 2];
    let mut g = vec![T2::zero(); n];
    unsafe {
        // Forward sweep
        *w.get_unchecked_mut(0) = *c.get_unchecked(0) / *b.get_unchecked(0);
        *g.get_unchecked_mut(0) = *d.get_unchecked(0) / *b.get_unchecked(0);
        if c.len() > 1 {
            *w.get_unchecked_mut(1) = *c.get_unchecked(1) / *b.get_unchecked(1);
        }
        *g.get_unchecked_mut(1) = *d.get_unchecked(1) / *b.get_unchecked(1);

        for i in 2..n - 2 {
            *w.get_unchecked_mut(i) = *c.get_unchecked(i)
                / (*b.get_unchecked(i) - *a.get_unchecked(i - 2) * *w.get_unchecked(i - 2));
        }
        for i in 2..n {
            *g.get_unchecked_mut(i) = (*d.get_unchecked(i)
                - *g.get_unchecked(i - 2) * *a.get_unchecked(i - 2))
                / (*b.get_unchecked(i) - *a.get_unchecked(i - 2) * *w.get_unchecked(i - 2));
        }

        // Back substitution
        *d.get_unchecked_mut(n - 1) = *g.get_unchecked(n - 1);
        *d.get_unchecked_mut(n - 2) = *g.get_unchecked(n - 2);
        for i in (1..n - 1).rev() {
            *d.get_unchecked_mut(i - 1) =
                *g.get_unchecked(i - 1) - *d.get_unchecked(i + 1) * *w.get_unchecked(i - 1);
        }
    }
}

/// Tridiagonal matrix solver
///     Ax = d
/// where A is banded with diagonals in offsets -2, 0, 2
///
/// a: sub-diagonal (-2)
/// b: main-diagonal
/// c: sub-diagonal (+2)
#[allow(clippy::many_single_char_names, dead_code)]
#[inline]
pub fn tdma_checked<T1, T2>(a: &[T1], b: &[T1], c: &[T1], d: &mut [T2])
where
    T1: Mul<Output = T1>
        + Sub<Output = T1>
        + Div<Output = T1>
        + Add<Output = T1>
        + Zero
        + Clone
        + Copy,
    T2: Mul<Output = T2>
        + Sub<Output = T2>
        + Div<Output = T2>
        + Add<Output = T2>
        + Add<T1, Output = T2>
        + Mul<T1, Output = T2>
        + Div<T1, Output = T2>
        + Sub<T1, Output = T2>
        + Zero
        + Clone
        + Copy,
{
    let n = d.len();
    assert!(n > 2, "Error in tdma: Size must be > 2, try a larger base!");
    let mut w = vec![T1::zero(); n - 2];
    let mut g = vec![T2::zero(); n];

    // Forward sweep
    w[0] = c[0] / b[0];
    g[0] = d[0] / b[0];
    if c.len() > 1 {
        w[1] = c[1] / b[1];
    }
    g[1] = d[1] / b[1];

    for i in 2..n - 2 {
        w[i] = c[i] / (b[i] - a[i - 2] * w[i - 2]);
    }
    for i in 2..n {
        g[i] = (d[i] - g[i - 2] * a[i - 2]) / (b[i] - a[i - 2] * w[i - 2]);
    }

    // Back substitution
    d[n - 1] = g[n - 1];
    d[n - 2] = g[n - 2];
    for i in (1..n - 1).rev() {
        d[i - 1] = g[i - 1] - d[i + 1] * w[i - 1];
    }
}

#[allow(clippy::many_single_char_names)]
#[inline]
pub fn pdma<T1, T2>(l2: &[T1], l1: &[T1], d0: &[T1], u1: &[T1], u2: &[T1], rhs: &mut [T2])
where
    T1: Mul<Output = T1>
        + Sub<Output = T1>
        + Div<Output = T1>
        + Add<Output = T1>
        + Zero
        + Clone
        + Copy,
    T2: Mul<Output = T2>
        + Sub<Output = T2>
        + Div<Output = T2>
        + Add<Output = T2>
        + Add<T1, Output = T2>
        + Mul<T1, Output = T2>
        + Div<T1, Output = T2>
        + Sub<T1, Output = T2>
        + Zero
        + Clone
        + Copy,
{
    let n = rhs.len();
    assert!(n > 3, "Error in pdma: Size must be > 3, try a larger base!");

    let mut al = vec![T1::zero(); n];
    let mut be = vec![T1::zero(); n];
    let mut ze = vec![T2::zero(); n];
    let mut ga = vec![T1::zero(); n];
    let mut mu = vec![T1::zero(); n];

    unsafe {
        *mu.get_unchecked_mut(0) = *d0.get_unchecked(0);
        *al.get_unchecked_mut(0) = *u1.get_unchecked(0) / *mu.get_unchecked(0);
        *be.get_unchecked_mut(0) = *u2.get_unchecked(0) / *mu.get_unchecked(0);
        *ze.get_unchecked_mut(0) = *rhs.get_unchecked(0) / *mu.get_unchecked(0);

        *ga.get_unchecked_mut(1) = *l1.get_unchecked(0);
        *mu.get_unchecked_mut(1) =
            *d0.get_unchecked(1) - *al.get_unchecked(0) * *ga.get_unchecked(1);
        *al.get_unchecked_mut(1) = (*u1.get_unchecked(1)
            - *be.get_unchecked(0) * *ga.get_unchecked(1))
            / *mu.get_unchecked(1);
        *be.get_unchecked_mut(1) = *u2.get_unchecked(1) / *mu.get_unchecked(1);
        *ze.get_unchecked_mut(1) = (*rhs.get_unchecked(1)
            - *ze.get_unchecked(0) * *ga.get_unchecked(1))
            / *mu.get_unchecked(1);

        for i in 2..n - 2 {
            *ga.get_unchecked_mut(i) =
                *l1.get_unchecked(i - 1) - *al.get_unchecked(i - 2) * *l2.get_unchecked(i - 2);
            *mu.get_unchecked_mut(i) = *d0.get_unchecked(i)
                - *be.get_unchecked(i - 2) * *l2.get_unchecked(i - 2)
                - *al.get_unchecked(i - 1) * *ga.get_unchecked(i);
            *al.get_unchecked_mut(i) = (*u1.get_unchecked(i)
                - *be.get_unchecked(i - 1) * *ga.get_unchecked(i))
                / *mu.get_unchecked(i);
            *be.get_unchecked_mut(i) = *u2.get_unchecked(i) / *mu.get_unchecked(i);
            *ze.get_unchecked_mut(i) = (*rhs.get_unchecked(i)
                - *ze.get_unchecked(i - 2) * *l2.get_unchecked(i - 2)
                - *ze.get_unchecked(i - 1) * *ga.get_unchecked(i))
                / *mu.get_unchecked(i);
        }

        *ga.get_unchecked_mut(n - 2) =
            *l1.get_unchecked(n - 3) - *al.get_unchecked(n - 4) * *l2.get_unchecked(n - 4);
        *mu.get_unchecked_mut(n - 2) = *d0.get_unchecked(n - 2)
            - *be.get_unchecked(n - 4) * *l2.get_unchecked(n - 4)
            - *al.get_unchecked(n - 3) * *ga.get_unchecked(n - 2);
        *al.get_unchecked_mut(n - 2) = (*u1.get_unchecked(n - 2)
            - *be.get_unchecked(n - 3) * *ga.get_unchecked(n - 2))
            / *mu.get_unchecked(n - 2);

        *ga.get_unchecked_mut(n - 1) =
            *l1.get_unchecked(n - 2) - *al.get_unchecked(n - 3) * *l2.get_unchecked(n - 3);
        *mu.get_unchecked_mut(n - 1) = *d0.get_unchecked(n - 1)
            - *be.get_unchecked(n - 3) * *l2.get_unchecked(n - 3)
            - *al.get_unchecked(n - 2) * *ga.get_unchecked(n - 1);

        *ze.get_unchecked_mut(n - 2) = (*rhs.get_unchecked(n - 2)
            - *ze.get_unchecked(n - 4) * *l2.get_unchecked(n - 4)
            - *ze.get_unchecked(n - 3) * *ga.get_unchecked(n - 2))
            / *mu.get_unchecked(n - 2);
        *ze.get_unchecked_mut(n - 1) = (*rhs.get_unchecked(n - 1)
            - *ze.get_unchecked(n - 3) * *l2.get_unchecked(n - 3)
            - *ze.get_unchecked(n - 2) * *ga.get_unchecked(n - 1))
            / *mu.get_unchecked(n - 1);

        // Backward substitution
        *rhs.get_unchecked_mut(n - 1) = *ze.get_unchecked(n - 1);
        *rhs.get_unchecked_mut(n - 2) =
            *ze.get_unchecked(n - 2) - *rhs.get_unchecked(n - 1) * *al.get_unchecked(n - 2);

        for i in (0..n - 2).rev() {
            *rhs.get_unchecked_mut(i) = *ze.get_unchecked(i)
                - *rhs.get_unchecked(i + 1) * *al.get_unchecked(i)
                - *rhs.get_unchecked(i + 2) * *be.get_unchecked(i);
        }
    }
}

#[allow(clippy::many_single_char_names, dead_code)]
#[inline]
pub fn pdma_checked<T1, T2>(l2: &[T1], l1: &[T1], d0: &[T1], u1: &[T1], u2: &[T1], rhs: &mut [T2])
where
    T1: Mul<Output = T1>
        + Sub<Output = T1>
        + Div<Output = T1>
        + Add<Output = T1>
        + Zero
        + Clone
        + Copy,
    T2: Mul<Output = T2>
        + Sub<Output = T2>
        + Div<Output = T2>
        + Add<Output = T2>
        + Add<T1, Output = T2>
        + Mul<T1, Output = T2>
        + Div<T1, Output = T2>
        + Sub<T1, Output = T2>
        + Zero
        + Clone
        + Copy,
{
    let n = rhs.len();
    assert!(n > 3, "Error in pdma: Size must be > 3, try a larger base!");

    let mut al = vec![T1::zero(); n];
    let mut be = vec![T1::zero(); n];
    let mut ze = vec![T2::zero(); n];
    let mut ga = vec![T1::zero(); n];
    let mut mu = vec![T1::zero(); n];

    mu[0] = d0[0];
    al[0] = u1[0] / mu[0];
    be[0] = u2[0] / mu[0];
    ze[0] = rhs[0] / mu[0];

    ga[1] = l1[0];
    mu[1] = d0[1] - al[0] * ga[1];
    al[1] = (u1[1] - be[0] * ga[1]) / mu[1];
    be[1] = u2[1] / mu[1];
    ze[1] = (rhs[1] - ze[0] * ga[1]) / mu[1];

    for i in 2..n - 2 {
        ga[i] = l1[i - 1] - al[i - 2] * l2[i - 2];
        mu[i] = d0[i] - be[i - 2] * l2[i - 2] - al[i - 1] * ga[i];
        al[i] = (u1[i] - be[i - 1] * ga[i]) / mu[i];
        be[i] = u2[i] / mu[i];
        ze[i] = (rhs[i] - ze[i - 2] * l2[i - 2] - ze[i - 1] * ga[i]) / mu[i];
    }

    ga[n - 2] = l1[n - 3] - al[n - 4] * l2[n - 4];
    mu[n - 2] = d0[n - 2] - be[n - 4] * l2[n - 4] - al[n - 3] * ga[n - 2];
    al[n - 2] = (u1[n - 2] - be[n - 3] * ga[n - 2]) / mu[n - 2];

    ga[n - 1] = l1[n - 2] - al[n - 3] * l2[n - 3];
    mu[n - 1] = d0[n - 1] - be[n - 3] * l2[n - 3] - al[n - 2] * ga[n - 1];

    ze[n - 2] = (rhs[n - 2] - ze[n - 4] * l2[n - 4] - ze[n - 3] * ga[n - 2]) / mu[n - 2];
    ze[n - 1] = (rhs[n - 1] - ze[n - 3] * l2[n - 3] - ze[n - 2] * ga[n - 1]) / mu[n - 1];

    // Backward substitution
    rhs[n - 1] = ze[n - 1];
    rhs[n - 2] = ze[n - 2] - rhs[n - 1] * al[n - 2];

    for i in (0..n - 2).rev() {
        rhs[i] = ze[i] - rhs[i + 1] * al[i] - rhs[i + 2] * be[i];
    }
}

#[allow(clippy::many_single_char_names, dead_code)]
#[inline]
pub fn pdma2<T1, T2>(l4: &[T1], l2: &[T1], d0: &[T1], u2: &[T1], u4: &[T1], rhs: &mut [T2])
where
    T1: Mul<Output = T1>
        + Sub<Output = T1>
        + Div<Output = T1>
        + Add<Output = T1>
        + Zero
        + Clone
        + Copy,
    T2: Mul<Output = T2>
        + Sub<Output = T2>
        + Div<Output = T2>
        + Add<Output = T2>
        + Add<T1, Output = T2>
        + Mul<T1, Output = T2>
        + Div<T1, Output = T2>
        + Sub<T1, Output = T2>
        + Zero
        + Clone
        + Copy,
{
    let n = rhs.len();
    assert!(
        n > 7,
        "Error in pdma2: Size must be > 7, try a larger base!"
    );

    let mut al = vec![T1::zero(); n];
    let mut be = vec![T1::zero(); n];
    let mut ze = vec![T2::zero(); n];
    let mut ga = vec![T1::zero(); n];
    let mut mu = vec![T1::zero(); n];
    unsafe {
        *mu.get_unchecked_mut(0) = *d0.get_unchecked(0);
        *al.get_unchecked_mut(0) = *u2.get_unchecked(0) / *mu.get_unchecked(0);
        *be.get_unchecked_mut(0) = *u4.get_unchecked(0) / *mu.get_unchecked(0);
        *ze.get_unchecked_mut(0) = *rhs.get_unchecked(0) / *mu.get_unchecked(0);

        *mu.get_unchecked_mut(1) = *d0.get_unchecked(1);
        *al.get_unchecked_mut(1) = *u2.get_unchecked(1) / *mu.get_unchecked(1);
        *be.get_unchecked_mut(1) = *u4.get_unchecked(1) / *mu.get_unchecked(1);
        *ze.get_unchecked_mut(1) = *rhs.get_unchecked(1) / *mu.get_unchecked(1);

        *ga.get_unchecked_mut(2) = *l2.get_unchecked(0);
        *mu.get_unchecked_mut(2) =
            *d0.get_unchecked(2) - *al.get_unchecked(0) * *ga.get_unchecked(2);
        *al.get_unchecked_mut(2) = (*u2.get_unchecked(2)
            - *be.get_unchecked(0) * *ga.get_unchecked(2))
            / *mu.get_unchecked(2);
        *be.get_unchecked_mut(2) = *u4.get_unchecked(2) / *mu.get_unchecked(2);
        *ze.get_unchecked_mut(2) = (*rhs.get_unchecked(2)
            - *ze.get_unchecked(0) * *ga.get_unchecked(2))
            / *mu.get_unchecked(2);

        *ga.get_unchecked_mut(3) = *l2.get_unchecked(1);
        *mu.get_unchecked_mut(3) =
            *d0.get_unchecked(3) - *al.get_unchecked(1) * *ga.get_unchecked(3);
        *al.get_unchecked_mut(3) = (*u2.get_unchecked(3)
            - *be.get_unchecked(1) * *ga.get_unchecked(3))
            / *mu.get_unchecked(3);
        *be.get_unchecked_mut(3) = *u4.get_unchecked(3) / *mu.get_unchecked(3);
        *ze.get_unchecked_mut(3) = (*rhs.get_unchecked(3)
            - *ze.get_unchecked(1) * *ga.get_unchecked(3))
            / *mu.get_unchecked(3);

        for i in 4..n - 4 {
            *ga.get_unchecked_mut(i) =
                *l2.get_unchecked(i - 2) - *al.get_unchecked(i - 4) * *l4.get_unchecked(i - 4);
            *mu.get_unchecked_mut(i) = *d0.get_unchecked(i)
                - *be.get_unchecked(i - 4) * *l4.get_unchecked(i - 4)
                - *al.get_unchecked(i - 2) * *ga.get_unchecked(i);
            *al.get_unchecked_mut(i) = (*u2.get_unchecked(i)
                - *be.get_unchecked(i - 2) * *ga.get_unchecked(i))
                / *mu.get_unchecked(i);
            *be.get_unchecked_mut(i) = *u4.get_unchecked(i) / *mu.get_unchecked(i);
            *ze.get_unchecked_mut(i) = (*rhs.get_unchecked(i)
                - *ze.get_unchecked(i - 4) * *l4.get_unchecked(i - 4)
                - *ze.get_unchecked(i - 2) * *ga.get_unchecked(i))
                / *mu.get_unchecked(i);
        }

        *ga.get_unchecked_mut(n - 4) =
            *l2.get_unchecked(n - 6) - *al.get_unchecked(n - 8) * *l4.get_unchecked(n - 8);
        *mu.get_unchecked_mut(n - 4) = *d0.get_unchecked(n - 4)
            - *be.get_unchecked(n - 8) * *l4.get_unchecked(n - 8)
            - *al.get_unchecked(n - 6) * *ga.get_unchecked(n - 4);
        *al.get_unchecked_mut(n - 4) = (*u2.get_unchecked(n - 4)
            - *be.get_unchecked(n - 6) * *ga.get_unchecked(n - 4))
            / *mu.get_unchecked(n - 4);

        *ga.get_unchecked_mut(n - 3) =
            *l2.get_unchecked(n - 5) - *al.get_unchecked(n - 7) * *l4.get_unchecked(n - 7);
        *mu.get_unchecked_mut(n - 3) = *d0.get_unchecked(n - 3)
            - *be.get_unchecked(n - 7) * *l4.get_unchecked(n - 7)
            - *al.get_unchecked(n - 5) * *ga.get_unchecked(n - 3);
        *al.get_unchecked_mut(n - 3) = (*u2.get_unchecked(n - 3)
            - *be.get_unchecked(n - 5) * *ga.get_unchecked(n - 3))
            / *mu.get_unchecked(n - 3);

        *ga.get_unchecked_mut(n - 2) =
            *l2.get_unchecked(n - 4) - *al.get_unchecked(n - 6) * *l4.get_unchecked(n - 6);
        *mu.get_unchecked_mut(n - 2) = *d0.get_unchecked(n - 2)
            - *be.get_unchecked(n - 6) * *l4.get_unchecked(n - 6)
            - *al.get_unchecked(n - 4) * *ga.get_unchecked(n - 2);

        *ga.get_unchecked_mut(n - 1) =
            *l2.get_unchecked(n - 3) - *al.get_unchecked(n - 5) * *l4.get_unchecked(n - 5);
        *mu.get_unchecked_mut(n - 1) = *d0.get_unchecked(n - 1)
            - *be.get_unchecked(n - 5) * *l4.get_unchecked(n - 5)
            - *al.get_unchecked(n - 3) * *ga.get_unchecked(n - 1);

        *ze.get_unchecked_mut(n - 4) = (*rhs.get_unchecked(n - 4)
            - *ze.get_unchecked(n - 8) * *l4.get_unchecked(n - 8)
            - *ze.get_unchecked(n - 6) * *ga.get_unchecked(n - 4))
            / *mu.get_unchecked(n - 4);
        *ze.get_unchecked_mut(n - 3) = (*rhs.get_unchecked(n - 3)
            - *ze.get_unchecked(n - 7) * *l4.get_unchecked(n - 7)
            - *ze.get_unchecked(n - 5) * *ga.get_unchecked(n - 3))
            / *mu.get_unchecked(n - 3);
        *ze.get_unchecked_mut(n - 2) = (*rhs.get_unchecked(n - 2)
            - *ze.get_unchecked(n - 6) * *l4.get_unchecked(n - 6)
            - *ze.get_unchecked(n - 4) * *ga.get_unchecked(n - 2))
            / *mu.get_unchecked(n - 2);
        *ze.get_unchecked_mut(n - 1) = (*rhs.get_unchecked(n - 1)
            - *ze.get_unchecked(n - 5) * *l4.get_unchecked(n - 5)
            - *ze.get_unchecked(n - 3) * *ga.get_unchecked(n - 1))
            / *mu.get_unchecked(n - 1);

        // Backward substitution
        *rhs.get_unchecked_mut(n - 1) = *ze.get_unchecked(n - 1);
        *rhs.get_unchecked_mut(n - 2) = *ze.get_unchecked(n - 2);
        *rhs.get_unchecked_mut(n - 3) =
            *ze.get_unchecked(n - 3) - *rhs.get_unchecked(n - 1) * *al.get_unchecked(n - 3);
        *rhs.get_unchecked_mut(n - 4) =
            *ze.get_unchecked(n - 4) - *rhs.get_unchecked(n - 2) * *al.get_unchecked(n - 4);

        for i in (0..n - 4).rev() {
            *rhs.get_unchecked_mut(i) = *ze.get_unchecked(i)
                - *rhs.get_unchecked(i + 2) * *al.get_unchecked(i)
                - *rhs.get_unchecked(i + 4) * *be.get_unchecked(i);
        }
    }
}

#[allow(clippy::many_single_char_names, dead_code)]
#[inline]
pub fn pdma2_checked<T1, T2>(l4: &[T1], l2: &[T1], d0: &[T1], u2: &[T1], u4: &[T1], rhs: &mut [T2])
where
    T1: Mul<Output = T1>
        + Sub<Output = T1>
        + Div<Output = T1>
        + Add<Output = T1>
        + Zero
        + Clone
        + Copy,
    T2: Mul<Output = T2>
        + Sub<Output = T2>
        + Div<Output = T2>
        + Add<Output = T2>
        + Add<T1, Output = T2>
        + Mul<T1, Output = T2>
        + Div<T1, Output = T2>
        + Sub<T1, Output = T2>
        + Zero
        + Clone
        + Copy,
{
    let n = rhs.len();
    assert!(
        n > 7,
        "Error in pdma2: Size must be > 7, try a larger base!"
    );

    let mut al = vec![T1::zero(); n];
    let mut be = vec![T1::zero(); n];
    let mut ze = vec![T2::zero(); n];
    let mut ga = vec![T1::zero(); n];
    let mut mu = vec![T1::zero(); n];

    mu[0] = d0[0];
    al[0] = u2[0] / mu[0];
    be[0] = u4[0] / mu[0];
    ze[0] = rhs[0] / mu[0];

    mu[1] = d0[1];
    al[1] = u2[1] / mu[1];
    be[1] = u4[1] / mu[1];
    ze[1] = rhs[1] / mu[1];

    ga[2] = l2[0];
    mu[2] = d0[2] - al[0] * ga[2];
    al[2] = (u2[2] - be[0] * ga[2]) / mu[2];
    be[2] = u4[2] / mu[2];
    ze[2] = (rhs[2] - ze[0] * ga[2]) / mu[2];

    ga[3] = l2[1];
    mu[3] = d0[3] - al[1] * ga[3];
    al[3] = (u2[3] - be[1] * ga[3]) / mu[3];
    be[3] = u4[3] / mu[3];
    ze[3] = (rhs[3] - ze[1] * ga[3]) / mu[3];

    for i in 4..n - 4 {
        ga[i] = l2[i - 2] - al[i - 4] * l4[i - 4];
        mu[i] = d0[i] - be[i - 4] * l4[i - 4] - al[i - 2] * ga[i];
        al[i] = (u2[i] - be[i - 2] * ga[i]) / mu[i];
        be[i] = u4[i] / mu[i];
        ze[i] = (rhs[i] - ze[i - 4] * l4[i - 4] - ze[i - 2] * ga[i]) / mu[i];
    }

    ga[n - 4] = l2[n - 6] - al[n - 8] * l4[n - 8];
    mu[n - 4] = d0[n - 4] - be[n - 8] * l4[n - 8] - al[n - 6] * ga[n - 4];
    al[n - 4] = (u2[n - 4] - be[n - 6] * ga[n - 4]) / mu[n - 4];

    ga[n - 3] = l2[n - 5] - al[n - 7] * l4[n - 7];
    mu[n - 3] = d0[n - 3] - be[n - 7] * l4[n - 7] - al[n - 5] * ga[n - 3];
    al[n - 3] = (u2[n - 3] - be[n - 5] * ga[n - 3]) / mu[n - 3];

    ga[n - 2] = l2[n - 4] - al[n - 6] * l4[n - 6];
    mu[n - 2] = d0[n - 2] - be[n - 6] * l4[n - 6] - al[n - 4] * ga[n - 2];

    ga[n - 1] = l2[n - 3] - al[n - 5] * l4[n - 5];
    mu[n - 1] = d0[n - 1] - be[n - 5] * l4[n - 5] - al[n - 3] * ga[n - 1];

    ze[n - 4] = (rhs[n - 4] - ze[n - 8] * l4[n - 8] - ze[n - 6] * ga[n - 4]) / mu[n - 4];
    ze[n - 3] = (rhs[n - 3] - ze[n - 7] * l4[n - 7] - ze[n - 5] * ga[n - 3]) / mu[n - 3];
    ze[n - 2] = (rhs[n - 2] - ze[n - 6] * l4[n - 6] - ze[n - 4] * ga[n - 2]) / mu[n - 2];
    ze[n - 1] = (rhs[n - 1] - ze[n - 5] * l4[n - 5] - ze[n - 3] * ga[n - 1]) / mu[n - 1];

    // Backward substitution
    rhs[n - 1] = ze[n - 1];
    rhs[n - 2] = ze[n - 2];
    rhs[n - 3] = ze[n - 3] - rhs[n - 1] * al[n - 3];
    rhs[n - 4] = ze[n - 4] - rhs[n - 2] * al[n - 4];

    for i in (0..n - 4).rev() {
        rhs[i] = ze[i] - rhs[i + 2] * al[i] - rhs[i + 4] * be[i];
    }
}
