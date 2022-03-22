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
