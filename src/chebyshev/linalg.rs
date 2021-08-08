//! # Linalg functions for chebyshev space
use crate::Scalar;
use ndarray::prelude::*;
use std::ops::{Add, Div, Mul, Sub};

/// Tridiagonal matrix solver
///     Ax = d
/// where A is banded with diagonals in offsets -2, 0, 2
///
/// a: sub-diagonal (-2)
/// b: main-diagonal
/// c: sub-diagonal (+2)
#[allow(clippy::many_single_char_names)]
pub fn tdma<S1, S2, T1, T2>(
    a: &ArrayBase<S1, Ix1>,
    b: &ArrayBase<S1, Ix1>,
    c: &ArrayBase<S1, Ix1>,
    d: &mut ArrayBase<S2, Ix1>,
) where
    S1: ndarray::Data<Elem = T1>,
    S2: ndarray::Data<Elem = T2> + ndarray::DataMut,
    T1: Scalar,
    T2: Scalar
        + Add<T1, Output = T2>
        + Mul<T1, Output = T2>
        + Div<T1, Output = T2>
        + Sub<T1, Output = T2>,
{
    let n = d.len();
    let mut x = Array1::<T2>::zeros(n);
    let mut w = Array1::<T1>::zeros(n - 2);
    let mut g = Array1::<T2>::zeros(n);

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
    x[n - 1] = g[n - 1];
    x[n - 2] = g[n - 2];
    for i in (1..n - 1).rev() {
        x[i - 1] = g[i - 1] - x[i + 1] * w[i - 1];
    }

    d.assign(&x);
}
