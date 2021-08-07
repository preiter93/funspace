//! # Linalg functions for chebyshev space
use crate::Scalar;
use ndarray::prelude::*;

/// Tridiagonal matrix solver
///     Ax = d
/// where A is banded with diagonals in offsets -2, 0, 2
///
/// a: sub-diagonal (-2)
/// b: main-diagonal
/// c: sub-diagonal (+2)
#[allow(clippy::many_single_char_names)]
pub fn tdma<S1, S2, T>(
    a: &ArrayBase<S1, Ix1>,
    b: &ArrayBase<S1, Ix1>,
    c: &ArrayBase<S1, Ix1>,
    d: &mut ArrayBase<S2, Ix1>,
) where
    S1: ndarray::Data<Elem = T>,
    S2: ndarray::Data<Elem = T> + ndarray::DataMut,
    T: Scalar,
{
    let n = d.len();
    let mut x = Array1::zeros(n);
    let mut w = Array1::zeros(n - 2);
    let mut g = Array1::zeros(n);

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
        g[i] = (d[i] - a[i - 2] * g[i - 2]) / (b[i] - a[i - 2] * w[i - 2]);
    }

    // Back substitution
    x[n - 1] = g[n - 1];
    x[n - 2] = g[n - 2];
    for i in (1..n - 1).rev() {
        x[i - 1] = g[i - 1] - w[i - 1] * x[i + 1];
    }

    d.assign(&x);
}
