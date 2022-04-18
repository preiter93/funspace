//! Tridiagonal matrix solver
//!     Ax = d
//! where A is banded with diagonals in offsets -2, 0, 2
use num_traits::Zero;
use std::clone::Clone;
use std::ops::{Add, Div, Mul, Sub};

/// Tridiagonal matrix solver
///     Ax = rhs
/// where A is banded with diagonals in offsets -2, 0, 2
///
/// l2: sub-diagonal (-2)
/// d0: main-diagonal
/// u2: sub-diagonal (+2)
#[allow(clippy::many_single_char_names)]
#[inline]
pub fn tdma<T1, T2>(l2: &[T1], d0: &[T1], u2: &[T1], rhs: &mut [T2])
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
    assert!(n > 2, "Error in tdma: Size = {} too small!", n);
    let mut w = vec![T1::zero(); n - 2];
    let mut g = vec![T2::zero(); n];
    unsafe {
        // Forward sweep
        *w.get_unchecked_mut(0) = *u2.get_unchecked(0) / *d0.get_unchecked(0);
        *g.get_unchecked_mut(0) = *rhs.get_unchecked(0) / *d0.get_unchecked(0);
        if u2.len() > 1 {
            *w.get_unchecked_mut(1) = *u2.get_unchecked(1) / *d0.get_unchecked(1);
        }
        *g.get_unchecked_mut(1) = *rhs.get_unchecked(1) / *d0.get_unchecked(1);

        for i in 2..n - 2 {
            let tmp = *d0.get_unchecked(i) - *l2.get_unchecked(i - 2) * *w.get_unchecked(i - 2);
            *w.get_unchecked_mut(i) = *u2.get_unchecked(i) / tmp;
            *g.get_unchecked_mut(i) =
                (*rhs.get_unchecked(i) - *g.get_unchecked(i - 2) * *l2.get_unchecked(i - 2)) / tmp;
        }

        if n > 3 {
            *g.get_unchecked_mut(n - 2) = (*rhs.get_unchecked(n - 2)
                - *g.get_unchecked(n - 4) * *l2.get_unchecked(n - 4))
                / (*d0.get_unchecked(n - 2) - *l2.get_unchecked(n - 4) * *w.get_unchecked(n - 4));
        }

        if n > 2 {
            *g.get_unchecked_mut(n - 1) = (*rhs.get_unchecked(n - 1)
                - *g.get_unchecked(n - 3) * *l2.get_unchecked(n - 3))
                / (*d0.get_unchecked(n - 1) - *l2.get_unchecked(n - 3) * *w.get_unchecked(n - 3));
        }

        // Back substitution
        *rhs.get_unchecked_mut(n - 1) = *g.get_unchecked(n - 1);
        *rhs.get_unchecked_mut(n - 2) = *g.get_unchecked(n - 2);
        for i in (1..n - 1).rev() {
            *rhs.get_unchecked_mut(i - 1) =
                *g.get_unchecked(i - 1) - *rhs.get_unchecked(i + 1) * *w.get_unchecked(i - 1);
        }
    }
}

/// Tridiagonal matrix solver
///     Ax = rhs
/// where A is banded with diagonals in offsets -2, 0, 2
///
/// l2: sub-diagonal (-2)
/// d0: main-diagonal
/// u2: sub-diagonal (+2)
#[allow(clippy::many_single_char_names, dead_code)]
#[inline]
pub fn tdma_checked<T1, T2>(l2: &[T1], d0: &[T1], u2: &[T1], rhs: &mut [T2])
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
    assert!(n > 2, "Error in tdma: Size = {} too small!", n);
    let mut w = vec![T1::zero(); n - 2];
    let mut g = vec![T2::zero(); n];

    // Forward sweep
    w[0] = u2[0] / d0[0];
    g[0] = rhs[0] / d0[0];
    if u2.len() > 1 {
        w[1] = u2[1] / d0[1];
    }
    g[1] = rhs[1] / d0[1];

    for i in 2..n - 2 {
        w[i] = u2[i] / (d0[i] - l2[i - 2] * w[i - 2]);
    }
    for i in 2..n {
        g[i] = (rhs[i] - g[i - 2] * l2[i - 2]) / (d0[i] - l2[i - 2] * w[i - 2]);
    }

    // Back substitution
    rhs[n - 1] = g[n - 1];
    rhs[n - 2] = g[n - 2];
    for i in (1..n - 1).rev() {
        rhs[i - 1] = g[i - 1] - rhs[i + 1] * w[i - 1];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};
    #[test]
    /// A x = b
    fn test_tdma() {
        let n = 6;
        let b = (1..n + 1).map(|x| x as f64).collect::<Vec<f64>>();
        // Diagonals (randomly chosen)
        let l2 = (1..n - 1).map(|x| 1.5 * x as f64).collect::<Vec<f64>>();
        let d0 = (1..n + 1).map(|x| 1.0 * x as f64).collect::<Vec<f64>>();
        let u2 = (1..n - 1).map(|x| -0.5 * x as f64).collect::<Vec<f64>>();
        // Fill diagonals in matrix, used for assertion of result
        let mut mat = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            mat[[i, i]] = d0[i];
        }
        for i in 0..n - 2 {
            mat[[i + 2, i]] = l2[i];
        }
        for i in 0..n - 2 {
            mat[[i, i + 2]] = u2[i];
        }

        // Solve
        let mut rhs = b.clone();
        tdma(&l2, &d0, &u2, &mut rhs);
        // Assert
        let b2 = mat.dot(&Array1::from_vec(rhs));
        for (v1, v2) in b.iter().zip(b2.iter()) {
            assert!((v1 - v2).abs() < 1e-6, "TDMA failed, {} /= {}.", v1, v2);
        }

        // Solve checked
        let mut rhs = b.clone();
        tdma_checked(&l2, &d0, &u2, &mut rhs);
        // Assert
        let b2 = mat.dot(&Array1::from_vec(rhs));
        for (v1, v2) in b.iter().zip(b2.iter()) {
            assert!((v1 - v2).abs() < 1e-6, "TDMA failed, {} /= {}.", v1, v2);
        }
    }
}
