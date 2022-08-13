//! Pentadiagonal matrix solver
//!     Ax = d
//! where A is banded with diagonals in offsets -2, -1, 0, 1, 2
use num_traits::Zero;
use std::clone::Clone;
use std::ops::{Add, Div, Mul, Sub};

/// Pentadiagonal matrix solver
///     Ax = d
/// where A is banded with diagonals in offsets -2, -1, 0, 1, 2
///
/// l2: sub-diagonal (-2)
/// l1: sub-diagonal (-1)
/// d0: main-diagonal
/// u1: sub-diagonal (+1)
/// u2: sub-diagonal (+2)
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
    assert!(n > 3, "Error in pdma: Size = {} too small!", n);

    let mut al = vec![T1::zero(); n - 1];
    let mut be = vec![T1::zero(); n - 2];
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
        *ze.get_unchecked_mut(n - 2) = (*rhs.get_unchecked(n - 2)
            - *ze.get_unchecked(n - 4) * *l2.get_unchecked(n - 4)
            - *ze.get_unchecked(n - 3) * *ga.get_unchecked(n - 2))
            / *mu.get_unchecked(n - 2);

        *ga.get_unchecked_mut(n - 1) =
            *l1.get_unchecked(n - 2) - *al.get_unchecked(n - 3) * *l2.get_unchecked(n - 3);
        *mu.get_unchecked_mut(n - 1) = *d0.get_unchecked(n - 1)
            - *be.get_unchecked(n - 3) * *l2.get_unchecked(n - 3)
            - *al.get_unchecked(n - 2) * *ga.get_unchecked(n - 1);
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

/// Pentadiagonal matrix solver
///     Ax = d
/// where A is banded with diagonals in offsets -2, -1, 0, 1, 2
///
/// l2: sub-diagonal (-2)
/// l1: sub-diagonal (-1)
/// d0: main-diagonal
/// u1: sub-diagonal (+1)
/// u2: sub-diagonal (+2)
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
    assert!(n > 3, "Error in pdma: Size = {} too small!", n);

    let mut al = vec![T1::zero(); n - 1];
    let mut be = vec![T1::zero(); n - 2];
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
    ze[n - 2] = (rhs[n - 2] - ze[n - 4] * l2[n - 4] - ze[n - 3] * ga[n - 2]) / mu[n - 2];

    ga[n - 1] = l1[n - 2] - al[n - 3] * l2[n - 3];
    mu[n - 1] = d0[n - 1] - be[n - 3] * l2[n - 3] - al[n - 2] * ga[n - 1];
    ze[n - 1] = (rhs[n - 1] - ze[n - 3] * l2[n - 3] - ze[n - 2] * ga[n - 1]) / mu[n - 1];

    // Backward substitution
    rhs[n - 1] = ze[n - 1];
    rhs[n - 2] = ze[n - 2] - rhs[n - 1] * al[n - 2];

    for i in (0..n - 2).rev() {
        rhs[i] = ze[i] - rhs[i + 1] * al[i] - rhs[i + 2] * be[i];
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use ndarray::{Array1, Array2};
//     #[test]
//     /// A x = b
//     fn test_pdma() {
//         let n = 8;
//         let b = (1..n + 1).map(|x| x as f64).collect::<Vec<f64>>();
//         // Diagonals (randomly chosen)
//         let l2 = (1..n - 1).map(|x| 1.5 * x as f64).collect::<Vec<f64>>();
//         let l1 = (1..n).map(|x| -2.5 * x as f64).collect::<Vec<f64>>();
//         let d0 = (1..n + 1).map(|x| 1.0 * x as f64).collect::<Vec<f64>>();
//         let u1 = (1..n).map(|x| 3.5 * x as f64).collect::<Vec<f64>>();
//         let u2 = (1..n - 1).map(|x| -0.5 * x as f64).collect::<Vec<f64>>();
//         // Fill diagonals in matrix, used for assertion of result
//         let mut mat = Array2::<f64>::zeros((n, n));
//         for i in 0..n {
//             mat[[i, i]] = d0[i];
//         }
//         for i in 0..n - 1 {
//             mat[[i + 1, i]] = l1[i];
//         }
//         for i in 0..n - 2 {
//             mat[[i + 2, i]] = l2[i];
//         }
//         for i in 0..n - 1 {
//             mat[[i, i + 1]] = u1[i];
//         }
//         for i in 0..n - 2 {
//             mat[[i, i + 2]] = u2[i];
//         }

//         // Solve
//         let mut rhs = b.clone();
//         pdma(&l2, &l1, &d0, &u1, &u2, &mut rhs);
//         // Assert
//         let b2 = mat.dot(&Array1::from_vec(rhs));
//         for (v1, v2) in b.iter().zip(b2.iter()) {
//             assert!((v1 - v2).abs() < 1e-6, "PDMA failed, {} /= {}.", v1, v2);
//         }

//         // Solve checked
//         let mut rhs = b.clone();
//         pdma_checked(&l2, &l1, &d0, &u1, &u2, &mut rhs);
//         // Assert
//         let b2 = mat.dot(&Array1::from_vec(rhs));
//         for (v1, v2) in b.iter().zip(b2.iter()) {
//             assert!((v1 - v2).abs() < 1e-6, "PDMA failed, {} /= {}.", v1, v2);
//         }
//     }
// }
