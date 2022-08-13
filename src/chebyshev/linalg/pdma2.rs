//! Pentadiagonal matrix solver
//!     Ax = d
//! where A is banded with diagonals in offsets -4, -2, 0, 2, 4
use num_traits::Zero;
use std::clone::Clone;
use std::ops::{Add, Div, Mul, Sub};

/// Pentadiagonal matrix solver
///     Ax = d
/// where A is banded with diagonals in offsets -4, -2, 0, 2, 4
///
/// l4: sub-diagonal (-4)
/// l2: sub-diagonal (-2)
/// d0: main-diagonal
/// u2: sub-diagonal (+2)
/// u4: sub-diagonal (+4)
#[allow(clippy::many_single_char_names, dead_code, clippy::too_many_lines)]
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
    assert!(n > 5, "Error in pdma2: Size = {} too small!", n);

    let mut al = vec![T1::zero(); n - 2];
    let mut be = vec![T1::zero(); n - 4];
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
        if u4.len() > 2 {
            *be.get_unchecked_mut(2) = *u4.get_unchecked(2) / *mu.get_unchecked(2);
        }
        *ze.get_unchecked_mut(2) = (*rhs.get_unchecked(2)
            - *ze.get_unchecked(0) * *ga.get_unchecked(2))
            / *mu.get_unchecked(2);

        *ga.get_unchecked_mut(3) = *l2.get_unchecked(1);
        *mu.get_unchecked_mut(3) =
            *d0.get_unchecked(3) - *al.get_unchecked(1) * *ga.get_unchecked(3);
        *al.get_unchecked_mut(3) = (*u2.get_unchecked(3)
            - *be.get_unchecked(1) * *ga.get_unchecked(3))
            / *mu.get_unchecked(3);
        if u4.len() > 3 {
            *be.get_unchecked_mut(3) = *u4.get_unchecked(3) / *mu.get_unchecked(3);
        }
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
        if l4.len() > 3 {
            *ga.get_unchecked_mut(n - 4) =
                *l2.get_unchecked(n - 6) - *al.get_unchecked(n - 8) * *l4.get_unchecked(n - 8);
            *mu.get_unchecked_mut(n - 4) = *d0.get_unchecked(n - 4)
                - *be.get_unchecked(n - 8) * *l4.get_unchecked(n - 8)
                - *al.get_unchecked(n - 6) * *ga.get_unchecked(n - 4);
            *al.get_unchecked_mut(n - 4) = (*u2.get_unchecked(n - 4)
                - *be.get_unchecked(n - 6) * *ga.get_unchecked(n - 4))
                / *mu.get_unchecked(n - 4);
            *ze.get_unchecked_mut(n - 4) = (*rhs.get_unchecked(n - 4)
                - *ze.get_unchecked(n - 8) * *l4.get_unchecked(n - 8)
                - *ze.get_unchecked(n - 6) * *ga.get_unchecked(n - 4))
                / *mu.get_unchecked(n - 4);
        }
        if l4.len() > 2 {
            *ga.get_unchecked_mut(n - 3) =
                *l2.get_unchecked(n - 5) - *al.get_unchecked(n - 7) * *l4.get_unchecked(n - 7);
            *mu.get_unchecked_mut(n - 3) = *d0.get_unchecked(n - 3)
                - *be.get_unchecked(n - 7) * *l4.get_unchecked(n - 7)
                - *al.get_unchecked(n - 5) * *ga.get_unchecked(n - 3);
            *al.get_unchecked_mut(n - 3) = (*u2.get_unchecked(n - 3)
                - *be.get_unchecked(n - 5) * *ga.get_unchecked(n - 3))
                / *mu.get_unchecked(n - 3);
            *ze.get_unchecked_mut(n - 3) = (*rhs.get_unchecked(n - 3)
                - *ze.get_unchecked(n - 7) * *l4.get_unchecked(n - 7)
                - *ze.get_unchecked(n - 5) * *ga.get_unchecked(n - 3))
                / *mu.get_unchecked(n - 3);
        }
        *ga.get_unchecked_mut(n - 2) =
            *l2.get_unchecked(n - 4) - *al.get_unchecked(n - 6) * *l4.get_unchecked(n - 6);
        *mu.get_unchecked_mut(n - 2) = *d0.get_unchecked(n - 2)
            - *be.get_unchecked(n - 6) * *l4.get_unchecked(n - 6)
            - *al.get_unchecked(n - 4) * *ga.get_unchecked(n - 2);
        *ze.get_unchecked_mut(n - 2) = (*rhs.get_unchecked(n - 2)
            - *ze.get_unchecked(n - 6) * *l4.get_unchecked(n - 6)
            - *ze.get_unchecked(n - 4) * *ga.get_unchecked(n - 2))
            / *mu.get_unchecked(n - 2);

        *ga.get_unchecked_mut(n - 1) =
            *l2.get_unchecked(n - 3) - *al.get_unchecked(n - 5) * *l4.get_unchecked(n - 5);
        *mu.get_unchecked_mut(n - 1) = *d0.get_unchecked(n - 1)
            - *be.get_unchecked(n - 5) * *l4.get_unchecked(n - 5)
            - *al.get_unchecked(n - 3) * *ga.get_unchecked(n - 1);

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

/// Pentadiagonal matrix solver
///     Ax = d
/// where A is banded with diagonals in offsets -4, -2, 0, 2, 4
///
/// l4: sub-diagonal (-4)
/// l2: sub-diagonal (-2)
/// d0: main-diagonal
/// u2: sub-diagonal (+2)
/// u4: sub-diagonal (+4)
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
    assert!(n > 5, "Error in pdma2: Size = {} too small!", n);

    let mut al = vec![T1::zero(); n - 2];
    let mut be = vec![T1::zero(); n - 4];
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
    if u4.len() > 2 {
        be[2] = u4[2] / mu[2];
    }
    ze[2] = (rhs[2] - ze[0] * ga[2]) / mu[2];

    ga[3] = l2[1];
    mu[3] = d0[3] - al[1] * ga[3];
    al[3] = (u2[3] - be[1] * ga[3]) / mu[3];
    if u4.len() > 3 {
        be[3] = u4[3] / mu[3];
    }
    ze[3] = (rhs[3] - ze[1] * ga[3]) / mu[3];

    for i in 4..n - 4 {
        ga[i] = l2[i - 2] - al[i - 4] * l4[i - 4];
        mu[i] = d0[i] - be[i - 4] * l4[i - 4] - al[i - 2] * ga[i];
        al[i] = (u2[i] - be[i - 2] * ga[i]) / mu[i];
        be[i] = u4[i] / mu[i];
        ze[i] = (rhs[i] - ze[i - 4] * l4[i - 4] - ze[i - 2] * ga[i]) / mu[i];
    }
    if l4.len() > 3 {
        ga[n - 4] = l2[n - 6] - al[n - 8] * l4[n - 8];
        mu[n - 4] = d0[n - 4] - be[n - 8] * l4[n - 8] - al[n - 6] * ga[n - 4];
        al[n - 4] = (u2[n - 4] - be[n - 6] * ga[n - 4]) / mu[n - 4];
        ze[n - 4] = (rhs[n - 4] - ze[n - 8] * l4[n - 8] - ze[n - 6] * ga[n - 4]) / mu[n - 4];
    }
    if l4.len() > 2 {
        ga[n - 3] = l2[n - 5] - al[n - 7] * l4[n - 7];
        mu[n - 3] = d0[n - 3] - be[n - 7] * l4[n - 7] - al[n - 5] * ga[n - 3];
        al[n - 3] = (u2[n - 3] - be[n - 5] * ga[n - 3]) / mu[n - 3];
        ze[n - 3] = (rhs[n - 3] - ze[n - 7] * l4[n - 7] - ze[n - 5] * ga[n - 3]) / mu[n - 3];
    }

    ga[n - 2] = l2[n - 4] - al[n - 6] * l4[n - 6];
    mu[n - 2] = d0[n - 2] - be[n - 6] * l4[n - 6] - al[n - 4] * ga[n - 2];
    ze[n - 2] = (rhs[n - 2] - ze[n - 6] * l4[n - 6] - ze[n - 4] * ga[n - 2]) / mu[n - 2];

    ga[n - 1] = l2[n - 3] - al[n - 5] * l4[n - 5];
    mu[n - 1] = d0[n - 1] - be[n - 5] * l4[n - 5] - al[n - 3] * ga[n - 1];
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

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use ndarray::{Array1, Array2};
//     #[test]
//     /// A x = b
//     fn test_pdma2() {
//         let n = 8;
//         let b = (1..n + 1).map(|x| x as f64).collect::<Vec<f64>>();
//         // Diagonals (randomly chosen)
//         let l4 = (1..n - 3).map(|x| 1.5 * x as f64).collect::<Vec<f64>>();
//         let l2 = (1..n - 1).map(|x| -2.5 * x as f64).collect::<Vec<f64>>();
//         let d0 = (1..n + 1).map(|x| 1.0 * x as f64).collect::<Vec<f64>>();
//         let u2 = (1..n - 1).map(|x| 3.5 * x as f64).collect::<Vec<f64>>();
//         let u4 = (1..n - 3).map(|x| -0.5 * x as f64).collect::<Vec<f64>>();
//         // Fill diagonals in matrix, used for assertion of result
//         let mut mat = Array2::<f64>::zeros((n, n));
//         for i in 0..n {
//             mat[[i, i]] = d0[i];
//         }
//         for i in 0..n - 2 {
//             mat[[i + 2, i]] = l2[i];
//         }
//         for i in 0..n - 4 {
//             mat[[i + 4, i]] = l4[i];
//         }
//         for i in 0..n - 2 {
//             mat[[i, i + 2]] = u2[i];
//         }
//         for i in 0..n - 4 {
//             mat[[i, i + 4]] = u4[i];
//         }

//         // Solve
//         let mut rhs = b.clone();
//         pdma2(&l4, &l2, &d0, &u2, &u4, &mut rhs);
//         // Assert
//         let b2 = mat.dot(&Array1::from_vec(rhs));
//         for (v1, v2) in b.iter().zip(b2.iter()) {
//             assert!((v1 - v2).abs() < 1e-6, "PDMA2 failed, {} /= {}.", v1, v2);
//         }

//         // Solve checked
//         let mut rhs = b.clone();
//         pdma2_checked(&l4, &l2, &d0, &u2, &u4, &mut rhs);
//         // Assert
//         let b2 = mat.dot(&Array1::from_vec(rhs));
//         for (v1, v2) in b.iter().zip(b2.iter()) {
//             assert!((v1 - v2).abs() < 1e-6, "PDMA2 failed, {} /= {}.", v1, v2);
//         }
//     }
// }
