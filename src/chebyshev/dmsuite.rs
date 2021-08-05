//! Additional differentaions routines for Chebyshev
//! Polynomials
//! They are not used inside `rustpde`, but good to have
//! for testing
use super::FloatNum;
use ndarray::{s, Array2};

/// Derivative matrix in spectral space of classical Chebyshev
/// polynomial on Gauss Lobattor points, see
/// Jan S. Hesthaven - Spectral Methods for Time-Dependent Problems (p. 256)
///
/// # Parameters
/// * `N` - Number of grid points
/// * `deriv` - Order of derivative
///
/// # Output
/// ndarray (N x N)
/// Derivative matrix, must be applied in spectral
/// space to chebyshev coefficients array
///
/// # Panics
/// **Panics** for deriv>2
#[allow(dead_code, clippy::must_use_candidate)]
pub fn diffmat_chebyshev<A: FloatNum>(n: usize, deriv: usize) -> Array2<A> {
    let mut dmat = Array2::<f64>::zeros((n, n));
    if deriv == 1 {
        for p in 0..n {
            for q in p + 1..n {
                if (p + q) % 2 != 0 {
                    dmat[[p, q]] = (q * 2) as f64;
                }
            }
        }
    } else if deriv == 2 {
        for p in 0..n {
            for q in p + 2..n {
                if (p + q) % 2 == 0 {
                    dmat[[p, q]] = (q * (q * q - p * p)) as f64;
                }
            }
        }
    } else {
        todo!()
    }
    for d in dmat.slice_mut(s![0, ..]).iter_mut() {
        *d *= 0.5;
    }
    dmat.mapv(|elem| A::from_f64(elem).unwrap())
}
