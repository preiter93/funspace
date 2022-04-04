//! # Orthogonal Chebyshev Base
use crate::enums::BaseKind;
use crate::traits::{FunspaceElemental, FunspaceExtended, FunspaceSize};
use crate::types::{FloatNum, ScalarNum};
use ndarray::{s, Array2};
use rustdct::{Dct1, DctPlanner};
use std::f64::consts::PI;
use std::sync::Arc;

/// # Container for chebyshev base
#[derive(Clone)]
pub struct Chebyshev<A> {
    /// Number of coefficients in physical space
    n: usize,
    /// Number of coefficients in spectral space ( equal to *n* for `Chebyshev` )
    m: usize,
    /// `DCT`-Plan
    plan_dct: Arc<dyn Dct1<A>>,
}

impl<A: FloatNum> Chebyshev<A> {
    /// Creates a new Basis.
    ///
    /// # Arguments
    /// * `n` - Length of array's dimension which shall live in chebyshev space.
    ///
    /// # Panics
    /// Panics when input type cannot be cast from f64.
    ///
    /// # Examples
    /// ```
    /// use funspace::chebyshev::Chebyshev;
    /// let cheby = Chebyshev::<f64>::new(10);
    /// ```
    #[must_use]
    pub fn new(n: usize) -> Self {
        let mut planner = DctPlanner::<A>::new();
        let dct1 = planner.plan_dct1(n);
        Self {
            n,
            m: n,
            plan_dct: Arc::clone(&dct1),
        }
    }

    // /// Chebyshev points of the second kind. $[-1, 1]$
    // /// $$$
    // /// x = - cos( pi*k/(npts - 1) )
    // /// $$$
    // #[allow(clippy::cast_precision_loss)]
    // fn chebyshev_nodes_2nd_kind(n: usize) -> Vec<A> {
    //     (0..n)
    //         .map(|k| -A::one() * A::from_f64(PI * k as f64 / (n - 1) as f64).unwrap().cos())
    //         .collect::<Vec<A>>()
    // }

    /// Chebyshev nodes of the second kind on intervall $[-1, 1]$
    /// $$$
    /// x = - cos( pi*k/(npts - 1) )
    /// $$$
    #[allow(clippy::cast_precision_loss)]
    fn chebyshev_nodes_2nd_kind(n: usize) -> Vec<A> {
        let m = (n - 1) as f64;
        (0..n)
            .map(|k| {
                let arg = A::from_f64(PI * (m - 2. * k as f64) / (2. * m)).unwrap();
                -A::one() * arg.sin()
            })
            .collect::<Vec<A>>()
    }

    /// Returns grid points
    #[must_use]
    pub fn nodes(n: usize) -> Vec<A> {
        Self::chebyshev_nodes_2nd_kind(n)
    }

    /// Differentation Matrix see [`chebyshev::dmsuite::diffmat_chebyshev`
    ///
    /// # Panics
    /// - Num conversion fails
    /// - deriv > 2: Not implemented
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn _dmat(n: usize, deriv: usize) -> Array2<A> {
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
    /// Pseudoinverse matrix of chebyshev spectral
    /// differentiation matrices
    ///
    /// When preconditioned with the pseudoinverse Matrix,
    /// systems become banded and thus efficient to solve.
    ///
    /// Literature:
    /// Sahuck Oh - An Efficient Spectral Method to Solve Multi-Dimensional
    /// Linear Partial Different Equations Using Chebyshev Polynomials
    ///
    /// Output:
    /// ndarray (n x n) matrix, acts in spectral space
    ///
    /// # Panics
    /// - Num conversion fails
    /// - deriv > 2: Not implemented
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn _pinv(n: usize, deriv: usize) -> Array2<A> {
        assert!(
            !(deriv > 2),
            "pinv does only support deriv's 1 & 2, got {}",
            deriv
        );
        let mut pinv = Array2::<f64>::zeros([n, n]);
        if deriv == 1 {
            pinv[[1, 0]] = 1.;
            for i in 2..n {
                pinv[[i, i - 1]] = 1. / (2. * i as f64); // diag - 1
            }
            for i in 1..n - 2 {
                pinv[[i, i + 1]] = -1. / (2. * i as f64); // diag + 1
            }
        } else if deriv == 2 {
            pinv[[2, 0]] = 0.25;
            for i in 3..n {
                pinv[[i, i - 2]] = 1. / (4 * i * (i - 1)) as f64; // diag - 2
            }
            for i in 2..n - 2 {
                pinv[[i, i]] = -1. / (2 * (i * i - 1)) as f64; // diag 0
            }
            for i in 2..n - 4 {
                pinv[[i, i + 2]] = 1. / (4 * i * (i + 1)) as f64; // diag + 2
            }
        }
        //pinv
        pinv.mapv(|elem| A::from_f64(elem).unwrap())
    }

    /// Returns eye matrix, where the n ( = deriv) upper rows
    ///
    /// # Panics
    /// Num conversion fails
    #[must_use]
    pub fn _pinv_eye(n: usize, deriv: usize) -> Array2<A> {
        let pinv_eye = Array2::<f64>::eye(n).slice(s![deriv.., ..]).to_owned();
        pinv_eye.mapv(|elem| A::from_f64(elem).unwrap())
    }
}

impl<A: FloatNum> FunspaceSize for Chebyshev<A> {
    /// Size in physical space
    #[must_use]
    fn len_phys(&self) -> usize {
        self.n
    }

    /// Size in spectral space
    #[must_use]
    fn len_spec(&self) -> usize {
        self.m
    }

    /// Size of orthogonal space
    #[must_use]
    fn len_orth(&self) -> usize {
        self.m
    }
}

impl<A: FloatNum> FunspaceExtended for Chebyshev<A> {
    type Real = A;

    type Spectral = A;

    /// Return kind of base
    fn base_kind(&self) -> BaseKind {
        BaseKind::Chebyshev
    }

    /// Coordinates in physical space
    fn get_nodes(&self) -> Vec<A> {
        Chebyshev::nodes(self.len_phys())
    }

    /// Mass matrix
    fn mass(&self) -> Array2<A> {
        Array2::<A>::eye(self.len_spec())
    }

    /// Inverse of mass matrix
    fn mass_inv(&self) -> Array2<A> {
        Array2::<A>::eye(self.len_spec())
    }

    /// Explicit differential operator
    fn diffmat(&self, deriv: usize) -> Array2<A> {
        Self::_dmat(self.n, deriv)
    }

    /// Laplacian $ L $
    fn laplace(&self) -> Array2<A> {
        Self::_dmat(self.n, 2)
    }

    /// Pseudoinverse mtrix of Laplacian $ L^{-1} $
    fn laplace_inv(&self) -> Array2<A> {
        Self::_pinv(self.n, 2)
    }

    /// Pseudoidentity matrix of laplacian $ L^{-1} L $
    fn laplace_inv_eye(&self) -> Array2<A> {
        Self::_pinv_eye(self.n, 2)
    }
}

impl<A: FloatNum> FunspaceElemental for Chebyshev<A> {
    type Physical = A;

    type Spectral = A;

    /// Differentiate Vector `n_times` using recurrence relation
    /// of chebyshev polynomials.
    ///
    /// # Panics
    /// Float conversion fails (unlikely)
    ///
    /// # Example
    /// Differentiate Chebyshev
    /// ```
    /// use funspace::traits::FunspaceElemental;
    /// use funspace::chebyshev::Chebyshev;
    /// use funspace::utils::approx_eq;
    /// let mut ch = Chebyshev::<f64>::new(4);
    /// let indata: Vec<f64> = vec![1., 2., 3., 4.];
    /// let mut outdata: Vec<f64> = vec![0.; 4];
    /// // Differentiate twice
    /// ch.differentiate_slice(&indata, &mut outdata, 2);
    /// approx_eq(&outdata, &vec![12., 96.,  0.,  0.]);
    /// ```
    fn differentiate_slice<T: ScalarNum>(&self, indata: &[T], outdata: &mut [T], n_times: usize) {
        assert!(outdata.len() == self.m);
        // Copy over
        for (y, x) in outdata.iter_mut().zip(indata.iter()) {
            *y = *x;
        }
        // Differentiate
        let two = T::one() + T::one();
        for _ in 0..n_times {
            // Forward
            unsafe {
                *outdata.get_unchecked_mut(0) = *outdata.get_unchecked(1);
                for i in 1..self.m - 1 {
                    *outdata.get_unchecked_mut(i) =
                        two * T::from_usize(i + 1).unwrap() * *outdata.get_unchecked(i + 1);
                }
                *outdata.get_unchecked_mut(self.m - 1) = T::zero();
                // Reverse
                for i in (1..self.m - 2).rev() {
                    *outdata.get_unchecked_mut(i) =
                        *outdata.get_unchecked(i) + *outdata.get_unchecked(i + 2);
                }
                *outdata.get_unchecked_mut(0) =
                    *outdata.get_unchecked(0) + *outdata.get_unchecked(2) / two;
            }
        }
    }

    fn forward_slice(&self, indata: &[Self::Physical], outdata: &mut [Self::Spectral]) {
        // Check input
        assert!(indata.len() == self.len_phys());
        assert!(indata.len() == outdata.len());

        // Copy and correct input data
        let cor = (A::one() + A::one()) * A::one() / A::from(self.n - 1).unwrap();

        // Reverse indata since it is defined on $[-1, 1]$, instead of $[1, -1]$
        for (y, x) in outdata.iter_mut().zip(indata.iter().rev()) {
            *y = *x * cor;
        }
        // Transform via dct
        self.plan_dct.process_dct1(outdata);

        // Correct first and last coefficient
        let half = A::from_f64(0.5).unwrap();
        outdata[0] *= half;
        outdata[self.n - 1] *= half;
    }

    fn backward_slice(&self, indata: &[Self::Spectral], outdata: &mut [Self::Physical]) {
        // Check input
        assert!(indata.len() == self.len_spec());
        assert!(indata.len() == outdata.len());

        // Copy and correct input data
        // Multiplying with [-1, 1, -1, ...] reverses order of output
        // data, such that outdata is defined on $[-1, 1]$ intervall
        let two = A::one() + A::one();
        for (i, (y, x)) in outdata.iter_mut().zip(indata.iter()).enumerate() {
            if i % 2 == 0 {
                *y = *x;
            } else {
                *y = *x * -A::one();
            }
        }
        outdata[0] *= two;
        outdata[self.m - 1] *= two;

        // Transform via dct
        self.plan_dct.process_dct1(outdata);
    }

    /// Composite space coefficients -> Orthogonal space coefficients
    fn to_ortho_slice<T: Copy>(&self, indata: &[T], outdata: &mut [T]) {
        // panic!("Function space Chebyshev is already orthogonal");
        for (y, x) in outdata.iter_mut().zip(indata.iter()) {
            *y = *x;
        }
    }

    /// Orthogonal space coefficients -> Composite space coefficients
    fn from_ortho_slice<T: Copy>(&self, indata: &[T], outdata: &mut [T]) {
        // panic!("Function space Chebyshev is already orthogonal");
        for (y, x) in outdata.iter_mut().zip(indata.iter()) {
            *y = *x;
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::utils::approx_eq;

    #[test]
    fn test_chebyshev_transform_1() {
        let cheby = Chebyshev::<f64>::new(4);
        let indata = vec![1., 2., 3., 4.];
        let mut outdata = vec![0.; 4];
        cheby.forward_slice(&indata, &mut outdata);
        approx_eq(&outdata, &vec![2.5, 1.33333333, 0., 0.16666667]);
    }
}
