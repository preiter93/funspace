use crate::types::FloatNum;
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

    /// Chebyshev points of the second kind. $[-1, 1]$
    /// $$$
    /// x = cos( pi*k/(npts - 1) )
    /// $$$
    fn chebyshev_nodes_2nd_kind(n: usize) -> Vec<A> {
        (0..n)
            .map(|k| {
                A::from_f64(PI * k as f64 / (n as isize - 1) as f64)
                    .unwrap()
                    .cos()
            })
            .collect::<Vec<A>>()
    }

    /// Returns grid points
    pub fn nodes(&self) -> Vec<A> {
        Self::chebyshev_nodes_2nd_kind(self.n)
    }

    /// Size in physical space
    pub fn len_phys(&self) -> usize {
        self.n
    }

    /// Size in spectral space
    pub fn len_spec(&self) -> usize {
        self.m
    }

    /// Is base orthogonal
    pub fn is_ortho(&self) -> bool {
        true
    }
}

impl<A: FloatNum> Chebyshev<A> {
    /// Transform: Physical values -> Spectral coefficients
    ///
    /// # Panics
    /// Float conversion fails (unlikely)
    pub fn forward_inplace(&self, indata: &[A], outdata: &mut [A]) {
        // Check input
        assert!(indata.len() == self.len_phys());
        assert!(indata.len() == outdata.len());

        // Copy and correct input data
        let cor = (A::one() + A::one()) * A::from_f64(1. / (self.n as isize - 1) as f64).unwrap();
        // Reverse indata since it is defined on $[-1, 1]$, instead of $[1, -1]$
        for (y, x) in outdata.iter_mut().zip(indata.iter().rev()) {
            *y = *x * cor;
        }

        // Transform via dct
        self.plan_dct.process_dct1(outdata);

        // Correct first and last coefficient
        let _05 = A::from_f64(0.5).unwrap();
        outdata[0] *= _05;
        outdata[self.n - 1] *= _05;
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test_utils::approx_eq;

    #[test]
    fn test_chebyshev_transform_1() {
        let cheby = Chebyshev::<f64>::new(4);
        let indata = vec![1., 2., 3., 4.];
        let mut outdata = vec![0.; 4];
        cheby.forward_inplace(&indata, &mut outdata);
        approx_eq(&outdata, &vec![2.5, 1.33333333, 0., 0.16666667]);
    }
}
