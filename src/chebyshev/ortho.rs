//! # Orthogonal Chebyshev Base
use crate::enums::{BaseKind, BaseType, TransformKind};
use crate::traits::{Differentiate, HasCoords, HasLength, HasType, ToOrtho, Transform};
use crate::types::{Real, Scalar};
use rustdct::{Dct1, DctPlanner};
use std::f64::consts::PI;
use std::sync::Arc;

/// # Container for chebyshev base
#[derive(Clone)]
pub struct Chebyshev<A> {
    /// Number of coefficients in physical space
    n: usize,
    /// `DCT`-Plan
    plan_dct: Arc<dyn Dct1<A>>,
}

impl<A: Real> Chebyshev<A> {
    /// Creates a new Basis.
    ///
    /// # Arguments
    /// * `n` - Length of array's dimension which shall live in chebyshev space.
    ///
    /// # Panics
    /// Panics when input type cannot be cast from f64.
    ///
    /// # Examples
    /// ```ignore
    /// use funspace::chebyshev::Chebyshev;
    /// let cheby = Chebyshev::<f64>::new(10);
    /// ```
    #[must_use]
    pub fn new(n: usize) -> Self {
        let mut planner = DctPlanner::<A>::new();
        let dct1 = planner.plan_dct1(n);
        Self {
            n,
            plan_dct: Arc::clone(&dct1),
        }
    }
}

impl<A: Real> HasLength for Chebyshev<A> {
    fn len_phys(&self) -> usize {
        self.n
    }

    fn len_spec(&self) -> usize {
        self.n
    }

    fn len_ortho(&self) -> usize {
        self.n
    }
}

impl<A: Real> HasCoords<A> for Chebyshev<A> {
    /// Chebyshev nodes of the second kind on intervall $[-1, 1]$
    /// $$$
    /// x = - cos( pi*k/(npts - 1) )
    /// $$$
    fn coords(&self) -> Vec<A> {
        let m = (self.n - 1) as f64;
        (0..self.n)
            .map(|k| {
                let arg = A::from_f64(PI * (m - 2. * k as f64) / (2. * m)).unwrap();
                -A::one() * arg.sin()
            })
            .collect()
    }
}

impl<A: Real> HasType for Chebyshev<A> {
    fn base_kind(&self) -> BaseKind {
        BaseKind::Chebyshev
    }

    fn base_type(&self) -> BaseType {
        BaseType::Orthogonal
    }

    fn transform_kind(&self) -> TransformKind {
        TransformKind::R2r
    }
}

impl<A, T> Differentiate<T> for Chebyshev<A>
where
    A: Real,
    T: Scalar,
{
    fn diff(&self, v: &[T], dv: &mut [T], order: usize) {
        dv.clone_from_slice(v);
        self.diff_inplace(dv, order);
    }

    /// Recursive differentiation
    fn diff_inplace(&self, v: &mut [T], order: usize) {
        let two = T::one() + T::one();
        for _ in 0..order {
            unsafe {
                *v.get_unchecked_mut(0) = *v.get_unchecked(1);
                for i in 1..self.n - 1 {
                    *v.get_unchecked_mut(i) =
                        two * T::from_usize(i + 1).unwrap() * *v.get_unchecked(i + 1);
                }
                *v.get_unchecked_mut(self.n - 1) = T::zero();
                for i in (1..self.n - 2).rev() {
                    *v.get_unchecked_mut(i) = *v.get_unchecked(i) + *v.get_unchecked(i + 2);
                }
                *v.get_unchecked_mut(0) = *v.get_unchecked(0) + *v.get_unchecked(2) / two;
            }
        }
    }
}

impl<A: Real> Transform for Chebyshev<A> {
    type Physical = A;

    type Spectral = A;

    fn forward(&self, phys: &[Self::Physical], spec: &mut [Self::Spectral]) {
        spec.clone_from_slice(phys);
        self.forward_inplace(spec)
    }

    fn backward(&self, spec: &[Self::Spectral], phys: &mut [Self::Physical]) {
        phys.clone_from_slice(spec);
        self.backward_inplace(phys)
    }

    fn forward_inplace(&self, v: &mut [Self::Spectral]) {
        // Correct
        let cor = (A::one() + A::one()) * A::one() / A::from(self.n - 1).unwrap();
        v.reverse();
        for vi in v.iter_mut() {
            *vi = *vi * cor;
        }

        // Transform via dct
        self.plan_dct.process_dct1(v);

        // Correct first and last coefficient
        let half = A::from_f64(0.5).unwrap();
        v[0] *= half;
        v[self.n - 1] *= half;
    }

    fn backward_inplace(&self, v: &mut [Self::Physical]) {
        // Correct
        let two = A::one() + A::one();
        for vi in v.iter_mut().skip(1).step_by(2) {
            *vi = *vi * -A::one();
        }
        v[0] *= two;
        v[self.n - 1] *= two;
        // Transform via dct
        self.plan_dct.process_dct1(v);
    }
}

impl<A, T> ToOrtho<T> for Chebyshev<A>
where
    A: Real,
    T: Scalar,
{
    fn to_ortho(&self, comp: &[T], ortho: &mut [T]) {
        ortho.clone_from_slice(comp);
    }

    fn from_ortho(&self, ortho: &[T], comp: &mut [T]) {
        comp.clone_from_slice(ortho);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::utils::approx_eq;

    #[test]
    fn test_ch_transform() {
        let cheby = Chebyshev::<f64>::new(4);
        let v = vec![1., 2., 3., 4.];
        let mut vhat = vec![0.; 4];
        cheby.forward(&v, &mut vhat);
        approx_eq(&vhat, &vec![2.5, 1.33333333, 0., 0.16666667]);
    }
}
