//! # Collection of Chebyshev-composite-stencils
//!
//! A stencil $S$ transforms from orthogonal space `u` to composite space `v`, i.e.
//! $$$
//! u = S v
//! $$&
//!
//! The stencil matrix is usually sparse, so we can define efficient methods
//! to get `v` from `u` and vice versa.
mod helper;
mod traits;
use crate::types::{Real, ScalarOperand};
use helper::{Helper, HelperDiag012, HelperDiag02, HelperDiag024};
pub(crate) use traits::StencilOperand;

#[derive(Clone)]
pub(super) enum StencilKind {
    Dirichlet,
    Neumann,
    DirichletNeumann,
    BiHarmonicA,
    BiHarmonicB,
    // Neumann(Neumann<A>),
    // DirichletNeumann(DirichletNeumann<A>),
    // BiHarmonicA(BiHarmonicA<A>),
    // BiHarmonicB(BiHarmonicB<A>),
}

#[derive(Clone)]
pub(super) struct Stencil<A: Real> {
    helper: Helper<A>,
    kind: StencilKind,
}

impl<A: Real> Stencil<A> {
    /// Return stencil of chebyshev dirichlet space
    /// ```text
    ///  \phi_k = T_k - T_{k+2}
    /// ```
    ///
    /// Reference:
    /// J. Shen: Effcient Spectral-Galerkin Method II.
    pub(super) fn dirichlet(n: usize) -> Self {
        let diag = vec![A::one(); n - 2];
        let low2 = vec![-A::one(); n - 2];
        let helper = Helper::HelperDiag02(HelperDiag02::<A>::new(n, n - 2, diag, low2));
        let kind = StencilKind::Dirichlet;
        Self { helper, kind }
    }

    /// Return stencil of chebyshev neumann space
    /// ```text
    ///  \phi_k = T_k - k^{2} \/ (k+2)^2 T_{k+2}
    /// ```
    ///
    /// Reference:
    /// J. Shen: Effcient Spectral-Galerkin Method II.
    #[allow(clippy::cast_precision_loss)]
    pub fn neumann(n: usize) -> Self {
        let diag = vec![A::one(); n - 2];
        let mut low2 = vec![A::zero(); n - 2];
        for (k, v) in low2.iter_mut().enumerate() {
            let k_ = A::from_f64(k.pow(2) as f64).unwrap();
            let k2_ = A::from_f64((k + 2).pow(2) as f64).unwrap();
            *v = -A::one() * k_ / k2_;
        }
        let helper = Helper::HelperDiag02(HelperDiag02::<A>::new(n, n - 2, diag, low2));
        let kind = StencilKind::Neumann;
        Self { helper, kind }
    }

    /// Return stencil of chebyshev neumann space
    /// ```text
    ///  \phi_k = T_k - k^{2} \/ (k+2)^2 T_{k+2}
    /// ```
    ///
    /// Reference:
    /// J. Shen: Effcient Spectral-Galerkin Method II.
    #[allow(clippy::cast_precision_loss)]
    pub fn dirichlet_neumann(n: usize) -> Self {
        let diag = vec![A::one(); n - 2];
        let mut low1 = vec![A::zero(); n - 2];
        let mut low2 = vec![A::zero(); n - 2];
        for (k, (v1, v2)) in low1.iter_mut().zip(low2.iter_mut()).enumerate() {
            let kf64 = k as f64;
            *v1 = A::from_f64(
                (-1. * kf64.powi(2) + (kf64 + 2.).powi(2))
                    / ((kf64 + 1.).powi(2) + (kf64 + 2.).powi(2)),
            )
            .unwrap();
            *v2 = A::from_f64(
                (-1. * kf64.powi(2) - (kf64 + 1.).powi(2))
                    / ((kf64 + 1.).powi(2) + (kf64 + 2.).powi(2)),
            )
            .unwrap();
        }
        let helper = Helper::HelperDiag012(HelperDiag012::<A>::new(n, n - 2, diag, low1, low2));
        let kind = StencilKind::DirichletNeumann;
        Self { helper, kind }
    }

    /// Return stencil of biarmonic space
    ///
    /// Reference:
    /// ```text
    /// F. Liu: doi: 10.4208/nmtma.2011.42s.5
    /// ```
    #[allow(clippy::cast_precision_loss)]
    pub fn biharmonic_a(n: usize) -> Self {
        let m = n - 4;
        let diag = vec![A::one(); m];
        let mut low2 = vec![A::zero(); m];
        let mut low4 = vec![A::zero(); m];
        for (ki, (v2, v4)) in low2.iter_mut().zip(low4.iter_mut()).enumerate() {
            let k = ki as f64;
            let d2 = -2. * (k + 2.) / (k + 3.);
            let d4 = (k + 1.) / (k + 3.);
            *v2 = A::from_f64(d2).unwrap();
            *v4 = A::from_f64(d4).unwrap();
        }
        let helper = Helper::HelperDiag024(HelperDiag024::<A>::new(n, m, diag, low2, low4));
        let kind = StencilKind::BiHarmonicA;
        Self { helper, kind }
    }

    /// Return stencil of biarmonic space
    ///
    /// Reference:
    /// ```text
    /// F. Liu: doi: 10.4208/nmtma.2011.42s.5
    /// ```
    #[allow(clippy::cast_precision_loss)]
    pub fn biharmonic_b(n: usize) -> Self {
        let m = n - 4;
        let diag = vec![A::one(); m];
        let mut low2 = vec![A::zero(); m];
        let mut low4 = vec![A::zero(); m];
        for (ki, (v2, v4)) in low2.iter_mut().zip(low4.iter_mut()).enumerate() {
            let k = ki as f64;
            let d2 = -1. * (2. * (k + 2.) * (15. + 2. * k * (k + 4.)))
                / ((k + 3.) * (19. + 2. * k * (6. + k)));
            let d4 = ((k + 1.) * (3. + 2. * k * (k + 2.))) / ((k + 3.) * (19. + 2. * k * (6. + k)));
            *v2 = A::from_f64(d2).unwrap();
            *v4 = A::from_f64(d4).unwrap();
        }
        let helper = Helper::HelperDiag024(HelperDiag024::<A>::new(n, m, diag, low2, low4));
        let kind = StencilKind::BiHarmonicB;
        Self { helper, kind }
    }

    pub(super) fn get_kind(&self) -> StencilKind {
        self.kind.clone()
    }
}

impl<A> StencilOperand<A> for Stencil<A>
where
    A: Real,
{
    fn matvec<T: ScalarOperand<A>>(&self, x: &[T], b: &mut [T]) {
        self.helper.matvec(x, b);
    }

    // fn matvec_inplace<T: ScalarOperand<A>>(&self, x: &mut [T]) {
    //     self.helper.matvec_inplace(x);
    // }

    fn solve<T: ScalarOperand<A>>(&self, b: &[T], x: &mut [T]) {
        self.helper.solve(b, x);
    }

    // fn solve_inplace<T: ScalarOperand<A>>(&self, x: &mut [T]) {
    //     self.helper.solve_inplace(x);
    // }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_matvec() {
        let d = Stencil::<f64>::dirichlet(5);
        let v = vec![1., 2., 3.];
        let mut u = vec![0., 0., 0., 0., 0.];
        d.helper.matvec(&v, &mut u);
        assert_eq!(u, [1., 2., 2., -2., -3.]);
    }

    #[test]
    fn test_solve() {
        let d = Stencil::<f64>::dirichlet(5);
        let u = vec![1., 2., 4., 1., 1.];
        let mut v = vec![0., 0., 0.];
        d.helper.solve(&u, &mut v);
        assert_eq!(v, [-1., 0.5, 1.]);
    }
}
