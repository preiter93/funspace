//! # Composite chebyshev spaces
use super::ortho::Chebyshev;
use super::stencils::{Stencil, StencilKind, StencilOperand};
use crate::enums::{BaseKind, BaseType, TransformKind};
use crate::traits::{Differentiate, HasCoords, HasLength, HasType, ToOrtho, Transform};
use crate::types::{Real, Scalar, ScalarOperand};
use std::clone::Clone;

#[derive(Clone)]
pub struct ChebyshevComposite<A: Real> {
    /// Number of coefficients in physical space
    n: usize,
    /// Number of coefficients in spectral space
    m: usize,
    /// Parent base
    ortho: Chebyshev<A>,
    /// Transform stencil
    stencil: Stencil<A>,
}

impl<A: Real> HasLength for ChebyshevComposite<A> {
    #[must_use]
    fn len_phys(&self) -> usize {
        self.n
    }

    #[must_use]
    fn len_spec(&self) -> usize {
        self.m
    }

    #[must_use]
    fn len_ortho(&self) -> usize {
        self.ortho.len_ortho()
    }
}

impl<A: Real> ChebyshevComposite<A> {
    /// Return function space of chebyshev space
    /// with *dirichlet* boundary conditions
    /// ```text
    ///  \phi_k = T_k - T_{k+2}
    /// ```
    ///
    ///```text
    /// u(-1)=0 and u(1)=0
    ///```
    ///
    /// Stencil has entries on diagonals 0, -2
    #[must_use]
    pub fn dirichlet(n: usize) -> Self {
        Self {
            n,
            m: n - 2,
            stencil: Stencil::dirichlet(n),
            ortho: Chebyshev::<A>::new(n),
        }
    }

    /// Return function space of chebyshev space
    /// with *neumann* boundary conditions
    /// ```text
    ///  \phi_k = T_k - k^{2} \/ (k+2)^2 T_{k+2}
    /// ```
    ///
    /// ```text
    /// u'(-1)=0 and u'(1)=0
    ///```
    ///
    /// Stencil has entries on diagonals 0, -2
    #[must_use]
    pub fn neumann(n: usize) -> Self {
        Self {
            n,
            m: n - 2,
            stencil: Stencil::neumann(n),
            ortho: Chebyshev::<A>::new(n),
        }
    }

    /// Return function space of chebyshev space
    /// with *dirichlet* boundary conditions at *x=-1*
    /// and *neumann* boundary conditions at *x=1*
    ///
    /// ```text
    /// u(-1)=0 and u'(1)=0
    ///```
    ///
    /// Stencil has entries on diagonals 0, -1, -2
    #[must_use]
    pub fn dirichlet_neumann(n: usize) -> Self {
        Self {
            n,
            m: n - 2,
            stencil: Stencil::dirichlet_neumann(n),
            ortho: Chebyshev::<A>::new(n),
        }
    }

    /// Return function space of chebyshev space
    /// with biharmonic boundary conditions, i.e.
    ///
    /// ```text
    /// u(-1)=0, u(1)=0, u'(-1)=0 and u'(1)=0
    ///```
    ///
    /// Stencil has entries on diagonals 0, -2, -4
    #[must_use]
    pub fn biharmonic_a(n: usize) -> Self {
        Self {
            n,
            m: n - 4,
            stencil: Stencil::biharmonic_a(n),
            ortho: Chebyshev::<A>::new(n),
        }
    }

    /// Return function space of chebyshev space
    /// with biharmonic boundary conditions, i.e.
    ///
    /// ```text
    /// u(-1)=0, u(1)=0, u''(-1)=0 and u''(1)=0
    ///```
    ///
    /// Stencil has entries on diagonals 0, -2, -4
    #[must_use]
    pub fn biharmonic_b(n: usize) -> Self {
        Self {
            n,
            m: n - 4,
            stencil: Stencil::biharmonic_b(n),
            ortho: Chebyshev::<A>::new(n),
        }
    }
}

impl<A: Real> HasCoords<A> for ChebyshevComposite<A> {
    /// Chebyshev nodes of the second kind on intervall $[-1, 1]$
    /// $$$
    /// x = - cos( pi*k/(npts - 1) )
    /// $$$
    fn coords(&self) -> Vec<A> {
        self.ortho.coords()
    }
}

impl<A: Real> HasType for ChebyshevComposite<A> {
    fn base_kind(&self) -> BaseKind {
        match self.stencil.get_kind() {
            StencilKind::Dirichlet => BaseKind::ChebDirichlet,
            StencilKind::Neumann => BaseKind::ChebNeumann,
            StencilKind::DirichletNeumann => BaseKind::ChebDirichletNeumann,
            StencilKind::BiHarmonicA => BaseKind::ChebBiHarmonicA,
            StencilKind::BiHarmonicB => BaseKind::ChebBiHarmonicB,
        }
    }

    fn base_type(&self) -> BaseType {
        BaseType::Composite
    }

    fn transform_kind(&self) -> TransformKind {
        TransformKind::R2r
    }
}

impl<A, T> ToOrtho<T> for ChebyshevComposite<A>
where
    A: Real,
    T: ScalarOperand<A>,
{
    fn to_ortho(&self, comp: &[T], ortho: &mut [T]) {
        self.stencil.matvec(comp, ortho);
    }

    fn from_ortho(&self, ortho: &[T], comp: &mut [T]) {
        self.stencil.solve(ortho, comp);
    }
}

impl<A, T> Differentiate<T> for ChebyshevComposite<A>
where
    A: Real,
    T: ScalarOperand<A>,
{
    fn diff(&self, v: &[T], dv: &mut [T], order: usize) {
        self.to_ortho(v, dv);
        self.ortho.diff_inplace(dv, order);
    }
}

impl<A: Real + Scalar> Transform for ChebyshevComposite<A> {
    type Physical = A;

    type Spectral = A;

    fn forward(&self, phys: &[Self::Physical], spec: &mut [Self::Spectral]) {
        let mut scratch = vec![Self::Spectral::zero(); self.len_ortho()];
        self.ortho.forward(phys, &mut scratch);
        self.from_ortho(&scratch, spec);
    }

    fn backward(&self, spec: &[Self::Spectral], phys: &mut [Self::Physical]) {
        self.to_ortho(spec, phys);
        self.ortho.backward_inplace(phys);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//                                          Tests
//////////////////////////////////////////////////////////////////////////////////////////////////
#[cfg(test)]
mod test {
    use super::*;
    use crate::utils::approx_eq;

    #[test]
    fn test_ch_dir_transform() {
        let ch = ChebyshevComposite::<f64>::dirichlet(6);
        let mut v: Vec<f64> = (0..ch.len_phys()).map(|x| x as f64).collect();
        let mut vhat: Vec<f64> = vec![0.; ch.len_spec()];
        ch.forward(&v, &mut vhat);
        approx_eq(
            &vhat,
            &vec![1.666666, 1.2610938576665822, 0.8333334, 0.7333334],
        );
        ch.backward(&vhat, &mut v);
        approx_eq(&v, &vec![0.0, 0.166666, 2., 2.166666, 4., 0.]);
    }

    #[test]
    fn test_ch_dir_diff() {
        let ch = ChebyshevComposite::<f64>::dirichlet(6);
        let v: Vec<f64> = (0..ch.len_phys()).map(|x| x as f64).collect();
        let mut vhat: Vec<f64> = vec![0.; ch.len_spec()];
        let mut dvhat: Vec<f64> = vec![0.; ch.len_ortho()];
        ch.forward(&v, &mut vhat);
        ch.diff(&vhat, &mut dvhat, 2);
        approx_eq(
            &dvhat,
            &vec![-30.0, -100.66625258399796, -40.0, -58.6666, 0.0, 0.0],
        );
    }

    #[test]
    fn test_ch_dir_neu_transform() {
        let ch = ChebyshevComposite::<f64>::dirichlet_neumann(6);
        let mut v: Vec<f64> = (0..ch.len_phys()).map(|x| x as f64).collect();
        let mut vhat: Vec<f64> = vec![0.; ch.len_spec()];
        ch.forward(&v, &mut vhat);
        approx_eq(
            &vhat,
            &vec![
                2.480497656739244,
                0.12173047156497377,
                0.37048237944528406,
                0.12385458652875704,
            ],
        );
        ch.backward(&vhat, &mut v);
        approx_eq(
            &v,
            &vec![
                0.,
                0.8809270232753428,
                2.107806302104389,
                2.846040171074075,
                4.282240563110729,
                4.570948448263369,
            ],
        );
    }

    #[test]
    fn test_ch_biharm_a_transform() {
        let n = 14;
        let ch = ChebyshevComposite::<f64>::biharmonic_a(n);
        let v: Vec<f64> = (0..n).map(|x| x as f64).collect();
        let mut vhat: Vec<f64> = vec![0.; n - 4];
        ch.forward(&v, &mut vhat);
        approx_eq(
            &vhat,
            &vec![
                4.56547619, 3.33647046, 4.23015873, 3.78717098, 3.62142857, 3.31016028, 2.43197279,
                2.21938133, 1.04034392, 0.9391508,
            ],
        );
    }

    #[test]
    fn test_ch_biharm_b_transform() {
        let n = 14;
        let ch = ChebyshevComposite::<f64>::biharmonic_b(n);
        let v: Vec<f64> = (0..n).map(|x| x as f64).collect();
        let mut vhat: Vec<f64> = vec![0.; n - 4];
        ch.forward(&v, &mut vhat);
        approx_eq(
            &vhat,
            &vec![
                5.08540138, 3.86188728, 3.9395884, 3.57256415, 3.16060956, 2.96883245, 2.14734963,
                2.0152583, 0.96481296, 0.89163043,
            ],
        );
    }
}
