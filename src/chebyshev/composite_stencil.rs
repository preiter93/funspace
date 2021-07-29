//! Transformation stencils from orthonormal chebyshev space to composite space
//! $$
//! p = S c
//! $$
//! where $S$ is a two-dimensional transform matrix.
#![allow(clippy::used_underscore_binding)]
use crate::FloatNum;
use ndarray::prelude::*;

/// Elementary methods for stencils
#[enum_dispatch]
pub trait Stencil<A> {
    /// Multiply stencil with a 1d array
    fn multiply_vec<S>(&self, composite_coeff: &ArrayBase<S, Ix1>) -> Array1<A>
    where
        S: ndarray::Data<Elem = A>;

    /// Multiply stencil with a 1d array (output must be supplied)
    fn multiply_vec_inplace<S1, S2>(
        &self,
        composite_coeff: &ArrayBase<S1, Ix1>,
        parent_coeff: &mut ArrayBase<S2, Ix1>,
    ) where
        S1: ndarray::Data<Elem = A>,
        S2: ndarray::Data<Elem = A> + ndarray::DataMut;

    /// Solve linear system $A c = p$, where stencil is matrix $A$.
    fn solve_vec<S>(&self, parent_coeff: &ArrayBase<S, Ix1>) -> Array1<A>
    where
        S: ndarray::Data<Elem = A>;

    /// Solve linear system $A c = p$, where stencil is matrix $A$ (output must be supplied)
    fn solve_vec_inplace<S1, S2>(
        &self,
        parent_coeff: &ArrayBase<S1, Ix1>,
        composite_coeff: &mut ArrayBase<S2, Ix1>,
    ) where
        S1: ndarray::Data<Elem = A>,
        S2: ndarray::Data<Elem = A> + ndarray::DataMut;

    /// Return stencil as 2d array
    fn to_array(&self) -> Array2<A>;
}

#[enum_dispatch(Stencil<A>)]
#[derive(Clone)]
pub enum ChebyshevStencil<A: FloatNum> {
    StencilChebyshev(StencilChebyshev<A>),
    StencilChebyshevBoundary(StencilChebyshevBoundary<A>),
}

/// Container for Chebyshev Stencil (internally used)
#[derive(Clone)]
pub struct StencilChebyshev<A> {
    /// Number of coefficients in parent space
    n: usize,
    /// Number of coefficients in parent space
    m: usize,
    /// Main diagonal
    diag: Array1<A>,
    /// Subdiagonal offset -2
    low2: Array1<A>,
    /// For tdma (diagonal)
    main: Array1<A>,
    /// For tdma (off-diagonal)
    off: Array1<A>,
}

/// Container for Boundary Condition Stencil
///
/// This stencil is fully defined by the
/// the 2 coefficients that act on $T_0$ and the
/// 2 coefficients that act on $T_1$, where $T$
/// are the basis function of the orthonormal
/// chebyshev basis.
#[derive(Clone)]
pub struct StencilChebyshevBoundary<A> {
    /// Number of coefficients in parent space
    n: usize,
    /// Number of coefficients in parent space
    m: usize,
    /// T0
    t0: Array1<A>,
    /// T1
    t1: Array1<A>,
}

impl<A: FloatNum> StencilChebyshev<A> {
    /// Return stencil of chebyshev dirichlet space
    /// $$
    ///  \phi_k = T_k - T_{k+2}
    /// $$
    ///
    /// Reference:
    /// J. Shen: Effcient Spectral-Galerkin Method II.
    pub fn dirichlet(n: usize) -> Self {
        let m = Self::get_m(n);
        let diag = Array::from_vec(vec![A::one(); m]);
        let low2 = Array::from_vec(vec![-A::one(); m]);
        let (main, off) = Self::_get_main_off(&diag.view(), &low2.view());
        Self {
            n,
            m,
            diag,
            low2,
            main,
            off,
        }
    }

    /// Return stencil of chebyshev neumann space
    /// $$
    /// \phi_k = T_k - k^{2} \/ (k+2)^2 T_{k+2}
    /// $$
    ///
    /// Reference:
    /// J. Shen: Effcient Spectral-Galerkin Method II.
    pub fn neumann(n: usize) -> Self {
        let m = Self::get_m(n);
        let diag = Array::from_vec(vec![A::one(); m]);
        let mut low2 = Array::from_vec(vec![A::zero(); m]);
        for (k, v) in low2.iter_mut().enumerate() {
            let k_ = A::from_f64(k.pow(2) as f64).unwrap();
            let k2_ = A::from_f64((k + 2).pow(2) as f64).unwrap();
            *v = -A::one() * k_ / k2_;
        }
        let (main, off) = Self::_get_main_off(&diag.view(), &low2.view());
        Self {
            n,
            m,
            diag,
            low2,
            main,
            off,
        }
    }

    /// Get main diagonal and off diagonal, used in [`StencilChebyshev::solve_vec_inplace`]
    fn _get_main_off(diag: &ArrayView1<A>, low2: &ArrayView1<A>) -> (Array1<A>, Array1<A>) {
        let m = diag.len();
        let mut main = Array::from_vec(vec![A::zero(); m]);
        let mut off = Array::from_vec(vec![A::zero(); m - 2]);
        for (i, v) in main.iter_mut().enumerate() {
            *v = diag[i] * diag[i] + low2[i] * low2[i];
        }
        for (i, v) in off.iter_mut().enumerate() {
            *v = diag[i + 2] * low2[i];
        }
        (main, off)
    }

    /// Composite spaces can be smaller than its orthonormal counterpart
    pub fn get_m(n: usize) -> usize {
        n - 2
    }
}

impl<A: FloatNum> Stencil<A> for StencilChebyshev<A> {
    /// Returns transform stencil as 2d ndarray
    fn to_array(&self) -> Array2<A> {
        let mut mat = Array2::<A>::zeros((self.n, self.m).f());
        for (i, (d, l)) in self.diag.iter().zip(self.low2.iter()).enumerate() {
            mat[[i, i]] = *d;
            mat[[i + 2, i]] = *l;
        }
        mat
    }

    /// Multiply stencil with a 1d array (transforms to parent coefficents)
    /// input and output array do usually differ in size.
    fn multiply_vec<S>(&self, composite_coeff: &ArrayBase<S, Ix1>) -> Array1<A>
    where
        S: ndarray::Data<Elem = A>,
    {
        let mut parent_coeff = Array1::<A>::zeros(self.n);
        self.multiply_vec_inplace(composite_coeff, &mut parent_coeff);
        parent_coeff
    }

    /// See [`StencilChebyshev::multiply_vec`]
    fn multiply_vec_inplace<S1, S2>(
        &self,
        composite_coeff: &ArrayBase<S1, Ix1>,
        parent_coeff: &mut ArrayBase<S2, Ix1>,
    ) where
        S1: ndarray::Data<Elem = A>,
        S2: ndarray::Data<Elem = A> + ndarray::DataMut,
    {
        parent_coeff.mapv_inplace(|x| x * A::zero());
        parent_coeff[0] = self.diag[0] * composite_coeff[0];
        parent_coeff[1] = self.diag[1] * composite_coeff[1];
        for i in 2..self.n - 2 {
            parent_coeff[i] =
                self.diag[i] * composite_coeff[i] + self.low2[i - 2] * composite_coeff[i - 2];
        }
        parent_coeff[self.n - 2] = self.low2[self.n - 4] * composite_coeff[self.n - 4];
        parent_coeff[self.n - 1] = self.low2[self.n - 3] * composite_coeff[self.n - 3];
    }

    /// Solve linear algebraic system $p = S c$ for $p$ with given composite
    /// coefficents $c$.
    ///
    /// Input and output array do usually differ in size.
    fn solve_vec<S>(&self, parent_coeff: &ArrayBase<S, Ix1>) -> Array1<A>
    where
        S: ndarray::Data<Elem = A>,
    {
        let mut composite_coeff = Array1::<A>::zeros(self.m);
        self.solve_vec_inplace(parent_coeff, &mut composite_coeff);
        composite_coeff
    }

    /// See [`StencilChebyshev::solve_vec`]
    fn solve_vec_inplace<S1, S2>(
        &self,
        parent_coeff: &ArrayBase<S1, Ix1>,
        composite_coeff: &mut ArrayBase<S2, Ix1>,
    ) where
        S1: ndarray::Data<Elem = A>,
        S2: ndarray::Data<Elem = A> + ndarray::DataMut,
    {
        use super::linalg::tdma;
        // Multiply right hand side
        for i in 0..self.m {
            composite_coeff[i] =
                self.diag[i] * parent_coeff[i] + self.low2[i] * parent_coeff[i + 2];
        }
        // Solve tridiagonal system
        tdma(
            &self.off.view(),
            &self.main.view(),
            &self.off.view(),
            &mut composite_coeff.view_mut(),
        );
    }
}

impl<A: FloatNum> StencilChebyshevBoundary<A> {
    /// dirichlet_bc basis
    /// $$
    ///     \phi_0 = 0.5 T_0 - 0.5 T_1
    /// $$
    /// $$
    ///     \phi_1 = 0.5 T_0 + 0.5 T_1
    /// $$
    pub fn dirichlet(n: usize) -> Self {
        let m = Self::get_m(n);
        let _05 = A::from_f64(0.5).unwrap();
        let t0 = Array::from_vec(vec![_05, _05]);
        let t1 = Array::from_vec(vec![-(_05), _05]);
        StencilChebyshevBoundary { n, m, t0, t1 }
    }

    /// neumann_bc basis
    /// $$
    ///     \phi_0 = 0.5T_0 - 1/8T_1
    /// $$
    /// $$
    ///     \phi_1 = 0.5T_0 + 1/8T_1
    /// $$
    pub fn neumann(n: usize) -> Self {
        let m = Self::get_m(n);
        let _05 = A::from_f64(0.5).unwrap();
        let _18 = A::from_f64(1. / 8.).unwrap();
        let t0 = Array::from_vec(vec![_05, _05]);
        let t1 = Array::from_vec(vec![-(_18), _18]);
        StencilChebyshevBoundary { n, m, t0, t1 }
    }

    /// Return size of spectral space (number of coefficients) from size in physical space
    pub fn get_m(_n: usize) -> usize {
        2
    }
}

impl<A: FloatNum> Stencil<A> for StencilChebyshevBoundary<A> {
    /// Returns transform stencil as 2d ndarray
    fn to_array(&self) -> Array2<A> {
        let mut mat = Array2::<A>::zeros((self.n, self.m).f());
        mat[[0, 0]] = self.t0[0];
        mat[[0, 1]] = self.t0[1];
        mat[[1, 0]] = self.t1[0];
        mat[[1, 1]] = self.t1[1];
        mat
    }

    /// Multiply stencil with a 1d array (transforms to parent coefficents)
    /// input and output array do usually differ in size.
    fn multiply_vec<S>(&self, composite_coeff: &ArrayBase<S, Ix1>) -> Array1<A>
    where
        S: ndarray::Data<Elem = A>,
    {
        let mut parent_coeff = Array1::<A>::zeros(self.n);
        self.multiply_vec_inplace(composite_coeff, &mut parent_coeff);
        parent_coeff
    }

    /// See [`StencilChebyshevBoundary::multiply_vec`]
    fn multiply_vec_inplace<S1, S2>(
        &self,
        composite_coeff: &ArrayBase<S1, Ix1>,
        parent_coeff: &mut ArrayBase<S2, Ix1>,
    ) where
        S1: ndarray::Data<Elem = A>,
        S2: ndarray::Data<Elem = A> + ndarray::DataMut,
    {
        parent_coeff.mapv_inplace(|x| x * A::zero());
        parent_coeff[0] = self.t0[0] * composite_coeff[0] + self.t0[1] * composite_coeff[1];
        parent_coeff[1] = self.t1[0] * composite_coeff[0] + self.t1[1] * composite_coeff[1];
    }

    /// Solve linear algebraic system $p = S c$ for $p$ with given composite
    /// coefficents $c$.
    ///
    /// Input and output array do usually differ in size.
    fn solve_vec<S>(&self, parent_coeff: &ArrayBase<S, Ix1>) -> Array1<A>
    where
        S: ndarray::Data<Elem = A>,
    {
        let mut composite_coeff = Array1::<A>::zeros(self.m);
        self.solve_vec_inplace(parent_coeff, &mut composite_coeff);
        composite_coeff
    }

    /// See [`StencilChebyshevBoundary::solve_vec`]
    fn solve_vec_inplace<S1, S2>(
        &self,
        parent_coeff: &ArrayBase<S1, Ix1>,
        composite_coeff: &mut ArrayBase<S2, Ix1>,
    ) where
        S1: ndarray::Data<Elem = A>,
        S2: ndarray::Data<Elem = A> + ndarray::DataMut,
    {
        let c0 = self.t0[0] * parent_coeff[0] + self.t1[0] * parent_coeff[1];
        let c1 = self.t0[1] * parent_coeff[0] + self.t1[1] * parent_coeff[1];
        // Determinante
        let a = self.t0[0] * self.t0[0] + self.t1[0] * self.t1[0];
        let b = self.t0[0] * self.t0[1] + self.t1[0] * self.t1[1];
        let c = self.t0[1] * self.t0[0] + self.t1[1] * self.t1[0];
        let d = self.t0[1] * self.t0[1] + self.t1[1] * self.t1[1];

        let det = A::one() / (a * d - b * c);
        composite_coeff[0] = det * (d * c0 - b * c1);
        composite_coeff[1] = det * (a * c1 - c * c0);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::utils::approx_eq;

    #[test]
    fn test_stench_cheb() {
        let stencil = StencilChebyshev::<f64>::dirichlet(5);
        let parent = Array::from_vec(vec![2., 0.7071, -1., -0.7071, -1.]);
        let composite = stencil.solve_vec(&parent);
        approx_eq(&composite, &array![2., 0.70710678, 1.]);

        let stencil = StencilChebyshev::<f64>::dirichlet(5);
        let composite = Array::from_vec(vec![2., 0.70710678, 1.]);
        let parent = stencil.multiply_vec(&composite);
        approx_eq(&parent, &array![2., 0.7071, -1., -0.7071, -1.]);
    }

    #[test]
    fn test_stench_cheb_boundary() {
        let stencil = StencilChebyshevBoundary::<f64>::dirichlet(4);
        let parent = Array::from_vec(vec![1., 2., 3., 4.]);
        let composite = stencil.solve_vec(&parent);
        approx_eq(&composite, &array![-1., 3.]);

        let stencil = StencilChebyshevBoundary::<f64>::dirichlet(4);
        let composite = Array::from_vec(vec![1., 2.]);
        let parent = stencil.multiply_vec(&composite);
        approx_eq(&parent, &array![1.5, 0.5, 0., 0.]);
    }
}
