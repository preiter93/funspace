//! # Composite chebyshev spaces
use super::ortho::Chebyshev;
use super::stencils::StencilOperations;
use super::stencils::{BiHarmonic, ChebyshevStencils, Dirichlet, DirichletNeumann, Neumann};
use crate::enums::BaseKind;
use crate::traits::{FunspaceElemental, FunspaceExtended, FunspaceSize};
use crate::types::{FloatNum, ScalarNum};
use ndarray::Array2;
use std::clone::Clone;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Clone)]
pub struct ChebyshevComposite<A: FloatNum> {
    /// Number of coefficients in physical space
    pub n: usize,
    /// Number of coefficients in spectral space
    pub m: usize,
    /// Parent base
    pub ortho: Chebyshev<A>,
    /// Transform stencil
    pub stencil: ChebyshevStencils<A>,
    // /// Kind of base
    // pub base_kind: BaseKind,
}

impl<A: FloatNum> FunspaceSize for ChebyshevComposite<A> {
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
        self.ortho.len_orth()
    }
}

impl<A: FloatNum> FunspaceExtended for ChebyshevComposite<A> {
    type Real = A;

    type Spectral = A;

    /// Return kind of base
    fn base_kind(&self) -> BaseKind {
        match self.stencil {
            ChebyshevStencils::Dirichlet(_) => BaseKind::ChebDirichlet,
            ChebyshevStencils::Neumann(_) => BaseKind::ChebNeumann,
            ChebyshevStencils::DirichletNeumann(_) => BaseKind::ChebDirichletNeumann,
            ChebyshevStencils::BiHarmonic(_) => BaseKind::ChebBiHarmonic,
        }
    }
    /// Coordinates in physical space
    fn get_nodes(&self) -> Vec<A> {
        Chebyshev::nodes(self.len_phys())
    }

    /// Mass matrix
    fn mass(&self) -> Array2<A> {
        self.stencil.to_array()
    }

    /// Explicit differential operator
    fn diffmat(&self, deriv: usize) -> Array2<A> {
        self.ortho.diffmat(deriv)
    }

    /// Laplacian $ L $
    fn laplace(&self) -> Array2<A> {
        self.ortho.laplace()
    }

    /// Pseudoinverse mtrix of Laplacian $ L^{-1} $
    fn laplace_inv(&self) -> Array2<A> {
        self.ortho.laplace_inv()
    }

    /// Pseudoidentity matrix of laplacian $ L^{-1} L $
    fn laplace_inv_eye(&self) -> Array2<A> {
        self.ortho.laplace_inv_eye()
    }
}

impl<A: FloatNum> ChebyshevComposite<A> {
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
        let stencil = Dirichlet::new(n);
        Self {
            n,
            m: Dirichlet::<A>::get_m(n),
            stencil: ChebyshevStencils::Dirichlet(stencil),
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
        let stencil = Neumann::new(n);
        Self {
            n,
            m: Neumann::<A>::get_m(n),
            stencil: ChebyshevStencils::Neumann(stencil),
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
        let stencil = DirichletNeumann::new(n);
        Self {
            n,
            m: DirichletNeumann::<A>::get_m(n),
            stencil: ChebyshevStencils::DirichletNeumann(stencil),
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
    pub fn biharmonic(n: usize) -> Self {
        let stencil = BiHarmonic::new(n);
        Self {
            n,
            m: BiHarmonic::<A>::get_m(n),
            stencil: ChebyshevStencils::BiHarmonic(stencil),
            ortho: Chebyshev::<A>::new(n),
        }
    }
}

impl<A: FloatNum + ScalarNum> FunspaceElemental for ChebyshevComposite<A> {
    type Physical = A;

    type Spectral = A;

    fn differentiate_slice<T>(&self, indata: &[T], outdata: &mut [T], n_times: usize)
    where
        T: ScalarNum
            + Add<A, Output = T>
            + Mul<A, Output = T>
            + Div<A, Output = T>
            + Sub<A, Output = T>,
    {
        let mut scratch: Vec<T> = vec![T::zero(); self.len_orth()];
        self.to_ortho_slice(indata, &mut scratch);
        self.ortho.differentiate_slice(&scratch, outdata, n_times);
    }

    fn forward_slice(&self, indata: &[Self::Physical], outdata: &mut [Self::Spectral]) {
        let mut scratch: Vec<Self::Spectral> = vec![Self::Spectral::zero(); self.len_orth()];
        self.ortho.forward_slice(indata, &mut scratch);
        self.from_ortho_slice(&scratch, outdata);
    }

    fn backward_slice(&self, indata: &[Self::Spectral], outdata: &mut [Self::Physical]) {
        let mut scratch: Vec<Self::Spectral> = vec![Self::Spectral::zero(); self.len_orth()];
        self.to_ortho_slice(indata, &mut scratch);
        self.ortho.backward_slice(&scratch, outdata);
    }

    fn to_ortho_slice<T>(&self, indata: &[T], outdata: &mut [T])
    where
        T: ScalarNum
            + Add<A, Output = T>
            + Mul<A, Output = T>
            + Div<A, Output = T>
            + Sub<A, Output = T>,
    {
        self.stencil.dot_inplace(indata, outdata);
    }

    fn from_ortho_slice<T>(&self, indata: &[T], outdata: &mut [T])
    where
        T: ScalarNum
            + Add<A, Output = T>
            + Mul<A, Output = T>
            + Div<A, Output = T>
            + Sub<A, Output = T>,
    {
        self.stencil.solve_inplace(indata, outdata);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//                                          Tests
//////////////////////////////////////////////////////////////////////////////////////////////////
#[cfg(test)]
mod test {
    use super::*;
    use crate::utils::{approx_eq, approx_eq_ndarray};
    use ndarray::{array, Array2};

    #[test]
    fn test_cheb_dirichlet_transform() {
        let ch = ChebyshevComposite::<f64>::dirichlet(6);
        let mut indata: Vec<f64> = (0..ch.len_phys()).map(|x| x as f64).collect();
        let mut outdata: Vec<f64> = vec![0.; ch.len_spec()];
        ch.forward_slice(&indata, &mut outdata);
        approx_eq(
            &outdata,
            &vec![1.666666, 1.2610938576665822, 0.8333334, 0.7333334],
        );
        ch.backward_slice(&outdata, &mut indata);
        approx_eq(&indata, &vec![0.0, 0.166666, 2., 2.166666, 4., 0.]);
    }

    #[test]
    fn test_cheb_dirichlet_neumann_transform() {
        let ch = ChebyshevComposite::<f64>::dirichlet_neumann(6);
        let mut indata: Vec<f64> = (0..ch.len_phys()).map(|x| x as f64).collect();
        let mut outdata: Vec<f64> = vec![0.; ch.len_spec()];
        ch.forward_slice(&indata, &mut outdata);
        approx_eq(
            &outdata,
            &vec![
                2.480497656739244,
                0.12173047156497377,
                0.37048237944528406,
                0.12385458652875704,
            ],
        );
        ch.backward_slice(&outdata, &mut indata);
        approx_eq(
            &indata,
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
    fn test_cheb_biharmonic_transform() {
        let n = 14;
        let ch = ChebyshevComposite::<f64>::biharmonic(n);
        let mut indata: Vec<f64> = (0..n).map(|x| x as f64).collect();
        let mut outdata: Vec<f64> = vec![0.; n - 4];
        ch.forward_slice(&indata, &mut outdata);
        approx_eq(
            &outdata,
            &vec![
                4.56547619, 3.33647046, 4.23015873, 3.78717098, 3.62142857, 3.31016028, 2.43197279,
                2.21938133, 1.04034392, 0.9391508,
            ],
        );
    }

    #[test]
    /// Differantiate 2d array along first and second axis
    fn test_cheb_dirichlet_to_ortho() {
        let (nx, ny) = (5, 4);
        let mut composite_coeff = Array2::<f64>::zeros((nx - 2, ny));
        let mut orthonorm_coeff = Array2::<f64>::zeros((nx, ny));

        // Axis 0
        let cheby = ChebyshevComposite::<f64>::dirichlet(nx);
        for (i, v) in composite_coeff.iter_mut().enumerate() {
            *v = i as f64;
        }
        let expected = array![
            [0., 1., 2., 3.],
            [4., 5., 6., 7.],
            [8., 8., 8., 8.],
            [-4., -5., -6., -7.],
            [-8., -9., -10., -11.],
        ];
        cheby.to_ortho_inplace(&composite_coeff, &mut orthonorm_coeff, 0);
        approx_eq_ndarray(&orthonorm_coeff, &expected);

        // Axis 1
        let mut composite_coeff = Array2::<f64>::zeros((nx, ny - 2));
        let cheby = ChebyshevComposite::<f64>::dirichlet(ny);
        for (i, v) in composite_coeff.iter_mut().enumerate() {
            *v = i as f64;
        }
        let expected = array![
            [0., 1., 0., -1.],
            [2., 3., -2., -3.],
            [4., 5., -4., -5.],
            [6., 7., -6., -7.],
            [8., 9., -8., -9.],
        ];
        cheby.to_ortho_inplace(&composite_coeff, &mut orthonorm_coeff, 1);
        approx_eq_ndarray(&orthonorm_coeff, &expected);
    }

    #[test]
    fn test_chebdirichlet_differentiate() {
        let ch = ChebyshevComposite::<f64>::dirichlet(6);
        let indata: Vec<f64> = (0..ch.len_phys()).map(|x| x as f64).collect();
        let mut outdata: Vec<f64> = vec![0.; ch.len_spec()];
        let mut deriv: Vec<f64> = vec![0.; ch.len_orth()];
        ch.forward_slice(&indata, &mut outdata);
        ch.differentiate_slice(&outdata, &mut deriv, 2);
        approx_eq(
            &deriv,
            &vec![-30.0, -100.66625258399796, -40.0, -58.6666, 0.0, 0.0],
        );
    }

    #[test]
    /// Differantiate ChebDirichlet (2d array) twice along first and second axis
    fn test_chebdirichlet_differentiate_2d() {
        let (nx, ny) = (6, 4);
        let mut data = Array2::<f64>::zeros((nx, ny));

        // Axis 0
        let cheby = ChebyshevComposite::<f64>::dirichlet(nx + 2);
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as f64;
        }
        let expected = array![
            [-1440.0, -1548.0, -1656.0, -1764.0],
            [-5568.0, -5904.0, -6240.0, -6576.0],
            [-2688.0, -2880.0, -3072.0, -3264.0],
            [-4960.0, -5240.0, -5520.0, -5800.0],
            [-1920.0, -2040.0, -2160.0, -2280.0],
            [-3360.0, -3528.0, -3696.0, -3864.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ];
        let diff = cheby.differentiate(&data, 2, 0);
        approx_eq_ndarray(&diff, &expected);

        // Axis 1
        let cheby = ChebyshevComposite::<f64>::dirichlet(ny + 2);
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as f64;
        }
        let expected = array![
            [-56.0, -312.0, -96.0, -240.0, 0.0, 0.0],
            [-184.0, -792.0, -288.0, -560.0, 0.0, 0.0],
            [-312.0, -1272.0, -480.0, -880.0, 0.0, 0.0],
            [-440.0, -1752.0, -672.0, -1200.0, 0.0, 0.0],
            [-568.0, -2232.0, -864.0, -1520.0, 0.0, 0.0],
            [-696.0, -2712.0, -1056.0, -1840.0, 0.0, 0.0],
        ];
        let diff = cheby.differentiate(&data, 2, 1);
        approx_eq_ndarray(&diff, &expected);
    }
}
