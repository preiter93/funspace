use funspace::{cheb_dirichlet, fourier_r2c, BaseKind, BaseSpace, FloatNum, ScalarNum, Space2};
use ndarray::Ix;
use ndarray::{prelude::*, Data};
use num_complex::Complex;
use std::convert::TryInto;

pub struct FieldBase<A, T1, T2, S, const N: usize> {
    /// Number of dimensions
    pub ndim: usize,
    /// Space
    pub space: S,
    /// Field in physical space
    pub v: Array<T1, Dim<[Ix; N]>>,
    /// Field in spectral space
    pub vhat: Array<T2, Dim<[Ix; N]>>,
    /// Grid coordinates
    pub x: [Array1<A>; N],
    /// Grid deltas
    pub dx: [Array1<A>; N],
    // /// Collection of numerical solvers (Poisson, Hholtz, ...)
    // pub solvers: HashMap<String, SolverField<T, N>>,
}

/// One dimensional Field (f64 in physical space and T2 in spectral space)
pub type Field1<T2, S> = FieldBase<f64, f64, T2, S, 1>;
/// Two dimensional Field  (f64 in physical space and T2 in spectral space)
pub type Field2<T2, S> = FieldBase<f64, f64, T2, S, 2>;

impl<A, T1, T2, S, const N: usize> FieldBase<A, T1, T2, S, N>
where
    A: FloatNum + ScalarNum,
    Complex<A>: ScalarNum,
    S: BaseSpace<A, N, Physical = T1, Spectral = T2>,
{
    pub fn new(space: &S) -> Self {
        Self {
            ndim: N,
            space: space.clone(),
            v: space.ndarray_physical(),
            vhat: space.ndarray_spectral(),
            x: space.coords(),
            dx: Self::get_dx(&space.coords()),
        }
    }

    /// Forward transformation
    pub fn forward(&mut self) {
        self.space.forward_inplace_par(&mut self.v, &mut self.vhat);
    }

    /// Backward transformation
    pub fn backward(&mut self) {
        self.space.backward_inplace_par(&mut self.vhat, &mut self.v);
    }

    /// Transform from composite to orthogonal space
    pub fn to_ortho(&self) -> Array<T2, Dim<[usize; N]>> {
        self.space.to_ortho_par(&self.vhat)
    }

    /// Transform from orthogonal to composite space
    pub fn from_ortho<S1>(&mut self, input: &ArrayBase<S1, Dim<[usize; N]>>)
    where
        S1: Data<Elem = T2>,
    {
        self.space.from_ortho_inplace_par(input, &mut self.vhat);
    }

    /// Gradient
    // #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    pub fn gradient(&self, deriv: [usize; N], scale: Option<[A; N]>) -> Array<T2, Dim<[usize; N]>> {
        self.space.gradient_par(&self.vhat, deriv, scale)
    }

    /// Generate grid deltas from coordinates
    ///
    /// ## Panics
    /// When vec to array convection fails
    fn get_dx(x_arr: &[Array1<A>; N]) -> [Array1<A>; N] {
        let mut dx_vec = Vec::new();
        let two = A::one() + A::one();
        for x in x_arr.iter() {
            let mut dx = Array1::<A>::zeros(x.len());
            for (i, dxi) in dx.iter_mut().enumerate() {
                let xs_left = if i == 0 {
                    x[0]
                } else {
                    (x[i] + x[i - 1]) / two
                };
                let xs_right = if i == x.len() - 1 {
                    x[x.len() - 1]
                } else {
                    (x[i + 1] + x[i]) / two
                };
                *dxi = xs_right - xs_left;
            }
            dx_vec.push(dx);
        }
        dx_vec.try_into().unwrap_or_else(|v: Vec<Array1<A>>| {
            panic!("Expected Vec of length {} but got {}", N, v.len())
        })
    }

    #[allow(unreachable_patterns)]
    pub fn ingredients_for_hholtz(&self, axis: usize) -> (Array2<A>, Array2<A>, Option<Array2<A>>) {
        let kind = self.space.base_kind(axis);
        let mass = self.space.mass(axis);
        let lap = self.space.laplace(axis);

        // Matrices and optional preconditioner
        match kind {
            BaseKind::Chebyshev => {
                let peye = self.space.laplace_inv_eye(axis);
                let pinv = peye.dot(&self.space.laplace_inv(axis));
                let mass_sliced = mass.slice(s![.., 2..]);
                (pinv.dot(&mass_sliced), peye.dot(&mass_sliced), Some(pinv))
            }
            BaseKind::ChebDirichlet | BaseKind::ChebNeumann | BaseKind::ChebDirichletNeumann => {
                let peye = self.space.laplace_inv_eye(axis);
                let pinv = peye.dot(&self.space.laplace_inv(axis));
                (pinv.dot(&mass), peye.dot(&mass), Some(pinv))
            }
            BaseKind::FourierR2c | BaseKind::FourierC2c => (mass, lap, None),
            _ => panic!("No ingredients found for Base kind: {}!", kind),
        }
    }

    #[allow(unreachable_patterns)]
    pub fn ingredients_for_poisson(
        &self,
        axis: usize,
    ) -> (Array2<A>, Array2<A>, Option<Array2<A>>, bool) {
        // Matrices and preconditioner
        let (mat_a, mat_b, precond) = self.ingredients_for_hholtz(axis);

        // Boolean, if laplacian is already diagonal
        // if not, a eigendecomposition will diagonalize mat a,
        // however, this is more expense.
        let kind = self.space.base_kind(axis);
        let is_diag = match kind {
            BaseKind::Chebyshev
            | BaseKind::ChebDirichlet
            | BaseKind::ChebNeumann
            | BaseKind::ChebDirichletNeumann => false,
            BaseKind::FourierR2c | BaseKind::FourierC2c => true,
            _ => panic!("No ingredients found for Base kind: {}!", kind),
        };

        (mat_a, mat_b, precond, is_diag)
    }
}

fn main() {
    let space = Space2::new(&fourier_r2c::<f64>(10), &cheb_dirichlet::<f64>(10));
    let mut field = Field2::new(&space);
    for v in field.v.iter_mut() {
        *v = 1.;
    }
    field.forward();
    field.backward();
}
