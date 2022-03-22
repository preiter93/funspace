use funspace::{
    cheb_dirichlet, fourier_r2c, BaseAll, BaseC2c, BaseR2c, BaseR2r, BaseSpace, Basics, FloatNum,
    LaplacianInverse, Space2,
};
use ndarray::{prelude::*, Data};
use ndarray::{Ix, ScalarOperand};
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
    A: FloatNum,
    Complex<A>: ScalarOperand,
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

    pub fn ingredients_for_hholtz(&self, axis: usize) -> (Array2<A>, Array2<A>, Option<Array2<A>>) {
        let x = &self.space.base_all()[axis];
        let mass = x.mass();
        let lap = x.laplace();
        let peye = x.laplace_inv_eye();
        let pinv = peye.dot(&x.laplace_inv());

        // Matrices
        let (mat_a, mat_b) = match x {
            BaseAll::BaseR2r(ref b) => match b {
                BaseR2r::Chebyshev(_) => {
                    let mass_sliced = mass.slice(s![.., 2..]);
                    (pinv.dot(&mass_sliced), peye.dot(&mass_sliced))
                }
                BaseR2r::CompositeChebyshev(_) => (pinv.dot(&mass), peye.dot(&mass)),
            },
            BaseAll::BaseR2c(ref b) => match b {
                BaseR2c::FourierR2c(_) => (mass, lap),
            },
            BaseAll::BaseC2c(ref b) => match b {
                BaseC2c::FourierC2c(_) => (mass, lap),
            },
        };
        // Preconditioner (optional)
        let precond = match x {
            BaseAll::BaseR2r(ref b) => match b {
                BaseR2r::Chebyshev(_) | BaseR2r::CompositeChebyshev(_) => Some(pinv),
            },
            BaseAll::BaseR2c(_) | BaseAll::BaseC2c(_) => None,
        };
        (mat_a, mat_b, precond)
    }

    pub fn ingredients_for_poisson(
        &self,
        axis: usize,
    ) -> (Array2<A>, Array2<A>, Option<Array2<A>>, bool) {
        let x = &self.space.base_all()[axis];

        // Matrices and preconditioner
        let (mat_a, mat_b, precond) = self.ingredients_for_hholtz(axis);

        // Boolean, if laplacian is already diagonal
        // if not, a eigendecomposition will diagonalize mat a,
        // however, this is more expense.
        let is_diag = match x {
            BaseAll::BaseR2r(_) => false,
            BaseAll::BaseR2c(_) | BaseAll::BaseC2c(_) => true,
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
