//! # Funspace
//! <img align="right" src="https://rustacean.net/assets/cuddlyferris.png" width="80">
//! Collection of function spaces.
//!
//! # Bases
//!
//! | Name                     | Transform-type | Orthogonal | Boundary conditions                  | Link                         |
//! |--------------------------|----------------|------------|--------------------------------------|------------------------------|
//! | ``Chebyshev``            | R2r            | True       | None                                 | [`chebyshev()`]              |
//! | ``ChebDirichlet ``       | R2r            | False      | u(-1) = u(1) = 0                     | [`cheb_dirichlet()`]         |
//! | ``ChebNeumann``          | R2r            | False      | u'(-1) = u'(1) = 0                   | [`cheb_neumann()`]           |
//! | ``ChebDirichletNeumann`` | R2r            | False      | u(-1) = u'(1) = 0                    | [`cheb_dirichlet_neumann()`] |
//! | ``ChebBiHarmonicA``      | R2r            | False      | u(-1)  = u'(-1) = u(1)= u'(1) = 0    | [`cheb_biharmonic_a()`]      |
//! | ``ChebBiHarmonicB``      | R2r            | False      | u(-1)  = u''(-1) = u(1)= u''(1) = 0  | [`cheb_biharmonic_b()`]      |
//! | ``FourierR2c``           | R2c            | True       | Periodic                             | [`fourier_r2c()`]            |
//! | ``FourierC2c``           | C2c            | True       | Periodic                             | [`fourier_c2c()`]            |

#![warn(clippy::pedantic)]
// #![allow(dead_code)]
// #![allow(unused_imports)]
#[macro_use]
extern crate enum_dispatch;
mod internal_macros;
// mod transpose;

pub mod chebyshev;
pub mod enums;
pub mod fourier;
pub mod space;
pub mod traits;
pub mod types;
pub mod utils;
pub use crate::enums::{BaseC2c, BaseKind, BaseR2c, BaseR2r};
use chebyshev::{Chebyshev, ChebyshevComposite};
use fourier::{FourierC2c, FourierR2c};
pub use types::{Real, Scalar};

// #[cfg(feature = "mpi")]
// pub mod space_mpi;

/// Function space for Chebyshev Polynomials
///
/// ```text
/// T_k
/// ```
///
/// ## Example
/// Transform array to function space.
/// ```
/// use funspace::chebyshev;
/// use funspace::traits::BaseTransform;
/// use ndarray::Array1;
/// let ch = chebyshev::<f64>(10);
/// let mut y = Array1::<f64>::linspace(0., 9., 10);
/// let yhat: Array1<f64> = ch.forward(&mut y, 0);
/// ```
#[must_use]
pub fn chebyshev<A: Real>(n: usize) -> BaseR2r<A> {
    BaseR2r::Chebyshev(Chebyshev::<A>::new(n))
}

/// Function space with Dirichlet boundary conditions
///
///```text
/// u(-1)=0 and u(1)=0
///```
///
///
/// ```text
///  \phi_k = T_k - T_{k+2}
/// ```
/// ## Example
/// Transform array to function space.
/// ```
/// use funspace::cheb_dirichlet;
/// use funspace::traits::BaseTransform;
/// use ndarray::Array1;
/// let cd = cheb_dirichlet::<f64>(10);
/// let mut y = Array1::<f64>::linspace(0., 9., 10);
/// let yhat: Array1<f64> = cd.forward(&mut y, 0);
/// ```
#[must_use]
pub fn cheb_dirichlet<A: Real>(n: usize) -> BaseR2r<A> {
    BaseR2r::ChebyshevComposite(ChebyshevComposite::<A>::dirichlet(n))
}

/// Function space with Neumann boundary conditions
///
///```text
/// u'(-1)=0 and u'(1)=0
///```
///
/// ```text
///  \phi_k = T_k - k^{2} \/ (k+2)^2 T_{k+2}
/// ```
///
/// ## Example
/// Transform array to function space.
/// ```
/// use funspace::cheb_neumann;
/// use funspace::traits::BaseTransform;
/// use ndarray::Array1;
/// let ch = cheb_neumann::<f64>(10);
/// let mut y = Array1::<f64>::linspace(0., 9., 10);
/// let yhat: Array1<f64> = ch.forward(&mut y, 0);
/// ```
#[must_use]
pub fn cheb_neumann<A: Real>(n: usize) -> BaseR2r<A> {
    BaseR2r::ChebyshevComposite(ChebyshevComposite::<A>::neumann(n))
}

/// Function space with Dirichlet boundary conditions at x=-1
/// and Neumann boundary conditions at x=1
///
///```text
/// u(-1)=0 and u'(1)=0
///```
#[must_use]
pub fn cheb_dirichlet_neumann<A: Real>(n: usize) -> BaseR2r<A> {
    BaseR2r::ChebyshevComposite(ChebyshevComposite::<A>::dirichlet_neumann(n))
}

/// Function space with biharmonic boundary conditions
///
/// ```text
/// u(-1)=0, u(1)=0, u'(-1)=0 and u'(1)=0
///```
#[must_use]
pub fn cheb_biharmonic_a<A: Real>(n: usize) -> BaseR2r<A> {
    BaseR2r::ChebyshevComposite(ChebyshevComposite::<A>::biharmonic_a(n))
}

/// Function space with biharmonic boundary conditions
///
/// ```text
/// u(-1)=0, u(1)=0, u''(-1)=0 and u''(1)=0
///```
#[must_use]
pub fn cheb_biharmonic_b<A: Real>(n: usize) -> BaseR2r<A> {
    BaseR2r::ChebyshevComposite(ChebyshevComposite::<A>::biharmonic_b(n))
}

/// Function space for Fourier Polynomials
///
/// ```text
/// F_k
/// ```
// /
// / ## Example
// / Transform array to function space.
// / ```
// / use funspace::fourier_c2c;
// / use funspace::traits::BaseTransform;
// / use num_complex::Complex;
// / let fo = fourier_c2c::<f64>(10);
// / let real = ndarray::Array::linspace(0., 9., 10);
// / let mut y = real.mapv(|x| Complex::new(x,x));
// / let yhat = fo.forward(&mut y, 0);
// / ```
#[must_use]
pub fn fourier_c2c<A: Real>(n: usize) -> BaseC2c<A> {
    BaseC2c::FourierC2c(FourierC2c::<A>::new(n))
}

/// Function space for Fourier Polynomials
/// (Real-to-complex)
///
/// ```text
/// F_k
/// ```
// /
// / ## Example
// / Transform array to function space.
// / ```
// / use funspace::fourier_r2c;
// / use funspace::traits::BaseTransform;
// / use num_complex::Complex;
// / use ndarray::Array1;
// / let fo = fourier_r2c::<f64>(10);
// / let mut y = ndarray::Array::linspace(0., 9., 10);
// / let yhat: Array1<Complex<f64>> = fo.forward(&mut y, 0);
// / ```
#[must_use]
pub fn fourier_r2c<A: Real>(n: usize) -> BaseR2c<A> {
    BaseR2c::FourierR2c(FourierR2c::<A>::new(n))
}
