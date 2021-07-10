//! # Funspace
//! <img align="right" src="https://rustacean.net/assets/cuddlyferris.png" width="80">
//!
//! Collection of function spaces.
//!
//! A function space is made up of elements of basis functions.
//! Every function in the function space can be represented as a
//! linear combination of basis functions, represented by real/complex
//! coefficients (spectral space).
//!
//! ## Transform
//! A transform describes a change from the physical space to the function
//! space. For example, a fourier transform describes a transform from
//! values of a function on a regular grid to coefficents of sine/cosine
//! polynomials. This concept is analogous to other function spaces.
//!
//! ## Differentiation
//! One key advantage of representation of a function with coefficents in
//! the function space is its ease of differentiation. Differentiation in
//! fourier space becomes multiplication with the wavenumbe vector.
//! Differentiation in Chebyshev space can be done easily by a recurrence
//! relation.
//! Each base implements a differentiation method, which must be applied on
//! an array of coefficents.
//!
//! ## Composite Bases
//! Bases like those of fourier polynomials or chebyshev polynomials are
//! considered orthonormal bases, i.e. the dot product of each individual
//! polynomial with any other of its set vanishes.
//! But other function spaces can be constructed by a linear combination
//! the orthonormal basis functions. This is often used
//! to construct bases which satisfy particular boundary conditions
//! like dirichlet (zero at the ends) or neumann (zero derivative at the ends).
//! This is usefull when solving partial differential equations. When expressed
//! in those composite function space, the boundary condition is automatically
//! satisfied. This may be understood under *Galerkin* Method.
//!
//! To switch from its composite form to the orthonormal form, each base implements
//! a *Parental* trait, which defines the transform *to_ortho* and *from_ortho*.
//! If the base is already orthogonal, the input will be returned, otherwise it
//! is returned. Note that the dimensionality of the composite space is often
//! less than its orthogonal counterpart.  Therefore the output array must
//! not maintain the same shape (but dimensionality is conserved).
//!
//! ## Implemented function spaces:
//! - Chebyshev (Orthogonal), see [`chebyshev()`]
//! - ChebDirichlet (Composite), see [`cheb_dirichlet()`]
//! - ChebNeumann (Composite), see [`cheb_neumann()`]
//!
//! # Example
//! Apply forward transform of 1d array in cheb_dirichlet space
//! ```
//! use funspace::{Transform, cheb_dirichlet};
//! use ndarray::prelude::*;
//! let mut cd = cheb_dirichlet::<f64>(5);
//! let mut input = array![1., 2., 3., 4., 5.];
//! let output = cd.forward(&mut input, 0);
//! ```
#![allow(clippy::just_underscores_and_digits)]
#[macro_use]
extern crate enum_dispatch;
pub mod chebyshev;
mod impl_transform;
mod traits;
pub mod utils;
use chebyshev::Chebyshev;
use chebyshev::CompositeChebyshev;
use ndarray::prelude::*;
use ndarray::ScalarOperand;
use num_traits::{Float, FromPrimitive, Signed, Zero};
use std::fmt::Debug;
pub use traits::{Differentiate, FromOrtho, LaplacianInverse, Mass, Size, Transform};

/// Generic floating point number, implemented for f32 and f64
pub trait FloatNum:
    Copy + Zero + FromPrimitive + Signed + Sync + Send + Float + Debug + 'static + ScalarOperand
{
}
impl FloatNum for f32 {}
impl FloatNum for f64 {}

/// Collection of all implemented basis functions.
///
/// This enum implements the traits
/// [`Differentiate`], [`Mass`], [`LaplacianInverse`], [`Size`], [`Transform`], [`FromOrtho`]
///
/// # Example
/// Apply diferentiation in ChebDirichlet space
/// ```
/// use funspace::{cheb_dirichlet};
/// use funspace::Differentiate;
/// use ndarray::prelude::*;
/// let cd = cheb_dirichlet::<f64>(5);
/// let input = array![1., 2., 3.,];
/// let output = cd.differentiate(&input, 2, 0);
/// ```
#[allow(clippy::large_enum_variant)]
#[enum_dispatch(Differentiate<T>, Mass<T>, LaplacianInverse<T>, Size, FromOrtho<T>)]
pub enum Base<T: FloatNum> {
    Chebyshev(Chebyshev<T>),
    CompositeChebyshev(CompositeChebyshev<T>),
}

/// Function space for Chebyshev Polynomials
///
/// $$
/// T_k
/// $$
///
/// ## Example
/// Transform array to function space.
/// ```
/// use funspace::chebyshev;
/// use funspace::Transform;
/// let mut ch = chebyshev::<f64>(10);
/// let mut y = ndarray::Array::linspace(0., 9., 10);
/// let yhat = ch.forward(&mut y, 0);
/// ```
pub fn chebyshev<A: FloatNum>(n: usize) -> Base<A> {
    Base::Chebyshev(Chebyshev::<A>::new(n))
}

/// Function space with Dirichlet boundary conditions
///
/// $$
///  \phi_k = T_k - T_{k+2}
/// $$
/// ## Example
/// Transform array to function space.
/// ```
/// use funspace::cheb_dirichlet;
/// use funspace::Transform;
/// let mut cd = cheb_dirichlet::<f64>(10);
/// let mut y = ndarray::Array::linspace(0., 9., 10);
/// let yhat = cd.forward(&mut y, 0);
/// ```
pub fn cheb_dirichlet<A: FloatNum>(n: usize) -> Base<A> {
    Base::CompositeChebyshev(CompositeChebyshev::<A>::dirichlet(n))
}

/// Function space with Neumann boundary conditions
///
/// $$
/// \phi_k = T_k - k^{2} \/ (k+2)^2 T_{k+2}
/// $$
/// ## Example
/// Transform array to function space.
/// ```
/// use funspace::cheb_neumann;
/// use funspace::Transform;
/// let mut cn = cheb_neumann::<f64>(10);
/// let mut y = ndarray::Array::linspace(0., 9., 10);
/// let yhat = cn.forward(&mut y, 0);
/// ```
pub fn cheb_neumann<A: FloatNum>(n: usize) -> Base<A> {
    Base::CompositeChebyshev(CompositeChebyshev::<A>::neumann(n))
}

/// Functions space for inhomogeneous Dirichlet
/// boundary conditions
///
/// $$
///     \phi_0 = 0.5 T_0 - 0.5 T_1
/// $$
/// $$
///     \phi_1 = 0.5 T_0 + 0.5 T_1
/// $$
pub fn cheb_dirichlet_bc<A: FloatNum>(n: usize) -> Base<A> {
    Base::CompositeChebyshev(CompositeChebyshev::<A>::dirichlet_bc(n))
}

/// Functions space for inhomogeneous Neumann
/// boundary conditions
///
/// $$
///     \phi_0 = 0.5T_0 - 1/8T_1
/// $$
/// $$
///     \phi_1 = 0.5T_0 + 1/8T_1
/// $$
pub fn cheb_neumann_bc<A: FloatNum>(n: usize) -> Base<A> {
    Base::CompositeChebyshev(CompositeChebyshev::<A>::neumann_bc(n))
}
