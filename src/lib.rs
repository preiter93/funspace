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
//! a *Parental* trait, which defines the transform `to_ortho` and `from_ortho`.
//! If the base is already orthogonal, the input will be returned, otherwise it
//! is returned. Note that the dimensionality of the composite space is often
//! less than its orthogonal counterpart.  Therefore the output array must
//! not maintain the same shape (but dimensionality is conserved).
//!
//! ## Implemented function spaces:
//! - `Chebyshev` (Orthogonal), see [`chebyshev()`]
//! - `ChebDirichlet` (Composite), see [`cheb_dirichlet()`]
//! - `ChebNeumann` (Composite), see [`cheb_neumann()`]
//! - `Fourier` (Orthogonal), see [`fourier()`]
//! - `FourierR2c` (Orthogonal), see [`fourier_r2c()`]
//!
//! # Example
//! Apply forward transform of 1d array in `cheb_dirichlet` space
//! ```
//! use funspace::{Transform, cheb_dirichlet};
//! use ndarray::prelude::*;
//! use ndarray::Array1;
//! let mut cd = cheb_dirichlet::<f64>(5);
//! let mut input = array![1., 2., 3., 4., 5.];
//! let output: Array1<f64> = cd.forward(&mut input, 0);
//! ```
#![allow(clippy::just_underscores_and_digits)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_precision_loss)]
#[macro_use]
extern crate enum_dispatch;
pub mod chebyshev;
pub mod fourier;
mod impl_differentiate;
mod impl_from_ortho;
mod impl_transform;
pub mod space;
mod traits;
pub mod types;
pub mod utils;
use chebyshev::Chebyshev;
use chebyshev::CompositeChebyshev;
use fourier::{Fourier, FourierR2c};
// use fourier::Fourier;
use ndarray::prelude::*;
pub use space::{Space1, Space2, SpaceBase};
pub use traits::{Differentiate, FromOrtho, LaplacianInverse, Mass, Size, Transform, TransformPar};
pub use types::{Complex, FloatNum, Scalar};

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
#[enum_dispatch(Mass<T>, LaplacianInverse<T>, Size)]
#[derive(Clone)]
pub enum Base<T: FloatNum> {
    Chebyshev(Chebyshev<T>),
    CompositeChebyshev(CompositeChebyshev<T>),
    Fourier(Fourier<T>),
    FourierR2c(FourierR2c<T>),
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
/// use ndarray::Array1;
/// let mut ch = chebyshev::<f64>(10);
/// let mut y = ndarray::Array::linspace(0., 9., 10);
/// let yhat: Array1<f64> = ch.forward(&mut y, 0);
/// ```
#[must_use]
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
/// use ndarray::Array1;
/// let mut cd = cheb_dirichlet::<f64>(10);
/// let mut y = ndarray::Array::linspace(0., 9., 10);
/// let yhat: Array1<f64> = cd.forward(&mut y, 0);
/// ```
#[must_use]
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
/// use ndarray::Array1;
/// let mut cn = cheb_neumann::<f64>(10);
/// let mut y = ndarray::Array::linspace(0., 9., 10);
/// let yhat: Array1<f64> = cn.forward(&mut y, 0);
/// ```
#[must_use]
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
#[must_use]
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
#[must_use]
pub fn cheb_neumann_bc<A: FloatNum>(n: usize) -> Base<A> {
    Base::CompositeChebyshev(CompositeChebyshev::<A>::neumann_bc(n))
}

/// Function space for Fourier Polynomials
///
/// $$
/// F_k
/// $$
///
/// ## Example
/// Transform array to function space.
/// ```
/// use funspace::fourier;
/// use funspace::Transform;
/// use funspace::Complex;
/// let mut fo = fourier::<f64>(10);
/// let real = ndarray::Array::linspace(0., 9., 10);
/// let mut y = real.mapv(|x| Complex::new(x,x));
/// let yhat = fo.forward(&mut y, 0);
/// ```
#[must_use]
pub fn fourier<A: FloatNum>(n: usize) -> Base<A> {
    Base::Fourier(Fourier::<A>::new(n))
}

/// Function space for Fourier Polynomials
/// (Real-to-complex)
///
/// $$
/// F_k
/// $$
///
/// ## Example
/// Transform array to function space.
/// ```
/// use funspace::fourier_r2c;
/// use funspace::Transform;
/// use funspace::Complex;
/// use ndarray::Array1;
/// let mut fo = fourier_r2c::<f64>(10);
/// let mut y = ndarray::Array::linspace(0., 9., 10);
/// let yhat: Array1<Complex<f64>> = fo.forward(&mut y, 0);
/// ```
#[must_use]
pub fn fourier_r2c<A: FloatNum>(n: usize) -> Base<A> {
    Base::FourierR2c(FourierR2c::<A>::new(n))
}
