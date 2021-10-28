//! # Funspace
//! <img align="right" src="https://rustacean.net/assets/cuddlyferris.png" width="80">
//!
//! Collection of function spaces.
//!
//! A function space is composed of elements of basis functions.
//! Each function in the function space can be represented as a
//! linear combination of basis functions, represented by real/complex
//! coefficients (spectral space).
//!
//! ## Implemented function spaces:
//! - `Chebyshev` (Orthogonal), see [`chebyshev()`]
//! - `ChebDirichlet` (Composite), see [`cheb_dirichlet()`]
//! - `ChebNeumann` (Composite), see [`cheb_neumann()`]
//! - `FourierC2c` (Orthogonal), see [`fourier_c2c()`]
//! - `FourierR2c` (Orthogonal), see [`fourier_r2c()`]
//!
//! ## Transform
//! A transformation describes a change from physical space to functional space.
//! For example, a Fourier transform describes a transformation of a function
//! on a regular grid into coefficients of sine/cosine polynomials.
//! This is analogous to other function spaces. The transformations are
//! implemented by the [`Transform`] trait.
//!
//! ### Example
//! Apply forward transform of 1d array in `cheb_dirichlet` space
//! ```
//! use funspace::{Transform, cheb_dirichlet};
//! use ndarray::prelude::*;
//! use ndarray::Array1;
//! let mut cd = cheb_dirichlet::<f64>(5);
//! let input = array![1., 2., 3., 4., 5.];
//! let output: Array1<f64> = cd.forward(&input, 0);
//! ```
//!
//! ## Differentiation
//! An essential advantage of the representation of a function with coefficients in
//! the function space is its ease of differentiation. Differentiation in
//! Fourier space becomes multiplication with the wavenumber vector.
//! Differentiation in Chebyshev space is done by a recurrence
//! relation and almost as fast as in Fourier space.
//! Each base implements a differentiation method, which is applied to
//! an array of coefficents. This is defined by the [`Differentiate`] trait.
//!
//! ### Example
//! Apply differentiation
//! ```
//! use funspace::{Transform, Differentiate, Basics, fourier_r2c};
//! use ndarray::prelude::*;
//! use ndarray::Array1;
//! use num_complex::Complex;
//! // Define base
//! let mut fo = fourier_r2c(8);
//! // Get coordinates in physical space
//! let x = fo.coords().clone();
//! let v = x.mapv(|xi: f64| (2. * xi).sin());
//! // Transform to physical space
//! let vhat: Array1<Complex<f64>> = fo.forward(&v, 0);
//!
//! // Apply differentiation twice along first axis
//! let dvhat = fo.differentiate(&vhat, 2, 0);
//! // Transform back to spectral space
//! let dv: Array1<f64> = fo.backward(&dvhat, 0);
//! // Compare with correct derivative
//! for (exp, ist) in x
//!     .mapv(|xi: f64| -4. * (2. * xi).sin())
//!     .iter()
//!     .zip(dv.iter())
//! {
//!     assert!((exp - ist).abs() < 1e-5);
//! }
//! ```
//!
//! ## Composite Bases
//! Bases such as those of Fourier polynomials or Chebyshev polynomials
//! are considered as orthogonal bases, i.e. the dot product of every single
//! polynomial with any other polynomial in its set vanishes. In these cases
//! the mass matrix is a diagonal matrix.
//! However, other function spaces can be constructed by a linear combination
//! of the orthogonal basis functions. In this way, bases can be constructed
//! construct bases that satisfy certain boundary conditions such as Dirichlet
//! (zero at the ends) or Neumann (zero derivative at the ends).
//! This is useful for solving partial differential equations. If they are
//! expressed in these composite function spaces, the boundary condition
//! is automatically satisfied. This is known as the *Galerkin* method.
//!
//! To switch from its composite form to the orthogonal form, each base implements
//! a [`FromOrtho`] trait, which defines the transform `to_ortho` and `from_ortho`.
//! If the base is already orthogonal, the input will be returned, otherwise it
//! is transformed from the composite space to the orthogonal space.
//! Note that the size of the composite space is usually
//! less than its orthogonal counterpart. In other words, the composite space is
//! usually a lower dimensional subspace of the orthogonal space. Therefore the output
//! array must not maintain the same shape.
//!
//! ### Example
//! Transform composite space `cheb_dirichlet` to its orthogonal counterpart
//! `chebyshev`. Note that `cheb_dirichlet` has 6 spectral coefficients,
//! while the `chebyshev` bases has 8.
//! ```
//! use funspace::{Transform, FromOrtho, Basics};
//! use funspace::{cheb_dirichlet, chebyshev};
//! use std::f64::consts::PI;
//! use ndarray::prelude::*;
//! use ndarray::Array1;
//! use num_complex::Complex;
//! // Define base
//! let mut ch = chebyshev(8);
//! let mut cd = cheb_dirichlet(8);
//! // Get coordinates in physical space
//! let x = ch.coords().clone();
//! let v = x.mapv(|xi: f64| (PI / 2. * xi).cos());
//! // Transform to physical space
//! let ch_vhat: Array1<f64> = ch.forward(&v, 0);
//! let cd_vhat: Array1<f64> = cd.forward(&v, 0);
//! // Send array to orthogonal space (cheb_dirichlet
//! // to chebyshev in this case)
//! let cd_vhat_ortho = cd.to_ortho(&cd_vhat, 0);
//! // Both arrays are equal, because field was
//! // initialized with correct boundary conditions,
//! // i.e. dirichlet ones
//! for (exp, ist) in ch_vhat.iter().zip(cd_vhat_ortho.iter()) {
//!     assert!((exp - ist).abs() < 1e-5);
//! }
//!
//! // However, if the physical field values do not
//! // satisfy dirichlet boundary conditions, they
//! // will be enforced by the transform to cheb_dirichle
//! // and ultimately the transformed values will deviate
//! // from a pure chebyshev transform (which does not)
//! // enfore the boundary conditions.
//! let mut v = x.mapv(|xi: f64| (PI / 2. * xi).sin());
//! let ch_vhat: Array1<f64> = ch.forward(&v, 0);
//! let cd_vhat: Array1<f64> = cd.forward(&v, 0);
//! let cd_vhat_ortho = cd.to_ortho(&cd_vhat, 0);
//! // They will deviate
//! println!("chebyshev     : {:?}", ch_vhat);
//! println!("cheb_dirichlet: {:?}", cd_vhat_ortho);
//! ```
//!
//! ## Multidimensional Spaces
//! A collection of bases forms a function space on which one can in turn define operations
//! along a specfic dimension (= axis). However, to transform a field from the physical space
//! into the spectral space, special attention must be paid to how the transformations are
//! concatenated  in a multidimensional space. Not all combinations are possible.
//! For example, `cheb_dirichlet` is a real-to-real transform,
//! while `fourier_r2c` defines a real-to-complex transform.
//! Thus, for a given real-valued physical field, the Chebyshev transform must precede
//! the fourier transform in the forward transform, and in reverse order in
//! the backward transform.
//!
//! **Note**: Currently `funspace` supports 1- 2- and 3 - dimensional spaces.
//!
//! ### Example
//! Apply transform from physical to spectral in a two-dimensional space
//! ```
//! use funspace::{fourier_r2c, cheb_dirichlet, Space2, BaseSpace};
//! use ndarray::prelude::*;
//! use std::f64::consts::PI;
//! use num_complex::Complex;
//! // Define the space and allocate arrays
//! let mut space = Space2::new(&fourier_r2c(5), &cheb_dirichlet(5));
//! let mut v: Array2<f64> = space.ndarray_physical();
//! // Set some field values
//! let x = space.coords_axis(0);
//! let y = space.coords_axis(1);
//! for (i,xi) in x.iter().enumerate() {
//!     for (j,yi) in y.iter().enumerate() {
//!         v[[i,j]] = xi.sin() * (PI/2.*yi).cos();
//!     }
//! }
//! // Transform forward (vhat is complex)
//! let mut vhat = space.forward(&v);
//! // Transform backward (v is real)
//! let v = space.backward(&vhat);
//! ```
//!
//! ## MPI Support
//! An mpi version of funspace can be found on
//! `https://github.com/preiter93/funspace-mpi`
//!
//! The mpi enabled version  will be merged
//! into this crate as soon as `rustmpi` releases
//! an updated version on crates.io
#![allow(clippy::just_underscores_and_digits)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_precision_loss)]
#[macro_use]
extern crate enum_dispatch;
mod macros;

pub mod chebyshev;
pub mod enums;
pub mod fourier;
pub mod space1;
pub mod space2;
pub mod space3;
pub mod space_traits;
pub mod traits;
pub mod types;
pub mod utils;
pub use crate::enums::{BaseAll, BaseC2c, BaseR2c, BaseR2r};
pub use crate::traits::BaseSize;
pub use crate::traits::Basics;
pub use crate::traits::Differentiate;
pub use crate::traits::DifferentiatePar;
pub use crate::traits::FromOrtho;
pub use crate::traits::FromOrthoPar;
pub use crate::traits::LaplacianInverse;
pub use crate::traits::Transform;
pub use crate::traits::TransformKind;
pub use crate::traits::TransformPar;
use chebyshev::Chebyshev;
use chebyshev::CompositeChebyshev;
use fourier::{FourierC2c, FourierR2c};
pub use space1::Space1;
pub use space2::Space2;
pub use space3::Space3;
pub use space_traits::BaseSpace;
pub use types::{FloatNum, Scalar};

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
pub fn chebyshev<A: FloatNum>(n: usize) -> BaseR2r<A> {
    BaseR2r::Chebyshev(Chebyshev::<A>::new(n))
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
pub fn cheb_dirichlet<A: FloatNum>(n: usize) -> BaseR2r<A> {
    BaseR2r::CompositeChebyshev(CompositeChebyshev::<A>::dirichlet(n))
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
pub fn cheb_neumann<A: FloatNum>(n: usize) -> BaseR2r<A> {
    BaseR2r::CompositeChebyshev(CompositeChebyshev::<A>::neumann(n))
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
pub fn cheb_dirichlet_bc<A: FloatNum>(n: usize) -> BaseR2r<A> {
    BaseR2r::CompositeChebyshev(CompositeChebyshev::<A>::dirichlet_bc(n))
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
pub fn cheb_neumann_bc<A: FloatNum>(n: usize) -> BaseR2r<A> {
    BaseR2r::CompositeChebyshev(CompositeChebyshev::<A>::neumann_bc(n))
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
/// use funspace::fourier_c2c;
/// use funspace::Transform;
/// use num_complex::Complex;
/// let mut fo = fourier_c2c::<f64>(10);
/// let real = ndarray::Array::linspace(0., 9., 10);
/// let mut y = real.mapv(|x| Complex::new(x,x));
/// let yhat = fo.forward(&mut y, 0);
/// ```
#[must_use]
pub fn fourier_c2c<A: FloatNum>(n: usize) -> BaseC2c<A> {
    BaseC2c::FourierC2c(FourierC2c::<A>::new(n))
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
/// use num_complex::Complex;
/// use ndarray::Array1;
/// let mut fo = fourier_r2c::<f64>(10);
/// let mut y = ndarray::Array::linspace(0., 9., 10);
/// let yhat: Array1<Complex<f64>> = fo.forward(&mut y, 0);
/// ```
#[must_use]
pub fn fourier_r2c<A: FloatNum>(n: usize) -> BaseR2c<A> {
    BaseR2c::FourierR2c(FourierR2c::<A>::new(n))
}
