//! # Funspace
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
//! - Chebyshev (Orthogonal)
//! - ChebDirichlet (Composite)
//! - ChebNeumann (Composite)
#![allow(clippy::just_underscores_and_digits)]
#[macro_use]
extern crate enum_dispatch;
pub mod chebyshev;
mod traits;
pub mod utils;
use chebyshev::Chebyshev;
use chebyshev::CompositeChebyshev;
pub use chebyshev::{cheb_dirichlet, cheb_neumann, chebyshev};
use ndarray::prelude::*;
use ndarray::ScalarOperand;
use num_traits::{Float, FromPrimitive, Signed, Zero};
use std::fmt::Debug;
pub use traits::{Differentiate, LaplacianInverse, Mass, Size, Transform};

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
/// Differentiate, Mass, LaplacianInverse, Size, (Transform)
///
/// # Example
/// Apply diferentioation in ChebDirichlet space
/// ```
/// use funspace::{cheb_dirichlet};
/// use funspace::Differentiate;
/// use ndarray::prelude::*;
/// let cd = cheb_dirichlet::<f64>(5);
/// let input = array![1., 2., 3.,];
/// let output = cd.differentiate(&input, 2, 0);
/// ```
#[allow(clippy::large_enum_variant)]
#[enum_dispatch(Differentiate<T>, Mass<T>, LaplacianInverse<T>, Size)]
pub enum Base<T: FloatNum> {
    Chebyshev(Chebyshev<T>),
    CompositeChebyshev(CompositeChebyshev<T>),
}

/// Implement transform trait per hand, can't be enum_dispatched
/// because of associated types.
impl<A: FloatNum + std::ops::MulAssign> Transform for Base<A> {
    type Physical = A;
    type Spectral = A;

    fn forward<S, D>(
        &mut self,
        input: &mut ArrayBase<S, D>,
        axis: usize,
    ) -> Array<Self::Spectral, D>
    where
        S: ndarray::Data<Elem = Self::Physical>,
        D: Dimension + ndarray::RemoveAxis,
    {
        match self {
            Self::Chebyshev(ref mut b) => b.forward(input, axis),
            Self::CompositeChebyshev(ref mut b) => b.forward(input, axis),
        }
    }

    fn forward_inplace<S1, S2, D>(
        &mut self,
        input: &mut ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = Self::Physical>,
        S2: ndarray::Data<Elem = Self::Spectral> + ndarray::DataMut,
        D: Dimension + ndarray::RemoveAxis,
    {
        match self {
            Self::Chebyshev(ref mut b) => b.forward_inplace(input, output, axis),
            Self::CompositeChebyshev(ref mut b) => b.forward_inplace(input, output, axis),
        }
    }

    fn backward<S, D>(
        &mut self,
        input: &mut ArrayBase<S, D>,
        axis: usize,
    ) -> Array<Self::Physical, D>
    where
        S: ndarray::Data<Elem = Self::Spectral>,
        D: Dimension + ndarray::RemoveAxis,
    {
        match self {
            Self::Chebyshev(ref mut b) => b.backward(input, axis),
            Self::CompositeChebyshev(ref mut b) => b.backward(input, axis),
        }
    }

    fn backward_inplace<S1, S2, D>(
        &mut self,
        input: &mut ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = Self::Spectral>,
        S2: ndarray::Data<Elem = Self::Physical> + ndarray::DataMut,
        D: Dimension + ndarray::RemoveAxis,
    {
        match self {
            Self::Chebyshev(ref mut b) => b.backward_inplace(input, output, axis),
            Self::CompositeChebyshev(ref mut b) => b.backward_inplace(input, output, axis),
        }
    }
}
