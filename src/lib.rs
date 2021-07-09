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
pub mod chebyshev;
pub mod utils;
pub use chebyshev::Chebyshev;
use ndarray::prelude::*;
use ndarray::ScalarOperand;
use num_traits::{Float, FromPrimitive, Signed, Zero};
use std::fmt::Debug;

/// Generic floating point number, implemented for f32 and f64
pub trait FloatNum:
    Copy + Zero + FromPrimitive + Signed + Sync + Send + Float + Debug + 'static + ScalarOperand
{
}
impl FloatNum for f32 {}
impl FloatNum for f64 {}

/// Returns function space for Chebyshev Polynomials $T_k$ (Orthogonal)
pub fn test() {
    todo!()
}

/// Defines size of basis
pub trait Size {
    /// Size in physical space
    fn len_phys(&self) -> usize;
    /// Size in spectral space
    fn len_spec(&self) -> usize;
}

/// Mass matrix expresses the connection (dot product)
/// of each basis of a funcion space.
///
/// Equals the identity matrix for orthonormal bases.
pub trait Mass<A> {
    /// Return mass matrix
    fn mass(&self) -> Array2<A>;
    /// Coordinates in physical space
    fn coords(&self) -> &Array1<A>;
}

/// Transform from physical to spectral space and vice versa.
///
/// The associated types *Physical* and *Spectral* refer
/// to the scalar types in the respective space.
/// For example, a fourier transforms from real-to-complex,
/// while chebyshev from real-to-real.
pub trait Transform {
    /// Scalar type in physical space (before transform)
    type Physical;
    /// Scalar type in spectral space (after transfrom)
    type Spectral;
    /// Transform physical -> spectral space along axis
    ///
    /// *input*: *n*-dimensional array of type Physical.
    /// Must be mutable, because
    /// some transform routines swap the axes back and
    /// forth, but it is effectively not altered.
    ///
    /// *axis*: Defines along which axis the array should be
    /// transformed.
    ///
    /// # Example
    /// Forward transform along first axis
    /// ```
    /// use funspace::{Chebyshev, Transform};
    /// use funspace::utils::approx_eq;
    /// use ndarray::prelude::*;
    /// let mut cheby = Chebyshev::new(4);
    /// let mut input = array![1., 2., 3., 4.];
    /// let output = cheby.forward(&mut input, 0);
    /// approx_eq(&output, &array![2.5, 1.33333333, 0. , 0.16666667]);
    /// ```
    fn forward<S, D>(
        &mut self,
        input: &mut ArrayBase<S, D>,
        axis: usize,
    ) -> Array<Self::Spectral, D>
    where
        S: ndarray::Data<Elem = Self::Physical>,
        D: Dimension + ndarray::RemoveAxis;

    /// Transform from spectral to physical space
    ///
    /// Same as *backward*, but no output array must
    /// be supplied instead of being created.
    fn forward_inplace<S1, S2, D>(
        &mut self,
        input: &mut ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = Self::Physical>,
        S2: ndarray::Data<Elem = Self::Spectral> + ndarray::DataMut,
        D: Dimension + ndarray::RemoveAxis;

    /// Transform spectral -> physical space along *axis*
    ///
    /// *input*: *n*-dimensional array of type Spectral.
    ///
    /// *axis*: Defines along which axis the array should be
    /// transformed.
    ///
    /// # Example
    /// Backward transform along first axis
    /// ```
    /// use funspace::{Chebyshev, Transform};
    /// use funspace::utils::approx_eq;
    /// use ndarray::prelude::*;
    /// let mut cheby = Chebyshev::new(4);
    /// let mut input = array![1., 2., 3., 4.];
    /// let output = cheby.backward(&mut input, 0);
    /// approx_eq(&output, &array![-2. ,  2.5, -3.5, 10.]);
    /// ```
    fn backward<S, D>(
        &mut self,
        input: &mut ArrayBase<S, D>,
        axis: usize,
    ) -> Array<Self::Physical, D>
    where
        S: ndarray::Data<Elem = Self::Spectral>,
        D: Dimension + ndarray::RemoveAxis;

    /// Transform from spectral to physical space
    ///
    /// Same as *backward*, but no output array must
    /// be supplied instead of being created.
    fn backward_inplace<S1, S2, D>(
        &mut self,
        input: &mut ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = Self::Spectral>,
        S2: ndarray::Data<Elem = Self::Physical> + ndarray::DataMut,
        D: Dimension + ndarray::RemoveAxis;
}

/// Perform differentiation in spectral space
pub trait Differentiate<T> {
    /// Return differentiated array
    fn differentiate<S, D>(
        &self,
        data: &ArrayBase<S, D>,
        n_times: usize,
        axis: usize,
    ) -> Array<T, D>
    where
        S: ndarray::Data<Elem = T>,
        D: Dimension;

    /// Differentiate on input array
    fn differentiate_inplace<S, D>(&self, data: &mut ArrayBase<S, D>, n_times: usize, axis: usize)
    where
        S: ndarray::Data<Elem = T> + ndarray::DataMut,
        D: Dimension;
}

/// Define (Pseudo-) Inverse of Laplacian
///
/// These operators are usefull when solving
/// second order equations
pub trait LaplacianInverse<T> {
    /// Pseudoinverse mtrix of Laplacian $ L^{-1} $
    fn laplace_inv(&self) -> Array2<T>;

    /// Pseudoidentity matrix of laplacian $ L^{-1} L $
    fn laplace_inv_eye(&self) -> Array2<T>;
}
