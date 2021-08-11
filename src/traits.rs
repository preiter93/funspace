//! Collection of usefull traits for function spaces
use crate::complex_to_complex::BaseC2c;
use crate::real_to_complex::BaseR2c;
use crate::real_to_real::BaseR2r;
use crate::Chebyshev;
use crate::CompositeChebyshev;
use crate::FloatNum;
use crate::FourierC2c;
use crate::FourierR2c;
use ndarray::prelude::*;

pub trait SuperBase<F, R, S>:
    BaseBasics<F>
    + Transform<R, S>
    + TransformPar<R, S>
    + FromOrtho<S>
    + FromOrthoPar<S>
    + Differentiate<S>
    + LaplacianInverse<F>
{
}

impl<T, F, R, S> SuperBase<F, R, S> for T where
    T: BaseBasics<F>
        + Transform<R, S>
        + TransformPar<R, S>
        + FromOrtho<S>
        + FromOrthoPar<S>
        + Differentiate<S>
        + LaplacianInverse<F>
{
}

/// Some basic  traits
#[enum_dispatch]
pub trait BaseBasics<T> {
    /// Coordinates in physical space
    fn coords(&self) -> &Array1<T>;
    /// Size in physical space
    fn len_phys(&self) -> usize;
    /// Size in spectral space
    fn len_spec(&self) -> usize;
    /// Return mass matrix
    fn mass(&self) -> Array2<T>;
    /// Return kind of transform
    fn get_transform_kind(&self) -> &TransformKind;
}

/// Transform from physical to spectral space and vice versa.
///
/// The associated types *Physical* and *Spectral* refer
/// to the scalar types in the respective space.
/// For example, a fourier transforms from real-to-complex,
/// while chebyshev from real-to-real.
pub trait Transform<T1, T2> {
    // /// Scalar type in physical space (before transform)
    type Physical;
    // /// Scalar type in spectral space (after transfrom)
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
    /// use funspace::Transform;
    /// use funspace::chebyshev::Chebyshev;
    /// use funspace::utils::approx_eq;
    /// use ndarray::prelude::*;
    /// let mut cheby = Chebyshev::new(4);
    /// let mut input = array![1., 2., 3., 4.];
    /// let output = cheby.forward(&mut input, 0);
    /// approx_eq(&output, &array![2.5, 1.33333333, 0. , 0.16666667]);
    /// ```
    fn forward<S, D>(&mut self, input: &mut ArrayBase<S, D>, axis: usize) -> Array<T2, D>
    where
        S: ndarray::Data<Elem = T1>,
        D: Dimension;

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
        S1: ndarray::Data<Elem = T1>,
        S2: ndarray::Data<Elem = T2> + ndarray::DataMut,
        D: Dimension;

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
    /// use funspace::Transform;
    /// use funspace::chebyshev::Chebyshev;
    /// use funspace::utils::approx_eq;
    /// use ndarray::prelude::*;
    /// let mut cheby = Chebyshev::new(4);
    /// let mut input = array![1., 2., 3., 4.];
    /// let output = cheby.backward(&mut input, 0);
    /// approx_eq(&output, &array![-2. ,  2.5, -3.5, 10.]);
    /// ```
    fn backward<S, D>(&mut self, input: &mut ArrayBase<S, D>, axis: usize) -> Array<T1, D>
    where
        S: ndarray::Data<Elem = T2>,
        D: Dimension;

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
        S1: ndarray::Data<Elem = T2>,
        S2: ndarray::Data<Elem = T1> + ndarray::DataMut,
        D: Dimension;
}

/// Transform from physical to spectral space and vice versa.
/// Parallel version of Transform, using Rayon,
///
/// The associated types *Physical* and *Spectral* refer
/// to the scalar types in the respective space.
/// For example, a fourier transforms from real-to-complex,
/// while chebyshev from real-to-real.
pub trait TransformPar<T1, T2> {
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
    /// use funspace::Transform;
    /// use funspace::chebyshev::Chebyshev;
    /// use funspace::utils::approx_eq;
    /// use ndarray::prelude::*;
    /// let mut cheby = Chebyshev::new(4);
    /// let mut input = array![1., 2., 3., 4.];
    /// let output = cheby.forward(&mut input, 0);
    /// approx_eq(&output, &array![2.5, 1.33333333, 0. , 0.16666667]);
    /// ```
    fn forward_par<S, D>(&mut self, input: &mut ArrayBase<S, D>, axis: usize) -> Array<T2, D>
    where
        S: ndarray::Data<Elem = T1>,
        D: Dimension;

    /// Transform from spectral to physical space
    ///
    /// Same as *backward*, but no output array must
    /// be supplied instead of being created.
    fn forward_inplace_par<S1, S2, D>(
        &mut self,
        input: &mut ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = T1>,
        S2: ndarray::Data<Elem = T2> + ndarray::DataMut,
        D: Dimension;

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
    /// use funspace::Transform;
    /// use funspace::chebyshev::Chebyshev;
    /// use funspace::utils::approx_eq;
    /// use ndarray::prelude::*;
    /// let mut cheby = Chebyshev::new(4);
    /// let mut input = array![1., 2., 3., 4.];
    /// let output = cheby.backward(&mut input, 0);
    /// approx_eq(&output, &array![-2. ,  2.5, -3.5, 10.]);
    /// ```
    fn backward_par<S, D>(&mut self, input: &mut ArrayBase<S, D>, axis: usize) -> Array<T1, D>
    where
        S: ndarray::Data<Elem = T2>,
        D: Dimension;

    /// Transform from spectral to physical space
    ///
    /// Same as *backward*, but no output array must
    /// be supplied instead of being created.
    fn backward_inplace_par<S1, S2, D>(
        &mut self,
        input: &mut ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = T2>,
        S2: ndarray::Data<Elem = T1> + ndarray::DataMut,
        D: Dimension;
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
#[enum_dispatch]
pub trait LaplacianInverse<T> {
    /// Laplacian $ L $
    fn laplace(&self) -> Array2<T>;

    /// Pseudoinverse mtrix of Laplacian $ L^{-1} $
    fn laplace_inv(&self) -> Array2<T>;

    /// Pseudoidentity matrix of laplacian $ L^{-1} L $
    fn laplace_inv_eye(&self) -> Array2<T>;
}

/// Define transformation from and to orthonormal space.
///
/// If the space is already the orthonormal (parent)
/// space, it copies and returns the input.
#[enum_dispatch]
pub trait FromOrtho<T> {
    /// Return coefficents in associated composite space
    ///
    /// ```
    /// use funspace::chebyshev::CompositeChebyshev;
    /// use ndarray::prelude::*;
    /// use funspace::utils::approx_eq;
    /// use funspace::FromOrtho;
    /// let (nx, ny) = (5, 4);
    /// let mut composite_coeff = Array2::<f64>::zeros((nx - 2, ny));
    /// for (i, v) in composite_coeff.iter_mut().enumerate() {
    ///     *v = i as f64;
    /// }
    /// let cd = CompositeChebyshev::<f64>::dirichlet(nx);
    ///
    /// let expected = array![
    ///     [0., 1., 2., 3.],
    ///     [4., 5., 6., 7.],
    ///     [8., 8., 8., 8.],
    ///     [-4., -5., -6., -7.],
    ///     [-8., -9., -10., -11.],
    /// ];
    /// let parent_coeff = cd.to_ortho(&composite_coeff, 0);
    /// approx_eq(&parent_coeff, &expected);
    /// ```
    fn to_ortho<S, D>(&self, input: &ArrayBase<S, D>, axis: usize) -> Array<T, D>
    where
        S: ndarray::Data<Elem = T>,
        D: Dimension;

    /// See *to_ortho*
    fn to_ortho_inplace<S1, S2, D>(
        &self,
        input: &ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = T>,
        S2: ndarray::Data<Elem = T> + ndarray::DataMut,
        D: Dimension;

    /// Return coefficents in associated composite space
    ///
    /// ```
    /// use funspace::chebyshev::CompositeChebyshev;
    /// use ndarray::prelude::*;
    /// use funspace::utils::approx_eq;
    /// use funspace::FromOrtho;
    /// let (nx, ny) = (5, 4);
    /// let mut parent_coeff = Array2::<f64>::zeros((nx, ny));
    /// for (i, v) in parent_coeff.iter_mut().enumerate() {
    ///     *v = i as f64;
    /// }
    /// let cd = CompositeChebyshev::<f64>::dirichlet(nx);
    ///
    /// let expected = array![
    ///     [-8., -8., -8., -8.],
    ///     [-4., -4., -4., -4.],
    ///     [-8., -8., -8., -8.],
    /// ];
    /// let composite_coeff = cd.from_ortho(&parent_coeff, 0);
    /// approx_eq(&composite_coeff, &expected);
    /// ```
    fn from_ortho<S, D>(&self, input: &ArrayBase<S, D>, axis: usize) -> Array<T, D>
    where
        S: ndarray::Data<Elem = T>,
        D: Dimension;

    /// See *fom_ortho*
    fn from_ortho_inplace<S1, S2, D>(
        &self,
        input: &ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = T>,
        S2: ndarray::Data<Elem = T> + ndarray::DataMut,
        D: Dimension;
}

/// Define transformation from and to orthonormal space.
/// (Parallel version which uses ndarrays `par_for_each` iterator)
#[enum_dispatch]
pub trait FromOrthoPar<T> {
    /// Parallel version of `to_ortho`
    fn to_ortho_par<S, D>(&self, input: &ArrayBase<S, D>, axis: usize) -> Array<T, D>
    where
        S: ndarray::Data<Elem = T>,
        D: Dimension;

    /// Parallel version of `to_ortho_inplace`
    fn to_ortho_inplace_par<S1, S2, D>(
        &self,
        input: &ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = T>,
        S2: ndarray::Data<Elem = T> + ndarray::DataMut,
        D: Dimension;

    /// Parallel version of `from_ortho`
    fn from_ortho_par<S, D>(&self, input: &ArrayBase<S, D>, axis: usize) -> Array<T, D>
    where
        S: ndarray::Data<Elem = T>,
        D: Dimension;

    /// Parallel version of `from_ortho_inplace`
    fn from_ortho_inplace_par<S1, S2, D>(
        &self,
        input: &ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = T>,
        S2: ndarray::Data<Elem = T> + ndarray::DataMut,
        D: Dimension;
}

/// Define which number format the
/// arrays have before and after
/// a transform (Type in phyical space
/// and type in spectral space)
#[derive(Clone)]
pub enum TransformKind {
    /// Real to real transform
    RealToReal,
    /// Complex to complex transform
    ComplexToComplex,
    /// Real to complex transform
    RealToComplex,
}

impl TransformKind {
    /// Return name of enum as str
    #[must_use]
    pub fn name(&self) -> &str {
        match *self {
            Self::RealToReal => "RealToReal",
            Self::ComplexToComplex => "ComplexToComplex",
            Self::RealToComplex => "RealToComplex",
        }
    }
}
