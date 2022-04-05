use crate::chebyshev::{Chebyshev, ChebyshevComposite};
use crate::fourier::{FourierC2c, FourierR2c};
use crate::traits::{
    BaseElements, BaseFromOrtho, BaseGradient, BaseMatOpGeneral, BaseMatOpLaplacian, BaseSize,
    BaseTransform,
};
use crate::types::{FloatNum, ScalarNum};
use ndarray::Array2;
use num_complex::Complex;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Clone)]
/// All bases who transform real-to-real
pub enum BaseR2r<T: FloatNum> {
    /// Chebyshev polynomials (orthogonal)
    Chebyshev(Chebyshev<T>),
    /// Chebyshev polynomials (composite)
    ChebyshevComposite(ChebyshevComposite<T>),
}

#[derive(Clone)]
/// All bases who transform: real-to-complex
pub enum BaseR2c<T: FloatNum> {
    /// Fourier polynomials
    FourierR2c(FourierR2c<T>),
}

#[derive(Clone)]
/// All bases who transform: complex-to-complex
pub enum BaseC2c<T: FloatNum> {
    /// Fourier polynomials
    FourierC2c(FourierC2c<T>),
}

impl_funspace_elemental_for_base!(BaseR2r, A, A, Chebyshev, ChebyshevComposite);

impl_funspace_elemental_for_base!(BaseR2c, A, Complex<A>, FourierR2c);

impl_funspace_elemental_for_base!(BaseC2c, Complex<A>, Complex<A>, FourierC2c);

/// Set of transform kinds
pub enum TransformKind {
    /// real-to-real
    R2r,
    /// real-to-complex
    R2c,
    /// complex-to-complex
    C2c,
}

/// All available function spaces
#[derive(Debug, Clone, Copy)]
pub enum BaseKind {
    /// Chebyshev orthogonal base
    Chebyshev,
    /// Chebyshev dirichlet base
    ChebDirichlet,
    /// Chebyshev neumann base
    ChebNeumann,
    /// Chebyshev dirichlet - neumann base
    ChebDirichletNeumann,
    /// Chebyshev biharmonic base
    ChebBiHarmonic,
    /// Fourier real to complex
    FourierR2c,
    /// Fourier complex to complex
    FourierC2c,
}

impl std::fmt::Display for BaseKind {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            BaseKind::Chebyshev => write!(f, "Chebyhev"),
            BaseKind::ChebDirichlet => write!(f, "ChebDirichlet"),
            BaseKind::ChebNeumann => write!(f, "ChebNeumann"),
            BaseKind::ChebDirichletNeumann => write!(f, "ChebDirichletNeumann"),
            BaseKind::ChebBiHarmonic => write!(f, "ChebBiHarmonic"),
            BaseKind::FourierR2c => write!(f, "FourierR2c"),
            BaseKind::FourierC2c => write!(f, "FourierC2c"),
        }
    }
}
