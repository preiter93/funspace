use crate::chebyshev::{Chebyshev, ChebyshevComposite};
use crate::fourier::{FourierC2c, FourierR2c};
use crate::traits::{Differentiate, HasCoords, HasLength, HasType, ToOrtho, Transform};
use crate::types::{Real, Scalar, ScalarOperand};
use num_complex::Complex;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Clone)]
/// All bases who transform real-to-real
pub enum BaseR2r<T: Real> {
    /// Chebyshev polynomials (orthogonal)
    Chebyshev(Chebyshev<T>),
    /// Chebyshev polynomials (composite)
    ChebyshevComposite(ChebyshevComposite<T>),
}

#[derive(Clone)]
/// All bases who transform: real-to-complex
pub enum BaseR2c<T: Real> {
    /// Fourier polynomials
    FourierR2c(FourierR2c<T>),
}

#[derive(Clone)]
/// All bases who transform: complex-to-complex
pub enum BaseC2c<T: Real> {
    /// Fourier polynomials
    FourierC2c(FourierC2c<T>),
}

impl_funspace_elemental_for_base!(BaseR2r, A, A, Chebyshev, ChebyshevComposite);

impl_funspace_elemental_for_base!(BaseR2c, A, Complex<A>, FourierR2c);

impl_funspace_elemental_for_base!(BaseC2c, Complex<A>, Complex<A>, FourierC2c);

/// Set of transform kinds
#[derive(Debug, Clone)]
pub enum TransformKind {
    /// real-to-real
    R2r,
    /// real-to-complex
    R2c,
    /// complex-to-complex
    C2c,
}

/// Type of base
#[derive(Debug, Clone)]
pub enum BaseType {
    Orthogonal,
    Composite,
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
    /// Chebyshev biharmonic base A
    ChebBiHarmonicA,
    /// Chebyshev biharmonic base B
    ChebBiHarmonicB,
    /// Fourier real to complex
    FourierR2c,
    /// Fourier complex to complex
    FourierC2c,
}

// impl std::fmt::Display for BaseKind {
//     fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
//         match *self {
//             BaseKind::Chebyshev => write!(f, "Chebyhev"),
//             BaseKind::ChebDirichlet => write!(f, "ChebDirichlet"),
//             BaseKind::ChebNeumann => write!(f, "ChebNeumann"),
//             BaseKind::ChebDirichletNeumann => write!(f, "ChebDirichletNeumann"),
//             BaseKind::ChebBiHarmonicA => write!(f, "ChebBiHarmonicA"),
//             BaseKind::ChebBiHarmonicB => write!(f, "ChebBiHarmonicB"),
//             BaseKind::FourierR2c => write!(f, "FourierR2c"),
//             BaseKind::FourierC2c => write!(f, "FourierC2c"),
//         }
//     }
// }
