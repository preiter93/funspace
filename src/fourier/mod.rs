//! # Function spaces of type Fourier
//!
//! Defined on the intervall $[0, 2pi]$, the coefficients of the fourier
//! polynomials are of complex type.
//!
//! Fourier polynomials are usefull in problems with periodic.
//!
//! See [`complex::Fourier`]
mod complex;
use crate::FloatNum;
pub use complex::Fourier;
use ndrustfft::Complex;
