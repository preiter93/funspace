//! # Function spaces of type Fourier
//!
//! Defined on the intervall $[0, 2pi]$, the coefficients of the fourier
//! polynomials are of complex type.
//!
//! Fourier polynomials are usefull in problems with periodic.
//!
//! See [`c2c::Fourier`]
mod c2c;
mod r2c;
use crate::FloatNum;
pub use c2c::Fourier;
use ndrustfft::Complex;
