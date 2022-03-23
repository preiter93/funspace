//! # Function spaces of type Fourier
//!
//! Defined on the intervall $[0, 2pi]$, the coefficients of the fourier
//! polynomials are of complex type.
//!
//! Fourier polynomials are usefull in problems with periodic.
//!
//! Complex-to-complex: [`c2c::FourierC2c`]
//! Real-to-complex: [`r2c::FourierR2c`]
#![allow(clippy::module_name_repetitions)]
mod c2c;
mod r2c;
pub use c2c::FourierC2c;
pub use r2c::FourierR2c;
