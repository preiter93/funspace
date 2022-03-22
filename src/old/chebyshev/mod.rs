//! # Function spaces of type Chebyshev
//!
//! Defined on the intervall $[-1, 1]$, the coefficients of the chebyshev
//! polynomials are of real (float) type.
//!
//! Chebyshev polynomials are for example usefull in problems with finite
//! domains and walls, for example wall bounded flows in fluid mechanics.
//!
//! See [`ortho::Chebyshev`]
#![allow(clippy::module_name_repetitions)]
mod composite;
mod composite_stencil;
mod dmsuite;
mod linalg;
mod ortho;
pub use composite::CompositeChebyshev;
pub use ortho::Chebyshev;
