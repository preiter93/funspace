//! # Funspace
//! <img align="right" src="https://rustacean.net/assets/cuddlyferris.png" width="80">
//! Collection of function spaces.
//!
//! # Bases
//!
//! ## Implemented bases:
//! - `Chebyshev` (Orthogonal), see [`chebyshev()`]
//! - `ChebDirichlet` (Composite), see [`cheb_dirichlet()`]
//! - `ChebNeumann` (Composite), see [`cheb_neumann()`]
//! - `ChebDirichletNeumann` (Composite), see [`cheb_dirichlet_neumann()`]
//! - `FourierC2c` (Orthogonal), see [`fourier_c2c()`]
//! - `FourierR2c` (Orthogonal), see [`fourier_r2c()`]
#[macro_use]
extern crate enum_dispatch;

pub mod chebyshev;
pub mod traits;
pub mod types;
pub mod utils;
