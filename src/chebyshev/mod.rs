#![allow(clippy::module_name_repetitions)]
pub mod compo;
pub mod ortho;
pub use compo::ChebyshevComposite;
pub use ortho::Chebyshev;
mod linalg;
mod stencils;
