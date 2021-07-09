//! # Function spaces of type Chebyshev
//!
//! Defined on the intervall $[-1, 1]$, the coefficients of the chebyshev
//! polynomials are of real (float) type.
//!
//! Chebyshev polynomials are for example usefull in problems with finite
//! domains and walls, for example wall bounded flows in fluid mechanics.
//!
//! See [`ortho::Chebyshev`]
mod composite;
mod composite_stencil;
mod linalg;
mod ortho;
use crate::Base;
use crate::FloatNum;
pub use composite::CompositeChebyshev;
pub use ortho::Chebyshev;

/// Function space for Chebyshev Polynomials
///
/// $$
/// T_k
/// $$
///
/// ```
/// use funspace::chebyshev;
/// let ch = chebyshev::<f64>(10);
/// ```
pub fn chebyshev<A: FloatNum>(n: usize) -> Base<A> {
    Base::Chebyshev(Chebyshev::<A>::new(n))
}

/// Function space with Dirichlet boundary conditions
///
/// $$
///  \phi_k = T_k - T_{k+2}
/// $$
/// ```
/// use funspace::cheb_dirichlet;
/// let cd = cheb_dirichlet::<f64>(10);
/// ```
pub fn cheb_dirichlet<A: FloatNum>(n: usize) -> Base<A> {
    Base::CompositeChebyshev(CompositeChebyshev::<A>::dirichlet(n))
}

// Function space with Neumann boundary conditions
///
/// $$
/// \phi_k = T_k - k^{2} \/ (k+2)^2 T_{k+2}
/// $$
/// ```
/// use funspace::cheb_neumann;
/// let cn = cheb_neumann::<f64>(10);
/// ```
pub fn cheb_neumann<A: FloatNum>(n: usize) -> Base<A> {
    Base::CompositeChebyshev(CompositeChebyshev::<A>::neumann(n))
}

/// Functions space for inhomogenoeus Dirichlet
/// boundary conditions
///
/// $$
///     \phi_0 = 0.5 T_0 - 0.5 T_1
/// $$
/// $$
///     \phi_1 = 0.5 T_0 + 0.5 T_1
/// $$
pub fn cheb_dirichlet_bc<A: FloatNum>(n: usize) -> Base<A> {
    Base::CompositeChebyshev(CompositeChebyshev::<A>::dirichlet_bc(n))
}

/// Functions space for inhomogenoeus Neumann
/// boundary conditions
///
/// $$
///     \phi_0 = 0.5T_0 - 1/8T_1
/// $$
/// $$
///     \phi_1 = 0.5T_0 + 1/8T_1
/// $$
pub fn cheb_neumann_bc<A: FloatNum>(n: usize) -> Base<A> {
    Base::CompositeChebyshev(CompositeChebyshev::<A>::neumann_bc(n))
}
