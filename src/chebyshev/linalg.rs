//! # Linalg methods for chebyshev base
pub(crate) mod pdma;
pub(crate) mod pdma2;
pub(crate) mod tdma;
pub(crate) use pdma::pdma;
pub(crate) use pdma2::pdma2;
pub(crate) use tdma::tdma;
