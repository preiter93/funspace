//! Feature: Mpi parallel spaces in two-dimensions
#![cfg(feature = "mpi")]
pub mod decomp_handler;
pub mod space2;
pub mod space_traits;
pub use decomp_handler::DecompHandler;
pub use rsmpi_decomp::mpi::environment::Universe;
pub use rsmpi_decomp::mpi::initialize;
pub use rsmpi_decomp::mpi::topology::Communicator;
pub use rsmpi_decomp::mpi::traits::Equivalence;
pub use rsmpi_decomp::Decomp2d;
pub use space2::Space2;
pub use space_traits::BaseSpaceMpi;
