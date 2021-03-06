//! Feature: Mpi parallel spaces in two-dimensions
#![cfg(feature = "mpi")]
pub mod decomp2d;
pub mod decomp_handler;
pub mod traits;
pub use decomp2d::functions::all_gather_sum;
pub use decomp2d::functions::broadcast_scalar;
pub use decomp2d::functions::gather_sum;
pub use decomp2d::Decomp2d;
pub use decomp_handler::DecompHandler;
pub use mpi_crate::collective::CommunicatorCollectives;
pub use mpi_crate::environment::Universe;
pub use mpi_crate::initialize;
pub use mpi_crate::raw::AsRaw;
pub use mpi_crate::topology::Communicator;
pub use mpi_crate::traits::Equivalence;
pub use space2::Space2Mpi;
pub use traits::BaseSpaceMpi;
pub mod space2;
