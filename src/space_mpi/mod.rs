//! Feature: Mpi parallel spaces in two-dimensions
#![cfg(feature = "mpi")]
#![allow(clippy::module_name_repetitions)]
pub mod traits;
pub use pencil_decomp::simple_comms::{all_gather_sum, gather_sum, broadcast_scalar};
pub use mpi_crate::{
    collective::CommunicatorCollectives, environment::Universe, initialize, raw::AsRaw,
    topology::Communicator, traits::Equivalence,
};
pub use space2_mpi::Space2Mpi;
pub use traits::BaseSpaceMpi;
pub mod space2_mpi;
