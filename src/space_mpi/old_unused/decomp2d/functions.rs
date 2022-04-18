//! Collection of simplified mpi routines
use mpi_crate::collective::CommunicatorCollectives;
use mpi_crate::collective::Root;
use mpi_crate::environment::Universe;
use mpi_crate::topology::Communicator;
use mpi_crate::traits::Equivalence;
use num_traits::Zero;

/// Broadcast scalar value from root to all processes
pub fn broadcast_scalar<T: Zero + Equivalence>(universe: &Universe, data: &mut T) {
    let world = universe.world();
    let root_rank = 0;
    let root_process = world.process_at_rank(root_rank);
    root_process.broadcast_into(data);
}

/// Gather sum of values on root
pub fn gather_sum<T: Zero + Equivalence + Clone + Copy + std::iter::Sum>(
    universe: &Universe,
    data: &T,
    result: &mut T,
) {
    let world = universe.world();
    let size = world.size() as usize;
    let root_rank = 0;
    let root_process = world.process_at_rank(root_rank);
    if world.rank() == root_rank {
        let mut a = vec![T::zero(); size];
        root_process.gather_into_root(data, &mut a[..]);
        *result = a.iter().copied().sum();
    } else {
        root_process.gather_into(data);
    }
}

/// Gather sum of values on all processes
pub fn all_gather_sum<T: Zero + Equivalence + Clone + Copy + std::iter::Sum>(
    universe: &Universe,
    data: &T,
    result: &mut T,
) {
    let world = universe.world();
    let size = world.size() as usize;
    let mut a = vec![T::zero(); size];
    world.all_gather_into(data, &mut a[..]);
    *result = a.iter().copied().sum();
}
