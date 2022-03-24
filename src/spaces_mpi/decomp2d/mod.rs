//! # `rsmpi-decomp`: Domain decomposition for mpi data distribution
//!
//! This library is work in progress...
//!
//! # Examples
//! See `examples/`
//!
//! # Supported:
//! - *2D domain*: Split 1 dimension.
//!
//!
//! ## x-pencil domain:
//! ```ignore
//! --------------
//! |     P1     |
//! --------------
//! |     P0     |
//! --------------
//! ```
//! ## y-pencil domain:
//! ```ignore
//! ---------------
//! |      |      |
//! |  P0  |  P1  |
//! |      |      |
//! ---------------
//! ```
//!
//! # `cargo-mpirun`
//! Install:
//! ```ignore
//! cargo install cargo-mpirun
//! ```
//! Run:
//! ```ignore
//! cargo mpirun --np 2 --example gather_root
//! ```
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::similar_names)]
//pub extern crate mpi;
pub mod distribute;
pub mod functions;
pub use distribute::Distribution;
use mpi_crate::collective::CommunicatorCollectives;
use mpi_crate::collective::Root;
use mpi_crate::datatype::{Partition, PartitionMut};
use mpi_crate::environment::Universe;
use mpi_crate::topology::Communicator;
use mpi_crate::topology::SystemCommunicator;
use mpi_crate::traits::Equivalence;
use ndarray::{Array2, ArrayBase, Data, DataMut, Dim, Ix2};
use num_traits::Zero;
use std::fmt::Debug;

/// Decomposition for 2D domain
#[derive(Debug, Clone)]
pub struct Decomp2d<'a> {
    // Mpi universe
    pub universe: &'a Universe,
    // Mpi world
    pub world: SystemCommunicator,
    // Number of processors
    pub nprocs: i32,
    // Processor id
    pub nrank: i32,
    // Total number of points
    pub n_global: [usize; 2],
    // Size, indices, counts and displacements for x-pencil
    pub x_pencil: Distribution<2>,
    // Size, indices, counts and displacements for y-pencil
    pub y_pencil: Distribution<2>,
}

impl<'a> Decomp2d<'a> {
    /// Initialize Decomp

    /// # Arguments
    /// * `universe`: `mpi::initialize().unwrap()`
    /// * `n_global`: Total number of grid points [nx global, ny global]
    #[must_use]
    pub fn new(universe: &'a Universe, n_global: [usize; 2]) -> Self {
        let world = universe.world();
        let nprocs = world.size();
        let nrank = world.rank();
        let x_pencil = Distribution::new(n_global, nprocs as usize, nrank as usize, 0);
        let y_pencil = Distribution::new(n_global, nprocs as usize, nrank as usize, 1);
        Self {
            universe,
            world,
            nprocs,
            nrank,
            n_global,
            x_pencil,
            y_pencil,
        }
    }

    /// Return shape of global array
    #[must_use]
    pub fn get_global_shape(&self) -> [usize; 2] {
        self.n_global
    }

    /// Return shape of x-pencil
    #[must_use]
    pub fn get_x_pencil_shape(&self) -> [usize; 2] {
        self.x_pencil.sz
    }

    /// Return shape of y-pencil
    #[must_use]
    pub fn get_y_pencil_shape(&self) -> [usize; 2] {
        self.y_pencil.sz
    }

    /// Distribute global data array to a x-pencil distribution.
    /// Each processors holds a sub-domain of the global array,
    /// where x is contiguous and y is broken
    #[must_use]
    pub fn split_array_x_pencil<S, T>(&self, data: &ArrayBase<S, Ix2>) -> Array2<T>
    where
        S: Data<Elem = T>,
        T: Copy,
    {
        self.x_pencil.split_array(data)
    }

    /// Distribute global data array to a y-pencil distribution.
    /// Each processors holds a sub-domain of the global array,
    /// where y is contiguous and x is broken
    #[must_use]
    pub fn split_array_y_pencil<S, T>(&self, data: &ArrayBase<S, Ix2>) -> Array2<T>
    where
        S: Data<Elem = T>,
        T: Copy,
    {
        self.y_pencil.split_array(data)
    }

    /// Transpose from x to y pencil
    pub fn transpose_x_to_y<S1, S2, T>(
        &self,
        snd: &ArrayBase<S1, Ix2>,
        rcv: &mut ArrayBase<S2, Ix2>,
    ) where
        S1: Data<Elem = T>,
        S2: DataMut<Elem = T>,
        T: Zero + Clone + Copy + Equivalence,
    {
        check_shape(snd, self.x_pencil.sz);
        check_shape(rcv, self.y_pencil.sz);

        // send & receive buffer
        let mut send = vec![T::zero(); self.x_pencil.length()];
        let mut recv = vec![T::zero(); self.y_pencil.length()];
        Self::split_xy(snd, &mut send);

        let (send_counts, send_displs) = self.x_pencil.get_counts_all_to_all(&self.y_pencil);
        let (recv_counts, recv_displs) = self.y_pencil.get_counts_all_to_all(&self.x_pencil);
        {
            let send_buffer = Partition::new(&send[..], &send_counts[..], &send_displs[..]);
            let mut recv_buffer =
                PartitionMut::new(&mut recv[..], &recv_counts[..], &recv_displs[..]);
            self.world
                .all_to_all_varcount_into(&send_buffer, &mut recv_buffer);
        }

        // copy receive buffer into array
        // *todo*: prevent copy by referencing?
        self.merge_xy(rcv, &recv);
    }

    /// Prepare send buffer for `transpose_x_to_y`
    fn split_xy<S, T>(data: &ArrayBase<S, Ix2>, buf: &mut [T])
    where
        S: Data<Elem = T>,
        T: Copy,
    {
        for (d, b) in data.iter().zip(buf.iter_mut()) {
            *b = *d;
        }
    }

    /// Redistribute recv buffer for `transpose_x_to_y`
    fn merge_xy<S, T>(&self, data: &mut ArrayBase<S, Ix2>, buf: &[T])
    where
        S: DataMut<Elem = T>,
        T: Copy,
    {
        let mut pos = 0;
        for proc in 0..self.nprocs as usize {
            let j1 = self.x_pencil.st_procs[proc][1];
            let j2 = self.x_pencil.en_procs[proc][1];
            for i in 0..self.y_pencil.sz[0] {
                for j in j1..=j2 {
                    data[[i, j]] = buf[pos];
                    pos += 1;
                }
            }
        }
    }

    /// Transpose from y to x pencil
    pub fn transpose_y_to_x<S1, S2, T>(
        &self,
        snd: &ArrayBase<S1, Ix2>,
        rcv: &mut ArrayBase<S2, Ix2>,
    ) where
        S1: Data<Elem = T>,
        S2: DataMut<Elem = T>,
        T: Zero + Clone + Copy + Equivalence,
    {
        check_shape(snd, self.y_pencil.sz);
        check_shape(rcv, self.x_pencil.sz);

        // send & receive buffer
        let mut send = vec![T::zero(); self.y_pencil.length()];
        let mut recv = vec![T::zero(); self.x_pencil.length()];
        Self::split_yx(snd, &mut send);

        let (send_counts, send_displs) = self.y_pencil.get_counts_all_to_all(&self.x_pencil);
        let (recv_counts, recv_displs) = self.x_pencil.get_counts_all_to_all(&self.y_pencil);
        {
            let send_buffer = Partition::new(&send[..], &send_counts[..], &send_displs[..]);
            let mut recv_buffer =
                PartitionMut::new(&mut recv[..], &recv_counts[..], &recv_displs[..]);
            self.world
                .all_to_all_varcount_into(&send_buffer, &mut recv_buffer);
        }

        // copy receive buffer into array
        // *todo*: prevent copy by referencing?
        self.merge_yx(rcv, &recv);
    }

    /// Prepare send buffer for `transpose_y_to_x`
    fn split_yx<S, T>(data: &ArrayBase<S, Ix2>, buf: &mut [T])
    where
        S: Data<Elem = T>,
        T: Copy,
    {
        let mut data_view = data.view();
        data_view.swap_axes(0, 1);
        for (d, b) in data_view.iter().zip(buf.iter_mut()) {
            *b = *d;
        }
    }

    /// Redistribute recv buffer for `transpose_y_to_x`
    fn merge_yx<S, T>(&self, data: &mut ArrayBase<S, Ix2>, buf: &[T])
    where
        S: DataMut<Elem = T>,
        T: Copy,
    {
        let mut pos = 0;
        for proc in 0..self.nprocs as usize {
            let i1 = self.y_pencil.st_procs[proc][0];
            let i2 = self.y_pencil.en_procs[proc][0];
            for j in 0..self.x_pencil.sz[1] {
                for i in i1..=i2 {
                    data[[i, j]] = buf[pos];
                    pos += 1;
                }
            }
        }
    }

    /// Gather data from x-pencil
    ///
    /// Call this routine from non-root,
    /// and call `gather_x_root` or `gather_x_inplace_root`
    /// from root
    ///
    /// # Panics
    /// If processor rank is root
    pub fn gather_x<S1, T>(&self, snd: &ArrayBase<S1, Ix2>)
    where
        S1: Data<Elem = T>,
        T: Zero + Clone + Copy + Equivalence,
    {
        let root_rank = 0;
        let root_process = self.world.process_at_rank(root_rank);
        // send buffer
        check_shape(snd, self.x_pencil.sz);
        let mut send = vec![T::zero(); self.x_pencil.length()];
        let mut snd_view = snd.view();
        snd_view.swap_axes(0, 1);
        for (s, m) in send.iter_mut().zip(snd_view.iter()) {
            *s = *m;
        }
        // gather
        if self.nrank == root_rank {
            panic!("Rank must not be root!");
        } else {
            root_process.gather_varcount_into(&send[..]);
        }
    }

    /// Gather data from x-pencil
    ///
    /// Call this routine from root
    ///
    /// # Panics
    /// If processor rank is not root
    pub fn gather_x_root<S1, T>(&self, snd: &ArrayBase<S1, Ix2>) -> Array2<T>
    where
        S1: Data<Elem = T>,
        T: Zero + Clone + Copy + Equivalence,
    {
        // call `gather_x_inplace_root`
        let root_rank = 0;
        if self.nrank == root_rank {
            let mut recv = Array2::<T>::zeros(self.n_global);
            self.gather_x_inplace_root(snd, &mut recv);
            recv
        } else {
            panic!("Rank must be root!");
        }
    }

    /// Gather data from x-pencil to root processor
    ///
    /// Call this routine from root
    ///
    /// # Panics
    /// If processor rank is not root
    pub fn gather_x_inplace_root<S1, S2, T>(
        &self,
        snd: &ArrayBase<S1, Ix2>,
        rcv: &mut ArrayBase<S2, Ix2>,
    ) where
        S1: Data<Elem = T>,
        S2: DataMut<Elem = T>,
        T: Zero + Clone + Copy + Equivalence,
    {
        let root_rank = 0;
        let root_process = self.world.process_at_rank(root_rank);

        // send buffer
        check_shape(snd, self.x_pencil.sz);
        let mut send = vec![T::zero(); self.x_pencil.length()];
        let mut snd_view = snd.view();
        snd_view.swap_axes(0, 1);
        for (s, m) in send.iter_mut().zip(snd_view.iter()) {
            *s = *m;
        }
        // gather on root
        if self.nrank == root_rank {
            let recv_length = self.n_global.iter().product::<usize>();
            let mut recv = vec![T::zero(); recv_length];

            let (counts, displs) = self.x_pencil.get_counts_all_gather();
            {
                let mut partition = PartitionMut::new(&mut recv[..], &counts[..], &displs[..]);
                root_process.gather_varcount_into_root(&send[..], &mut partition);
            }
            // copy receive buffer into receiving array
            // *todo*: prevent copy by referencing?
            check_shape(rcv, self.n_global);
            rcv.swap_axes(0, 1);
            for (s, m) in rcv.iter_mut().zip(recv.iter()) {
                *s = *m;
            }
            rcv.swap_axes(0, 1);
        } else {
            panic!("Rank must be root!");
        }
    }

    /// Gather data from y-pencil
    ///
    /// Call this routine from non-root,
    /// and call `gather_y_root` or `gather_y_inplace_root`
    /// from root
    ///
    /// # Panics
    /// If processor rank is root
    pub fn gather_y<S1, T>(&self, snd: &ArrayBase<S1, Ix2>)
    where
        S1: Data<Elem = T>,
        T: Zero + Clone + Copy + Equivalence,
    {
        let root_rank = 0;
        let root_process = self.world.process_at_rank(root_rank);
        // send buffer
        check_shape(snd, self.y_pencil.sz);
        let mut send = vec![T::zero(); self.y_pencil.length()];
        for (s, m) in send.iter_mut().zip(snd.iter()) {
            *s = *m;
        }
        // gather
        if self.nrank == root_rank {
            panic!("Rank must not be root!");
        } else {
            root_process.gather_varcount_into(&send[..]);
        }
    }

    /// Gather data from y-pencil
    ///
    /// Call this routine from root
    ///
    /// # Panics
    /// If processor rank is not root
    #[must_use]
    pub fn gather_y_root<S1, T>(&self, snd: &ArrayBase<S1, Ix2>) -> Array2<T>
    where
        S1: Data<Elem = T>,
        T: Zero + Clone + Copy + Equivalence,
    {
        // call `gather_x_inplace_root`
        let root_rank = 0;
        if self.nrank == root_rank {
            let mut recv = Array2::<T>::zeros(self.n_global);
            self.gather_y_inplace_root(snd, &mut recv);
            recv
        } else {
            panic!("Rank must be root!");
        }
    }

    /// Gather data from y-pencil to root processor
    ///
    /// Call this routine from root
    ///
    /// # Panics
    /// If processor rank is not root
    pub fn gather_y_inplace_root<S1, S2, T>(
        &self,
        snd: &ArrayBase<S1, Ix2>,
        rcv: &mut ArrayBase<S2, Ix2>,
    ) where
        S1: Data<Elem = T>,
        S2: DataMut<Elem = T>,
        T: Zero + Clone + Copy + Equivalence,
    {
        let root_rank = 0;
        let root_process = self.world.process_at_rank(root_rank);
        // send buffer
        check_shape(snd, self.y_pencil.sz);
        let mut send = vec![T::zero(); self.y_pencil.length()];
        for (s, m) in send.iter_mut().zip(snd.iter()) {
            *s = *m;
        }
        // gather on root
        if self.nrank == root_rank {
            let recv_length = self.n_global.iter().product::<usize>();
            let mut recv = vec![T::zero(); recv_length];

            let (counts, displs) = self.y_pencil.get_counts_all_gather();
            {
                let mut partition = PartitionMut::new(&mut recv[..], &counts[..], &displs[..]);
                root_process.gather_varcount_into_root(&send[..], &mut partition);
            }
            // copy receive buffer into receiving array
            // *todo*: prevent copy by referencing?
            check_shape(rcv, self.n_global);
            for (s, m) in rcv.iter_mut().zip(recv.iter()) {
                *s = *m;
            }
        } else {
            panic!("Rank must be root!");
        }
    }

    /// Scatter data from x-pencil
    ///
    /// Call this routine from non-root
    /// and call `scatter_x_root` or `scatter_x_inplace_root`
    /// from root
    ///
    /// # Panics
    /// If processor rank is root
    #[must_use]
    pub fn scatter_x<T>(&self) -> Array2<T>
    where
        T: Zero + Clone + Copy + Equivalence,
    {
        let root_rank = 0;
        if self.nrank == root_rank {
            panic!("Rank must not be root!");
        } else {
            let mut recv = Array2::<T>::zeros(self.x_pencil.sz);
            self.scatter_x_inplace(&mut recv);
            recv
        }
    }

    /// Scatter data from x-pencil
    ///
    /// Call this routine from non-root,
    /// and call `scatter_x_root` or `scatter_x_inplace_root`
    /// from root
    ///
    /// # Panics
    /// If processor rank is root
    pub fn scatter_x_inplace<S2, T>(&self, rcv: &mut ArrayBase<S2, Ix2>)
    where
        S2: DataMut<Elem = T>,
        T: Zero + Clone + Copy + Equivalence,
    {
        check_shape(rcv, self.x_pencil.sz);

        let root_rank = 0;
        let root_process = self.world.process_at_rank(root_rank);

        // recv buffer
        let mut recv = vec![T::zero(); self.x_pencil.length()];

        if self.nrank == root_rank {
            panic!("Rank must not be root!");
        } else {
            root_process.scatter_varcount_into(&mut recv[..]);
        }

        // copy receive buffer into array
        // *todo*: prevent copy by referencing?
        rcv.swap_axes(0, 1);
        for (s, m) in rcv.iter_mut().zip(recv.iter()) {
            *s = *m;
        }
        rcv.swap_axes(0, 1);
    }

    /// Scatter data from x-pencil
    ///
    /// Call this routine from root,
    /// and call `scatter_x_inplace` or `scatter_x`
    /// from non root
    ///
    /// # Panics
    /// If processor rank is not root
    #[must_use]
    pub fn scatter_x_root<S1, T>(&self, snd: &ArrayBase<S1, Ix2>) -> Array2<T>
    where
        S1: Data<Elem = T>,
        T: Zero + Clone + Copy + Equivalence,
    {
        let root_rank = 0;
        if self.nrank == root_rank {
            let mut recv = Array2::<T>::zeros(self.x_pencil.sz);
            self.scatter_x_inplace_root(snd, &mut recv);
            recv
        } else {
            panic!("Rank must be root!");
        }
    }

    /// Scatter data from x-pencil
    ///
    /// Call this routine from root,
    /// and call `scatter_x_inplace` or `scatter_x`
    /// from non root
    ///
    /// # Panics
    /// If processor rank is not root
    pub fn scatter_x_inplace_root<S1, S2, T>(
        &self,
        snd: &ArrayBase<S1, Ix2>,
        rcv: &mut ArrayBase<S2, Ix2>,
    ) where
        S1: Data<Elem = T>,
        S2: DataMut<Elem = T>,
        T: Zero + Clone + Copy + Equivalence,
    {
        check_shape(snd, self.n_global);
        check_shape(rcv, self.x_pencil.sz);

        let root_rank = 0;
        let root_process = self.world.process_at_rank(root_rank);

        // recv buffer
        let mut recv = vec![T::zero(); self.x_pencil.length()];

        if self.nrank == root_rank {
            // send buffer
            let sendv_length = self.n_global.iter().product::<usize>();
            let mut send = vec![T::zero(); sendv_length];
            let mut snd_view = snd.view();
            snd_view.swap_axes(0, 1);
            for (s, m) in send.iter_mut().zip(snd_view.iter()) {
                *s = *m;
            }
            let (counts, displs) = self.x_pencil.get_counts_all_gather();
            {
                let partition = Partition::new(&send[..], &counts[..], &displs[..]);
                root_process.scatter_varcount_into_root(&partition, &mut recv[..]);
            }
        } else {
            panic!("Rank must be root!");
        }

        // copy receive buffer into array
        // *todo*: prevent copy by referencing?
        rcv.swap_axes(0, 1);
        for (s, m) in rcv.iter_mut().zip(recv.iter()) {
            *s = *m;
        }
        rcv.swap_axes(0, 1);
    }

    /// Scatter data from y-pencil
    ///
    /// Call this routine from non-root
    /// and call `scatter_y_root` or `scatter_y_inplace_root`
    /// from root
    ///
    /// # Panics
    /// If processor rank is root
    #[must_use]
    pub fn scatter_y<T>(&self) -> Array2<T>
    where
        T: Zero + Clone + Copy + Equivalence,
    {
        let root_rank = 0;
        if self.nrank == root_rank {
            panic!("Rank must not be root!");
        } else {
            let mut recv = Array2::<T>::zeros(self.y_pencil.sz);
            self.scatter_y_inplace(&mut recv);
            recv
        }
    }

    /// Scatter data from y-pencil
    ///
    /// Call this routine from non-root,
    /// and call `scatter_y_root` or `scatter_y_inplace_root`
    /// from root
    ///
    /// # Panics
    /// If processor rank is root
    pub fn scatter_y_inplace<S2, T>(&self, rcv: &mut ArrayBase<S2, Ix2>)
    where
        S2: DataMut<Elem = T>,
        T: Zero + Clone + Copy + Equivalence,
    {
        check_shape(rcv, self.y_pencil.sz);

        let root_rank = 0;
        let root_process = self.world.process_at_rank(root_rank);

        // recv buffer
        let mut recv = vec![T::zero(); self.y_pencil.length()];

        if self.nrank == root_rank {
            panic!("Rank must not be root!");
        } else {
            root_process.scatter_varcount_into(&mut recv[..]);
        }
        // copy receive buffer into array
        // *todo*: prevent copy by referencing?
        for (s, m) in rcv.iter_mut().zip(recv.iter()) {
            *s = *m;
        }
    }

    /// Scatter data from y-pencil
    ///
    /// Call this routine from root,
    /// and call `scatter_y_inplace` or `scatter_x`
    /// from non root
    ///
    /// # Panics
    /// If processor rank is not root
    pub fn scatter_y_root<S1, T>(&self, snd: &ArrayBase<S1, Ix2>) -> Array2<T>
    where
        S1: Data<Elem = T>,
        T: Zero + Clone + Copy + Equivalence,
    {
        let root_rank = 0;
        if self.nrank == root_rank {
            let mut recv = Array2::<T>::zeros(self.y_pencil.sz);
            self.scatter_y_inplace_root(snd, &mut recv);
            recv
        } else {
            panic!("Rank must be root!");
        }
    }

    /// Scatter data from y-pencil
    ///
    /// Call this routine from root,
    /// and call `scatter_y_inplace` or `scatter_x`
    /// from non root
    ///
    /// # Panics
    /// If processor rank is not root
    pub fn scatter_y_inplace_root<S1, S2, T>(
        &self,
        snd: &ArrayBase<S1, Ix2>,
        rcv: &mut ArrayBase<S2, Ix2>,
    ) where
        S1: Data<Elem = T>,
        S2: DataMut<Elem = T>,
        T: Zero + Clone + Copy + Equivalence,
    {
        check_shape(snd, self.n_global);
        check_shape(rcv, self.y_pencil.sz);

        let root_rank = 0;
        let root_process = self.world.process_at_rank(root_rank);

        // recv buffer
        let mut recv = vec![T::zero(); self.y_pencil.length()];

        if self.nrank == root_rank {
            // send buffer
            let sendv_length = self.n_global.iter().product::<usize>();
            let mut send = vec![T::zero(); sendv_length];
            for (s, m) in send.iter_mut().zip(snd.iter()) {
                *s = *m;
            }
            let (counts, displs) = self.y_pencil.get_counts_all_gather();
            {
                let partition = Partition::new(&send[..], &counts[..], &displs[..]);
                root_process.scatter_varcount_into_root(&partition, &mut recv[..]);
            }
        } else {
            panic!("Rank must be root!");
        }

        // copy receive buffer into array
        // *todo*: prevent copy by referencing?
        for (s, m) in rcv.iter_mut().zip(recv.iter()) {
            *s = *m;
        }
    }

    /// Gather data from x-pencil to all participating processors
    pub fn all_gather_x<S1, S2, T>(&self, snd: &ArrayBase<S1, Ix2>, rcv: &mut ArrayBase<S2, Ix2>)
    where
        S1: Data<Elem = T>,
        S2: DataMut<Elem = T>,
        T: Zero + Clone + Copy + Equivalence + Debug,
    {
        check_shape(snd, self.x_pencil.sz);
        check_shape(rcv, self.n_global);
        // send & receive buffer
        let mut send = vec![T::zero(); self.x_pencil.length()];
        let mut snd_view = snd.view();
        snd_view.swap_axes(0, 1);
        for (s, m) in send.iter_mut().zip(snd_view.iter()) {
            *s = *m;
        }

        let recv_length = self.n_global.iter().product::<usize>();
        let mut recv = vec![T::zero(); recv_length];

        let (counts, displs) = self.x_pencil.get_counts_all_gather();
        {
            let mut partition = PartitionMut::new(&mut recv[..], &counts[..], &displs[..]);
            self.world
                .all_gather_varcount_into(&send[..], &mut partition);
        }
        // copy receive buffer into array
        // *todo*: prevent copy by referencing?
        rcv.swap_axes(0, 1);
        for (s, m) in rcv.iter_mut().zip(recv.iter()) {
            *s = *m;
        }
        rcv.swap_axes(0, 1);
    }

    /// Gather data from y-pencil to all participating processors
    pub fn all_gather_y<S1, S2, T>(&self, snd: &ArrayBase<S1, Ix2>, rcv: &mut ArrayBase<S2, Ix2>)
    where
        S1: Data<Elem = T>,
        S2: DataMut<Elem = T>,
        T: Zero + Clone + Copy + Equivalence,
    {
        check_shape(snd, self.y_pencil.sz);
        check_shape(rcv, self.n_global);
        // send & receive buffer
        let mut send = vec![T::zero(); self.y_pencil.length()];
        for (s, m) in send.iter_mut().zip(snd.iter()) {
            *s = *m;
        }

        let recv_length = self.n_global.iter().product::<usize>();
        let mut recv = vec![T::zero(); recv_length];

        let (counts, displs) = self.y_pencil.get_counts_all_gather();
        {
            let mut partition = PartitionMut::new(&mut recv[..], &counts[..], &displs[..]);
            self.world
                .all_gather_varcount_into(&send[..], &mut partition);
        }
        // copy receive buffer into array
        // *todo*: prevent copy by referencing?
        for (s, m) in rcv.iter_mut().zip(recv.iter()) {
            *s = *m;
        }
    }
}

/// # Panics
/// Panics if array shape does not conform with pencil distribution
fn check_shape<A, S, const N: usize>(data: &ArrayBase<S, Dim<[usize; N]>>, shape: [usize; N])
where
    S: Data<Elem = A>,
    Dim<[usize; N]>: ndarray::Dimension,
{
    if data.shape() != shape {
        panic!(
            "Shape mismatch, got {:?} expected {:?}.",
            data.shape(),
            shape
        );
    }
}
