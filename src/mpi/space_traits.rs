//! Common traits for mpi space
//use crate::BaseAll;
#![cfg(feature = "mpi")]
use super::Decomp2d;
use super::Universe;
use crate::BaseSpace;
use crate::FloatNum;
use ndarray::{Array, ArrayBase, Data, DataMut, Dim};

pub trait BaseSpaceMpi<A, const N: usize>: Clone + BaseSpace<A, N>
where
    A: FloatNum,
{
    /// Return reference to mpi universe
    fn get_universe(&self) -> &Universe;

    /// Return processor rank
    fn get_nrank(&self) -> usize;

    /// Return number of processors in universe
    fn get_nprocs(&self) -> usize;

    /// Return decomposition which matches a given global arrays shape.
    fn get_decomp_from_global(&self, shape: &[usize]) -> &Decomp2d;

    /// Return decomposition which matches a given x-pencil arrays shape.
    fn get_decomp_from_x_pencil(&self, shape: &[usize]) -> &Decomp2d;

    /// Return decomposition which matches a given x-pencil arrays shape.
    fn get_decomp_from_y_pencil(&self, shape: &[usize]) -> &Decomp2d;

    /// Shape of physical space (x pencil distribution)
    fn shape_physical_x_pen(&self) -> [usize; N];

    /// Shape of physical space (y pencil distribution)
    fn shape_physical_y_pen(&self) -> [usize; N];

    /// Shape of spectral space (x pencil distribution)
    fn shape_spectral_x_pen(&self) -> [usize; N];

    /// Shape of spectral space (y pencil distribution)
    fn shape_spectral_y_pen(&self) -> [usize; N];

    /// Return array where size and type matches physical field (x pencil distribution)
    fn ndarray_physical_x_pen(&self)
        -> Array<<Self as BaseSpace<A, N>>::Physical, Dim<[usize; N]>>;

    /// Return array where size and type matches physical field (y pencil distribution)
    fn ndarray_physical_y_pen(&self)
        -> Array<<Self as BaseSpace<A, N>>::Physical, Dim<[usize; N]>>;

    /// Return array where size and type matches spectral field (x pencil distribution)
    fn ndarray_spectral_x_pen(&self)
        -> Array<<Self as BaseSpace<A, N>>::Spectral, Dim<[usize; N]>>;

    /// Return array where size and type matches spectral field (y pencil distribution)
    fn ndarray_spectral_y_pen(&self)
        -> Array<<Self as BaseSpace<A, N>>::Spectral, Dim<[usize; N]>>;

    /// Transformation from composite and to orthonormal space.
    ///
    /// # Arguments
    ///
    /// * `input` - *ndarray* with num type of spectral space
    fn to_ortho_mpi<S>(
        &self,
        input: &ArrayBase<S, Dim<[usize; N]>>,
    ) -> Array<<Self as BaseSpace<A, N>>::Spectral, Dim<[usize; N]>>
    where
        S: Data<Elem = <Self as BaseSpace<A, N>>::Spectral>;

    /// Transformation from composite and to orthonormal space (inplace).
    ///
    /// # Arguments
    ///
    /// * `input` - *ndarray* with num type of spectral space
    /// * `output` - *ndarray* with num type of spectral space
    fn to_ortho_inplace_mpi<S1, S2>(
        &self,
        input: &ArrayBase<S1, Dim<[usize; N]>>,
        output: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = <Self as BaseSpace<A, N>>::Spectral>,
        S2: Data<Elem = <Self as BaseSpace<A, N>>::Spectral> + DataMut;

    /// Transformation from orthonormal and to composite space.
    ///
    /// # Arguments
    ///
    /// * `input` - *ndarray* with num type of spectral space
    fn from_ortho_mpi<S>(
        &self,
        input: &ArrayBase<S, Dim<[usize; N]>>,
    ) -> Array<<Self as BaseSpace<A, N>>::Spectral, Dim<[usize; N]>>
    where
        S: Data<Elem = <Self as BaseSpace<A, N>>::Spectral>;

    /// Transformation from orthonormal and to composite space (inplace).
    ///
    /// # Arguments
    ///
    /// * `input` - *ndarray* with num type of spectral space
    /// * `output` - *ndarray* with num type of spectral space
    fn from_ortho_inplace_mpi<S1, S2>(
        &self,
        input: &ArrayBase<S1, Dim<[usize; N]>>,
        output: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = <Self as BaseSpace<A, N>>::Spectral>,
        S2: Data<Elem = <Self as BaseSpace<A, N>>::Spectral> + DataMut;

    /// Take gradient. Optional: Rescale result by a constant. (Parallel)
    ///
    /// # Arguments
    ///
    /// * `input` - *ndarray* with num type of spectral space
    /// * `deriv` - [usize; N], derivative along each axis
    /// * `scale` - [float; N], scaling factor along each axis (default [1.;n])
    /// (necessary if domain size is normalized)
    fn gradient_mpi<S>(
        &self,
        input: &ArrayBase<S, Dim<[usize; N]>>,
        deriv: [usize; N],
        scale: Option<[A; N]>,
    ) -> Array<<Self as BaseSpace<A, N>>::Spectral, Dim<[usize; N]>>
    where
        S: Data<Elem = <Self as BaseSpace<A, N>>::Spectral>;

    /// Transform physical -> spectral space
    ///
    /// # Arguments
    ///
    /// * `input` - *ndarray* with num type of physical space
    /// * `output` - *ndarray* with num type of spectral space
    fn forward_mpi<S>(
        &mut self,
        input: &ArrayBase<S, Dim<[usize; N]>>,
    ) -> Array<<Self as BaseSpace<A, N>>::Spectral, Dim<[usize; N]>>
    where
        S: Data<Elem = <Self as BaseSpace<A, N>>::Physical>;

    /// Transform physical -> spectral space (inplace)
    ///
    /// # Arguments
    ///
    /// * `input` - *ndarray* with num type of physical space
    /// * `output` - *ndarray* with num type of spectral space
    fn forward_inplace_mpi<S1, S2>(
        &mut self,
        input: &ArrayBase<S1, Dim<[usize; N]>>,
        output: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = <Self as BaseSpace<A, N>>::Physical>,
        S2: Data<Elem = <Self as BaseSpace<A, N>>::Spectral> + DataMut;

    /// Transform spectral -> physical space
    ///
    /// # Arguments
    ///
    /// * `input` - *ndarray* with num type of spectral space
    /// * `output` - *ndarray* with num type of physical space
    fn backward_mpi<S>(
        &mut self,
        input: &ArrayBase<S, Dim<[usize; N]>>,
    ) -> Array<<Self as BaseSpace<A, N>>::Physical, Dim<[usize; N]>>
    where
        S: Data<Elem = <Self as BaseSpace<A, N>>::Spectral>;

    /// Transform spectral -> physical space (inplace)
    ///
    /// # Arguments
    ///
    /// * `input` - *ndarray* with num type of spectral space
    /// * `output` - *ndarray* with num type of physical space
    fn backward_inplace_mpi<S1, S2>(
        &mut self,
        input: &ArrayBase<S1, Dim<[usize; N]>>,
        output: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = <Self as BaseSpace<A, N>>::Spectral>,
        S2: Data<Elem = <Self as BaseSpace<A, N>>::Physical> + DataMut;

    /// Gather data from all processors (x-pencil distributed) onto root
    fn gather_from_x_pencil_phys<S1, S2>(
        &self,
        pencil_data: &ArrayBase<S1, Dim<[usize; N]>>,
        global_data: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = <Self as BaseSpace<A, N>>::Physical>,
        S2: Data<Elem = <Self as BaseSpace<A, N>>::Physical> + DataMut;

    /// Gather data from all processors (y-pencil distributed) onto root
    fn gather_from_y_pencil_phys<S1, S2>(
        &self,
        pencil_data: &ArrayBase<S1, Dim<[usize; N]>>,
        global_data: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = <Self as BaseSpace<A, N>>::Physical>,
        S2: Data<Elem = <Self as BaseSpace<A, N>>::Physical> + DataMut;

    /// Gather data from all processors (x-pencil distributed) onto root
    fn gather_from_x_pencil_spec<S1, S2>(
        &self,
        pencil_data: &ArrayBase<S1, Dim<[usize; N]>>,
        global_data: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = <Self as BaseSpace<A, N>>::Spectral>,
        S2: Data<Elem = <Self as BaseSpace<A, N>>::Spectral> + DataMut;

    /// Gather data from all processors (x-pencil distributed) onto root
    fn gather_from_y_pencil_spec<S1, S2>(
        &self,
        pencil_data: &ArrayBase<S1, Dim<[usize; N]>>,
        global_data: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = <Self as BaseSpace<A, N>>::Spectral>,
        S2: Data<Elem = <Self as BaseSpace<A, N>>::Spectral> + DataMut;

    /// Gather data from all processors (x-pencil distributed) to all
    /// participating processors
    fn all_gather_from_x_pencil_phys<S1, S2>(
        &self,
        pencil_data: &ArrayBase<S1, Dim<[usize; N]>>,
        global_data: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = <Self as BaseSpace<A, N>>::Physical>,
        S2: Data<Elem = <Self as BaseSpace<A, N>>::Physical> + DataMut;

    /// Gather data from all processors (y-pencil distributed) to all
    /// participating processors
    fn all_gather_from_y_pencil_phys<S1, S2>(
        &self,
        pencil_data: &ArrayBase<S1, Dim<[usize; N]>>,
        global_data: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = <Self as BaseSpace<A, N>>::Physical>,
        S2: Data<Elem = <Self as BaseSpace<A, N>>::Physical> + DataMut;
    /// Gather data from all processors (x-pencil distributed) to all
    /// participating processors
    fn all_gather_from_x_pencil_spec<S1, S2>(
        &self,
        pencil_data: &ArrayBase<S1, Dim<[usize; N]>>,
        global_data: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = <Self as BaseSpace<A, N>>::Spectral>,
        S2: Data<Elem = <Self as BaseSpace<A, N>>::Spectral> + DataMut;

    /// Gather data from all processors (y-pencil distributed) to all
    /// participating processors
    fn all_gather_from_y_pencil_spec<S1, S2>(
        &self,
        pencil_data: &ArrayBase<S1, Dim<[usize; N]>>,
        global_data: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = <Self as BaseSpace<A, N>>::Spectral>,
        S2: Data<Elem = <Self as BaseSpace<A, N>>::Spectral> + DataMut;

    /// Scatter data from root to all processors (x-pencil distributed)
    fn scatter_to_x_pencil_phys<S1, S2>(
        &self,
        global_data: &ArrayBase<S1, Dim<[usize; N]>>,
        pencil_data: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = <Self as BaseSpace<A, N>>::Physical>,
        S2: Data<Elem = <Self as BaseSpace<A, N>>::Physical> + DataMut;

    /// Scatter data from root to all processors (y-pencil distributed)
    fn scatter_to_y_pencil_phys<S1, S2>(
        &self,
        global_data: &ArrayBase<S1, Dim<[usize; N]>>,
        pencil_data: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = <Self as BaseSpace<A, N>>::Physical>,
        S2: Data<Elem = <Self as BaseSpace<A, N>>::Physical> + DataMut;

    /// Scatter data from root to all processors (x-pencil distributed)
    fn scatter_to_x_pencil_spec<S1, S2>(
        &self,
        global_data: &ArrayBase<S1, Dim<[usize; N]>>,
        pencil_data: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = <Self as BaseSpace<A, N>>::Spectral>,
        S2: Data<Elem = <Self as BaseSpace<A, N>>::Spectral> + DataMut;

    /// Scatter data from root to all processors (y-pencil distributed)
    fn scatter_to_y_pencil_spec<S1, S2>(
        &self,
        global_data: &ArrayBase<S1, Dim<[usize; N]>>,
        pencil_data: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = <Self as BaseSpace<A, N>>::Spectral>,
        S2: Data<Elem = <Self as BaseSpace<A, N>>::Spectral> + DataMut;

    /// Transpose from x pencil to y pencil for
    /// data in physical space
    fn transpose_x_to_y_phys<S1, S2>(
        &self,
        x_pencil: &ArrayBase<S1, Dim<[usize; N]>>,
        y_pencil: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = <Self as BaseSpace<A, N>>::Physical>,
        S2: Data<Elem = <Self as BaseSpace<A, N>>::Physical> + DataMut;

    /// Transpose from y pencil to x pencil for
    /// data in physical space
    fn transpose_y_to_x_phys<S1, S2>(
        &self,
        y_pencil: &ArrayBase<S1, Dim<[usize; N]>>,
        x_pencil: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = <Self as BaseSpace<A, N>>::Physical>,
        S2: Data<Elem = <Self as BaseSpace<A, N>>::Physical> + DataMut;

    /// Transpose from x pencil to y pencil for
    /// data in spectral space
    fn transpose_x_to_y_spec<S1, S2>(
        &self,
        x_pencil: &ArrayBase<S1, Dim<[usize; N]>>,
        y_pencil: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = <Self as BaseSpace<A, N>>::Spectral>,
        S2: Data<Elem = <Self as BaseSpace<A, N>>::Spectral> + DataMut;

    /// Transpose from y pencil to x pencil for
    /// data in spectral space
    fn transpose_y_to_x_spec<S1, S2>(
        &self,
        y_pencil: &ArrayBase<S1, Dim<[usize; N]>>,
        x_pencil: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = <Self as BaseSpace<A, N>>::Spectral>,
        S2: Data<Elem = <Self as BaseSpace<A, N>>::Spectral> + DataMut;
}

// /// Split data to all processors as x pencil.
// /// Input data must conform physical space array.
// fn split_to_x_pencil_phys<S1, S2>(
//     &mut self,
//     global_data: &ArrayBase<S1, Dim<[usize; N]>>,
//     pencil_data: &mut ArrayBase<S2, Dim<[usize; N]>>,
// ) where
//     S1: Data<Elem = <Self as BaseSpace<A, N>>::Physical>,
//     S2: Data<Elem = <Self as BaseSpace<A, N>>::Physical> + DataMut;
//
// /// Split data to all processors as x pencil.
// /// Input data must conform spectral space array.
// fn split_to_x_pencil_spec<S1, S2>(
//     &mut self,
//     global_data: &ArrayBase<S1, Dim<[usize; N]>>,
//     pencil_data: &mut ArrayBase<S2, Dim<[usize; N]>>,
// ) where
//     S1: Data<Elem = <Self as BaseSpace<A, N>>::Spectral>,
//     S2: Data<Elem = <Self as BaseSpace<A, N>>::Spectral> + DataMut;
//
// /// Split data to all processors as y pencil.
// /// Input data must conform physical space array.
// fn split_to_y_pencil_phys<S1, S2>(
//     &mut self,
//     global_data: &ArrayBase<S1, Dim<[usize; N]>>,
//     pencil_data: &mut ArrayBase<S2, Dim<[usize; N]>>,
// ) where
//     S1: Data<Elem = <Self as BaseSpace<A, N>>::Physical>,
//     S2: Data<Elem = <Self as BaseSpace<A, N>>::Physical> + DataMut;
//
// /// Split data to all processors as y pencil.
// /// Input data must conform spectral space array.
// fn split_to_y_pencil_spec<S1, S2>(
//     &mut self,
//     global_data: &ArrayBase<S1, Dim<[usize; N]>>,
//     pencil_data: &mut ArrayBase<S2, Dim<[usize; N]>>,
// ) where
//     S1: Data<Elem = <Self as BaseSpace<A, N>>::Spectral>,
//     S2: Data<Elem = <Self as BaseSpace<A, N>>::Spectral> + DataMut;
