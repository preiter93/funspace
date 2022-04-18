//! Common traits for mpi space
//use crate::BaseAll;
#![cfg(feature = "mpi")]
use crate::types::{FloatNum, ScalarNum};
use crate::space::traits::{BaseSpaceSize, BaseSpaceTransform};
use ndarray::{Array, ArrayBase, Data, DataMut, Dim};

/// # Base space supertrait
pub trait BaseSpaceMpi<A, const N: usize>:
    Clone
    + BaseSpaceMpiFromOrtho<A, Self::Spectral, N>
    + BaseSpaceMpiGradient<A, Self::Spectral, N>
    + BaseSpaceMpiTransform<A, N>
where
    A: FloatNum,
{
}

impl<A, T, const N: usize> BaseSpaceMpi<A, N> for T
where
    A: FloatNum,
    T: Clone
        + BaseSpaceMpiFromOrtho<A, Self::Spectral, N>
        + BaseSpaceMpiGradient<A, Self::Spectral, N>
        + BaseSpaceMpiTransform<A, N>,
{
}

/// Dimensions
pub trait BaseSpaceMpiSize<const N: usize>: BaseSpaceSize<N> {
    /// Shape of physical space that this processors holds
    fn shape_physical_mpi(&self) -> [usize; N];

    /// Shape of spectral space that this processors holds
    fn shape_spectral_mpi(&self) -> [usize; N];

    // Shape of orthogonal spectral space that this processors holds
    fn shape_spectral_ortho_mpi(&self) -> [usize; N];
}

pub trait BaseSpaceMpiGradient<A, T, const N: usize>: BaseSpaceMpiSize<N> {
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
    ) -> Array<T, Dim<[usize; N]>>
    where
        S: Data<Elem = T>;
}

pub trait BaseSpaceMpiFromOrtho<A, T, const N: usize>: BaseSpaceMpiSize<N>
where
    T: ScalarNum,
{
    /// Transformation from composite to orthonormal space.
    fn to_ortho_mpi<S>(&self, input: &ArrayBase<S, Dim<[usize; N]>>) -> Array<T, Dim<[usize; N]>>
    where
        S: Data<Elem = T>,
    {
        let mut output = self.ndarray_from_shape::<T>(self.shape_spectral_ortho_mpi());
        self.to_ortho_inplace_mpi(input, &mut output);
        output
    }

    /// Transformation from composite to orthogonal space (inplace).
    fn to_ortho_inplace_mpi<S1, S2>(
        &self,
        input: &ArrayBase<S1, Dim<[usize; N]>>,
        output: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = T>,
        S2: DataMut<Elem = T>;

    /// Transformation from orthonormal and to composite space.
    fn from_ortho_mpi<S>(&self, input: &ArrayBase<S, Dim<[usize; N]>>) -> Array<T, Dim<[usize; N]>>
    where
        S: Data<Elem = T>,
    {
        let mut output = self.ndarray_from_shape::<T>(self.shape_spectral_mpi());
        self.from_ortho_inplace_mpi(input, &mut output);
        output
    }

    /// Transformation from orthogonal to composite space (inplace).
    fn from_ortho_inplace_mpi<S1, S2>(
        &self,
        input: &ArrayBase<S1, Dim<[usize; N]>>,
        output: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = T>,
        S2: DataMut<Elem = T>;
}

pub trait BaseSpaceMpiTransform<A, const N: usize>:
    BaseSpaceTransform<A, N> + BaseSpaceMpiSize<N>
where
    A: FloatNum,
{
    /// Transform physical -> spectral space
    fn forward_mpi<S>(
        &self,
        input: &ArrayBase<S, Dim<[usize; N]>>,
    ) -> Array<Self::Spectral, Dim<[usize; N]>>
    where
        S: ndarray::Data<Elem = Self::Physical>,
    {
        let mut output = self.ndarray_from_shape::<Self::Spectral>(self.shape_spectral_mpi());
        self.forward_inplace(input, &mut output);
        output
    }

    /// Transform physical -> spectral space (inplace)
    ///
    /// # Arguments
    ///
    /// * `input` - *ndarray* with num type of physical space
    /// * `output` - *ndarray* with num type of spectral space
    fn forward_inplace_mpi<S1, S2>(
        &self,
        input: &ArrayBase<S1, Dim<[usize; N]>>,
        output: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = <Self as BaseSpaceTransform<A, N>>::Physical>,
        S2: Data<Elem = <Self as BaseSpaceTransform<A, N>>::Spectral> + DataMut;

    /// Transform spectral -> physical space
    fn backward_mpi<S>(
        &self,
        input: &ArrayBase<S, Dim<[usize; N]>>,
    ) -> Array<Self::Physical, Dim<[usize; N]>>
    where
        S: ndarray::Data<Elem = Self::Spectral>,
    {
        let mut output = self.ndarray_from_shape::<Self::Physical>(self.shape_physical_mpi());
        self.backward_inplace(input, &mut output);
        output
    }

    /// Transform spectral -> physical space (inplace)
    ///
    /// # Arguments
    ///
    /// * `input`  - *ndarray* with num type of spectral space
    /// * `output` - *ndarray* with num type of physical space
    fn backward_inplace_mpi<S1, S2>(
        &self,
        input: &ArrayBase<S1, Dim<[usize; N]>>,
        output: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = <Self as BaseSpaceTransform<A, N>>::Spectral>,
        S2: Data<Elem = <Self as BaseSpaceTransform<A, N>>::Physical> + DataMut;
}

/*
pub trait MpiScatter<A, const N: usize>: Clone + BaseSpace<A, N>
where
    A: FloatNum,
{
    /// Scatter data from root to all processors (x-pencil distributed)
    ///
    /// # Info
    /// Call this routine from non-root
    fn scatter_to_x_pencil_phys<S2>(&self, pencil_data: &mut ArrayBase<S2, Dim<[usize; N]>>)
    where
        S2: Data<Elem = <Self as BaseSpaceTransform<A, N>>::Physical> + DataMut;

    /// Scatter data from root to all processors (x-pencil distributed)
    ///
    /// # Info
    /// Call this routine from root
    fn scatter_to_x_pencil_phys_root<S1, S2>(
        &self,
        global_data: &ArrayBase<S1, Dim<[usize; N]>>,
        pencil_data: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = <Self as BaseSpaceTransform<A, N>>::Physical>,
        S2: Data<Elem = <Self as BaseSpaceTransform<A, N>>::Physical> + DataMut;

    /// Scatter data from root to all processors (y-pencil distributed)
    ///
    /// # Info
    /// Call this routine from non-root
    fn scatter_to_y_pencil_phys<S2>(&self, pencil_data: &mut ArrayBase<S2, Dim<[usize; N]>>)
    where
        S2: Data<Elem = <Self as BaseSpaceTransform<A, N>>::Physical> + DataMut;

    /// Scatter data from root to all processors (y-pencil distributed)
    ///
    /// # Info
    /// Call this routine from root
    fn scatter_to_y_pencil_phys_root<S1, S2>(
        &self,
        global_data: &ArrayBase<S1, Dim<[usize; N]>>,
        pencil_data: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = <Self as BaseSpaceTransform<A, N>>::Physical>,
        S2: Data<Elem = <Self as BaseSpaceTransform<A, N>>::Physical> + DataMut;

    /// Scatter data from root to all processors (x-pencil distributed)
    ///
    /// # Info
    /// Call this routine from non-root
    fn scatter_to_x_pencil_spec<S2>(&self, pencil_data: &mut ArrayBase<S2, Dim<[usize; N]>>)
    where
        S2: Data<Elem = <Self as BaseSpaceTransform<A, N>>::Spectral> + DataMut;

    /// Scatter data from root to all processors (x-pencil distributed)
    ///
    /// # Info
    /// Call this routine from root
    fn scatter_to_x_pencil_spec_root<S1, S2>(
        &self,
        global_data: &ArrayBase<S1, Dim<[usize; N]>>,
        pencil_data: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = <Self as BaseSpaceTransform<A, N>>::Spectral>,
        S2: Data<Elem = <Self as BaseSpaceTransform<A, N>>::Spectral> + DataMut;

    /// Scatter data from root to all processors (y-pencil distributed)
    ///
    /// # Info
    /// Call this routine from non-root
    fn scatter_to_y_pencil_spec<S2>(&self, pencil_data: &mut ArrayBase<S2, Dim<[usize; N]>>)
    where
        S2: Data<Elem = <Self as BaseSpaceTransform<A, N>>::Spectral> + DataMut;

    /// Scatter data from root to all processors (y-pencil distributed)
    ///
    /// # Info
    /// Call this routine from root
    fn scatter_to_y_pencil_spec_root<S1, S2>(
        &self,
        global_data: &ArrayBase<S1, Dim<[usize; N]>>,
        pencil_data: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = <Self as BaseSpaceTransform<A, N>>::Spectral>,
        S2: Data<Elem = <Self as BaseSpaceTransform<A, N>>::Spectral> + DataMut;
}

pub trait MpiTranspose<A, const N: usize>: Clone + BaseSpace<A, N>
where
    A: FloatNum,
{
    /// Transpose from x pencil to y pencil for
    /// data in physical space
    fn transpose_x_to_y_phys<S1, S2>(
        &self,
        x_pencil: &ArrayBase<S1, Dim<[usize; N]>>,
        y_pencil: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = <Self as BaseSpaceTransform<A, N>>::Physical>,
        S2: Data<Elem = <Self as BaseSpaceTransform<A, N>>::Physical> + DataMut;

    /// Transpose from y pencil to x pencil for
    /// data in physical space
    fn transpose_y_to_x_phys<S1, S2>(
        &self,
        y_pencil: &ArrayBase<S1, Dim<[usize; N]>>,
        x_pencil: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = <Self as BaseSpaceTransform<A, N>>::Physical>,
        S2: Data<Elem = <Self as BaseSpaceTransform<A, N>>::Physical> + DataMut;

    /// Transpose from x pencil to y pencil for
    /// data in spectral space
    fn transpose_x_to_y_spec<S1, S2>(
        &self,
        x_pencil: &ArrayBase<S1, Dim<[usize; N]>>,
        y_pencil: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = <Self as BaseSpaceTransform<A, N>>::Spectral>,
        S2: Data<Elem = <Self as BaseSpaceTransform<A, N>>::Spectral> + DataMut;

    /// Transpose from y pencil to x pencil for
    /// data in spectral space
    fn transpose_y_to_x_spec<S1, S2>(
        &self,
        y_pencil: &ArrayBase<S1, Dim<[usize; N]>>,
        x_pencil: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = <Self as BaseSpaceTransform<A, N>>::Spectral>,
        S2: Data<Elem = <Self as BaseSpaceTransform<A, N>>::Spectral> + DataMut;
}
*/
