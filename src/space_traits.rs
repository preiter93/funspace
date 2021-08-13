//! Common traits for space, independent of dimensionality
use crate::BaseAll;
use crate::FloatNum;
use ndarray::{prelude::*, Data, DataMut};

pub trait BaseSpace<A, const N: usize>: Clone
where
    A: FloatNum,
{
    /// Number type in physical space (float or complex)
    type Physical;

    /// Number type in spectral space (float or complex)
    type Spectral;

    /// Laplacian
    ///
    /// # Arguments
    ///
    /// * `axis` - usize
    fn laplace(&self, axis: usize) -> Array2<A>;

    /// Pseudoinverse matrix of Laplacian
    ///
    /// # Arguments
    ///
    /// * `axis` - usize
    fn laplace_inv(&self, axis: usize) -> Array2<A>;

    /// Pseudoidentity matrix of laplacian
    ///
    /// # Arguments
    ///
    /// * `axis` - usize
    fn laplace_inv_eye(&self, axis: usize) -> Array2<A>;

    /// Mass matrix
    ///
    /// # Arguments
    ///
    /// * `axis` - usize
    fn mass(&self, axis: usize) -> Array2<A>;

    /// Coordinates of grid points (in physical space)
    ///
    /// # Arguments
    ///
    /// * `axis` - usize
    fn coords_axis(&self, axis: usize) -> Array1<A>;

    /// Array of coordinates
    fn coords(&self) -> [Array1<A>; N];

    /// Shape of physical space
    fn shape_physical(&self) -> [usize; N];

    /// Shape of spectral space
    fn shape_spectral(&self) -> [usize; N];

    /// Return array where size and type matches physical field
    fn ndarray_physical(&self) -> Array<Self::Physical, Dim<[usize; N]>>;

    /// Return array where size and type matches spectral field
    fn ndarray_spectral(&self) -> Array<Self::Spectral, Dim<[usize; N]>>;

    /// Transformation from composite and to orthonormal space.
    ///
    /// # Arguments
    ///
    /// * `input` - *ndarray* with num type of spectral space
    fn to_ortho<S>(
        &self,
        input: &ArrayBase<S, Dim<[usize; N]>>,
    ) -> Array<Self::Spectral, Dim<[usize; N]>>
    where
        S: Data<Elem = Self::Spectral>;

    /// Transformation from composite and to orthonormal space (inplace).
    ///
    /// # Arguments
    ///
    /// * `input` - *ndarray* with num type of spectral space
    /// * `output` - *ndarray* with num type of spectral space
    fn to_ortho_inplace<S1, S2>(
        &self,
        input: &ArrayBase<S1, Dim<[usize; N]>>,
        output: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = Self::Spectral>,
        S2: Data<Elem = Self::Spectral> + DataMut;

    /// Transformation from orthonormal and to composite space.
    ///
    /// # Arguments
    ///
    /// * `input` - *ndarray* with num type of spectral space
    fn from_ortho<S>(
        &self,
        input: &ArrayBase<S, Dim<[usize; N]>>,
    ) -> Array<Self::Spectral, Dim<[usize; N]>>
    where
        S: Data<Elem = Self::Spectral>;

    /// Transformation from orthonormal and to composite space (inplace).
    ///
    /// # Arguments
    ///
    /// * `input` - *ndarray* with num type of spectral space
    /// * `output` - *ndarray* with num type of spectral space
    fn from_ortho_inplace<S1, S2>(
        &self,
        input: &ArrayBase<S1, Dim<[usize; N]>>,
        output: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = Self::Spectral>,
        S2: Data<Elem = Self::Spectral> + DataMut;

    /// Transformation from composite and to orthonormal space.
    ///
    /// # Arguments
    ///
    /// * `input` - *ndarray* with num type of spectral space
    fn to_ortho_par<S>(
        &self,
        input: &ArrayBase<S, Dim<[usize; N]>>,
    ) -> Array<Self::Spectral, Dim<[usize; N]>>
    where
        S: Data<Elem = Self::Spectral>;

    /// Transformation from composite and to orthonormal space (inplace).
    ///
    /// # Arguments
    ///
    /// * `input` - *ndarray* with num type of spectral space
    /// * `output` - *ndarray* with num type of spectral space
    fn to_ortho_inplace_par<S1, S2>(
        &self,
        input: &ArrayBase<S1, Dim<[usize; N]>>,
        output: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = Self::Spectral>,
        S2: Data<Elem = Self::Spectral> + DataMut;

    /// Transformation from orthonormal and to composite space.
    ///
    /// # Arguments
    ///
    /// * `input` - *ndarray* with num type of spectral space
    fn from_ortho_par<S>(
        &self,
        input: &ArrayBase<S, Dim<[usize; N]>>,
    ) -> Array<Self::Spectral, Dim<[usize; N]>>
    where
        S: Data<Elem = Self::Spectral>;

    /// Transformation from orthonormal and to composite space (inplace).
    ///
    /// # Arguments
    ///
    /// * `input` - *ndarray* with num type of spectral space
    /// * `output` - *ndarray* with num type of spectral space
    fn from_ortho_inplace_par<S1, S2>(
        &self,
        input: &ArrayBase<S1, Dim<[usize; N]>>,
        output: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = Self::Spectral>,
        S2: Data<Elem = Self::Spectral> + DataMut;

    /// Take gradient. Optional: Rescale result by a constant.
    ///
    /// # Arguments
    ///
    /// * `input` - *ndarray* with num type of spectral space
    /// * `deriv` - [usize; N], derivative along each axis
    /// * `scale` - [float; N], scaling factor along each axis (default [1.;n])
    fn gradient<S>(
        &self,
        input: &ArrayBase<S, Dim<[usize; N]>>,
        deriv: [usize; N],
        scale: Option<[A; N]>,
    ) -> Array<Self::Spectral, Dim<[usize; N]>>
    where
        S: Data<Elem = Self::Spectral>;

    /// Take gradient. Optional: Rescale result by a constant. (Parallel)
    ///
    /// # Arguments
    ///
    /// * `input` - *ndarray* with num type of spectral space
    /// * `deriv` - [usize; N], derivative along each axis
    /// * `scale` - [float; N], scaling factor along each axis (default [1.;n])
    fn gradient_par<S>(
        &self,
        input: &ArrayBase<S, Dim<[usize; N]>>,
        deriv: [usize; N],
        scale: Option<[A; N]>,
    ) -> Array<Self::Spectral, Dim<[usize; N]>>
    where
        S: Data<Elem = Self::Spectral>;

    /// Return bases as array of enums
    fn base_all(&self) -> [BaseAll<A>; N];

    /// Transform physical -> spectral space
    ///
    /// # Arguments
    ///
    /// * `input` - *ndarray* with num type of physical space
    /// * `output` - *ndarray* with num type of spectral space
    fn forward<S>(
        &mut self,
        input: &mut ArrayBase<S, Dim<[usize; N]>>,
    ) -> Array<Self::Spectral, Dim<[usize; N]>>
    where
        S: ndarray::Data<Elem = Self::Physical>;

    /// Transform physical -> spectral space (inplace)
    ///
    /// # Arguments
    ///
    /// * `input` - *ndarray* with num type of physical space
    /// * `output` - *ndarray* with num type of spectral space
    fn forward_inplace<S1, S2>(
        &mut self,
        input: &mut ArrayBase<S1, Dim<[usize; N]>>,
        output: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = Self::Physical>,
        S2: Data<Elem = Self::Spectral> + DataMut;

    /// Transform spectral -> physical space
    ///
    /// # Arguments
    ///
    /// * `input` - *ndarray* with num type of spectral space
    /// * `output` - *ndarray* with num type of physical space
    fn backward<S>(
        &mut self,
        input: &mut ArrayBase<S, Dim<[usize; N]>>,
    ) -> Array<Self::Physical, Dim<[usize; N]>>
    where
        S: Data<Elem = Self::Spectral>;

    /// Transform spectral -> physical space (inplace)
    ///
    /// # Arguments
    ///
    /// * `input` - *ndarray* with num type of spectral space
    /// * `output` - *ndarray* with num type of physical space
    fn backward_inplace<S1, S2>(
        &mut self,
        input: &mut ArrayBase<S1, Dim<[usize; N]>>,
        output: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = Self::Spectral>,
        S2: Data<Elem = Self::Physical> + DataMut;

    /// Transform physical -> spectral space
    ///
    /// # Arguments
    ///
    /// * `input` - *ndarray* with num type of physical space
    /// * `output` - *ndarray* with num type of spectral space
    fn forward_par<S>(
        &mut self,
        input: &mut ArrayBase<S, Dim<[usize; N]>>,
    ) -> Array<Self::Spectral, Dim<[usize; N]>>
    where
        S: Data<Elem = Self::Physical>;

    /// Transform physical -> spectral space (inplace)
    ///
    /// # Arguments
    ///
    /// * `input` - *ndarray* with num type of physical space
    /// * `output` - *ndarray* with num type of spectral space
    fn forward_inplace_par<S1, S2>(
        &mut self,
        input: &mut ArrayBase<S1, Dim<[usize; N]>>,
        output: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = Self::Physical>,
        S2: Data<Elem = Self::Spectral> + DataMut;

    /// Transform spectral -> physical space
    ///
    /// # Arguments
    ///
    /// * `input` - *ndarray* with num type of spectral space
    /// * `output` - *ndarray* with num type of physical space
    fn backward_par<S>(
        &mut self,
        input: &mut ArrayBase<S, Dim<[usize; N]>>,
    ) -> Array<Self::Physical, Dim<[usize; N]>>
    where
        S: Data<Elem = Self::Spectral>;

    /// Transform spectral -> physical space (inplace)
    ///
    /// # Arguments
    ///
    /// * `input` - *ndarray* with num type of spectral space
    /// * `output` - *ndarray* with num type of physical space
    fn backward_inplace_par<S1, S2>(
        &mut self,
        input: &mut ArrayBase<S1, Dim<[usize; N]>>,
        output: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = Self::Spectral>,
        S2: Data<Elem = Self::Physical> + DataMut;
}
