//! Common traits for space, independent of dimensionality
use crate::BaseAll;
use crate::FloatNum;
use ndarray::{prelude::*, Data, DataMut};

pub trait BaseSpace<A, const N: usize>: Clone
where
    A: FloatNum,
{
    type Physical;

    type Spectral;

    /// Laplacian
    fn laplace(&self, axis: usize) -> Array2<A>;

    /// Pseudoinverse mtrix of Laplacian
    fn laplace_inv(&self, axis: usize) -> Array2<A>;

    /// Pseudoidentity matrix of laplacian
    fn laplace_inv_eye(&self, axis: usize) -> Array2<A>;

    /// Mass matrix
    fn mass(&self, axis: usize) -> Array2<A>;

    /// Coordinates of grid points (in physical space)
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
    fn to_ortho<S>(
        &self,
        input: &ArrayBase<S, Dim<[usize; N]>>,
    ) -> Array<Self::Spectral, Dim<[usize; N]>>
    where
        S: Data<Elem = Self::Spectral>;

    /// Transformation from composite and to orthonormal space (inplace).
    fn to_ortho_inplace<S1, S2>(
        &self,
        input: &ArrayBase<S1, Dim<[usize; N]>>,
        output: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = Self::Spectral>,
        S2: Data<Elem = Self::Spectral> + DataMut;

    /// Transformation from orthonormal and to composite space.
    fn from_ortho<S>(
        &self,
        input: &ArrayBase<S, Dim<[usize; N]>>,
    ) -> Array<Self::Spectral, Dim<[usize; N]>>
    where
        S: Data<Elem = Self::Spectral>;

    /// Transformation from orthonormal and to composite space (inplace).
    fn from_ortho_inplace<S1, S2>(
        &self,
        input: &ArrayBase<S1, Dim<[usize; N]>>,
        output: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = Self::Spectral>,
        S2: Data<Elem = Self::Spectral> + DataMut;

    /// Transformation from composite and to orthonormal space.
    fn to_ortho_par<S>(
        &self,
        input: &ArrayBase<S, Dim<[usize; N]>>,
    ) -> Array<Self::Spectral, Dim<[usize; N]>>
    where
        S: Data<Elem = Self::Spectral>;

    /// Transformation from composite and to orthonormal space (inplace).
    fn to_ortho_inplace_par<S1, S2>(
        &self,
        input: &ArrayBase<S1, Dim<[usize; N]>>,
        output: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = Self::Spectral>,
        S2: Data<Elem = Self::Spectral> + DataMut;

    /// Transformation from orthonormal and to composite space.
    fn from_ortho_par<S>(
        &self,
        input: &ArrayBase<S, Dim<[usize; N]>>,
    ) -> Array<Self::Spectral, Dim<[usize; N]>>
    where
        S: Data<Elem = Self::Spectral>;

    /// Transformation from orthonormal and to composite space (inplace).
    fn from_ortho_inplace_par<S1, S2>(
        &self,
        input: &ArrayBase<S1, Dim<[usize; N]>>,
        output: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = Self::Spectral>,
        S2: Data<Elem = Self::Spectral> + DataMut;

    /// Take gradient. Optional: Rescale result by a constant.
    fn gradient<S>(
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
    fn forward<S>(
        &mut self,
        input: &mut ArrayBase<S, Dim<[usize; N]>>,
    ) -> Array<Self::Spectral, Dim<[usize; N]>>
    where
        S: ndarray::Data<Elem = Self::Physical>;

    /// Transform physical -> spectral space (inplace)
    fn forward_inplace<S1, S2>(
        &mut self,
        input: &mut ArrayBase<S1, Dim<[usize; N]>>,
        output: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = Self::Physical>,
        S2: Data<Elem = Self::Spectral> + DataMut;

    /// Transform spectral -> physical space
    fn backward<S>(
        &mut self,
        input: &mut ArrayBase<S, Dim<[usize; N]>>,
    ) -> Array<Self::Physical, Dim<[usize; N]>>
    where
        S: Data<Elem = Self::Spectral>;

    /// Transform spectral -> physical space (inplace)
    fn backward_inplace<S1, S2>(
        &mut self,
        input: &mut ArrayBase<S1, Dim<[usize; N]>>,
        output: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = Self::Spectral>,
        S2: Data<Elem = Self::Physical> + DataMut;

    /// Transform physical -> spectral space
    fn forward_par<S>(
        &mut self,
        input: &mut ArrayBase<S, Dim<[usize; N]>>,
    ) -> Array<Self::Spectral, Dim<[usize; N]>>
    where
        S: Data<Elem = Self::Physical>;

    /// Transform physical -> spectral space (inplace)
    fn forward_inplace_par<S1, S2>(
        &mut self,
        input: &mut ArrayBase<S1, Dim<[usize; N]>>,
        output: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = Self::Physical>,
        S2: Data<Elem = Self::Spectral> + DataMut;

    /// Transform spectral -> physical space
    fn backward_par<S>(
        &mut self,
        input: &mut ArrayBase<S, Dim<[usize; N]>>,
    ) -> Array<Self::Physical, Dim<[usize; N]>>
    where
        S: Data<Elem = Self::Spectral>;

    /// Transform spectral -> physical space (inplace)
    fn backward_inplace_par<S1, S2>(
        &mut self,
        input: &mut ArrayBase<S1, Dim<[usize; N]>>,
        output: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = Self::Spectral>,
        S2: Data<Elem = Self::Physical> + DataMut;
}
