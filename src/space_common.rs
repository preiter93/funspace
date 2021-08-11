//! Common traits for space, independent of dimensionality
use crate::BaseAll;
use crate::FloatNum;
use ndarray::prelude::*;

pub trait SpaceCommon<A, const N: usize>
where
    A: FloatNum,
{
    type Output;

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

    /// Transformation from composite and to orthonormal space.
    fn to_ortho(
        &self,
        input: &Array<Self::Output, Dim<[usize; N]>>,
    ) -> Array<Self::Output, Dim<[usize; N]>>;

    /// Transformation from composite and to orthonormal space (inplace).
    fn to_ortho_inplace(
        &self,
        input: &Array<Self::Output, Dim<[usize; N]>>,
        output: &mut Array<Self::Output, Dim<[usize; N]>>,
    );

    /// Transformation from orthonormal and to composite space.
    fn from_ortho(
        &self,
        input: &Array<Self::Output, Dim<[usize; N]>>,
    ) -> Array<Self::Output, Dim<[usize; N]>>;

    /// Transformation from orthonormal and to composite space (inplace).
    fn from_ortho_inplace(
        &self,
        input: &Array<Self::Output, Dim<[usize; N]>>,
        output: &mut Array<Self::Output, Dim<[usize; N]>>,
    );

    /// Transformation from composite and to orthonormal space.
    fn to_ortho_par(
        &self,
        input: &Array<Self::Output, Dim<[usize; N]>>,
    ) -> Array<Self::Output, Dim<[usize; N]>>;

    /// Transformation from composite and to orthonormal space (inplace).
    fn to_ortho_inplace_par(
        &self,
        input: &Array<Self::Output, Dim<[usize; N]>>,
        output: &mut Array<Self::Output, Dim<[usize; N]>>,
    );

    /// Transformation from orthonormal and to composite space.
    fn from_ortho_par(
        &self,
        input: &Array<Self::Output, Dim<[usize; N]>>,
    ) -> Array<Self::Output, Dim<[usize; N]>>;

    /// Transformation from orthonormal and to composite space (inplace).
    fn from_ortho_inplace_par(
        &self,
        input: &Array<Self::Output, Dim<[usize; N]>>,
        output: &mut Array<Self::Output, Dim<[usize; N]>>,
    );

    /// Take gradient. Optional: Rescale result by a constant.
    fn gradient(
        &self,
        input: &Array<Self::Output, Dim<[usize; N]>>,
        deriv: [usize; N],
        scale: Option<[A; N]>,
    ) -> Array<Self::Output, Dim<[usize; N]>>;

    /// Return bases as array of enums
    fn base_all(&self) -> [BaseAll<A>; N];
}
