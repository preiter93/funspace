//! Common traits for space, independent of dimensionality
use crate::enums::{BaseKind, TransformKind};
use crate::types::{FloatNum, ScalarNum};
use ndarray::{prelude::*, Data, DataMut};
use num_traits::Zero;

/// # Base space supertrait
pub trait BaseSpace<A, const N: usize>:
    Clone
    + BaseSpaceElements<N, RealNum = A>
    + BaseSpaceMatOpStencil<NumType = A>
    + BaseSpaceMatOpLaplacian<NumType = A>
    + BaseSpaceFromOrtho<A, Self::Spectral, N>
    + BaseSpaceGradient<A, Self::Spectral, N>
    + BaseSpaceTransform<A, N>
where
    A: FloatNum,
{
}

impl<A, T, const N: usize> BaseSpace<A, N> for T
where
    A: FloatNum,
    T: Clone
        + BaseSpaceElements<N, RealNum = A>
        + BaseSpaceMatOpStencil<NumType = A>
        + BaseSpaceMatOpLaplacian<NumType = A>
        + BaseSpaceFromOrtho<A, Self::Spectral, N>
        + BaseSpaceGradient<A, Self::Spectral, N>
        + BaseSpaceTransform<A, N>,
{
}

/// Dimensions
pub trait BaseSpaceSize<const N: usize> {
    /// Shape of physical space
    fn shape_physical(&self) -> [usize; N];

    /// Shape of spectral space
    fn shape_spectral(&self) -> [usize; N];

    /// Shape of spectral space (orthogonal bases)
    fn shape_spectral_ortho(&self) -> [usize; N];

    /// Return array from shape
    fn ndarray_from_shape<T: Clone + Zero>(&self, shape: [usize; N]) -> Array<T, Dim<[usize; N]>>;
}

/// Coordinates and base functions
pub trait BaseSpaceElements<const N: usize> {
    /// Real valued scalar type
    type RealNum;

    /// Return base key
    fn base_kind(&self, axis: usize) -> BaseKind;

    /// Return transform kind
    fn transform_kind(&self, axis: usize) -> TransformKind;

    /// Coordinates of grid points (in physical space)
    ///
    /// # Arguments
    ///
    /// * `axis` - usize
    fn coords_axis(&self, axis: usize) -> Array1<Self::RealNum>;

    /// Array of coordinates
    fn coords(&self) -> [Array1<Self::RealNum>; N];
}

/// Collection of matrix operators
pub trait BaseSpaceMatOpStencil {
    /// Scalar type of matrix
    type NumType;

    /// Transformation stencil
    ///
    /// Multiplication of this matrix with a coefficient vector has
    /// the same effect as  [`BaseSpaceFromOrtho::to_ortho()`],
    /// but is less efficient.
    ///
    /// Returns identity matrix for orthogonal bases
    ///
    /// # Arguments
    ///
    /// * `axis` - usize
    fn stencil(&self, axis: usize) -> Array2<Self::NumType>;

    /// Inverse of transformation stencil
    ///
    /// Multiplication of this matrix with a coefficient vector has
    /// the same effect as  [`BaseSpaceFromOrtho::from_ortho()`],
    /// but is less efficient.
    ///
    /// Returns identity matrix for orthogonal bases
    ///
    /// # Arguments
    ///
    /// * `axis` - usize
    fn stencil_inv(&self, axis: usize) -> Array2<Self::NumType>;
}

/// Collection of *Laplacian* matrix operators
pub trait BaseSpaceMatOpLaplacian {
    /// Scalar type of laplacian matrix
    type NumType;

    /// Laplacian `L`
    ///
    /// ```text
    /// L_pinv @ L = I_pinv
    /// ```
    ///
    /// # Arguments
    ///
    /// * `axis` - usize
    fn laplacian(&self, axis: usize) -> Array2<Self::NumType>;

    /// Pseudoinverse matrix `L_pinv` of Laplacian
    ///
    /// Returns (`L_pinv`, `I_pinv`)
    ///
    /// ```text
    /// L_pinv @ L = I_pinv
    /// ```
    ///
    /// # Arguments
    ///
    /// * `axis` - usize
    fn laplacian_pinv(&self, axis: usize) -> (Array2<Self::NumType>, Array2<Self::NumType>);
}

/// # Transform from orthogonal <-> composite base
pub trait BaseSpaceFromOrtho<A, T, const N: usize>: BaseSpaceSize<N>
where
    T: ScalarNum,
{
    /// Transformation from composite and to orthonormal space.
    ///
    /// # Arguments
    ///
    /// * `input` - *ndarray* with num type of spectral space
    fn to_ortho<S>(&self, input: &ArrayBase<S, Dim<[usize; N]>>) -> Array<T, Dim<[usize; N]>>
    where
        S: Data<Elem = T>,
        [usize; N]: Dimension,
    {
        let mut output = self.ndarray_from_shape::<T>(self.shape_spectral_ortho());
        self.to_ortho_inplace(input, &mut output);
        output
    }

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
        S1: Data<Elem = T>,
        S2: Data<Elem = T> + DataMut;

    /// Transformation from orthonormal and to composite space.
    ///
    /// # Arguments
    ///
    /// * `input` - *ndarray* with num type of spectral space
    fn from_ortho<S>(&self, input: &ArrayBase<S, Dim<[usize; N]>>) -> Array<T, Dim<[usize; N]>>
    where
        S: Data<Elem = T>,
    {
        let mut output = self.ndarray_from_shape::<T>(self.shape_spectral());
        self.from_ortho_inplace(input, &mut output);
        output
    }

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
        S1: Data<Elem = T>,
        S2: Data<Elem = T> + DataMut;

    /// Transformation from composite and to orthonormal space.
    ///
    /// # Arguments
    ///
    /// * `input` - *ndarray* with num type of spectral space
    fn to_ortho_par<S>(&self, input: &ArrayBase<S, Dim<[usize; N]>>) -> Array<T, Dim<[usize; N]>>
    where
        S: Data<Elem = T>,
    {
        let mut output = self.ndarray_from_shape::<T>(self.shape_spectral_ortho());
        self.to_ortho_inplace_par(input, &mut output);
        output
    }

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
        S1: Data<Elem = T>,
        S2: Data<Elem = T> + DataMut;

    /// Transformation from orthonormal and to composite space.
    ///
    /// # Arguments
    ///
    /// * `input` - *ndarray* with num type of spectral space
    fn from_ortho_par<S>(&self, input: &ArrayBase<S, Dim<[usize; N]>>) -> Array<T, Dim<[usize; N]>>
    where
        S: Data<Elem = T>,
    {
        let mut output = self.ndarray_from_shape::<T>(self.shape_spectral());
        self.from_ortho_inplace_par(input, &mut output);
        output
    }

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
        S1: Data<Elem = T>,
        S2: Data<Elem = T> + DataMut;
}

/// # Differentiation in spectral space
pub trait BaseSpaceGradient<A, T, const N: usize>: BaseSpaceSize<N> {
    /// Number type in spectral space (float or complex)
    // type Spectral2;

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
    ) -> Array<T, Dim<[usize; N]>>
    where
        S: Data<Elem = T>;

    /// Take gradient. Optional: Rescale result by a constant.
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
    ) -> Array<T, Dim<[usize; N]>>
    where
        S: Data<Elem = T>;
}

/// Collection of explicit differential operators
/// in matrix form
pub trait BaseSpaceExplicitOperators<A> {
    /// Laplacian `L`
    ///
    /// ```text
    /// L_pinv @ L = I_pinv
    /// ```
    ///
    /// # Arguments
    ///
    /// * `axis` - usize
    fn laplace(&self, axis: usize) -> Array2<A>;

    /// Pseudoinverse matrix `L_pinv` of Laplacian
    ///
    /// Returns (`L_pinv`, `I_pinv`)
    ///
    /// ```text
    /// L_pinv @ L = I_pinv
    /// ```
    ///
    /// # Arguments
    ///
    /// * `axis` - usize
    fn laplace_pinv(&self, axis: usize) -> (Array2<A>, Array2<A>);

    /// Transformation stencil
    ///
    /// Multiplication of this matrix with a coefficient vector has
    /// the same effect as  [`BaseSpaceFromOrtho::to_ortho()`],
    /// but is less efficient.
    ///
    /// Returns identity matrix for orthogonal bases
    ///
    /// # Arguments
    ///
    /// * `axis` - usize
    fn stencil(&self, axis: usize) -> Array2<A>;

    /// Inverse of transformation stencil
    ///
    /// Multiplication of this matrix with a coefficient vector has
    /// the same effect as  [`BaseSpaceFromOrtho::from_ortho()`],
    /// but is less efficient.
    ///
    /// Returns identity matrix for orthogonal bases
    ///
    /// # Arguments
    ///
    /// * `axis` - usize
    fn stencil_inv(&self, axis: usize) -> Array2<A>;
}

/// # Transformation from physical values to spectral coefficients
pub trait BaseSpaceTransform<A, const N: usize>: BaseSpaceSize<N>
where
    A: FloatNum,
    Self::Physical: ScalarNum,
    Self::Spectral: ScalarNum,
{
    // Number type in physical space (float or complex)
    type Physical;

    // Number type in spectral space (float or complex)
    type Spectral;

    /// Return array where size and type matches physical field
    fn ndarray_physical(&self) -> Array<Self::Physical, Dim<[usize; N]>> {
        self.ndarray_from_shape::<Self::Physical>(self.shape_physical())
    }

    /// Return array where size and type matches spectral field
    fn ndarray_spectral(&self) -> Array<Self::Spectral, Dim<[usize; N]>> {
        self.ndarray_from_shape::<Self::Spectral>(self.shape_spectral())
    }

    /// Transform physical -> spectral space
    ///
    /// # Arguments
    ///
    /// * `input` - *ndarray* with num type of physical space
    /// * `output` - *ndarray* with num type of spectral space
    fn forward<S>(
        &self,
        input: &ArrayBase<S, Dim<[usize; N]>>,
    ) -> Array<Self::Spectral, Dim<[usize; N]>>
    where
        S: ndarray::Data<Elem = Self::Physical>,
    {
        let mut output = self.ndarray_from_shape::<Self::Spectral>(self.shape_spectral());
        self.forward_inplace(input, &mut output);
        output
    }

    /// Transform physical -> spectral space (inplace)
    ///
    /// # Arguments
    ///
    /// * `input` - *ndarray* with num type of physical space
    /// * `output` - *ndarray* with num type of spectral space
    fn forward_inplace<S1, S2>(
        &self,
        input: &ArrayBase<S1, Dim<[usize; N]>>,
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
        &self,
        input: &ArrayBase<S, Dim<[usize; N]>>,
    ) -> Array<Self::Physical, Dim<[usize; N]>>
    where
        S: Data<Elem = Self::Spectral>,
    {
        let mut output = self.ndarray_from_shape::<Self::Physical>(self.shape_physical());
        self.backward_inplace(input, &mut output);
        output
    }

    /// Transform spectral -> physical space (inplace)
    ///
    /// # Arguments
    ///
    /// * `input` - *ndarray* with num type of spectral space
    /// * `output` - *ndarray* with num type of physical space
    fn backward_inplace<S1, S2>(
        &self,
        input: &ArrayBase<S1, Dim<[usize; N]>>,
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
        &self,
        input: &ArrayBase<S, Dim<[usize; N]>>,
    ) -> Array<Self::Spectral, Dim<[usize; N]>>
    where
        S: Data<Elem = Self::Physical>,
    {
        let mut output = self.ndarray_from_shape::<Self::Spectral>(self.shape_spectral());
        self.forward_inplace_par(input, &mut output);
        output
    }

    /// Transform physical -> spectral space (inplace)
    ///
    /// # Arguments
    ///
    /// * `input` - *ndarray* with num type of physical space
    /// * `output` - *ndarray* with num type of spectral space
    fn forward_inplace_par<S1, S2>(
        &self,
        input: &ArrayBase<S1, Dim<[usize; N]>>,
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
        &self,
        input: &ArrayBase<S, Dim<[usize; N]>>,
    ) -> Array<Self::Physical, Dim<[usize; N]>>
    where
        S: Data<Elem = Self::Spectral>,
    {
        let mut output = self.ndarray_from_shape::<Self::Physical>(self.shape_physical());
        self.backward_inplace_par(input, &mut output);
        output
    }

    /// Transform spectral -> physical space (inplace)
    ///
    /// # Arguments
    ///
    /// * `input` - *ndarray* with num type of spectral space
    /// * `output` - *ndarray* with num type of physical space
    fn backward_inplace_par<S1, S2>(
        &self,
        input: &ArrayBase<S1, Dim<[usize; N]>>,
        output: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = Self::Spectral>,
        S2: Data<Elem = Self::Physical> + DataMut;
}
