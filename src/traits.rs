//! # Collection of traits that bases must implement
use crate::enums::{BaseKind, TransformKind};
use crate::utils::{array_resized_axis, check_array_axis};
use ndarray::{Array, Array2, ArrayBase, Axis, Data, DataMut, Dimension, Zip};
use num_traits::identities::Zero;

/// Base super trait
pub trait Base<T>:
    BaseSize
    + BaseMatOpLaplacian
    + BaseMatOpGeneral
    + BaseElements
    + BaseGradient<T>
    + BaseFromOrtho<T>
    + BaseTransform
where
    T: Zero + Copy,
{
}

impl<A, T> Base<T> for A
where
    T: Zero + Copy,
    A: BaseSize
        + BaseMatOpLaplacian
        + BaseMatOpGeneral
        + BaseElements
        + BaseGradient<T>
        + BaseFromOrtho<T>
        + BaseTransform,
{
}

/// Dimensions
pub trait BaseSize {
    /// Size in physical space
    fn len_phys(&self) -> usize;

    /// Size in spectral space
    fn len_spec(&self) -> usize;

    /// Size of orthogonal space
    fn len_orth(&self) -> usize;
}

/// Coordinates and base functions
pub trait BaseElements {
    /// Real valued scalar type
    type RealNum;

    /// Return kind of base
    fn base_kind(&self) -> BaseKind;

    /// Return kind of transform
    fn transform_kind(&self) -> TransformKind;

    /// Coordinates in physical space
    fn coords(&self) -> Vec<Self::RealNum>;
}

/// Collection of matrix operators
pub trait BaseMatOpGeneral {
    /// Real valued scalar type
    type RealNum;

    /// Scalar type of spectral coefficients
    type SpectralNum;

    /// Explicit differential operator $ D $
    ///
    /// Matrix-based version of [`BaseGradient::gradient()`]
    fn diffmat(&self, _deriv: usize) -> Array2<Self::SpectralNum>;

    /// Explicit inverse of differential operator $ D^* $
    ///
    /// Returns ``(D_pinv, I_pinv)``, where `D_pinv` is the pseudoinverse
    /// and ``I_pinv`` the corresponding pseudoidentity matrix, such
    /// that
    ///
    /// ```text
    /// D_pinv @ D = I_pinv
    /// ```
    ///
    /// Can be used as a preconditioner.
    fn diffmat_pinv(&self, _deriv: usize)
        -> (Array2<Self::SpectralNum>, Array2<Self::SpectralNum>);

    /// Transformation stencil composite -> orthogonal space
    fn stencil(&self) -> Array2<Self::RealNum>;

    /// Inverse of transformation stencil
    fn stencil_inv(&self) -> Array2<Self::RealNum>;
}

/// Collection of *Laplacian* matrix operators
pub trait BaseMatOpLaplacian {
    /// Scalar type of laplacian matrix
    type ScalarNum;

    /// Laplacian $ L $
    fn laplace(&self) -> Array2<Self::ScalarNum>;

    /// Pseudoinverse matrix of Laplacian $ L^{-1} $
    ///
    /// Returns pseudoinverse and pseudoidentity,i.e
    /// ``(D_pinv, I_pinv)``
    ///
    /// ```text
    /// D_pinv @ D = I_pinv
    /// ```
    fn laplace_pinv(&self) -> (Array2<Self::ScalarNum>, Array2<Self::ScalarNum>);
}

/// # Transform from orthogonal <-> composite base
pub trait BaseFromOrtho<T>: BaseSize
where
    T: Zero + Copy,
{
    /// ## Composite coefficients -> Orthogonal coefficients
    fn to_ortho_slice(&self, indata: &[T], outdata: &mut [T]);

    /// ## Orthogonal coefficients -> Composite coefficients
    fn from_ortho_slice(&self, indata: &[T], outdata: &mut [T]);

    /// ## Composite coefficients -> Orthogonal coefficients
    fn to_ortho<S, D>(&self, indata: &ArrayBase<S, D>, axis: usize) -> Array<T, D>
    where
        S: Data<Elem = T>,
        D: Dimension,
    {
        let mut outdata = array_resized_axis(indata, self.len_orth(), axis);
        self.to_ortho_inplace(indata, &mut outdata, axis);
        outdata
    }

    apply_along_axis!(
        /// ## Composite coefficients -> Orthogonal coefficients
        to_ortho_inplace,
        T,
        T,
        to_ortho_slice,
        len_spec,
        len_orth,
        "to_ortho"
    );

    /// ## Composite coefficients -> Orthogonal coefficients
    fn from_ortho<S, D>(&self, indata: &ArrayBase<S, D>, axis: usize) -> Array<T, D>
    where
        S: Data<Elem = T>,
        D: Dimension,
    {
        let mut outdata = array_resized_axis(indata, self.len_spec(), axis);
        self.from_ortho_inplace(indata, &mut outdata, axis);
        outdata
    }

    apply_along_axis!(
        /// ## Composite coefficients -> Orthogonal coefficients
        from_ortho_inplace,
        T,
        T,
        from_ortho_slice,
        len_orth,
        len_spec,
        "from_ortho"
    );

    /// ## Composite coefficients -> Orthogonal coefficients  (Parallel)
    fn to_ortho_par<S, D>(&self, indata: &ArrayBase<S, D>, axis: usize) -> Array<T, D>
    where
        S: Data<Elem = T>,
        D: Dimension,
        Self: Sync,
        T: Send + Sync,
    {
        let mut outdata = array_resized_axis(indata, self.len_orth(), axis);
        self.to_ortho_inplace_par(indata, &mut outdata, axis);
        outdata
    }

    par_apply_along_axis!(
        /// ## Composite coefficients -> Orthogonal coefficients (Parallel)
        to_ortho_inplace_par,
        T,
        T,
        to_ortho_slice,
        len_spec,
        len_orth,
        "to_ortho"
    );

    /// ## Composite coefficients -> Orthogonal coefficients  (Parallel)
    fn from_ortho_par<S, D>(&self, indata: &ArrayBase<S, D>, axis: usize) -> Array<T, D>
    where
        S: Data<Elem = T>,
        D: Dimension,
        Self: Sync,
        T: Send + Sync,
    {
        let mut outdata = array_resized_axis(indata, self.len_spec(), axis);
        self.from_ortho_inplace_par(indata, &mut outdata, axis);
        outdata
    }

    par_apply_along_axis!(
        /// ## Composite coefficients -> Orthogonal coefficients (Parallel)
        from_ortho_inplace_par,
        T,
        T,
        from_ortho_slice,
        len_orth,
        len_spec,
        "from_ortho"
    );
}

/// The associated types *Physical* and *Spectral* refer
/// to the scalar types in the physical and spectral space.
/// For example, Fourier transforms from real-to-complex,
/// while Chebyshev transforms from real-to-real.
pub trait BaseTransform: BaseSize {
    /// ## Scalar type in physical space
    type Physical;

    /// ## Scalar type in spectral space
    type Spectral;

    /// ## Physical values -> Spectral coefficients
    ///
    /// Transforms a one-dimensional slice.
    fn forward_slice(&self, indata: &[Self::Physical], outdata: &mut [Self::Spectral]);

    /// ## Spectral coefficients -> Physical values
    ///
    /// Transforms a one-dimensional slice.
    fn backward_slice(&self, indata: &[Self::Spectral], outdata: &mut [Self::Physical]);

    /// ## Physical values -> Spectral coefficients
    fn forward<S, D>(&self, indata: &ArrayBase<S, D>, axis: usize) -> Array<Self::Spectral, D>
    where
        S: Data<Elem = Self::Physical>,
        D: Dimension,
        Self::Physical: Clone,
        Self::Spectral: Zero + Clone + Copy,
    {
        let mut outdata = array_resized_axis(indata, self.len_spec(), axis);
        self.forward_inplace(indata, &mut outdata, axis);
        outdata
    }

    apply_along_axis!(
        /// ## Physical values -> Spectral coefficients
        forward_inplace,
        Self::Physical,
        Self::Spectral,
        forward_slice,
        len_phys,
        len_spec,
        "forward"
    );

    /// ## Spectral coefficients -> Physical values
    fn backward<S, D>(&self, indata: &ArrayBase<S, D>, axis: usize) -> Array<Self::Physical, D>
    where
        S: Data<Elem = Self::Spectral>,
        D: Dimension,
        Self::Spectral: Clone,
        Self::Physical: Zero + Clone + Copy,
    {
        let mut outdata = array_resized_axis(indata, self.len_phys(), axis);
        self.backward_inplace(indata, &mut outdata, axis);
        outdata
    }

    apply_along_axis!(
        /// ## Spectral coefficients -> Physical values
        backward_inplace,
        Self::Spectral,
        Self::Physical,
        backward_slice,
        len_spec,
        len_phys,
        "backward"
    );

    /// ## Physical values -> Spectral coefficients (Parallel)
    fn forward_par<S, D>(&self, indata: &ArrayBase<S, D>, axis: usize) -> Array<Self::Spectral, D>
    where
        S: Data<Elem = Self::Physical>,
        D: Dimension,
        Self::Physical: Clone + Send + Sync,
        Self::Spectral: Zero + Clone + Copy + Send + Sync,
        Self: Sync,
    {
        let mut outdata = array_resized_axis(indata, self.len_spec(), axis);
        self.forward_inplace_par(indata, &mut outdata, axis);
        outdata
    }

    par_apply_along_axis!(
        /// ## Physical values -> Spectral coefficients (Parallel)
        forward_inplace_par,
        Self::Physical,
        Self::Spectral,
        forward_slice,
        len_phys,
        len_spec,
        "forward"
    );

    /// ## Spectral coefficients -> Physical values (Parallel)
    fn backward_par<S, D>(&self, indata: &ArrayBase<S, D>, axis: usize) -> Array<Self::Physical, D>
    where
        S: Data<Elem = Self::Spectral>,
        D: Dimension,
        Self::Spectral: Clone + Send + Sync,
        Self::Physical: Zero + Clone + Copy + Send + Sync,
        Self: Sync,
    {
        let mut outdata = array_resized_axis(indata, self.len_phys(), axis);
        self.backward_inplace_par(indata, &mut outdata, axis);
        outdata
    }

    par_apply_along_axis!(
        /// ## Spectral coefficients -> Physical values (Parallel)
        backward_inplace_par,
        Self::Spectral,
        Self::Physical,
        backward_slice,
        len_spec,
        len_phys,
        "backward"
    );
}

/// # Gradient
pub trait BaseGradient<T>: BaseSize
where
    T: Zero + Copy,
{
    /// Differentiate in spectral space
    fn gradient_slice(&self, indata: &[T], outdata: &mut [T], n_times: usize);

    /// Differentiate in spectral space
    fn gradient<S, D>(&self, indata: &ArrayBase<S, D>, n_times: usize, axis: usize) -> Array<T, D>
    where
        S: Data<Elem = T>,
        D: Dimension,
    {
        let mut outdata = array_resized_axis(indata, self.len_orth(), axis);
        self.gradient_inplace(indata, &mut outdata, n_times, axis);
        outdata
    }

    /// Differentiate in spectral space
    fn gradient_inplace<S1, S2, D>(
        &self,
        indata: &ArrayBase<S1, D>,
        outdata: &mut ArrayBase<S2, D>,
        n_times: usize,
        axis: usize,
    ) where
        S1: Data<Elem = T>,
        S2: Data<Elem = T> + DataMut,
        D: Dimension,
    {
        assert!(indata.is_standard_layout());
        assert!(outdata.is_standard_layout());
        check_array_axis(indata, self.len_spec(), axis, "gradient");
        check_array_axis(outdata, self.len_orth(), axis, "gradient");

        let outer_axis = outdata.ndim() - 1;
        if axis == outer_axis {
            // Data is contiguous in memory
            Zip::from(indata.rows())
                .and(outdata.rows_mut())
                .for_each(|x, mut y| {
                    self.gradient_slice(x.as_slice().unwrap(), y.as_slice_mut().unwrap(), n_times);
                });
        } else {
            // Data is *not* contiguous in memory.
            let mut scratch: Vec<T> = vec![T::zero(); outdata.shape()[axis]];
            Zip::from(indata.lanes(Axis(axis)))
                .and(outdata.lanes_mut(Axis(axis)))
                .for_each(|x, mut y| {
                    self.gradient_slice(&x.to_vec(), &mut scratch, n_times);
                    for (yi, si) in y.iter_mut().zip(scratch.iter()) {
                        *yi = *si;
                    }
                });
        }
    }

    /// Differentiate in spectral space
    fn gradient_par<S, D>(
        &self,
        indata: &ArrayBase<S, D>,
        n_times: usize,
        axis: usize,
    ) -> Array<T, D>
    where
        S: Data<Elem = T>,
        D: Dimension,
        T: Send + Sync,
        Self: Sync,
    {
        let mut outdata = array_resized_axis(indata, self.len_orth(), axis);
        self.gradient_inplace_par(indata, &mut outdata, n_times, axis);
        outdata
    }

    /// Differentiate in spectral space
    fn gradient_inplace_par<S1, S2, D>(
        &self,
        indata: &ArrayBase<S1, D>,
        outdata: &mut ArrayBase<S2, D>,
        n_times: usize,
        axis: usize,
    ) where
        S1: Data<Elem = T>,
        S2: Data<Elem = T> + DataMut,
        D: Dimension,
        T: Send + Sync,
        Self: Sync,
    {
        assert!(indata.is_standard_layout());
        assert!(outdata.is_standard_layout());
        check_array_axis(indata, self.len_spec(), axis, "gradient");
        check_array_axis(outdata, self.len_orth(), axis, "gradient");

        let outer_axis = outdata.ndim() - 1;
        if axis == outer_axis {
            // Data is contiguous in memory
            Zip::from(indata.rows())
                .and(outdata.rows_mut())
                .par_for_each(|x, mut y| {
                    self.gradient_slice(x.as_slice().unwrap(), y.as_slice_mut().unwrap(), n_times);
                });
        } else {
            // Data is *not* contiguous in memory.
            let scratch_len = outdata.shape()[axis];
            Zip::from(indata.lanes(Axis(axis)))
                .and(outdata.lanes_mut(Axis(axis)))
                .par_for_each(|x, mut y| {
                    let mut scratch: Vec<T> = vec![T::zero(); scratch_len];
                    self.gradient_slice(&x.to_vec(), &mut scratch, n_times);
                    for (yi, si) in y.iter_mut().zip(scratch.iter()) {
                        *yi = *si;
                    }
                });
        }
    }
}

// /// Applys function along lanes of *axis*
// fn apply_along_axis<F, S1, S2, D, T1, T2>(
//     &self,
//     indata: &ArrayBase<S1, D>,
//     outdata: &mut ArrayBase<S2, D>,
//     axis: usize,
//     function: &F,
// ) where
//     S1: Data<Elem = T1>,
//     S2: Data<Elem = T2> + DataMut,
//     D: Dimension,
//     F: Fn(&Self, &[T1], &mut [T2]),
//     T1: Clone,
//     T2: Clone + Zero + Copy,
// {
//     let outer_axis = indata.ndim() - 1;
//     if axis == outer_axis {
//         // Data is contiguous in memory
//         Zip::from(indata.lanes(Axis(axis)))
//             .and(outdata.lanes_mut(Axis(axis)))
//             .for_each(|x, mut y| {
//                 function(self, x.as_slice().unwrap(), y.as_slice_mut().unwrap());
//             });
//     } else {
//         // Data is *not* contiguous in memory.
//         let mut scratch: Vec<T2> = vec![T2::zero(); outdata.shape()[axis]];
//         Zip::from(indata.lanes(Axis(axis)))
//             .and(outdata.lanes_mut(Axis(axis)))
//             .for_each(|x, mut y| {
//                 function(self, &x.to_vec(), &mut scratch);
//                 for (yi, si) in y.iter_mut().zip(scratch.iter()) {
//                     *yi = *si;
//                 }
//             });
//     }
// }
