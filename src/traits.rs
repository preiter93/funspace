//! # Collection of traits that bases must implement
use crate::enums::BaseKind;
use crate::types::ScalarNum;
use ndarray::{Array, Array2, ArrayBase, Axis, Data, DataMut, Dimension, Zip};
use num_traits::identities::Zero;
use std::ops::{Add, Div, Mul, Sub};

// pub trait FunspaceElemental {
//     /// Coordinates in physical space
//     fn nodes<T: FloatNum>(&self) -> &[T];
// }
/// Transform from physical to spectral space and vice versa.
///

/// More helpful functions with function spaces
pub trait FunspaceExtended {
    /// Real valued scalar type
    type Real;

    /// Scalar type in spectral space
    type Spectral;

    /// Return kind of base
    fn base_kind(&self) -> BaseKind;

    /// Coordinates in physical space
    fn get_nodes(&self) -> Vec<Self::Real>;

    /// Mass matrix (=Stencil for non-orthogonal matrices)
    fn mass(&self) -> Array2<Self::Real>;

    /// Inverse of mass matrix, might be unimplemented
    fn mass_inv(&self) -> Array2<Self::Real>;

    /// Explicit differential operator $ D $
    fn diffmat(&self, deriv: usize) -> Array2<Self::Spectral>;

    /// Laplacian $ L $
    fn laplace(&self) -> Array2<Self::Real>;

    /// Pseudoinverse mtrix of Laplacian $ L^{-1} $
    fn laplace_inv(&self) -> Array2<Self::Real>;

    /// Pseudoidentity matrix of laplacian $ L^{-1} L $
    fn laplace_inv_eye(&self) -> Array2<Self::Real>;
}

pub trait FunspaceSize {
    /// Size in physical space
    fn len_phys(&self) -> usize;

    /// Size in spectral space
    fn len_spec(&self) -> usize;

    /// Size of orthogonal space
    fn len_orth(&self) -> usize;
}

/// The associated types *Physical* and *Spectral* refer
/// to the scalar types in the physical and spectral space.
/// For example, Fourier transforms from real-to-complex,
/// while Chebyshev transforms from real-to-real.
pub trait FunspaceElemental: FunspaceSize {
    /// Scalar type in physical space
    type Physical;

    /// Scalar type in spectral space
    type Spectral;

    /// Physical values -> Spectral coefficients
    ///
    /// Transforms a one-dimensional slice.
    fn forward_slice(&self, indata: &[Self::Physical], outdata: &mut [Self::Spectral]);

    /// Spectral coefficients -> Physical values
    ///
    /// Transforms a one-dimensional slice.
    fn backward_slice(&self, indata: &[Self::Spectral], outdata: &mut [Self::Physical]);

    /// Differentiate in spectral space
    ///
    /// Type can deviate from Spectral type
    fn differentiate_slice<T>(&self, indata: &[T], outdata: &mut [T], n_times: usize)
    where
        T: ScalarNum
            + Add<Self::Spectral, Output = T>
            + Mul<Self::Spectral, Output = T>
            + Div<Self::Spectral, Output = T>
            + Sub<Self::Spectral, Output = T>;

    /// Physical values -> Spectral coefficients
    fn forward<S, D>(&self, indata: &ArrayBase<S, D>, axis: usize) -> Array<Self::Spectral, D>
    where
        S: Data<Elem = Self::Physical>,
        D: Dimension,
        Self::Physical: Clone,
        Self::Spectral: Zero + Clone + Copy,
    {
        use crate::utils::array_resized_axis;
        let mut outdata = array_resized_axis(indata, self.len_spec(), axis);
        self.forward_inplace(indata, &mut outdata, axis);
        outdata
    }

    /// Physical values -> Spectral coefficients
    fn forward_par<S, D>(&self, indata: &ArrayBase<S, D>, axis: usize) -> Array<Self::Spectral, D>
    where
        S: Data<Elem = Self::Physical>,
        D: Dimension,
        Self::Physical: Clone + Send + Sync,
        Self::Spectral: Clone + Zero + Copy + Send + Sync,
        Self: Sync,
    {
        use crate::utils::array_resized_axis;
        let mut outdata = array_resized_axis(indata, self.len_spec(), axis);
        self.forward_inplace_par(indata, &mut outdata, axis);
        outdata
    }

    /// Physical values -> Spectral coefficients
    ///
    /// Transforms (multidimensional) `ndarray` along `axis`.
    /// `axis` must be smaller than `ndim` - 1
    fn forward_inplace<S1, S2, D>(
        &self,
        indata: &ArrayBase<S1, D>,
        outdata: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: Data<Elem = Self::Physical>,
        S2: Data<Elem = Self::Spectral> + DataMut,
        D: Dimension,
        Self::Physical: Clone,
        Self::Spectral: Clone + Zero + Copy,
    {
        assert!(indata.is_standard_layout());
        assert!(outdata.is_standard_layout());

        let outer_axis = indata.ndim() - 1;
        if axis == outer_axis {
            // Data is contiguous in memory
            Zip::from(indata.rows())
                .and(outdata.rows_mut())
                .for_each(|x, mut y| {
                    self.forward_slice(x.as_slice().unwrap(), y.as_slice_mut().unwrap());
                });
        } else {
            // Data is *not* contiguous in memory.
            let scratch_size = outdata.shape()[axis];
            let mut scratch: Vec<Self::Spectral> = vec![Self::Spectral::zero(); scratch_size];
            Zip::from(indata.lanes(Axis(axis)))
                .and(outdata.lanes_mut(Axis(axis)))
                .for_each(|x, mut y| {
                    self.forward_slice(&x.to_vec(), &mut scratch);
                    for (yi, si) in y.iter_mut().zip(scratch.iter()) {
                        *yi = *si;
                    }
                });
        }
    }

    /// Physical values -> Spectral coefficients
    ///
    /// Transforms (multidimensional) `ndarray` along `axis`.
    /// `axis` must be smaller than `ndim` - 1
    fn forward_inplace_par<S1, S2, D>(
        &self,
        indata: &ArrayBase<S1, D>,
        outdata: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: Data<Elem = Self::Physical>,
        S2: Data<Elem = Self::Spectral> + DataMut,
        D: Dimension,
        Self::Physical: Clone + Send + Sync,
        Self::Spectral: Clone + Zero + Copy + Send + Sync,
        Self: Sync,
    {
        assert!(indata.is_standard_layout());
        assert!(outdata.is_standard_layout());

        let outer_axis = indata.ndim() - 1;
        if axis == outer_axis {
            // Data is contiguous in memory
            Zip::from(indata.rows())
                .and(outdata.rows_mut())
                .par_for_each(|x, mut y| {
                    self.forward_slice(x.as_slice().unwrap(), y.as_slice_mut().unwrap());
                });
        } else {
            // Data is *not* contiguous in memory.
            let scratch_size = outdata.shape()[axis];
            Zip::from(indata.lanes(Axis(axis)))
                .and(outdata.lanes_mut(Axis(axis)))
                .par_for_each(|x, mut y| {
                    let mut scratch: Vec<Self::Spectral> =
                        vec![Self::Spectral::zero(); scratch_size];
                    self.forward_slice(&x.to_vec(), &mut scratch);
                    for (yi, si) in y.iter_mut().zip(scratch.iter()) {
                        *yi = *si;
                    }
                });
        }
    }

    /// Spectral coefficients -> Physical values
    fn backward<S, D>(&self, indata: &ArrayBase<S, D>, axis: usize) -> Array<Self::Physical, D>
    where
        S: Data<Elem = Self::Spectral>,
        D: Dimension,
        Self::Spectral: Clone,
        Self::Physical: Zero + Clone + Copy,
    {
        use crate::utils::array_resized_axis;
        let mut outdata = array_resized_axis(indata, self.len_phys(), axis);
        self.backward_inplace(indata, &mut outdata, axis);
        outdata
    }

    /// Spectral coefficients -> Physical values
    fn backward_par<S, D>(&self, indata: &ArrayBase<S, D>, axis: usize) -> Array<Self::Physical, D>
    where
        S: Data<Elem = Self::Spectral>,
        D: Dimension,
        Self::Spectral: Clone + Send + Sync,
        Self::Physical: Clone + Zero + Copy + Send + Sync,
        Self: Sync,
    {
        use crate::utils::array_resized_axis;
        let mut outdata = array_resized_axis(indata, self.len_phys(), axis);
        self.backward_inplace_par(indata, &mut outdata, axis);
        outdata
    }

    /// Spectral coefficients -> Physical values
    ///
    /// Transforms (multidimensional) `ndarray` along `axis`.
    /// `axis` must be smaller than `ndim` - 1
    fn backward_inplace<S1, S2, D>(
        &self,
        indata: &ArrayBase<S1, D>,
        outdata: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: Data<Elem = Self::Spectral>,
        S2: Data<Elem = Self::Physical> + DataMut,
        D: Dimension,
        Self::Spectral: Clone,
        Self::Physical: Clone + Zero + Copy,
    {
        assert!(indata.is_standard_layout());
        assert!(outdata.is_standard_layout());

        let outer_axis = indata.ndim() - 1;
        if axis == outer_axis {
            // Data is contiguous in memory
            Zip::from(indata.rows())
                .and(outdata.rows_mut())
                .for_each(|x, mut y| {
                    self.backward_slice(x.as_slice().unwrap(), y.as_slice_mut().unwrap());
                });
        } else {
            // Data is *not* contiguous in memory.
            let mut scratch: Vec<Self::Physical> =
                vec![Self::Physical::zero(); outdata.shape()[axis]];
            Zip::from(indata.lanes(Axis(axis)))
                .and(outdata.lanes_mut(Axis(axis)))
                .for_each(|x, mut y| {
                    self.backward_slice(&x.to_vec(), &mut scratch);
                    for (yi, si) in y.iter_mut().zip(scratch.iter()) {
                        *yi = *si;
                    }
                });
        }
    }

    /// Spectral coefficients -> Physical values
    ///
    /// Transforms (multidimensional) `ndarray` along `axis`.
    /// `axis` must be smaller than `ndim` - 1
    fn backward_inplace_par<S1, S2, D>(
        &self,
        indata: &ArrayBase<S1, D>,
        outdata: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: Data<Elem = Self::Spectral>,
        S2: Data<Elem = Self::Physical> + DataMut,
        D: Dimension,
        Self::Spectral: Clone + Send + Sync,
        Self::Physical: Clone + Zero + Copy + Send + Sync,
        Self: Sync,
    {
        assert!(indata.is_standard_layout());
        assert!(outdata.is_standard_layout());

        let outer_axis = indata.ndim() - 1;
        if axis == outer_axis {
            // Data is contiguous in memory
            Zip::from(indata.lanes(Axis(axis)))
                .and(outdata.lanes_mut(Axis(axis)))
                .par_for_each(|x, mut y| {
                    self.backward_slice(x.as_slice().unwrap(), y.as_slice_mut().unwrap());
                });
        } else {
            // Data is *not* contiguous in memory.
            let scratch_len = outdata.shape()[axis];
            Zip::from(indata.lanes(Axis(axis)))
                .and(outdata.lanes_mut(Axis(axis)))
                .par_for_each(|x, mut y| {
                    let mut scratch: Vec<Self::Physical> =
                        vec![Self::Physical::zero(); scratch_len];
                    self.backward_slice(&x.to_vec(), &mut scratch);
                    for (yi, si) in y.iter_mut().zip(scratch.iter()) {
                        *yi = *si;
                    }
                });
        }
    }

    /// Differentiate in spectral space
    fn differentiate<T, S, D>(
        &self,
        indata: &ArrayBase<S, D>,
        n_times: usize,
        axis: usize,
    ) -> Array<T, D>
    where
        T: ScalarNum
            + Add<Self::Spectral, Output = T>
            + Mul<Self::Spectral, Output = T>
            + Div<Self::Spectral, Output = T>
            + Sub<Self::Spectral, Output = T>,
        S: Data<Elem = T>,
        D: Dimension,
    {
        use crate::utils::array_resized_axis;
        let mut outdata = array_resized_axis(indata, self.len_orth(), axis);
        self.differentiate_inplace(indata, &mut outdata, n_times, axis);
        outdata
    }

    /// Differentiate in spectral space
    fn differentiate_inplace<T, S1, S2, D>(
        &self,
        indata: &ArrayBase<S1, D>,
        outdata: &mut ArrayBase<S2, D>,
        n_times: usize,
        axis: usize,
    ) where
        T: ScalarNum
            + Add<Self::Spectral, Output = T>
            + Mul<Self::Spectral, Output = T>
            + Div<Self::Spectral, Output = T>
            + Sub<Self::Spectral, Output = T>,
        S1: Data<Elem = T>,
        S2: Data<Elem = T> + DataMut,
        D: Dimension,
    {
        assert!(indata.is_standard_layout());
        assert!(outdata.is_standard_layout());
        let outer_axis = outdata.ndim() - 1;
        if axis == outer_axis {
            // Data is contiguous in memory
            Zip::from(indata.rows())
                .and(outdata.rows_mut())
                .for_each(|x, mut y| {
                    self.differentiate_slice(
                        x.as_slice().unwrap(),
                        y.as_slice_mut().unwrap(),
                        n_times,
                    );
                });
        } else {
            // Data is *not* contiguous in memory.
            let mut scratch: Vec<T> = vec![T::zero(); outdata.shape()[axis]];
            Zip::from(indata.lanes(Axis(axis)))
                .and(outdata.lanes_mut(Axis(axis)))
                .for_each(|x, mut y| {
                    self.differentiate_slice(&x.to_vec(), &mut scratch, n_times);
                    for (yi, si) in y.iter_mut().zip(scratch.iter()) {
                        *yi = *si;
                    }
                });
        }
    }

    /// Differentiate in spectral space
    fn differentiate_par<T, S, D>(
        &self,
        indata: &ArrayBase<S, D>,
        n_times: usize,
        axis: usize,
    ) -> Array<T, D>
    where
        T: ScalarNum
            + Add<Self::Spectral, Output = T>
            + Mul<Self::Spectral, Output = T>
            + Div<Self::Spectral, Output = T>
            + Sub<Self::Spectral, Output = T>,
        S: Data<Elem = T>,
        D: Dimension,
        T: Send + Sync,
        Self: Sync,
    {
        use crate::utils::array_resized_axis;
        let mut outdata = array_resized_axis(indata, self.len_orth(), axis);
        self.differentiate_inplace_par(indata, &mut outdata, n_times, axis);
        outdata
    }

    /// Differentiate in spectral space
    fn differentiate_inplace_par<T, S1, S2, D>(
        &self,
        indata: &ArrayBase<S1, D>,
        outdata: &mut ArrayBase<S2, D>,
        n_times: usize,
        axis: usize,
    ) where
        T: ScalarNum
            + Add<Self::Spectral, Output = T>
            + Mul<Self::Spectral, Output = T>
            + Div<Self::Spectral, Output = T>
            + Sub<Self::Spectral, Output = T>,
        S1: Data<Elem = T>,
        S2: Data<Elem = T> + DataMut,
        D: Dimension,
        T: Send + Sync,
        Self: Sync,
    {
        assert!(indata.is_standard_layout());
        assert!(outdata.is_standard_layout());
        let outer_axis = outdata.ndim() - 1;
        if axis == outer_axis {
            // Data is contiguous in memory
            Zip::from(indata.rows())
                .and(outdata.rows_mut())
                .par_for_each(|x, mut y| {
                    self.differentiate_slice(
                        x.as_slice().unwrap(),
                        y.as_slice_mut().unwrap(),
                        n_times,
                    );
                });
        } else {
            // Data is *not* contiguous in memory.
            let scratch_len = outdata.shape()[axis];
            Zip::from(indata.lanes(Axis(axis)))
                .and(outdata.lanes_mut(Axis(axis)))
                .par_for_each(|x, mut y| {
                    let mut scratch: Vec<T> = vec![T::zero(); scratch_len];
                    self.differentiate_slice(&x.to_vec(), &mut scratch, n_times);
                    for (yi, si) in y.iter_mut().zip(scratch.iter()) {
                        *yi = *si;
                    }
                });
        }
    }

    /// Composite space coefficients -> Orthogonal space coefficients
    ///
    /// Todo: Are all these trait bounds necessary?
    fn to_ortho_slice<T>(&self, indata: &[T], outdata: &mut [T])
    where
        T: ScalarNum
            + Add<Self::Spectral, Output = T>
            + Mul<Self::Spectral, Output = T>
            + Div<Self::Spectral, Output = T>
            + Sub<Self::Spectral, Output = T>;

    /// Orthogonal space coefficients -> Composite space coefficients
    ///
    /// Todo: Are all these trait bounds necessary?
    fn from_ortho_slice<T>(&self, indata: &[T], outdata: &mut [T])
    where
        T: ScalarNum
            + Add<Self::Spectral, Output = T>
            + Mul<Self::Spectral, Output = T>
            + Div<Self::Spectral, Output = T>
            + Sub<Self::Spectral, Output = T>;

    /// Composite space coefficients -> Orthogonal space coefficients
    fn to_ortho<T, S, D>(&self, indata: &ArrayBase<S, D>, axis: usize) -> Array<T, D>
    where
        T: ScalarNum
            + Add<Self::Spectral, Output = T>
            + Mul<Self::Spectral, Output = T>
            + Div<Self::Spectral, Output = T>
            + Sub<Self::Spectral, Output = T>,
        S: Data<Elem = T>,
        D: Dimension,
    {
        use crate::utils::array_resized_axis;
        let mut outdata = array_resized_axis(indata, self.len_orth(), axis);
        self.to_ortho_inplace(indata, &mut outdata, axis);
        outdata
    }

    /// Composite space coefficients -> Orthogonal space coefficients
    fn to_ortho_inplace<T, S1, S2, D>(
        &self,
        indata: &ArrayBase<S1, D>,
        outdata: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        T: ScalarNum
            + Add<Self::Spectral, Output = T>
            + Mul<Self::Spectral, Output = T>
            + Div<Self::Spectral, Output = T>
            + Sub<Self::Spectral, Output = T>,
        S1: Data<Elem = T>,
        S2: Data<Elem = T> + DataMut,
        D: Dimension,
    {
        assert!(indata.is_standard_layout());
        assert!(outdata.is_standard_layout());

        let outer_axis = indata.ndim() - 1;
        if axis == outer_axis {
            // Data is contiguous in memory
            Zip::from(indata.lanes(Axis(axis)))
                .and(outdata.lanes_mut(Axis(axis)))
                .for_each(|x, mut y| {
                    self.to_ortho_slice(x.as_slice().unwrap(), y.as_slice_mut().unwrap());
                });
        } else {
            // Data is *not* contiguous in memory.
            let mut scratch: Vec<T> = vec![T::zero(); outdata.shape()[axis]];
            Zip::from(indata.lanes(Axis(axis)))
                .and(outdata.lanes_mut(Axis(axis)))
                .for_each(|x, mut y| {
                    self.to_ortho_slice(&x.to_vec(), &mut scratch);
                    for (yi, si) in y.iter_mut().zip(scratch.iter()) {
                        *yi = *si;
                    }
                });
        }
    }

    /// Composite space coefficients -> Orthogonal space coefficients
    fn to_ortho_par<T, S, D>(&self, indata: &ArrayBase<S, D>, axis: usize) -> Array<T, D>
    where
        T: ScalarNum
            + Add<Self::Spectral, Output = T>
            + Mul<Self::Spectral, Output = T>
            + Div<Self::Spectral, Output = T>
            + Sub<Self::Spectral, Output = T>
            + Send
            + Sync,
        S: Data<Elem = T>,
        D: Dimension,
        Self: Sync,
    {
        use crate::utils::array_resized_axis;
        let mut outdata = array_resized_axis(indata, self.len_orth(), axis);
        self.to_ortho_inplace_par(indata, &mut outdata, axis);
        outdata
    }

    /// Composite space coefficients -> Orthogonal space coefficients
    fn to_ortho_inplace_par<T, S1, S2, D>(
        &self,
        indata: &ArrayBase<S1, D>,
        outdata: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        T: ScalarNum
            + Add<Self::Spectral, Output = T>
            + Mul<Self::Spectral, Output = T>
            + Div<Self::Spectral, Output = T>
            + Sub<Self::Spectral, Output = T>
            + Send
            + Sync,
        S1: Data<Elem = T>,
        S2: Data<Elem = T> + DataMut,
        D: Dimension,
        Self: Sync,
    {
        assert!(indata.is_standard_layout());
        assert!(outdata.is_standard_layout());

        let outer_axis = indata.ndim() - 1;
        if axis == outer_axis {
            // Data is contiguous in memory
            Zip::from(indata.lanes(Axis(axis)))
                .and(outdata.lanes_mut(Axis(axis)))
                .par_for_each(|x, mut y| {
                    self.to_ortho_slice(x.as_slice().unwrap(), y.as_slice_mut().unwrap());
                });
        } else {
            // Data is *not* contiguous in memory.

            let scratch_len = outdata.shape()[axis];
            Zip::from(indata.lanes(Axis(axis)))
                .and(outdata.lanes_mut(Axis(axis)))
                .par_for_each(|x, mut y| {
                    let mut scratch: Vec<T> = vec![T::zero(); scratch_len];
                    self.to_ortho_slice(&x.to_vec(), &mut scratch);
                    for (yi, si) in y.iter_mut().zip(scratch.iter()) {
                        *yi = *si;
                    }
                });
        }
    }

    /// Composite space coefficients -> Orthogonal space coefficients
    fn from_ortho<T, S, D>(&self, indata: &ArrayBase<S, D>, axis: usize) -> Array<T, D>
    where
        T: ScalarNum
            + Add<Self::Spectral, Output = T>
            + Mul<Self::Spectral, Output = T>
            + Div<Self::Spectral, Output = T>
            + Sub<Self::Spectral, Output = T>,
        S: Data<Elem = T>,
        D: Dimension,
    {
        use crate::utils::array_resized_axis;
        let mut outdata = array_resized_axis(indata, self.len_spec(), axis);
        self.from_ortho_inplace(indata, &mut outdata, axis);
        outdata
    }

    /// Orthogonal space coefficients -> Composite space coefficients
    fn from_ortho_inplace<T, S1, S2, D>(
        &self,
        indata: &ArrayBase<S1, D>,
        outdata: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        T: ScalarNum
            + Add<Self::Spectral, Output = T>
            + Mul<Self::Spectral, Output = T>
            + Div<Self::Spectral, Output = T>
            + Sub<Self::Spectral, Output = T>,
        S1: Data<Elem = T>,
        S2: Data<Elem = T> + DataMut,
        D: Dimension,
    {
        assert!(indata.is_standard_layout());
        assert!(outdata.is_standard_layout());

        let outer_axis = indata.ndim() - 1;
        if axis == outer_axis {
            // Data is contiguous in memory
            Zip::from(indata.lanes(Axis(axis)))
                .and(outdata.lanes_mut(Axis(axis)))
                .for_each(|x, mut y| {
                    self.from_ortho_slice(x.as_slice().unwrap(), y.as_slice_mut().unwrap());
                });
        } else {
            // Data is *not* contiguous in memory.
            let mut scratch: Vec<T> = vec![T::zero(); outdata.shape()[axis]];
            Zip::from(indata.lanes(Axis(axis)))
                .and(outdata.lanes_mut(Axis(axis)))
                .for_each(|x, mut y| {
                    self.from_ortho_slice(&x.to_vec(), &mut scratch);
                    for (yi, si) in y.iter_mut().zip(scratch.iter()) {
                        *yi = *si;
                    }
                });
        }
    }

    /// Composite space coefficients -> Orthogonal space coefficients
    fn from_ortho_par<T, S, D>(&self, indata: &ArrayBase<S, D>, axis: usize) -> Array<T, D>
    where
        T: ScalarNum
            + Add<Self::Spectral, Output = T>
            + Mul<Self::Spectral, Output = T>
            + Div<Self::Spectral, Output = T>
            + Sub<Self::Spectral, Output = T>
            + Send
            + Sync,
        S: Data<Elem = T>,
        D: Dimension,
        Self: Sync,
    {
        use crate::utils::array_resized_axis;
        let mut outdata = array_resized_axis(indata, self.len_spec(), axis);
        self.from_ortho_inplace_par(indata, &mut outdata, axis);
        outdata
    }

    /// Orthogonal space coefficients -> Composite space coefficients
    fn from_ortho_inplace_par<T, S1, S2, D>(
        &self,
        indata: &ArrayBase<S1, D>,
        outdata: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        T: ScalarNum
            + Add<Self::Spectral, Output = T>
            + Mul<Self::Spectral, Output = T>
            + Div<Self::Spectral, Output = T>
            + Sub<Self::Spectral, Output = T>
            + Send
            + Sync,
        S1: Data<Elem = T>,
        S2: Data<Elem = T> + DataMut,
        D: Dimension,
        Self: Sync,
    {
        assert!(indata.is_standard_layout());
        assert!(outdata.is_standard_layout());

        let outer_axis = indata.ndim() - 1;
        if axis == outer_axis {
            // Data is contiguous in memory
            Zip::from(indata.lanes(Axis(axis)))
                .and(outdata.lanes_mut(Axis(axis)))
                .par_for_each(|x, mut y| {
                    self.from_ortho_slice(x.as_slice().unwrap(), y.as_slice_mut().unwrap());
                });
        } else {
            // Data is *not* contiguous in memory.
            let scratch_len = outdata.shape()[axis];
            Zip::from(indata.lanes(Axis(axis)))
                .and(outdata.lanes_mut(Axis(axis)))
                .par_for_each(|x, mut y| {
                    let mut scratch: Vec<T> = vec![T::zero(); scratch_len];
                    self.from_ortho_slice(&x.to_vec(), &mut scratch);
                    for (yi, si) in y.iter_mut().zip(scratch.iter()) {
                        *yi = *si;
                    }
                });
        }
    }
}
