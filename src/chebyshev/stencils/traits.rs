//! # Stencil traits
//!
//! A stencil $S$ transforms from orthogonal space `u` to composite space `v`, i.e.
//! $$$
//! u = S v
//! $$$
//!
//! Most importantly, a stencil type must implement
//! transforms v -> u and u-> v.
use ndarray::Array2;
use num_traits::Zero;
use std::clone::Clone;
use std::ops::{Add, Div, Mul, Sub};

/// Elementary methods for stencils
#[enum_dispatch]
pub trait StencilOperations<A> {
    /// Multiply stencil with a 1d vector
    fn dot_inplace<T>(&self, v: &[T], u: &mut [T])
    where
        T: Mul<Output = T>
            + Sub<Output = T>
            + Div<Output = T>
            + Add<Output = T>
            + Add<A, Output = T>
            + Mul<A, Output = T>
            + Div<A, Output = T>
            + Sub<A, Output = T>
            + Zero
            + Clone
            + Copy;

    /// Solve linear system $S v = u$
    fn solve_inplace<T>(&self, u: &[T], v: &mut [T])
    where
        T: Mul<Output = T>
            + Sub<Output = T>
            + Div<Output = T>
            + Add<Output = T>
            + Add<A, Output = T>
            + Mul<A, Output = T>
            + Div<A, Output = T>
            + Sub<A, Output = T>
            + Zero
            + Clone
            + Copy;

    /// Return stencil as 2d array
    fn to_array(&self) -> Array2<A>;

    /// Return pseudo inverse of stencil as 2d array
    ///
    /// Might be unimplemented!
    fn pinv(&self) -> Array2<A>;
}
