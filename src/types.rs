//! Collection of Types for `funspace`
//!
//! FloatNum: Floating Point number
//! Scalar: Generic type for linalg operations
use num_traits::{Float, FromPrimitive, Signed};
use num_traits::{One, Zero};
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};
// use std::ops::{AddAssign, DivAssign, MulAssign, RemAssign, SubAssign};
use ndarray::ScalarOperand;
pub use num_complex::Complex;

/// Generic floating point number, implemented for f32 and f64
pub trait FloatNum:
    Copy + Zero + FromPrimitive + Signed + Sync + Send + Float + Debug + 'static + ScalarOperand
{
}
impl FloatNum for f32 {}
impl FloatNum for f64 {}

/// Elements that support linear algebra operations.
///
/// `'static` for type-based specialization, `Copy` so that they don't need move
/// semantics or destructors, and the rest are numerical traits.
pub trait Scalar:
    'static
    + Copy
    + Zero
    + One
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
{
}

impl<T> Scalar for T where
    T: 'static
        + Copy
        + Zero
        + One
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
{
}
