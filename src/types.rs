//! # Custom types
//!
//! `FloatNum`:   Floating Point number
//! `ScalarNum`:  Generic type for linalg operations
use num_complex::Complex;
use num_traits::{Float, FloatConst, FromPrimitive, One, Signed, Zero};
use rustdct::DctNum;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

/// Generic floating point number, implemented for f32 and f64
pub trait FloatNum: Float
// DCT
+ FloatConst
+ Debug
+ Send
+ Sync
+ Signed
+ FromPrimitive
+ DctNum
// Assign Ops
+ MulAssign
+ DivAssign
+ AddAssign
+ SubAssign
{
}
impl FloatNum for f32 {}
impl FloatNum for f64 {}

/// Elements that support linear algebra operations.
///
/// `'static` for type-based specialization, `Copy` so that they don't need move
/// semantics or destructors, and the rest are numerical traits.
pub trait ScalarNum:
    'static
    + Copy
    + Zero
    + One
    + FromPrimitive
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + MulAssign
    + DivAssign
    + AddAssign
    + SubAssign
{
}

impl ScalarNum for f32 {}
impl ScalarNum for f64 {}
impl ScalarNum for Complex<f32> {}
impl ScalarNum for Complex<f64> {}
