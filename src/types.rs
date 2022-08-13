//! # Custom types
//!
//! `FloatNum`:   Floating Point number
//! `ScalarNum`:  Generic type for linalg operations
use num_complex::Complex;
use num_traits::{Float, FloatConst, FromPrimitive, One, Signed, Zero};
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

/// Generic real number, implemented for f32 and f64
pub trait Real:
    Float
    + FloatConst
    + Debug
    + Send
    + Sync
    + Signed
    + FromPrimitive
    + MulAssign
    + DivAssign
    + AddAssign
    + SubAssign
    + 'static
{
}
impl Real for f32 {}
impl Real for f64 {}

/// Elements that support linear algebra operations.
///
/// `'static` for type-based specialization, `Copy` so that they don't need move
/// semantics or destructors, and the rest are numerical traits.
pub trait Scalar:
    'static
    + Copy
    + Zero
    + One
    + Debug
    + Send
    + Sync
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

impl Scalar for f32 {}
impl Scalar for f64 {}
impl Scalar for Complex<f32> {}
impl Scalar for Complex<f64> {}

// /// Generic floating point number, implemented for f32 and f64
// pub trait RealNum:
//     Float
//     + FloatConst
//     + Debug
//     + Send
//     + Sync
//     + Signed
//     + FromPrimitive
//     + DctNum
//     + MulAssign
//     + DivAssign
//     + AddAssign
//     + SubAssign
// {
// }
// impl RealNum for f32 {}
// impl RealNum for f64 {}

pub trait ScalarOperand<A>:
    Scalar
    + Add<A, Output = Self>
    + Sub<A, Output = Self>
    + Mul<A, Output = Self>
    + Div<A, Output = Self>
{
}

impl<A> ScalarOperand<A> for A where A: Scalar {}
