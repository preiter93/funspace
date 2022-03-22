//! # Custom types
//!
//! `FloatNum`: 	Floating Point number
//! `ScalarNum`: 	Generic type for linalg operations
use num_traits::{Float, FloatConst, FromPrimitive, Signed};
use rustdct::DctNum;
use std::fmt::Debug;
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

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
