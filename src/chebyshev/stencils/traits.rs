use super::helper::{Helper, HelperDiag012, HelperDiag02, HelperDiag024};
use crate::types::{Real, ScalarOperand};

#[enum_dispatch]
pub(crate) trait StencilOperand<A> {
    fn matvec<T: ScalarOperand<A>>(&self, x: &[T], b: &mut [T]);
    // fn matvec_inplace<T: ScalarOperand<A>>(&self, x: &mut [T]);
    fn solve<T: ScalarOperand<A>>(&self, b: &[T], x: &mut [T]);
    // fn solve_inplace<T: ScalarOperand<A>>(&self, x: &mut [T]);
}
