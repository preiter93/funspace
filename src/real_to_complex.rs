//! Enum for all bases who transform real-to-complex
//!
//! Real-to-complex transforms implement the differentiate and
//! from ortho trait only for complex numbers
use crate::fourier::FourierR2c;
use crate::traits::Differentiate;
use crate::traits::FromOrtho;
use crate::traits::FromOrthoPar;
use crate::traits::Transform;
use crate::traits::TransformPar;
use crate::types::FloatNum;
use ndarray::prelude::*;
use num_complex::Complex;

#[enum_dispatch(BaseBasics<T>, LaplacianInverse<T>)]
#[derive(Clone)]
pub enum BaseR2c<T: FloatNum> {
    FourierR2c(FourierR2c<T>),
}

impl_transform_trait_for_base!(BaseR2c, A, Complex<A>, FourierR2c);

impl_differentiate_trait_for_base!(BaseR2c, Complex<A>, FourierR2c);

impl_from_ortho_trait_for_base!(BaseR2c, Complex<A>, FourierR2c);
