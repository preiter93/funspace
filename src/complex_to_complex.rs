//! Enum for all bases who transform complex-to-complex
//!
//! Complex-to-complex transforms implement the differentiate and
//! from ortho trait only for complex numbers
use crate::fourier::FourierC2c;
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
pub enum BaseC2c<T: FloatNum> {
    FourierC2c(FourierC2c<T>),
}

impl_transform_trait_for_base!(BaseC2c, Complex<A>, Complex<A>, FourierC2c);

impl_differentiate_trait_for_base!(BaseC2c, Complex<A>, FourierC2c);

impl_from_ortho_trait_for_base!(BaseC2c, Complex<A>, FourierC2c);
