//! Enum for all bases who transform real-to-real
//!
//! Real-to-real transforms implement the differentiate and
//! from ortho trait for both, real and complex values
use crate::chebyshev::Chebyshev;
use crate::chebyshev::CompositeChebyshev;
use crate::traits::Differentiate;
use crate::traits::FromOrtho;
use crate::traits::FromOrthoPar;
use crate::traits::Transform;
use crate::traits::TransformPar;
use crate::types::FloatNum;
use ndarray::prelude::*;
use num_complex::Complex;

#[allow(clippy::large_enum_variant)]
#[enum_dispatch(BaseBasics<T>, LaplacianInverse<T>)]
#[derive(Clone)]
pub enum BaseR2r<T: FloatNum> {
    Chebyshev(Chebyshev<T>),
    CompositeChebyshev(CompositeChebyshev<T>),
}

impl_transform_trait_for_base!(BaseR2r, A, A, Chebyshev, CompositeChebyshev);

impl_differentiate_trait_for_base!(BaseR2r, A, Chebyshev, CompositeChebyshev);
impl_differentiate_trait_for_base!(BaseR2r, Complex<A>, Chebyshev, CompositeChebyshev);

impl_from_ortho_trait_for_base!(BaseR2r, A, Chebyshev, CompositeChebyshev);
impl_from_ortho_trait_for_base!(BaseR2r, Complex<A>, Chebyshev, CompositeChebyshev);
