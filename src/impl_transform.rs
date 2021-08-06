//! Implement transform trait.
use crate::Base;
use crate::FloatNum;
use crate::Transform;
use crate::TransformPar;
use ndarray::prelude::*;
use num_complex::Complex;

/// Implement for Real-to-real
impl<A: FloatNum> Transform<A, A> for Base<A> {
    type Physical = A;
    type Spectral = A;

    fn forward<S, D>(
        &mut self,
        input: &mut ArrayBase<S, D>,
        axis: usize,
    ) -> Array<Self::Spectral, D>
    where
        S: ndarray::Data<Elem = Self::Physical>,
        D: Dimension,
    {
        match self {
            Self::Chebyshev(ref mut b) => b.forward(input, axis),
            Self::CompositeChebyshev(ref mut b) => b.forward(input, axis),
            Self::Fourier(_) | Self::FourierR2c(_) => {
                panic!("Expected real-to-real transform, but Fourier is complex-to-complex.")
            }
        }
    }

    fn forward_inplace<S1, S2, D>(
        &mut self,
        input: &mut ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = Self::Physical>,
        S2: ndarray::Data<Elem = Self::Spectral> + ndarray::DataMut,
        D: Dimension,
    {
        match self {
            Self::Chebyshev(ref mut b) => b.forward_inplace(input, output, axis),
            Self::CompositeChebyshev(ref mut b) => b.forward_inplace(input, output, axis),
            Self::Fourier(_) | Self::FourierR2c(_) => {
                panic!("Expected real-to-real transform, but Fourier is complex-to-complex.")
            }
        }
    }

    fn backward<S, D>(
        &mut self,
        input: &mut ArrayBase<S, D>,
        axis: usize,
    ) -> Array<Self::Physical, D>
    where
        S: ndarray::Data<Elem = Self::Spectral>,
        D: Dimension,
    {
        match self {
            Self::Chebyshev(ref mut b) => b.backward(input, axis),
            Self::CompositeChebyshev(ref mut b) => b.backward(input, axis),
            Self::Fourier(_) | Self::FourierR2c(_) => {
                panic!("Expected real-to-real transform, but Fourier is complex-to-complex.")
            }
        }
    }

    fn backward_inplace<S1, S2, D>(
        &mut self,
        input: &mut ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = Self::Spectral>,
        S2: ndarray::Data<Elem = Self::Physical> + ndarray::DataMut,
        D: Dimension,
    {
        match self {
            Self::Chebyshev(ref mut b) => b.backward_inplace(input, output, axis),
            Self::CompositeChebyshev(ref mut b) => b.backward_inplace(input, output, axis),
            Self::Fourier(_) | Self::FourierR2c(_) => {
                panic!("Expected real-to-real transform, but Fourier is complex-to-complex.")
            }
        }
    }
}

/// Implement for Real-to-real
impl<A: FloatNum> TransformPar<A, A> for Base<A> {
    type Physical = A;
    type Spectral = A;

    fn forward_par<S, D>(
        &mut self,
        input: &mut ArrayBase<S, D>,
        axis: usize,
    ) -> Array<Self::Spectral, D>
    where
        S: ndarray::Data<Elem = Self::Physical>,
        D: Dimension,
    {
        match self {
            Self::Chebyshev(ref mut b) => b.forward_par(input, axis),
            Self::CompositeChebyshev(ref mut b) => b.forward_par(input, axis),
            Self::Fourier(_) | Self::FourierR2c(_) => {
                panic!("Expected real-to-real transform, but Fourier is complex-to-complex.")
            }
        }
    }

    fn forward_inplace_par<S1, S2, D>(
        &mut self,
        input: &mut ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = Self::Physical>,
        S2: ndarray::Data<Elem = Self::Spectral> + ndarray::DataMut,
        D: Dimension,
    {
        match self {
            Self::Chebyshev(ref mut b) => b.forward_inplace_par(input, output, axis),
            Self::CompositeChebyshev(ref mut b) => b.forward_inplace_par(input, output, axis),
            Self::Fourier(_) | Self::FourierR2c(_) => {
                panic!("Expected real-to-real transform, but Fourier is complex-to-complex.")
            }
        }
    }

    fn backward_par<S, D>(
        &mut self,
        input: &mut ArrayBase<S, D>,
        axis: usize,
    ) -> Array<Self::Physical, D>
    where
        S: ndarray::Data<Elem = Self::Spectral>,
        D: Dimension,
    {
        match self {
            Self::Chebyshev(ref mut b) => b.backward_par(input, axis),
            Self::CompositeChebyshev(ref mut b) => b.backward_par(input, axis),
            Self::Fourier(_) | Self::FourierR2c(_) => {
                panic!("Expected real-to-real transform, but Fourier is complex-to-complex.")
            }
        }
    }

    fn backward_inplace_par<S1, S2, D>(
        &mut self,
        input: &mut ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = Self::Spectral>,
        S2: ndarray::Data<Elem = Self::Physical> + ndarray::DataMut,
        D: Dimension,
    {
        match self {
            Self::Chebyshev(ref mut b) => b.backward_inplace_par(input, output, axis),
            Self::CompositeChebyshev(ref mut b) => b.backward_inplace_par(input, output, axis),
            Self::Fourier(_) | Self::FourierR2c(_) => {
                panic!("Expected real-to-real transform, but Fourier is complex-to-complex.")
            }
        }
    }
}

/// Implement for Complex-to-complex
impl<A: FloatNum> Transform<Complex<A>, Complex<A>> for Base<A> {
    type Physical = Complex<A>;
    type Spectral = Complex<A>;

    fn forward<S, D>(
        &mut self,
        input: &mut ArrayBase<S, D>,
        axis: usize,
    ) -> Array<Self::Spectral, D>
    where
        S: ndarray::Data<Elem = Self::Physical>,
        D: Dimension,
    {
        match self {
            Self::Chebyshev(_) | Self::CompositeChebyshev(_) => {
                panic!("Expected complex-to-complex transform, but Chebyshev is real-to-real.")
            }
            Self::Fourier(ref mut b) => b.forward(input, axis),
            Self::FourierR2c(_) => {
                panic!("Expected complex-to-complex transform, but FourierR2c is real-to-complex.")
            }
        }
    }

    fn forward_inplace<S1, S2, D>(
        &mut self,
        input: &mut ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = Self::Physical>,
        S2: ndarray::Data<Elem = Self::Spectral> + ndarray::DataMut,
        D: Dimension,
    {
        match self {
            Self::Chebyshev(_) | Self::CompositeChebyshev(_) => {
                panic!("Expected complex-to-complex transform, but Chebyshev is real-to-real.")
            }
            Self::Fourier(ref mut b) => b.forward_inplace(input, output, axis),
            Self::FourierR2c(_) => {
                panic!("Expected complex-to-complex transform, but FourierR2c is real-to-complex.")
            }
        }
    }

    fn backward<S, D>(
        &mut self,
        input: &mut ArrayBase<S, D>,
        axis: usize,
    ) -> Array<Self::Physical, D>
    where
        S: ndarray::Data<Elem = Self::Spectral>,
        D: Dimension,
    {
        match self {
            Self::Chebyshev(_) | Self::CompositeChebyshev(_) => {
                panic!("Expected complex-to-complex transform, but Chebyshev is real-to-real.")
            }
            Self::Fourier(ref mut b) => b.backward(input, axis),
            Self::FourierR2c(_) => {
                panic!("Expected complex-to-complex transform, but FourierR2c is real-to-complex.")
            }
        }
    }

    fn backward_inplace<S1, S2, D>(
        &mut self,
        input: &mut ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = Self::Spectral>,
        S2: ndarray::Data<Elem = Self::Physical> + ndarray::DataMut,
        D: Dimension,
    {
        match self {
            Self::Chebyshev(_) | Self::CompositeChebyshev(_) => {
                panic!("Expected complex-to-complex transform, but Chebyshev is real-to-real.")
            }
            Self::Fourier(ref mut b) => b.backward_inplace(input, output, axis),
            Self::FourierR2c(_) => {
                panic!("Expected complex-to-complex transform, but FourierR2c is real-to-complex.")
            }
        }
    }
}
/// Implement for Complex-to-complex
impl<A: FloatNum> TransformPar<Complex<A>, Complex<A>> for Base<A> {
    type Physical = Complex<A>;
    type Spectral = Complex<A>;

    fn forward_par<S, D>(
        &mut self,
        input: &mut ArrayBase<S, D>,
        axis: usize,
    ) -> Array<Self::Spectral, D>
    where
        S: ndarray::Data<Elem = Self::Physical>,
        D: Dimension,
    {
        match self {
            Self::Chebyshev(_) | Self::CompositeChebyshev(_) => {
                panic!("Expected complex-to-complex transform, but Chebyshev is real-to-real.")
            }
            Self::Fourier(ref mut b) => b.forward_par(input, axis),
            Self::FourierR2c(_) => {
                panic!("Expected complex-to-complex transform, but FourierR2c is real-to-complex.")
            }
        }
    }

    fn forward_inplace_par<S1, S2, D>(
        &mut self,
        input: &mut ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = Self::Physical>,
        S2: ndarray::Data<Elem = Self::Spectral> + ndarray::DataMut,
        D: Dimension,
    {
        match self {
            Self::Chebyshev(_) | Self::CompositeChebyshev(_) => {
                panic!("Expected complex-to-complex transform, but Chebyshev is real-to-real.")
            }
            Self::Fourier(ref mut b) => b.forward_inplace_par(input, output, axis),
            Self::FourierR2c(_) => {
                panic!("Expected complex-to-complex transform, but FourierR2c is real-to-complex.")
            }
        }
    }

    fn backward_par<S, D>(
        &mut self,
        input: &mut ArrayBase<S, D>,
        axis: usize,
    ) -> Array<Self::Physical, D>
    where
        S: ndarray::Data<Elem = Self::Spectral>,
        D: Dimension,
    {
        match self {
            Self::Chebyshev(_) | Self::CompositeChebyshev(_) => {
                panic!("Expected complex-to-complex transform, but Chebyshev is real-to-real.")
            }
            Self::Fourier(ref mut b) => b.backward_par(input, axis),
            Self::FourierR2c(_) => {
                panic!("Expected complex-to-complex transform, but FourierR2c is real-to-complex.")
            }
        }
    }

    fn backward_inplace_par<S1, S2, D>(
        &mut self,
        input: &mut ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = Self::Spectral>,
        S2: ndarray::Data<Elem = Self::Physical> + ndarray::DataMut,
        D: Dimension,
    {
        match self {
            Self::Chebyshev(_) | Self::CompositeChebyshev(_) => {
                panic!("Expected complex-to-complex transform, but Chebyshev is real-to-real.")
            }
            Self::Fourier(ref mut b) => b.backward_inplace_par(input, output, axis),
            Self::FourierR2c(_) => {
                panic!("Expected complex-to-complex transform, but FourierR2c is real-to-complex.")
            }
        }
    }
}

/// Implement for Real-to-complex
impl<A: FloatNum> Transform<A, Complex<A>> for Base<A> {
    type Physical = A;
    type Spectral = Complex<A>;

    fn forward<S, D>(
        &mut self,
        input: &mut ArrayBase<S, D>,
        axis: usize,
    ) -> Array<Self::Spectral, D>
    where
        S: ndarray::Data<Elem = Self::Physical>,
        D: Dimension,
    {
        match self {
            Self::Chebyshev(_) | Self::CompositeChebyshev(_) => {
                panic!("Expected real-to-complex transform, but Chebyshev is real-to-real.")
            }
            Self::Fourier(_) => {
                panic!("Expected real-to-complex transform, but Fourier is complex-to-complex.")
            }
            Self::FourierR2c(ref mut b) => b.forward(input, axis),
        }
    }

    fn forward_inplace<S1, S2, D>(
        &mut self,
        input: &mut ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = Self::Physical>,
        S2: ndarray::Data<Elem = Self::Spectral> + ndarray::DataMut,
        D: Dimension,
    {
        match self {
            Self::Chebyshev(_) | Self::CompositeChebyshev(_) => {
                panic!("Expected real-to-complex transform, but Chebyshev is real-to-real.")
            }
            Self::Fourier(_) => {
                panic!("Expected real-to-complex transform, but Fourier is complex-to-complex.")
            }
            Self::FourierR2c(ref mut b) => b.forward_inplace(input, output, axis),
        }
    }

    fn backward<S, D>(
        &mut self,
        input: &mut ArrayBase<S, D>,
        axis: usize,
    ) -> Array<Self::Physical, D>
    where
        S: ndarray::Data<Elem = Self::Spectral>,
        D: Dimension,
    {
        match self {
            Self::Chebyshev(_) | Self::CompositeChebyshev(_) => {
                panic!("Expected real-to-complex transform, but Chebyshev is real-to-real.")
            }
            Self::Fourier(_) => {
                panic!("Expected real-to-complex transform, but Fourier is complex-to-complex.")
            }
            Self::FourierR2c(ref mut b) => b.backward(input, axis),
        }
    }

    fn backward_inplace<S1, S2, D>(
        &mut self,
        input: &mut ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = Self::Spectral>,
        S2: ndarray::Data<Elem = Self::Physical> + ndarray::DataMut,
        D: Dimension,
    {
        match self {
            Self::Chebyshev(_) | Self::CompositeChebyshev(_) => {
                panic!("Expected real-to-complex transform, but Chebyshev is real-to-real.")
            }
            Self::Fourier(_) => {
                panic!("Expected real-to-complex transform, but Fourier is complex-to-complex.")
            }
            Self::FourierR2c(ref mut b) => b.backward_inplace(input, output, axis),
        }
    }
}
/// Implement for Real-to-complex
impl<A: FloatNum> TransformPar<A, Complex<A>> for Base<A> {
    type Physical = A;
    type Spectral = Complex<A>;

    fn forward_par<S, D>(
        &mut self,
        input: &mut ArrayBase<S, D>,
        axis: usize,
    ) -> Array<Self::Spectral, D>
    where
        S: ndarray::Data<Elem = Self::Physical>,
        D: Dimension,
    {
        match self {
            Self::Chebyshev(_) | Self::CompositeChebyshev(_) => {
                panic!("Expected real-to-complex transform, but Chebyshev is real-to-real.")
            }
            Self::Fourier(_) => {
                panic!("Expected real-to-complex transform, but Fourier is complex-to-complex.")
            }
            Self::FourierR2c(ref mut b) => b.forward_par(input, axis),
        }
    }

    fn forward_inplace_par<S1, S2, D>(
        &mut self,
        input: &mut ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = Self::Physical>,
        S2: ndarray::Data<Elem = Self::Spectral> + ndarray::DataMut,
        D: Dimension,
    {
        match self {
            Self::Chebyshev(_) | Self::CompositeChebyshev(_) => {
                panic!("Expected real-to-complex transform, but Chebyshev is real-to-real.")
            }
            Self::Fourier(_) => {
                panic!("Expected real-to-complex transform, but Fourier is complex-to-complex.")
            }
            Self::FourierR2c(ref mut b) => b.forward_inplace_par(input, output, axis),
        }
    }

    fn backward_par<S, D>(
        &mut self,
        input: &mut ArrayBase<S, D>,
        axis: usize,
    ) -> Array<Self::Physical, D>
    where
        S: ndarray::Data<Elem = Self::Spectral>,
        D: Dimension,
    {
        match self {
            Self::Chebyshev(_) | Self::CompositeChebyshev(_) => {
                panic!("Expected real-to-complex transform, but Chebyshev is real-to-real.")
            }
            Self::Fourier(_) => {
                panic!("Expected real-to-complex transform, but Fourier is complex-to-complex.")
            }
            Self::FourierR2c(ref mut b) => b.backward_par(input, axis),
        }
    }

    fn backward_inplace_par<S1, S2, D>(
        &mut self,
        input: &mut ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = Self::Spectral>,
        S2: ndarray::Data<Elem = Self::Physical> + ndarray::DataMut,
        D: Dimension,
    {
        match self {
            Self::Chebyshev(_) | Self::CompositeChebyshev(_) => {
                panic!("Expected real-to-complex transform, but Chebyshev is real-to-real.")
            }
            Self::Fourier(_) => {
                panic!("Expected real-to-complex transform, but Fourier is complex-to-complex.")
            }
            Self::FourierR2c(ref mut b) => b.backward_inplace_par(input, output, axis),
        }
    }
}
