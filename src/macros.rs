//! Collection of macros which implement the base traits
//! on the base enums `BaseR2r`, `BaseR2c`, `BaseC2c`
#![macro_use]

/// Implement transform trait across Base enums
macro_rules! impl_transform_trait_for_base {
    ($base: ident, $a: ty, $b: ty, $($var:ident),*) => {
        impl<A: FloatNum> Transform for $base<A> {
            type Physical = $a;
            type Spectral = $b;

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
                    $(Self::$var(ref mut b) => b.forward(input, axis),)*
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
                    $(Self::$var(ref mut b) => b.forward_inplace(input,  output, axis),)*
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
                    $(Self::$var(ref mut b) => b.backward(input, axis),)*
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
                    $(Self::$var(ref mut b) => b.backward_inplace(input, output, axis),)*
                }
            }
        }

        impl<A: FloatNum> TransformPar for $base<A> {
            type Physical = $a;
            type Spectral = $b;

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
                    $(Self::$var(ref mut b) => b.forward_par(input, axis),)*
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
                    $(Self::$var(ref mut b) => b.forward_inplace_par(input, output, axis),)*
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
                    $(Self::$var(ref mut b) => b.backward_par(input, axis),)*
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
                    $(Self::$var(ref mut b) => b.backward_inplace_par(input, output, axis),)*
                }
            }
        }
    };
}

/// Implement differentiate trait across Base enums
macro_rules! impl_differentiate_trait_for_base {
    ($base: ident, $a: ty, $($var:ident),*) => {
        impl<A: FloatNum> Differentiate<$a> for $base<A> {
            fn differentiate<S, D>(
                &self,
                data: &ArrayBase<S, D>,
                n_times: usize,
                axis: usize,
            ) -> Array<$a, D>
            where
                S: ndarray::Data<Elem = $a>,
                D: Dimension,
            {
                match self {
                    $(Self::$var(ref b) => b.differentiate(data, n_times, axis),)*
                }
            }

            fn differentiate_inplace<S, D>(
                &self,
                data: &mut ArrayBase<S, D>,
                n_times: usize,
                axis: usize,
            ) where
                S: ndarray::Data<Elem = $a> + ndarray::DataMut,
                D: Dimension,
            {
                match self {
                    $(Self::$var(ref b) => b.differentiate_inplace(data, n_times, axis),)*
                }
            }
        }
    };
}

/// Implement from_ortho trait across Base enums
macro_rules! impl_from_ortho_trait_for_base {
    ($base: ident, $a: ty, $($var:ident),*) => {
        impl<A: FloatNum> FromOrtho<$a> for $base<A> {
            fn to_ortho<S, D>(&self, input: &ArrayBase<S, D>, axis: usize) -> Array<$a, D>
            where
                S: ndarray::Data<Elem = $a>,
                D: Dimension,
            {
                match self {
                    $(Self::$var(ref b) => b.to_ortho(input, axis),)*
                }
            }

            fn to_ortho_inplace<S1, S2, D>(
                &self,
                input: &ArrayBase<S1, D>,
                output: &mut ArrayBase<S2, D>,
                axis: usize,
            ) where
                S1: ndarray::Data<Elem = $a>,
                S2: ndarray::Data<Elem = $a> + ndarray::DataMut,
                D: Dimension,
            {
                match self {
                    $(Self::$var(ref b) => b.to_ortho_inplace(input, output, axis),)*
                }
            }

            fn from_ortho<S, D>(&self, input: &ArrayBase<S, D>, axis: usize) -> Array<$a, D>
            where
                S: ndarray::Data<Elem = $a>,
                D: Dimension,
            {
                match self {
                    $(Self::$var(ref b) => b.from_ortho(input, axis),)*
                }
            }

            fn from_ortho_inplace<S1, S2, D>(
                &self,
                input: &ArrayBase<S1, D>,
                output: &mut ArrayBase<S2, D>,
                axis: usize,
            ) where
                S1: ndarray::Data<Elem = $a>,
                S2: ndarray::Data<Elem = $a> + ndarray::DataMut,
                D: Dimension,
            {
                match self {
                    $(Self::$var(ref b) => b.from_ortho_inplace(input, output, axis),)*
                }
            }
        }

        impl<A: FloatNum> FromOrthoPar<$a> for $base<A> {
            fn to_ortho_par<S, D>(&self, input: &ArrayBase<S, D>, axis: usize) -> Array<$a, D>
            where
                S: ndarray::Data<Elem = $a>,
                D: Dimension,
            {
                match self {
                    $(Self::$var(ref b) => b.to_ortho_par(input, axis),)*
                }
            }

            fn to_ortho_inplace_par<S1, S2, D>(
                &self,
                input: &ArrayBase<S1, D>,
                output: &mut ArrayBase<S2, D>,
                axis: usize,
            ) where
                S1: ndarray::Data<Elem = $a>,
                S2: ndarray::Data<Elem = $a> + ndarray::DataMut,
                D: Dimension,
            {
                match self {
                    $(Self::$var(ref b) => b.to_ortho_inplace_par(input, output, axis),)*
                }
            }

            fn from_ortho_par<S, D>(&self, input: &ArrayBase<S, D>, axis: usize) -> Array<$a, D>
            where
                S: ndarray::Data<Elem = $a>,
                D: Dimension,
            {
                match self {
                    $(Self::$var(ref b) => b.from_ortho_par(input, axis),)*
                }
            }

            fn from_ortho_inplace_par<S1, S2, D>(
                &self,
                input: &ArrayBase<S1, D>,
                output: &mut ArrayBase<S2, D>,
                axis: usize,
            ) where
                S1: ndarray::Data<Elem = $a>,
                S2: ndarray::Data<Elem = $a> + ndarray::DataMut,
                D: Dimension,
            {
                match self {
                    $(Self::$var(ref b) => b.from_ortho_inplace_par(input, output, axis),)*
                }
            }
        }
    };
}
