//! Collection of macros which implement the base traits
//! on the base enums `BaseR2r`, `BaseR2c`, `BaseC2c`
#![macro_use]

/// Implement transform trait across Base enums
macro_rules! impl_funspace_elemental_for_base {
    ($base: ident, $a: ty, $b: ty, $($var:ident),*) => {

        impl<A: FloatNum> FunspaceSize for $base<A> {
            /// Size in physical space
            fn len_phys(&self) -> usize {
                match self {
                    $(Self::$var(ref b) => b.len_phys(),)*
                }
            }

            /// Size in spectral space
            fn len_spec(&self) -> usize {
                match self {
                    $(Self::$var(ref b) => b.len_spec(),)*
                }
            }

            /// Size of orthogonal space
            fn len_orth(&self) -> usize {
                match self {
                    $(Self::$var(ref b) => b.len_orth(),)*
                }
            }
        }

        impl<A: FloatNum> FunspaceExtended for $base<A> {
            type Real = A;

            type Spectral = $b;

            /// Return kind of base
            fn base_kind(&self) -> BaseKind{
                match self {
                    $(Self::$var(ref b) => b.base_kind(),)*
                }
            }

            /// Grid coordinates
            fn get_nodes(&self) -> Vec<A> {
                match self {
                    $(Self::$var(ref b) => b.get_nodes(),)*
                }
            }

            /// Mass matrix
            fn mass(&self) -> Array2<A> {
                match self {
                    $(Self::$var(ref b) => b.mass(),)*
                }
            }

            /// Inverse of mass matrix
            fn mass_inv(&self) -> Array2<A> {
                match self {
                    $(Self::$var(ref b) => b.mass_inv(),)*
                }
            }

            /// Explicit differential operator
            fn diffmat(&self, deriv: usize) -> Array2<Self::Spectral> {
                match self {
                    $(Self::$var(ref b) => b.diffmat(deriv),)*
                }
            }

            /// Laplacian $ L $
            fn laplace(&self) -> Array2<A> {
                match self {
                    $(Self::$var(ref b) => b.laplace(),)*
                }
            }

            /// Pseudoinverse mtrix of Laplacian $ L^{-1} $
            fn laplace_inv(&self) -> Array2<A> {
                match self {
                    $(Self::$var(ref b) => b.laplace_inv(),)*
                }
            }

            /// Pseudoidentity matrix of laplacian $ L^{-1} L $
            fn laplace_inv_eye(&self) -> Array2<A> {
                match self {
                    $(Self::$var(ref b) => b.laplace_inv_eye(),)*
                }
            }
        }

        impl<A: FloatNum + ScalarNum> FunspaceElemental for $base<A> {
            type Physical = $a;
            type Spectral = $b;

            fn forward_slice(&self, indata: &[Self::Physical], outdata: &mut [Self::Spectral])
            {
                match self {
                    $(Self::$var(ref b) => b.forward_slice(indata,  outdata),)*
                }
            }

            fn backward_slice(&self, indata: &[Self::Spectral], outdata: &mut [Self::Physical])
            {
                match self {
                    $(Self::$var(ref b) => b.backward_slice(indata,  outdata),)*
                }
            }

            fn differentiate_slice<T>(&self, indata: &[T], outdata: &mut [T], n_times: usize)
            where
                T: ScalarNum
                    + Add<Self::Spectral, Output = T>
                    + Mul<Self::Spectral, Output = T>
                    + Div<Self::Spectral, Output = T>
                    + Sub<Self::Spectral, Output = T>
            {
                match self {
                    $(Self::$var(ref b) => b.differentiate_slice(indata,  outdata, n_times),)*
                }
            }

            fn to_ortho_slice<T>(&self, indata: &[T], outdata: &mut [T])
            where
                T: ScalarNum
                + Add<Self::Spectral, Output = T>
                + Mul<Self::Spectral, Output = T>
                + Div<Self::Spectral, Output = T>
                + Sub<Self::Spectral, Output = T>
            {
                match self {
                    $(Self::$var(ref b) => b.to_ortho_slice(indata,  outdata),)*
                }
            }

            fn from_ortho_slice<T>(&self, indata: &[T], outdata: &mut [T])
            where
                T: ScalarNum
                + Add<Self::Spectral, Output = T>
                + Mul<Self::Spectral, Output = T>
                + Div<Self::Spectral, Output = T>
                + Sub<Self::Spectral, Output = T>,
            {
                match self {
                    $(Self::$var(ref b) => b.from_ortho_slice(indata,  outdata),)*
                }
            }
        }
    };
}
