//! Collection of macros which implement the base traits
//! on the base enums `BaseR2r`, `BaseR2c`, `BaseC2c`
#![macro_use]

/// Implement transform trait across Base enums
macro_rules! impl_funspace_elemental_for_base {
    ($base: ident, $a: ty, $b: ty, $($var:ident),*) => {

        impl<A: FloatNum> BaseSize for $base<A> {
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

        impl<A: FloatNum> BaseElements for $base<A> {
            /// Real valued scalar type
            type RealNum = A;

            /// Return kind of base
            fn base_kind(&self) -> BaseKind{
                match self {
                    $(Self::$var(ref b) => b.base_kind(),)*
                }
            }

            /// Return kind of transform
            fn transform_kind(&self) -> TransformKind {
                match self {
                    $(Self::$var(ref b) => b.transform_kind(),)*
                }
            }

            /// Grid coordinates
            fn coords(&self) -> Vec<Self::RealNum> {
                match self {
                    $(Self::$var(ref b) => b.coords(),)*
                }
            }
        }

        impl<A: FloatNum> BaseMatOpGeneral for $base<A> {
            /// Real valued scalar type
            type RealNum = A;

            /// Scalar type of spectral coefficients
            type SpectralNum = $b;

            /// Explicit differential operator $ D $
            fn diffmat(&self, deriv: usize) -> Array2<Self::SpectralNum> {
                match self {
                    $(Self::$var(ref b) => b.diffmat(deriv),)*
                }
            }

            /// Explicit inverse of differential operator $ D^* $
            fn diffmat_pinv(&self, deriv: usize) -> (Array2<Self::SpectralNum>, Array2<Self::SpectralNum>) {
                match self {
                    $(Self::$var(ref b) => b.diffmat_pinv(deriv),)*
                }
            }

            /// Transformation stencil composite -> orthogonal space
            fn stencil(&self) -> Array2<Self::RealNum> {
                match self {
                    $(Self::$var(ref b) => b.stencil(),)*
                }
            }

            /// Inverse of transformation stencil
            fn stencil_inv(&self) -> Array2<Self::RealNum> {
                match self {
                    $(Self::$var(ref b) => b.stencil_inv(),)*
                }
            }
        }

        impl<A: FloatNum> BaseMatOpLaplacian for $base<A> {
            /// Scalar type of laplacian matrix
            type ScalarNum = A;

            /// Laplacian $ L $
            fn laplace(&self) -> Array2<Self::ScalarNum> {
                match self {
                    $(Self::$var(ref b) => b.laplace(),)*
                }
            }

            /// Pseudoinverse matrix of Laplacian $ L^{-1} $
            ///
            /// Returns pseudoinverse and pseudoidentity,i.e
            /// ``(D_pinv, I_pinv)``
            ///
            /// ```text
            /// D_pinv @ D = I_pinv
            /// ``
            fn laplace_pinv(&self) -> (Array2<Self::ScalarNum>, Array2<Self::ScalarNum>) {
                match self {
                    $(Self::$var(ref b) => b.laplace_pinv(),)*
                }
            }
        }

        impl<A, T> BaseFromOrtho<T> for $base<A>
        where
            A: FloatNum,
            T: ScalarNum
            + Add<$b, Output = T>
            + Mul<$b, Output = T>
            + Div<$b, Output = T>
            + Sub<$b, Output = T>,
        {
            fn to_ortho_slice(&self, indata: &[T], outdata: &mut [T])
            {
                match self {
                    $(Self::$var(ref b) => b.to_ortho_slice(indata,  outdata),)*
                }
            }

            fn from_ortho_slice(&self, indata: &[T], outdata: &mut [T])
            {
                match self {
                    $(Self::$var(ref b) => b.from_ortho_slice(indata,  outdata),)*
                }
            }
        }

        impl<A, T> BaseGradient<T> for $base<A>
        where
            A: FloatNum,
            T: ScalarNum
            + Add<$b, Output = T>
            + Mul<$b, Output = T>
            + Div<$b, Output = T>
            + Sub<$b, Output = T>,
        {
            fn gradient_slice(&self, indata: &[T], outdata: &mut [T], n_times: usize)
            {
                match self {
                    $(Self::$var(ref b) => b.gradient_slice(indata,  outdata, n_times),)*
                }
            }
        }

        impl<A: FloatNum + ScalarNum> BaseTransform for $base<A> {
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
        }
    };
}

/// Apply a method that takes a slice on multidimensional arrays
macro_rules! apply_along_axis {
    (
        $(#[$meta:meta])* $i: ident, $t1: ty, $t2: ty, $f: ident, $l1: ident, $l2: ident, $e:expr
    ) => {
        $(#[$meta])*
        fn $i<S1, S2, D>(
            &self,
            indata: &ArrayBase<S1, D>,
            outdata: &mut ArrayBase<S2, D>,
            axis: usize,
        ) where
            S1: Data<Elem = $t1>,
            S2: Data<Elem = $t2> + DataMut,
            D: Dimension,
            $t1: Clone,
            $t2: Clone + Zero + Copy,
        {
            assert!(indata.is_standard_layout());
            assert!(outdata.is_standard_layout());
            check_array_axis(indata, self.$l1(), axis, $e);
            check_array_axis(outdata, self.$l2(), axis, $e);

            let outer_axis = indata.ndim() - 1;
            if axis == outer_axis {
                // Data is contiguous in memory
                Zip::from(indata.lanes(Axis(axis)))
                    .and(outdata.lanes_mut(Axis(axis)))
                    .for_each(|x, mut y| {
                        self.$f(x.as_slice().unwrap(), y.as_slice_mut().unwrap());
                    });
            } else {
                // Data is *not* contiguous in memory.
                let mut scratch: Vec<$t2> = vec![<$t2>::zero(); outdata.shape()[axis]];
                Zip::from(indata.lanes(Axis(axis)))
                    .and(outdata.lanes_mut(Axis(axis)))
                    .for_each(|x, mut y| {
                        self.$f(&x.to_vec(), &mut scratch);
                        for (yi, si) in y.iter_mut().zip(scratch.iter()) {
                            *yi = *si;
                        }
                    });
            }
        }
    };
}

/// Apply a method that takes a slice on multidimensional arrays.
/// Uses parallel iterators
macro_rules! par_apply_along_axis {
    (
        $(#[$meta:meta])* $i: ident, $t1: ty, $t2: ty, $f: ident, $l1: ident, $l2: ident, $e:expr
    ) => {
        $(#[$meta])*
        fn $i<S1, S2, D>(
            &self,
            indata: &ArrayBase<S1, D>,
            outdata: &mut ArrayBase<S2, D>,
            axis: usize,
        ) where
            S1: Data<Elem = $t1>,
            S2: Data<Elem = $t2> + DataMut,
            D: Dimension,
            $t1: Clone + Send + Sync,
            $t2: Clone + Zero + Copy + Send + Sync,
            Self: Sync,
        {
            assert!(indata.is_standard_layout());
            assert!(outdata.is_standard_layout());
            check_array_axis(indata, self.$l1(), axis, $e);
            check_array_axis(outdata, self.$l2(), axis, $e);

            let outer_axis = indata.ndim() - 1;
            if axis == outer_axis {
                // Data is contiguous in memory
                Zip::from(indata.lanes(Axis(axis)))
                    .and(outdata.lanes_mut(Axis(axis)))
                    .par_for_each(|x, mut y| {
                        self.$f(x.as_slice().unwrap(), y.as_slice_mut().unwrap());
                    });
            } else {
                // Data is *not* contiguous in memory.
                let scratch_len = outdata.shape()[axis];
                Zip::from(indata.lanes(Axis(axis)))
                    .and(outdata.lanes_mut(Axis(axis)))
                    .par_for_each(|x, mut y| {
                        let mut scratch: Vec<$t2> = vec![<$t2>::zero(); scratch_len];
                        self.$f(&x.to_vec(), &mut scratch);
                        for (yi, si) in y.iter_mut().zip(scratch.iter()) {
                            *yi = *si;
                        }
                    });
            }
        }
    };
}
