//! Collection of macros which implement the base traits
//! on the base enums `BaseR2r`, `BaseR2c`, `BaseC2c`
#![macro_use]

/// Implement transform trait across Base enums
macro_rules! impl_funspace_elemental_for_base {
    ($base: ident, $a: ty, $b: ty, $($var:ident),*) => {

        impl<A: Real> HasLength for $base<A> {
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
            fn len_ortho(&self) -> usize {
                match self {
                    $(Self::$var(ref b) => b.len_ortho(),)*
                }
            }

        }

        impl<A: Real> HasType for $base<A> {

            /// Return kind of base (e.g. FourierR2c, Chebyshev, ...)
            fn base_kind(&self) -> BaseKind{
                match self {
                    $(Self::$var(ref b) => b.base_kind(),)*
                }
            }

            /// Return type of base (e.g. Orthogonal or Composite)
            fn base_type(&self) -> BaseType {
                match self {
                    $(Self::$var(ref b) => b.base_type(),)*
                }
            }

            /// Return kind of transform (e.g, real-to-real, real-to-complex ...)
            fn transform_kind(&self) -> TransformKind {
                match self {
                    $(Self::$var(ref b) => b.transform_kind(),)*
                }
            }
        }


        impl<A: Real> HasCoords<A> for $base<A> {
            fn coords(&self) -> Vec<A>{
                match self {
                    $(Self::$var(ref b) => b.coords(),)*
                }
            }
        }

        impl<A, T> Differentiate<T> for $base<A>
        where
            A: Real,
            T: ScalarOperand<A>
            + Add<$b, Output = T>
            + Mul<$b, Output = T>
            + Div<$b, Output = T>
            + Sub<$b, Output = T>,
        {
            fn diff(&self, v: &[T], dv: &mut [T], order: usize)
            {
                match self {
                    $(Self::$var(ref b) => b.diff(v, dv,  order),)*
                }
            }

            fn diff_inplace(&self, v: &mut [T], order: usize)
            {
                match self {
                    $(Self::$var(ref b) => b.diff_inplace(v,  order),)*
                }
            }
        }

        impl<A, T> ToOrtho<T> for $base<A>
        where
            A: Real,
            T: ScalarOperand<A>,
        {
            fn to_ortho(&self, comp: &[T], ortho: &mut [T])
            {
                match self {
                    $(Self::$var(ref b) => b.to_ortho(comp,  ortho),)*
                }
            }

            fn from_ortho(&self, ortho: &[T], comp: &mut [T])
            {
                match self {
                    $(Self::$var(ref b) => b.from_ortho(ortho,  comp),)*
                }
            }
        }

        impl<A: Real + Scalar> Transform for $base<A> {
            type Physical = $a;
            type Spectral = $b;

            fn forward(&self, phys: &[Self::Physical], spec: &mut [Self::Spectral])
            {
                match self {
                    $(Self::$var(ref b) => b.forward(phys,  spec),)*
                }
            }

            fn backward(&self, spec: &[Self::Spectral], phys: &mut [Self::Physical])
            {
                match self {
                    $(Self::$var(ref b) => b.backward(spec,  phys),)*
                }
            }
        }
    };
}

// /// Apply a method that takes a slice on multidimensional arrays
// macro_rules! apply_along_axis {
//     (
//         $(#[$meta:meta])* $i: ident, $t1: ty, $t2: ty, $f: ident, $l1: ident, $l2: ident, $e:expr
//     ) => {
//         $(#[$meta])*
//         fn $i<S1, S2, D>(
//             &self,
//             indata: &ArrayBase<S1, D>,
//             outdata: &mut ArrayBase<S2, D>,
//             axis: usize,
//         ) where
//             S1: Data<Elem = $t1>,
//             S2: Data<Elem = $t2> + DataMut,
//             D: Dimension,
//             $t1: Clone,
//             $t2: Clone + Zero + Copy,
//         {
//             assert!(indata.is_standard_layout());
//             assert!(outdata.is_standard_layout());
//             check_array_axis(indata, self.$l1(), axis, $e);
//             check_array_axis(outdata, self.$l2(), axis, $e);

//             let outer_axis = indata.ndim() - 1;
//             if axis == outer_axis {
//                 // Data is contiguous in memory
//                 Zip::from(indata.lanes(Axis(axis)))
//                     .and(outdata.lanes_mut(Axis(axis)))
//                     .for_each(|x, mut y| {
//                         self.$f(x.as_slice().unwrap(), y.as_slice_mut().unwrap());
//                     });
//             } else {
//                 // Data is *not* contiguous in memory.
//                 let mut scratch: Vec<$t2> = vec![<$t2>::zero(); outdata.shape()[axis]];
//                 Zip::from(indata.lanes(Axis(axis)))
//                     .and(outdata.lanes_mut(Axis(axis)))
//                     .for_each(|x, mut y| {
//                         self.$f(&x.to_vec(), &mut scratch);
//                         for (yi, si) in y.iter_mut().zip(scratch.iter()) {
//                             *yi = *si;
//                         }
//                     });
//             }
//         }
//     };
// }

// /// Apply a method that takes a slice on multidimensional arrays.
// /// Uses parallel iterators
// macro_rules! par_apply_along_axis {
//     (
//         $(#[$meta:meta])* $i: ident, $t1: ty, $t2: ty, $f: ident, $l1: ident, $l2: ident, $e:expr
//     ) => {
//         $(#[$meta])*
//         fn $i<S1, S2, D>(
//             &self,
//             indata: &ArrayBase<S1, D>,
//             outdata: &mut ArrayBase<S2, D>,
//             axis: usize,
//         ) where
//             S1: Data<Elem = $t1>,
//             S2: Data<Elem = $t2> + DataMut,
//             D: Dimension,
//             $t1: Clone + Send + Sync,
//             $t2: Clone + Zero + Copy + Send + Sync,
//             Self: Sync,
//         {
//             assert!(indata.is_standard_layout());
//             assert!(outdata.is_standard_layout());
//             check_array_axis(indata, self.$l1(), axis, $e);
//             check_array_axis(outdata, self.$l2(), axis, $e);

//             let outer_axis = indata.ndim() - 1;
//             if axis == outer_axis {
//                 // Data is contiguous in memory
//                 Zip::from(indata.lanes(Axis(axis)))
//                     .and(outdata.lanes_mut(Axis(axis)))
//                     .par_for_each(|x, mut y| {
//                         self.$f(x.as_slice().unwrap(), y.as_slice_mut().unwrap());
//                     });
//             } else {
//                 // Data is *not* contiguous in memory.
//                 let scratch_len = outdata.shape()[axis];
//                 Zip::from(indata.lanes(Axis(axis)))
//                     .and(outdata.lanes_mut(Axis(axis)))
//                     .par_for_each(|x, mut y| {
//                         let mut scratch: Vec<$t2> = vec![<$t2>::zero(); scratch_len];
//                         self.$f(&x.to_vec(), &mut scratch);
//                         for (yi, si) in y.iter_mut().zip(scratch.iter()) {
//                             *yi = *si;
//                         }
//                     });
//             }
//         }
//     };
// }
