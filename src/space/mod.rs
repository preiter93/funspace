pub mod traits;
use crate::enums::{BaseC2c, BaseR2c, BaseR2r};
use crate::traits::{Differentiate, HasLength, ToOrtho, Transform};
use crate::types::{Real, Scalar, ScalarOperand};
use ndarray::{Array2, ArrayBase, Axis, Data, DataMut, Dim, Zip};
use num_complex::Complex;
use num_traits::Zero;
use std::ops::{Add, Div, Mul, Sub};
use traits::{HasShape, SpaceDifferentiate, SpaceToOrtho, SpaceTransform};

/// Create two-dimensional space
#[derive(Clone)]
pub struct Space2<B0, B1> {
    // Intermediate -> Spectral
    pub base0: B0,
    // Phsical -> Intermediate
    pub base1: B1,
}

impl<B0, B1> Space2<B0, B1>
where
    B0: Clone,
    B1: Clone,
{
    /// Create a new space
    pub fn new(base0: &B0, base1: &B1) -> Self {
        Self {
            base0: base0.clone(),
            base1: base1.clone(),
        }
    }

    /// Get reference to base0
    pub fn get_base0<'a>(&'a self) -> &'a B0 {
        &self.base0
    }

    /// Get reference to base1
    pub fn get_base1<'a>(&'a self) -> &'a B1 {
        &self.base1
    }
}

impl<B0, B1> Space2<B0, B1>
where
    B0: Clone + Transform,
    B1: Clone + Transform,
    B0::Physical: Zero + Clone + Copy,
    B0::Spectral: Zero + Clone + Copy,
{
    pub fn forward_axis0<S1, S2>(
        &self,
        phys: &ArrayBase<S1, Dim<[usize; 2]>>,
        spec: &mut ArrayBase<S2, Dim<[usize; 2]>>,
    ) where
        S1: Data<Elem = B0::Physical>,
        S2: Data<Elem = B0::Spectral> + DataMut,
    {
        assert!(phys.is_standard_layout());
        assert!(spec.is_standard_layout());
        let mut scratch = vec![B0::Spectral::zero(); spec.shape()[0]];
        Zip::from(phys.lanes(Axis(0)))
            .and(spec.lanes_mut(Axis(0)))
            .for_each(|x, mut y| {
                self.base0.forward(&x.to_vec(), &mut scratch);
                for (yi, si) in y.iter_mut().zip(scratch.iter()) {
                    *yi = *si;
                }
            });
    }

    pub fn forward_axis1<S1, S2>(
        &self,
        phys: &ArrayBase<S1, Dim<[usize; 2]>>,
        spec: &mut ArrayBase<S2, Dim<[usize; 2]>>,
    ) where
        S1: Data<Elem = B1::Physical>,
        S2: Data<Elem = B1::Spectral> + DataMut,
    {
        assert!(phys.is_standard_layout());
        assert!(spec.is_standard_layout());
        Zip::from(phys.lanes(Axis(1)))
            .and(spec.lanes_mut(Axis(1)))
            .for_each(|x, mut y| {
                self.base1
                    .forward(x.as_slice().unwrap(), y.as_slice_mut().unwrap());
            });
    }

    pub fn backward_axis0<S1, S2>(
        &self,
        spec: &ArrayBase<S1, Dim<[usize; 2]>>,
        phys: &mut ArrayBase<S2, Dim<[usize; 2]>>,
    ) where
        S1: Data<Elem = B0::Spectral>,
        S2: Data<Elem = B0::Physical> + DataMut,
    {
        assert!(phys.is_standard_layout());
        assert!(spec.is_standard_layout());
        let mut scratch = vec![B0::Physical::zero(); phys.shape()[0]];
        Zip::from(spec.lanes(Axis(0)))
            .and(phys.lanes_mut(Axis(0)))
            .for_each(|x, mut y| {
                self.base0.backward(&x.to_vec(), &mut scratch);
                for (yi, si) in y.iter_mut().zip(scratch.iter()) {
                    *yi = *si;
                }
            });
    }

    pub fn backward_axis1<S1, S2>(
        &self,
        spec: &ArrayBase<S1, Dim<[usize; 2]>>,
        phys: &mut ArrayBase<S2, Dim<[usize; 2]>>,
    ) where
        S1: Data<Elem = B1::Spectral>,
        S2: Data<Elem = B1::Physical> + DataMut,
    {
        assert!(phys.is_standard_layout());
        assert!(spec.is_standard_layout());
        Zip::from(spec.lanes(Axis(1)))
            .and(phys.lanes_mut(Axis(1)))
            .for_each(|x, mut y| {
                self.base1
                    .backward(x.as_slice().unwrap(), y.as_slice_mut().unwrap());
            });
    }
}

/// Apply a method that takes a slice on multidimensional arrays
macro_rules! apply_along_axis {
    (
        $(#[$meta:meta])* $i: ident, $t1: ty, $t2: ty, $f: ident, $e:expr, $($args:ident: $t:ty,),*
    ) => {
        $(#[$meta])*
        fn $i<S1, S2>(
            &self,
            src: &ArrayBase<S1, Dim<[usize; 2]>>,
            dst: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            $($args: $t,)*
            axis: usize
        ) where
            S1: Data<Elem = $t1>,
            S2: Data<Elem = $t2> + DataMut,
            // D: Dimension,
            $t1: Clone,
            $t2: Clone + Zero + Copy,
        {
            assert!(src.is_standard_layout());
            assert!(dst.is_standard_layout());
            assert!(axis < 2);
            if axis == 1 {
                // Data is contiguous in memory
                Zip::from(src.lanes(Axis(axis)))
                    .and(dst.lanes_mut(Axis(axis)))
                    .for_each(|x, mut y| {
                        self.base1.$f(x.as_slice().unwrap(), y.as_slice_mut().unwrap(), $($args,)*);
                    });
            } else {
                // Data is *not* contiguous in memory.
                let mut scratch: Vec<$t2> = vec![<$t2>::zero(); dst.shape()[axis]];
                Zip::from(src.lanes(Axis(axis)))
                    .and(dst.lanes_mut(Axis(axis)))
                    .for_each(|x, mut y| {
                        self.base0.$f(&x.to_vec(), &mut scratch, $($args,)*);
                        for (yi, si) in y.iter_mut().zip(scratch.iter()) {
                            *yi = *si;
                        }
                    });
            }
        }
    };
}

macro_rules! impl_space2 {
    ($space: ident, $base0: ident, $base1: ident, $p: ty, $s: ty) => {
        impl<A> HasShape<2> for $space<$base0<A>, $base1<A>>
        where
            A: Real,
        {
            fn shape_phys(&self) -> [usize; 2] {
                [self.base0.len_phys(), self.base1.len_phys()]
            }

            fn shape_spec(&self) -> [usize; 2] {
                [self.base0.len_spec(), self.base1.len_spec()]
            }

            fn shape_spec_ortho(&self) -> [usize; 2] {
                [self.base0.len_ortho(), self.base1.len_ortho()]
            }
        }

        impl<A, T> SpaceToOrtho<T, 2> for $space<$base0<A>, $base1<A>>
        where
            A: Real,
            T: ScalarOperand<A>,
        {
            apply_along_axis!(to_ortho_axis, T, T, to_ortho, "to_ortho",);

            apply_along_axis!(from_ortho_axis, T, T, from_ortho, "from_ortho",);

            fn to_ortho<S1, S2>(
                &self,
                comp: &ArrayBase<S1, Dim<[usize; 2]>>,
                ortho: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = T>,
                S2: Data<Elem = T> + DataMut,
            {
                let mut scratch = Array2::zeros((self.base0.len_spec(), self.base1.len_ortho()));
                self.to_ortho_axis(comp, &mut scratch, 1);
                self.to_ortho_axis(&scratch, ortho, 0);
            }

            fn from_ortho<S1, S2>(
                &self,
                ortho: &ArrayBase<S1, Dim<[usize; 2]>>,
                comp: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = T>,
                S2: Data<Elem = T> + DataMut,
            {
                let mut scratch = Array2::zeros((self.base0.len_ortho(), self.base1.len_spec()));
                self.from_ortho_axis(ortho, &mut scratch, 1);
                self.from_ortho_axis(&scratch, comp, 0);
            }
        }

        impl<A, T> SpaceDifferentiate<T, 2> for $space<$base0<A>, $base1<A>>
        where
            A: Real,
            T: ScalarOperand<A>
                + Add<$s, Output = T>
                + Mul<$s, Output = T>
                + Div<$s, Output = T>
                + Sub<$s, Output = T>,
        {
            apply_along_axis!(diff_axis, T, T, diff, "diff", order: usize,);
        }

        impl<A> SpaceTransform<A, 2> for $space<$base0<A>, $base1<A>>
        where
            A: Real + Scalar,
            Complex<A>: Scalar,
        {
            type Physical = $p;

            type Spectral = $s;

            fn forward<S1, S2>(
                &self,
                phys: &ArrayBase<S1, Dim<[usize; 2]>>,
                spec: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = Self::Physical>,
                S2: Data<Elem = Self::Spectral> + DataMut,
            {
                let mut scratch = Array2::zeros((self.base0.len_phys(), self.base1.len_spec()));
                self.forward_axis1(phys, &mut scratch);
                self.forward_axis0(&scratch, spec);
            }

            fn backward<S1, S2>(
                &self,
                spec: &ArrayBase<S1, Dim<[usize; 2]>>,
                phys: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = Self::Spectral>,
                S2: Data<Elem = Self::Physical> + DataMut,
            {
                let mut scratch = Array2::zeros((self.base0.len_phys(), self.base1.len_spec()));
                self.backward_axis0(spec, &mut scratch);
                self.backward_axis1(&scratch, phys);
            }
        }
    };
}
impl_space2!(Space2, BaseR2r, BaseR2r, A, A);
impl_space2!(Space2, BaseR2c, BaseR2r, A, Complex<A>);
impl_space2!(Space2, BaseC2c, BaseR2c, A, Complex<A>);
impl_space2!(Space2, BaseC2c, BaseC2c, Complex<A>, Complex<A>);
