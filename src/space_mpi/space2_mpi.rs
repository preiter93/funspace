//! Two-dimensional function space with mpi support
mod impl_serial;
use super::traits::{
    BaseSpaceMpiFromOrtho, BaseSpaceMpiGradient, BaseSpaceMpiSize, BaseSpaceMpiTransform,
};
use crate::space::traits::BaseSpaceSize;
use crate::traits::{BaseFromOrtho, BaseGradient, BaseSize, BaseTransform};
use crate::types::{FloatNum, ScalarNum};
use crate::{BaseC2c, BaseR2c, BaseR2r};
use mpi_crate::{datatype::Equivalence, environment::Universe, topology::Communicator};
use ndarray::{Array2, ArrayBase, Data, DataMut, Dim};
use num_complex::Complex;
use num_traits::Zero;
use pencil_decomp::Decomp2;
use std::collections::HashMap;
use std::ops::{Add, Div, Mul, Sub};

/// Organizes multiple pencil decompositions which is necessary to communicate
/// different sized array. The dimensionality of a space can change
/// when it is transformed from physical to spectral space,
/// or if transformed from galerkin to orthogonal space.
#[derive(Debug, Clone)]
pub struct Decomp2Handler<'a> {
    /// Map global array shape to ``Decomp2d``
    map: HashMap<[usize; 2], Decomp2<'a>>,
}

impl<'a> Decomp2Handler<'a> {
    /// Initialize multidecomp with empty decomp hashmap.
    /// Use `Self::insert` to add decompositions.
    #[must_use]
    pub fn empty() -> Self {
        let map: HashMap<[usize; 2], Decomp2<'a>> = HashMap::new();
        Self { map }
    }

    /// Add decomposition.
    // / # Panics
    // / Decomposition must have a unique domain shape.
    pub fn insert(&mut self, decomp: Decomp2<'a>) {
        let shape = decomp.shape_global();
        // Insert if not exist
        self.map.entry(shape).or_insert_with(|| decomp);
    }

    /// Return `Decomp2` which matches a given array shape.
    ///
    /// # Panics
    /// Shape not previously inserted to `Decomp2Handler`.
    /// See insert method.
    #[must_use]
    pub fn get(&self, shape: &[usize]) -> &Decomp2 {
        self.map.get(shape).unwrap()
    }
}

#[derive(Clone)]
pub struct Space2Mpi<'a, B0, B1> {
    // Intermediate <-> Spectral
    pub base0: B0,
    // Phsical <-> Intermediate
    pub base1: B1,
    // Mpi decomp Handler
    pub dcp: Decomp2Handler<'a>,
}

impl<'a, B0, B1> Space2Mpi<'a, B0, B1>
where
    B0: Clone + BaseSize,
    B1: Clone + BaseSize,
{
    /// Create a new space
    pub fn new(base0: &B0, base1: &B1, universe: &'a Universe) -> Self {
        // Initialize mpi domain decompositions.
        // Several decompositions may be required because the shape of the field in physical
        // space may be different from that in spectral space.
        let mut dcp = Decomp2Handler::empty();
        // Physical / Physical
        dcp.insert(Decomp2::new(
            universe,
            [base0.len_phys(), base1.len_phys()],
            [universe.world().size()],
            [false],
        ));
        // Spectral / Physical
        if base0.len_spec() != base0.len_phys() {
            dcp.insert(Decomp2::new(
                universe,
                [base0.len_spec(), base1.len_phys()],
                [universe.world().size()],
                [false],
            ));
        }
        // Physical / Spectral
        if base1.len_spec() != base1.len_phys() {
            dcp.insert(Decomp2::new(
                universe,
                [base0.len_phys(), base1.len_spec()],
                [universe.world().size()],
                [false],
            ));
        }
        // Spectral / Spectral
        if base0.len_spec() != base0.len_phys() && base1.len_spec() != base1.len_phys() {
            dcp.insert(Decomp2::new(
                universe,
                [base0.len_spec(), base1.len_spec()],
                [universe.world().size()],
                [false],
            ));
        }
        // Return
        Self {
            base0: base0.clone(),
            base1: base1.clone(),
            dcp,
        }
    }
}

impl<'a, B0, B1> Space2Mpi<'a, B0, B1>
where
    B0: Clone + BaseSize,
    B1: Clone + BaseSize,
    Self: BaseSpaceSize<2>,
{
    /// Shape in physical space (x pencil)
    pub fn shape_physical_x_pen(&self) -> [usize; 2] {
        let dcp = self.dcp.get(&self.shape_physical());
        dcp.x_pencil.shape()
    }

    /// Shape in physical space (y pencil)
    pub fn shape_physical_y_pen(&self) -> [usize; 2] {
        let dcp = self.dcp.get(&self.shape_physical());
        dcp.y_pencil.shape()
    }

    /// Shape in spectral space (x pencil)
    pub fn shape_spectral_x_pen(&self) -> [usize; 2] {
        let dcp = self.dcp.get(&self.shape_spectral());
        dcp.x_pencil.shape()
    }

    /// Shape in spectral space (y pencil)
    pub fn shape_spectral_y_pen(&self) -> [usize; 2] {
        let dcp = self.dcp.get(&self.shape_spectral());
        dcp.y_pencil.shape()
    }

    /// Transpose from x to y-pencil in *physical* space
    pub fn transpose_x_to_y_phys<S1, S2, T>(
        &self,
        x_pencil: &ArrayBase<S1, Dim<[usize; 2]>>,
        y_pencil: &mut ArrayBase<S2, Dim<[usize; 2]>>,
    ) where
        S1: Data<Elem = T>,
        S2: Data<Elem = T> + DataMut,
        T: Equivalence + Copy + Zero,
    {
        let shape = self.shape_physical();
        let dcp = self.dcp.get(&shape);
        dcp.transpose_x_to_y(x_pencil, y_pencil);
    }

    /// Transpose from y to x-pencil in *physical* space
    pub fn transpose_y_to_x_phys<S1, S2, T>(
        &self,
        x_pencil: &ArrayBase<S1, Dim<[usize; 2]>>,
        y_pencil: &mut ArrayBase<S2, Dim<[usize; 2]>>,
    ) where
        S1: Data<Elem = T>,
        S2: Data<Elem = T> + DataMut,
        T: Equivalence + Copy + Zero,
    {
        let shape = self.shape_physical();
        let dcp = self.dcp.get(&shape);
        dcp.transpose_y_to_x(x_pencil, y_pencil);
    }

    /// Transpose from x to y-pencil in *spectral* space
    pub fn transpose_x_to_y_spec<S1, S2, T>(
        &self,
        x_pencil: &ArrayBase<S1, Dim<[usize; 2]>>,
        y_pencil: &mut ArrayBase<S2, Dim<[usize; 2]>>,
    ) where
        S1: Data<Elem = T>,
        S2: Data<Elem = T> + DataMut,
        T: Equivalence + Copy + Zero,
    {
        let shape = self.shape_spectral();
        let dcp = self.dcp.get(&shape);
        dcp.transpose_x_to_y(x_pencil, y_pencil);
    }

    /// Transpose from y to x-pencil in *spectral* space
    pub fn transpose_y_to_x_spec<S1, S2, T>(
        &self,
        x_pencil: &ArrayBase<S1, Dim<[usize; 2]>>,
        y_pencil: &mut ArrayBase<S2, Dim<[usize; 2]>>,
    ) where
        S1: Data<Elem = T>,
        S2: Data<Elem = T> + DataMut,
        T: Equivalence + Copy + Zero,
    {
        let shape = self.shape_spectral();
        let dcp = self.dcp.get(&shape);
        dcp.transpose_y_to_x(x_pencil, y_pencil);
    }
}

macro_rules! impl_space2_mpi {
    ($space: ident, $base0: ident, $base1: ident, $p: ty, $s: ty) => {
        impl<A, T> BaseSpaceMpiGradient<A, T, 2> for $space<'_, $base0<A>, $base1<A>>
        where
            A: FloatNum + ScalarNum,
            T: ScalarNum
                + From<A>
                + Add<A, Output = T>
                + Mul<A, Output = T>
                + Div<A, Output = T>
                + Sub<A, Output = T>
                + Add<$s, Output = T>
                + Mul<$s, Output = T>
                + Div<$s, Output = T>
                + Sub<$s, Output = T>
                + Equivalence,
        {
            fn gradient_mpi<S>(
                &self,
                input: &ArrayBase<S, Dim<[usize; 2]>>,
                deriv: [usize; 2],
                scale: Option<[A; 2]>,
            ) -> Array2<T>
            where
                S: Data<Elem = T>,
            {
                // differentiate
                let scratch_y_pen = {
                    let dcp = self
                        .dcp
                        .get(&[self.base0.len_orth(), self.base1.len_spec()]);
                    let mut work_y_pen = Array2::zeros(dcp.y_pencil.shape());
                    dcp.transpose_x_to_y(&self.base0.gradient(input, deriv[0], 0), &mut work_y_pen);
                    self.base1.gradient(&work_y_pen, deriv[1], 1)
                };

                // transform back to x pencil
                let dcp = self
                    .dcp
                    .get(&[self.base0.len_orth(), self.base1.len_orth()]);
                let mut output = Array2::zeros(dcp.x_pencil.shape());
                dcp.transpose_y_to_x(&scratch_y_pen, &mut output);

                // rescale
                if let Some(s) = scale {
                    let sc: T = (s[0].powi(deriv[0] as i32) * s[1].powi(deriv[1] as i32)).into();
                    for x in output.iter_mut() {
                        *x /= sc;
                    }
                }
                output
            }
        }

        impl<A, T> BaseSpaceMpiFromOrtho<A, T, 2> for $space<'_, $base0<A>, $base1<A>>
        where
            A: FloatNum + ScalarNum,
            T: ScalarNum
                + From<A>
                + Add<A, Output = T>
                + Mul<A, Output = T>
                + Div<A, Output = T>
                + Sub<A, Output = T>
                + Add<$s, Output = T>
                + Mul<$s, Output = T>
                + Div<$s, Output = T>
                + Sub<$s, Output = T>
                + Equivalence,
        {
            fn to_ortho_inplace_mpi<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 2]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = T>,
                S2: DataMut<Elem = T>,
            {
                // composite base coefficients -> orthogonal base coefficients
                let scratch_y_pen = {
                    let dcp = self
                        .dcp
                        .get(&[self.base0.len_orth(), self.base1.len_spec()]);
                    let mut work_y_pen = Array2::zeros(dcp.y_pencil.shape());
                    dcp.transpose_x_to_y(&self.base0.to_ortho(input, 0), &mut work_y_pen);
                    self.base1.to_ortho(&work_y_pen, 1)
                };

                // transform back to x pencil
                {
                    let dcp = self
                        .dcp
                        .get(&[self.base0.len_orth(), self.base1.len_orth()]);
                    dcp.transpose_y_to_x(&scratch_y_pen, output);
                }
            }

            fn from_ortho_inplace_mpi<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 2]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = T>,
                S2: DataMut<Elem = T>,
            {
                // orthogonal base coefficients -> composite base coefficients
                let scratch_y_pen = {
                    let dcp = self
                        .dcp
                        .get(&[self.base0.len_spec(), self.base1.len_orth()]);
                    let mut work_y_pen = Array2::zeros(dcp.y_pencil.shape());
                    dcp.transpose_x_to_y(&self.base0.from_ortho(input, 0), &mut work_y_pen);
                    self.base1.from_ortho(&work_y_pen, 1)
                };

                // transform back to x pencil
                {
                    let dcp = self
                        .dcp
                        .get(&[self.base0.len_spec(), self.base1.len_spec()]);
                    dcp.transpose_y_to_x(&scratch_y_pen, output);
                }
            }
        }

        impl<A> BaseSpaceMpiTransform<A, 2> for $space<'_, $base0<A>, $base1<A>>
        where
            A: FloatNum + ScalarNum + Equivalence,
            Complex<A>: ScalarNum + Equivalence,
        {
            fn forward_inplace_mpi<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 2]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = Self::Physical>,
                S2: Data<Elem = Self::Spectral> + DataMut,
            {
                // axis 1
                let scratch_y_pen = self.base1.forward(input, 1);

                // Transpose
                let dcp = self
                    .dcp
                    .get(&[self.base0.len_phys(), self.base1.len_spec()]);
                let mut scratch_x_pen = Array2::zeros(dcp.x_pencil.shape());
                dcp.transpose_y_to_x(&scratch_y_pen, &mut scratch_x_pen);

                // axis 0
                self.base0.forward_inplace(&scratch_x_pen, output, 0);
            }

            fn backward_inplace_mpi<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 2]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = Self::Spectral>,
                S2: Data<Elem = Self::Physical> + DataMut,
            {
                // axis 0
                let scratch_x_pen = self.base0.backward(input, 0);

                // Transpose
                let dcp = self
                    .dcp
                    .get(&[self.base0.len_phys(), self.base1.len_spec()]);
                let mut scratch_y_pen = Array2::zeros(dcp.y_pencil.shape());
                dcp.transpose_x_to_y(&scratch_x_pen, &mut scratch_y_pen);

                // axis 0
                self.base1.backward_inplace(&scratch_y_pen, output, 1);
            }
        }

        impl<A> BaseSpaceMpiSize<2> for $space<'_, $base0<A>, $base1<A>>
        where
            A: FloatNum,
        {
            fn shape_physical_mpi(&self) -> [usize; 2] {
                let dcp = self.dcp.get(&self.shape_physical());
                dcp.y_pencil.shape()
            }

            fn shape_spectral_mpi(&self) -> [usize; 2] {
                let dcp = self.dcp.get(&self.shape_spectral());
                dcp.x_pencil.shape()
            }

            fn shape_spectral_ortho_mpi(&self) -> [usize; 2] {
                let dcp = self.dcp.get(&self.shape_spectral_ortho());
                dcp.x_pencil.shape()
            }
        }
    };
}

impl_space2_mpi!(Space2Mpi, BaseR2r, BaseR2r, A, A);
impl_space2_mpi!(Space2Mpi, BaseR2c, BaseR2r, A, Complex<A>);
impl_space2_mpi!(Space2Mpi, BaseC2c, BaseR2c, A, Complex<A>);
/*impl<'a, B0, B1> Space2Mpi<'a, B0, B1>
where
    B0: Clone + BaseSize,
    B1: Clone + BaseSize,
{
    /// Create a new space
    pub fn new(base0: &B0, base1: &B1, universe: &'a Universe) -> Self {
        // Initialize mpi domain decompositions.
        // Several decompositions may be required because the shape of the field in physical
        // space may be different from that in spectral space.
        let mut decomp_handler = DecompHandler::new(&universe);
        decomp_handler.insert(Decomp2d::new(
            universe,
            [base0.len_phys(), base1.len_phys()],
        ));
        if base0.len_spec() != base0.len_phys() {
            decomp_handler.insert(Decomp2d::new(
                universe,
                [base0.len_spec(), base1.len_phys()],
            ));
        }
        if base1.len_spec() != base1.len_phys() {
            decomp_handler.insert(Decomp2d::new(
                universe,
                [base0.len_phys(), base1.len_spec()],
            ));
        }
        if base0.len_spec() != base0.len_phys() && base1.len_spec() != base1.len_phys() {
            decomp_handler.insert(Decomp2d::new(
                universe,
                [base0.len_spec(), base1.len_spec()],
            ));
        }
        Self {
            base0: base0.clone(),
            base1: base1.clone(),
            decomp_handler,
        }
    }
}*/
