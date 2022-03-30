//! Two-dimensional function space with mpi support
#![cfg(feature = "mpi")]
use super::traits::BaseSpaceMpi;
use super::Communicator;
use super::Decomp2d;
use super::DecompHandler;
use super::Equivalence;
use super::Universe;
use crate::traits::{FunspaceElemental, FunspaceExtended, FunspaceSize};
use crate::BaseKind;
use crate::BaseSpace;
use crate::{BaseC2c, BaseR2c, BaseR2r, FloatNum, ScalarNum};
use ndarray::{Array, Array1, Array2, ArrayBase, Data, DataMut, Dim};
use num_complex::Complex;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Clone)]
pub struct Space2<'a, B0, B1> {
    // Intermediate <-> Spectral
    pub base0: B0,
    // Phsical <-> Intermediate
    pub base1: B1,
    // Mpi Handler
    pub decomp_handler: DecompHandler<'a>,
}

impl<'a, B0, B1> Space2<'a, B0, B1>
where
    B0: Clone + FunspaceSize,
    B1: Clone + FunspaceSize,
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
}

/// Exactly as implementation in 'crate::space2'
macro_rules! impl_space2 {
    ($base0: ident, $base1: ident, $p: ty, $s: ty) => {
        impl<A> BaseSpace<A, 2> for Space2<'_, $base0<A>, $base1<A>>
        where
            A: FloatNum + ScalarNum,
            Complex<A>: ScalarNum,
        {
            type Physical = $p;

            type Spectral = $s;

            fn shape_physical(&self) -> [usize; 2] {
                [self.base0.len_phys(), self.base1.len_phys()]
            }

            fn shape_spectral(&self) -> [usize; 2] {
                [self.base0.len_spec(), self.base1.len_spec()]
            }

            fn ndarray_physical(&self) -> Array2<Self::Physical> {
                let shape = [self.base0.len_phys(), self.base1.len_phys()];
                Array2::zeros(shape)
            }

            fn ndarray_spectral(&self) -> Array2<Self::Spectral> {
                let shape = [self.base0.len_spec(), self.base1.len_spec()];
                Array2::zeros(shape)
            }

            fn laplace(&self, axis: usize) -> Array2<A> {
                if axis == 0 {
                    self.base0.laplace()
                } else {
                    self.base1.laplace()
                }
            }

            fn laplace_inv(&self, axis: usize) -> Array2<A> {
                if axis == 0 {
                    self.base0.laplace_inv()
                } else {
                    self.base1.laplace_inv()
                }
            }

            fn laplace_inv_eye(&self, axis: usize) -> Array2<A> {
                if axis == 0 {
                    self.base0.laplace_inv_eye()
                } else {
                    self.base1.laplace_inv_eye()
                }
            }

            fn mass(&self, axis: usize) -> Array2<A> {
                if axis == 0 {
                    self.base0.mass()
                } else {
                    self.base1.mass()
                }
            }

            // fn diff_op(&self, deriv: usize, axis: usize) -> Array2<Self::Spectral> {
            //     if axis == 0 {
            //         self.base0.diff_op(deriv)
            //     } else {
            //         self.base1.diff_op(deriv)
            //     }
            // }
            //
            // fn stencil(&self, axis: usize) -> Option<Array2<Self::Spectral>> {
            //     if axis == 0 {
            //         self.base0.stencil()
            //     } else {
            //         self.base1.stencil()
            //     }
            // }

            fn coords(&self) -> [Array1<A>; 2] {
                [self.coords_axis(0), self.coords_axis(1)]
            }

            fn coords_axis(&self, axis: usize) -> Array1<A> {
                if axis == 0 {
                    self.base0.get_nodes().into()
                } else {
                    self.base1.get_nodes().into()
                }
            }

            fn base_kind(&self, axis: usize) -> BaseKind {
                if axis == 0 {
                    self.base0.base_kind()
                } else {
                    self.base1.base_kind()
                }
            }

            fn to_ortho<T, S>(
                &self,
                input: &ArrayBase<S, Dim<[usize; 2]>>,
            ) -> Array<T, Dim<[usize; 2]>>
            where
                T: ScalarNum
                    + Add<A, Output = T>
                    + Mul<A, Output = T>
                    + Div<A, Output = T>
                    + Sub<A, Output = T>
                    + Add<Self::Spectral, Output = T>
                    + Mul<Self::Spectral, Output = T>
                    + Div<Self::Spectral, Output = T>
                    + Sub<Self::Spectral, Output = T>,
                S: Data<Elem = T>,
            {
                let buffer = self.base0.to_ortho(input, 0);
                self.base1.to_ortho(&buffer, 1)
            }

            fn to_ortho_inplace<T, S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 2]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                T: ScalarNum
                    + Add<A, Output = T>
                    + Mul<A, Output = T>
                    + Div<A, Output = T>
                    + Sub<A, Output = T>
                    + Add<Self::Spectral, Output = T>
                    + Mul<Self::Spectral, Output = T>
                    + Div<Self::Spectral, Output = T>
                    + Sub<Self::Spectral, Output = T>,
                S1: Data<Elem = T>,
                S2: Data<Elem = T> + DataMut,
            {
                let buffer = self.base0.to_ortho(input, 0);
                self.base1.to_ortho_inplace(&buffer, output, 1);
            }

            fn from_ortho<T, S>(
                &self,
                input: &ArrayBase<S, Dim<[usize; 2]>>,
            ) -> Array<T, Dim<[usize; 2]>>
            where
                T: ScalarNum
                    + Add<A, Output = T>
                    + Mul<A, Output = T>
                    + Div<A, Output = T>
                    + Sub<A, Output = T>
                    + Add<Self::Spectral, Output = T>
                    + Mul<Self::Spectral, Output = T>
                    + Div<Self::Spectral, Output = T>
                    + Sub<Self::Spectral, Output = T>,
                S: Data<Elem = T>,
            {
                let buffer = self.base0.from_ortho(input, 0);
                self.base1.from_ortho(&buffer, 1)
            }

            fn from_ortho_inplace<T, S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 2]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                T: ScalarNum
                    + Add<A, Output = T>
                    + Mul<A, Output = T>
                    + Div<A, Output = T>
                    + Sub<A, Output = T>
                    + Add<Self::Spectral, Output = T>
                    + Mul<Self::Spectral, Output = T>
                    + Div<Self::Spectral, Output = T>
                    + Sub<Self::Spectral, Output = T>,
                S1: Data<Elem = T>,
                S2: Data<Elem = T> + DataMut,
            {
                let buffer = self.base0.from_ortho(input, 0);
                self.base1.from_ortho_inplace(&buffer, output, 1);
            }

            fn to_ortho_par<T, S>(
                &self,
                input: &ArrayBase<S, Dim<[usize; 2]>>,
            ) -> Array<T, Dim<[usize; 2]>>
            where
                T: ScalarNum
                    + Add<A, Output = T>
                    + Mul<A, Output = T>
                    + Div<A, Output = T>
                    + Sub<A, Output = T>
                    + Add<Self::Spectral, Output = T>
                    + Mul<Self::Spectral, Output = T>
                    + Div<Self::Spectral, Output = T>
                    + Sub<Self::Spectral, Output = T>
                    + Send
                    + Sync,
                S: Data<Elem = T>,
            {
                let buffer = self.base0.to_ortho_par(input, 0);
                self.base1.to_ortho_par(&buffer, 1)
            }

            fn to_ortho_inplace_par<T, S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 2]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                T: ScalarNum
                    + Add<A, Output = T>
                    + Mul<A, Output = T>
                    + Div<A, Output = T>
                    + Sub<A, Output = T>
                    + Add<Self::Spectral, Output = T>
                    + Mul<Self::Spectral, Output = T>
                    + Div<Self::Spectral, Output = T>
                    + Sub<Self::Spectral, Output = T>
                    + Send
                    + Sync,
                S1: Data<Elem = T>,
                S2: Data<Elem = T> + DataMut,
            {
                let buffer = self.base0.to_ortho_par(input, 0);
                self.base1.to_ortho_inplace_par(&buffer, output, 1);
            }

            fn from_ortho_par<T, S>(
                &self,
                input: &ArrayBase<S, Dim<[usize; 2]>>,
            ) -> Array<T, Dim<[usize; 2]>>
            where
                T: ScalarNum
                    + Add<A, Output = T>
                    + Mul<A, Output = T>
                    + Div<A, Output = T>
                    + Sub<A, Output = T>
                    + Add<Self::Spectral, Output = T>
                    + Mul<Self::Spectral, Output = T>
                    + Div<Self::Spectral, Output = T>
                    + Sub<Self::Spectral, Output = T>
                    + Send
                    + Sync,
                S: Data<Elem = T>,
            {
                let buffer = self.base0.from_ortho_par(input, 0);
                self.base1.from_ortho_par(&buffer, 1)
            }

            fn from_ortho_inplace_par<T, S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 2]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                T: ScalarNum
                    + Add<A, Output = T>
                    + Mul<A, Output = T>
                    + Div<A, Output = T>
                    + Sub<A, Output = T>
                    + Add<Self::Spectral, Output = T>
                    + Mul<Self::Spectral, Output = T>
                    + Div<Self::Spectral, Output = T>
                    + Sub<Self::Spectral, Output = T>
                    + Send
                    + Sync,
                S1: Data<Elem = T>,
                S2: Data<Elem = T> + DataMut,
            {
                let buffer = self.base0.from_ortho_par(input, 0);
                self.base1.from_ortho_inplace_par(&buffer, output, 1);
            }

            fn gradient<T, S>(
                &self,
                input: &ArrayBase<S, Dim<[usize; 2]>>,
                deriv: [usize; 2],
                scale: Option<[A; 2]>,
            ) -> Array<T, Dim<[usize; 2]>>
            where
                T: ScalarNum
                    + Add<A, Output = T>
                    + Mul<A, Output = T>
                    + Div<A, Output = T>
                    + Sub<A, Output = T>
                    + Add<Self::Spectral, Output = T>
                    + Mul<Self::Spectral, Output = T>
                    + Div<Self::Spectral, Output = T>
                    + Sub<Self::Spectral, Output = T>
                    + From<A>,
                S: Data<Elem = T>,
            {
                let buffer = self.base0.differentiate(input, deriv[0], 0);
                let mut output = self.base1.differentiate(&buffer, deriv[1], 1);
                if let Some(s) = scale {
                    let sc: T = (s[0].powi(deriv[0] as i32) * s[1].powi(deriv[1] as i32)).into();
                    for x in output.iter_mut() {
                        *x /= sc;
                    }
                }
                output
            }

            fn gradient_par<T, S>(
                &self,
                input: &ArrayBase<S, Dim<[usize; 2]>>,
                deriv: [usize; 2],
                scale: Option<[A; 2]>,
            ) -> Array<T, Dim<[usize; 2]>>
            where
                T: ScalarNum
                    + Add<A, Output = T>
                    + Mul<A, Output = T>
                    + Div<A, Output = T>
                    + Sub<A, Output = T>
                    + Add<Self::Spectral, Output = T>
                    + Mul<Self::Spectral, Output = T>
                    + Div<Self::Spectral, Output = T>
                    + Sub<Self::Spectral, Output = T>
                    + From<A>
                    + Send
                    + Sync,
                S: Data<Elem = T>,
            {
                let buffer = self.base0.differentiate_par(input, deriv[0], 0);
                let mut output = self.base1.differentiate_par(&buffer, deriv[1], 1);
                if let Some(s) = scale {
                    let sc: T = (s[0].powi(deriv[0] as i32) * s[1].powi(deriv[1] as i32)).into();
                    for x in output.iter_mut() {
                        *x /= sc;
                    }
                }
                output
            }

            fn forward<S>(
                &self,
                input: &ArrayBase<S, Dim<[usize; 2]>>,
            ) -> Array<Self::Spectral, Dim<[usize; 2]>>
            where
                S: Data<Elem = Self::Physical>,
            {
                let buffer = self.base1.forward(input, 1);
                self.base0.forward(&buffer, 0)
            }

            fn forward_inplace<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 2]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = Self::Physical>,
                S2: Data<Elem = Self::Spectral> + DataMut,
            {
                let buffer = self.base1.forward(input, 1);
                self.base0.forward_inplace(&buffer, output, 0);
            }

            fn backward<S>(
                &self,
                input: &ArrayBase<S, Dim<[usize; 2]>>,
            ) -> Array<Self::Physical, Dim<[usize; 2]>>
            where
                S: Data<Elem = Self::Spectral>,
            {
                let buffer = self.base0.backward(input, 0);
                self.base1.backward(&buffer, 1)
            }

            fn backward_inplace<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 2]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = Self::Spectral>,
                S2: Data<Elem = Self::Physical> + DataMut,
            {
                let buffer = self.base0.backward(input, 0);
                self.base1.backward_inplace(&buffer, output, 1);
            }

            fn forward_par<S>(
                &self,
                input: &ArrayBase<S, Dim<[usize; 2]>>,
            ) -> Array<Self::Spectral, Dim<[usize; 2]>>
            where
                S: Data<Elem = Self::Physical>,
            {
                let buffer = self.base1.forward_par(input, 1);
                self.base0.forward_par(&buffer, 0)
            }

            fn forward_inplace_par<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 2]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = Self::Physical>,
                S2: Data<Elem = Self::Spectral> + DataMut,
            {
                let buffer = self.base1.forward_par(input, 1);
                self.base0.forward_inplace_par(&buffer, output, 0);
            }

            fn backward_par<S>(
                &self,
                input: &ArrayBase<S, Dim<[usize; 2]>>,
            ) -> Array<Self::Physical, Dim<[usize; 2]>>
            where
                S: Data<Elem = Self::Spectral>,
            {
                let buffer = self.base0.backward_par(input, 0);
                self.base1.backward_par(&buffer, 1)
            }

            fn backward_inplace_par<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 2]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = Self::Spectral>,
                S2: Data<Elem = Self::Physical> + DataMut,
            {
                let buffer = self.base0.backward_par(input, 0);
                self.base1.backward_inplace_par(&buffer, output, 1);
            }
        }
    };
}
impl_space2!(BaseR2r, BaseR2r, A, A);
impl_space2!(BaseR2c, BaseR2r, A, Complex<A>);
impl_space2!(BaseC2c, BaseR2c, A, Complex<A>);
impl_space2!(BaseC2c, BaseC2c, Complex<A>, Complex<A>);

macro_rules! impl_space2_mpi {
    ($base0: ident, $base1: ident, $p: ty, $s: ty) => {
        impl<A> BaseSpaceMpi<A, 2> for Space2<'_, $base0<A>, $base1<A>>
        where
            A: FloatNum + ScalarNum + Equivalence,
            Complex<A>: ScalarNum + Equivalence,
        {
            fn get_universe(&self) -> &Universe {
                self.decomp_handler.universe
            }

            fn get_nrank(&self) -> usize {
                let world = self.get_universe().world();
                world.rank() as usize
            }

            fn get_nprocs(&self) -> usize {
                let world = self.get_universe().world();
                world.size() as usize
            }

            /// Return decomposition which matches a given global arrays shape.
            fn get_decomp_from_global_shape(&self, shape: &[usize]) -> &Decomp2d {
                self.decomp_handler.get_decomp_from_global_shape(shape)
            }

            fn shape_physical_x_pen(&self) -> [usize; 2] {
                let dcp = self.get_decomp_from_global_shape(&self.shape_physical());
                dcp.x_pencil.sz
            }

            fn shape_physical_y_pen(&self) -> [usize; 2] {
                let dcp = self.get_decomp_from_global_shape(&self.shape_physical());
                dcp.y_pencil.sz
            }

            fn shape_spectral_x_pen(&self) -> [usize; 2] {
                let dcp = self.get_decomp_from_global_shape(&self.shape_spectral());
                dcp.x_pencil.sz
            }

            fn shape_spectral_y_pen(&self) -> [usize; 2] {
                let dcp = self.get_decomp_from_global_shape(&self.shape_spectral());
                dcp.y_pencil.sz
            }

            fn ndarray_physical_x_pen(&self) -> Array2<Self::Physical> {
                let shape = self.shape_physical_x_pen();
                Array2::zeros(shape)
            }

            fn ndarray_physical_y_pen(&self) -> Array2<Self::Physical> {
                let shape = self.shape_physical_y_pen();
                Array2::zeros(shape)
            }

            fn ndarray_spectral_x_pen(&self) -> Array2<Self::Spectral> {
                let shape = self.shape_spectral_x_pen();
                Array2::zeros(shape)
            }

            fn ndarray_spectral_y_pen(&self) -> Array2<Self::Spectral> {
                let shape = self.shape_spectral_y_pen();
                Array2::zeros(shape)
            }

            fn to_ortho_mpi<S>(
                &self,
                input: &ArrayBase<S, Dim<[usize; 2]>>,
            ) -> Array<Self::Spectral, Dim<[usize; 2]>>
            where
                S: Data<Elem = Self::Spectral>,
            {
                // composite base coefficients -> orthogonal base coefficients
                let work = {
                    let dcp = self.get_decomp_from_global_shape(&[
                        self.base0.len_orth(),
                        self.base1.len_spec(),
                    ]);
                    let mut buffer_ypen = Array2::zeros(dcp.y_pencil.sz);
                    dcp.transpose_x_to_y(&self.base0.to_ortho(input, 0), &mut buffer_ypen);
                    self.base1.to_ortho(&buffer_ypen, 1)
                };

                // transform back to x pencil
                {
                    let dcp = self.get_decomp_from_global_shape(&[
                        self.base0.len_orth(),
                        self.base1.len_orth(),
                    ]);
                    let mut buffer_xpen = Array2::zeros(dcp.x_pencil.sz);
                    dcp.transpose_y_to_x(&work, &mut buffer_xpen);
                    buffer_xpen
                }
            }

            fn to_ortho_inplace_mpi<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 2]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = Self::Spectral>,
                S2: Data<Elem = Self::Spectral> + DataMut,
            {
                // composite base coefficients -> orthogonal base coefficients
                let work = {
                    let dcp = self.get_decomp_from_global_shape(&[
                        self.base0.len_orth(),
                        self.base1.len_spec(),
                    ]);
                    let mut buffer_ypen = Array2::zeros(dcp.y_pencil.sz);
                    dcp.transpose_x_to_y(&self.base0.to_ortho(input, 0), &mut buffer_ypen);
                    self.base1.to_ortho(&buffer_ypen, 1)
                };

                // transform back to x pencil
                {
                    let dcp = self.get_decomp_from_global_shape(&[
                        self.base0.len_orth(),
                        self.base1.len_orth(),
                    ]);
                    dcp.transpose_y_to_x(&work, output);
                }
            }

            fn from_ortho_mpi<S>(
                &self,
                input: &ArrayBase<S, Dim<[usize; 2]>>,
            ) -> Array<Self::Spectral, Dim<[usize; 2]>>
            where
                S: Data<Elem = Self::Spectral>,
            {
                // orthogonal base coefficients -> composite base coefficients
                let work = {
                    let dcp = self.get_decomp_from_global_shape(&[
                        self.base0.len_spec(),
                        self.base1.len_orth(),
                    ]);
                    let mut buffer_ypen = Array2::zeros(dcp.y_pencil.sz);
                    dcp.transpose_x_to_y(&self.base0.from_ortho(input, 0), &mut buffer_ypen);
                    self.base1.from_ortho(&buffer_ypen, 1)
                };

                // transform back to x pencil
                {
                    let dcp = self.get_decomp_from_global_shape(&[
                        self.base0.len_spec(),
                        self.base1.len_spec(),
                    ]);
                    let mut buffer_xpen = Array2::zeros(dcp.x_pencil.sz);
                    dcp.transpose_y_to_x(&work, &mut buffer_xpen);
                    buffer_xpen
                }
            }

            fn from_ortho_inplace_mpi<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 2]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = Self::Spectral>,
                S2: Data<Elem = Self::Spectral> + DataMut,
            {
                // orthogonal base coefficients -> composite base coefficients
                let work = {
                    let dcp = self.get_decomp_from_global_shape(&[
                        self.base0.len_spec(),
                        self.base1.len_orth(),
                    ]);
                    let mut buffer_ypen = Array2::zeros(dcp.y_pencil.sz);
                    dcp.transpose_x_to_y(&self.base0.from_ortho(input, 0), &mut buffer_ypen);
                    self.base1.from_ortho(&buffer_ypen, 1)
                };

                // transform back to x pencil
                {
                    let dcp = self.get_decomp_from_global_shape(&[
                        self.base0.len_spec(),
                        self.base1.len_spec(),
                    ]);
                    dcp.transpose_y_to_x(&work, output);
                }
            }

            fn gradient_mpi<S>(
                &self,
                input: &ArrayBase<S, Dim<[usize; 2]>>,
                deriv: [usize; 2],
                scale: Option<[A; 2]>,
            ) -> Array<Self::Spectral, Dim<[usize; 2]>>
            where
                S: Data<Elem = Self::Spectral>,
            {
                // differentiate
                let work = {
                    let dcp = self.get_decomp_from_global_shape(&[
                        self.base0.len_orth(),
                        self.base1.len_spec(),
                    ]);
                    let mut buffer_ypen = Array2::zeros(dcp.y_pencil.sz);
                    dcp.transpose_x_to_y(
                        &self.base0.differentiate(input, deriv[0], 0),
                        &mut buffer_ypen,
                    );
                    self.base1.differentiate(&buffer_ypen, deriv[1], 1)
                };

                // transform back to x pencil
                {
                    let dcp = self.get_decomp_from_global_shape(&[
                        self.base0.len_orth(),
                        self.base1.len_orth(),
                    ]);
                    let mut output = Array2::zeros(dcp.x_pencil.sz);
                    dcp.transpose_y_to_x(&work, &mut output);

                    // rescale (optional)
                    if let Some(s) = scale {
                        let sc: Self::Spectral =
                            (s[0].powi(deriv[0] as i32) * s[1].powi(deriv[1] as i32)).into();
                        for x in output.iter_mut() {
                            *x /= sc;
                        }
                    }
                    output
                }
            }

            fn forward_mpi<S>(
                &self,
                input: &ArrayBase<S, Dim<[usize; 2]>>,
            ) -> Array<Self::Spectral, Dim<[usize; 2]>>
            where
                S: Data<Elem = Self::Physical>,
            {
                // axis 1
                let buffer = self.base1.forward(input, 1);
                let dcp = self
                    .get_decomp_from_global_shape(&[self.base0.len_phys(), self.base1.len_spec()]);
                let mut buffer_xpen = Array2::zeros(dcp.x_pencil.sz);
                // axis 0
                dcp.transpose_y_to_x(&buffer, &mut buffer_xpen);
                self.base0.forward(&buffer_xpen, 0)
            }

            fn forward_inplace_mpi<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 2]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = Self::Physical>,
                S2: Data<Elem = Self::Spectral> + DataMut,
            {
                // axis 1
                let buffer = self.base1.forward(input, 1);
                let dcp = self
                    .get_decomp_from_global_shape(&[self.base0.len_phys(), self.base1.len_spec()]);
                let mut buffer_xpen = Array2::zeros(dcp.x_pencil.sz);
                // axis 0
                dcp.transpose_y_to_x(&buffer, &mut buffer_xpen);
                self.base0.forward_inplace(&buffer_xpen, output, 0);
            }

            fn backward_mpi<S>(
                &self,
                input: &ArrayBase<S, Dim<[usize; 2]>>,
            ) -> Array<Self::Physical, Dim<[usize; 2]>>
            where
                S: Data<Elem = Self::Spectral>,
            {
                // axis 0
                let buffer = self.base0.backward(input, 0);
                let dcp = self
                    .get_decomp_from_global_shape(&[self.base0.len_phys(), self.base1.len_spec()]);
                let mut buffer_ypen = Array2::zeros(dcp.y_pencil.sz);
                // axis 1
                dcp.transpose_x_to_y(&buffer, &mut buffer_ypen);
                self.base1.backward(&buffer_ypen, 1)
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
                let buffer = self.base0.backward(input, 0);
                let dcp = self
                    .get_decomp_from_global_shape(&[self.base0.len_phys(), self.base1.len_spec()]);
                let mut buffer_ypen = Array2::zeros(dcp.y_pencil.sz);
                // axis 1
                dcp.transpose_x_to_y(&buffer, &mut buffer_ypen);
                self.base1.backward_inplace(&buffer_ypen, output, 1)
            }

            fn gather_from_x_pencil_phys<S1>(&self, pencil_data: &ArrayBase<S1, Dim<[usize; 2]>>)
            where
                S1: Data<Elem = Self::Physical>,
            {
                let shape = self.shape_physical();
                let dcp = self.get_decomp_from_global_shape(&shape);
                dcp.gather_x(pencil_data);
            }

            fn gather_from_x_pencil_phys_root<S1, S2>(
                &self,
                pencil_data: &ArrayBase<S1, Dim<[usize; 2]>>,
                global_data: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = Self::Physical>,
                S2: Data<Elem = Self::Physical> + DataMut,
            {
                let shape = self.shape_physical();
                let dcp = self.get_decomp_from_global_shape(&shape);
                dcp.gather_x_inplace_root(pencil_data, global_data);
            }

            fn gather_from_y_pencil_phys<S1>(&self, pencil_data: &ArrayBase<S1, Dim<[usize; 2]>>)
            where
                S1: Data<Elem = Self::Physical>,
            {
                let shape = self.shape_physical();
                let dcp = self.get_decomp_from_global_shape(&shape);
                dcp.gather_y(pencil_data);
            }

            fn gather_from_y_pencil_phys_root<S1, S2>(
                &self,
                pencil_data: &ArrayBase<S1, Dim<[usize; 2]>>,
                global_data: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = Self::Physical>,
                S2: Data<Elem = Self::Physical> + DataMut,
            {
                let shape = self.shape_physical();
                let dcp = self.get_decomp_from_global_shape(&shape);
                dcp.gather_y_inplace_root(pencil_data, global_data);
            }

            fn gather_from_x_pencil_spec<S1>(&self, pencil_data: &ArrayBase<S1, Dim<[usize; 2]>>)
            where
                S1: Data<Elem = Self::Spectral>,
            {
                let shape = self.shape_spectral();
                let dcp = self.get_decomp_from_global_shape(&shape);
                dcp.gather_x(pencil_data);
            }

            fn gather_from_x_pencil_spec_root<S1, S2>(
                &self,
                pencil_data: &ArrayBase<S1, Dim<[usize; 2]>>,
                global_data: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = Self::Spectral>,
                S2: Data<Elem = Self::Spectral> + DataMut,
            {
                let shape = self.shape_spectral();
                let dcp = self.get_decomp_from_global_shape(&shape);
                dcp.gather_x_inplace_root(pencil_data, global_data);
            }

            fn gather_from_y_pencil_spec<S1>(&self, pencil_data: &ArrayBase<S1, Dim<[usize; 2]>>)
            where
                S1: Data<Elem = Self::Spectral>,
            {
                let shape = self.shape_spectral();
                let dcp = self.get_decomp_from_global_shape(&shape);
                dcp.gather_y(pencil_data);
            }

            fn gather_from_y_pencil_spec_root<S1, S2>(
                &self,
                pencil_data: &ArrayBase<S1, Dim<[usize; 2]>>,
                global_data: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = Self::Spectral>,
                S2: Data<Elem = Self::Spectral> + DataMut,
            {
                let shape = self.shape_spectral();
                let dcp = self.get_decomp_from_global_shape(&shape);
                dcp.gather_y_inplace_root(pencil_data, global_data);
            }

            fn all_gather_from_x_pencil_phys<S1, S2>(
                &self,
                pencil_data: &ArrayBase<S1, Dim<[usize; 2]>>,
                global_data: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = Self::Physical>,
                S2: Data<Elem = Self::Physical> + DataMut,
            {
                let shape = self.shape_physical();
                let dcp = self.get_decomp_from_global_shape(&shape);
                dcp.all_gather_x(pencil_data, global_data);
            }

            fn all_gather_from_y_pencil_phys<S1, S2>(
                &self,
                pencil_data: &ArrayBase<S1, Dim<[usize; 2]>>,
                global_data: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = Self::Physical>,
                S2: Data<Elem = Self::Physical> + DataMut,
            {
                let shape = self.shape_physical();
                let dcp = self.get_decomp_from_global_shape(&shape);
                dcp.all_gather_y(pencil_data, global_data);
            }

            fn all_gather_from_x_pencil_spec<S1, S2>(
                &self,
                pencil_data: &ArrayBase<S1, Dim<[usize; 2]>>,
                global_data: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = Self::Spectral>,
                S2: Data<Elem = Self::Spectral> + DataMut,
            {
                let shape = self.shape_spectral();
                let dcp = self.get_decomp_from_global_shape(&shape);
                dcp.all_gather_x(pencil_data, global_data);
            }

            fn all_gather_from_y_pencil_spec<S1, S2>(
                &self,
                pencil_data: &ArrayBase<S1, Dim<[usize; 2]>>,
                global_data: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = Self::Spectral>,
                S2: Data<Elem = Self::Spectral> + DataMut,
            {
                let shape = self.shape_spectral();
                let dcp = self.get_decomp_from_global_shape(&shape);
                dcp.all_gather_y(pencil_data, global_data);
            }

            fn scatter_to_x_pencil_phys<S2>(&self, pencil_data: &mut ArrayBase<S2, Dim<[usize; 2]>>)
            where
                S2: Data<Elem = Self::Physical> + DataMut,
            {
                let shape = self.shape_physical();
                let dcp = self.get_decomp_from_global_shape(&shape);
                dcp.scatter_x_inplace(pencil_data);
            }

            fn scatter_to_x_pencil_phys_root<S1, S2>(
                &self,
                global_data: &ArrayBase<S1, Dim<[usize; 2]>>,
                pencil_data: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = Self::Physical>,
                S2: Data<Elem = Self::Physical> + DataMut,
            {
                let shape = self.shape_physical();
                let dcp = self.get_decomp_from_global_shape(&shape);
                dcp.scatter_x_inplace_root(global_data, pencil_data);
            }

            fn scatter_to_y_pencil_phys<S2>(&self, pencil_data: &mut ArrayBase<S2, Dim<[usize; 2]>>)
            where
                S2: Data<Elem = Self::Physical> + DataMut,
            {
                let shape = self.shape_physical();
                let dcp = self.get_decomp_from_global_shape(&shape);
                dcp.scatter_y_inplace(pencil_data);
            }

            fn scatter_to_y_pencil_phys_root<S1, S2>(
                &self,
                global_data: &ArrayBase<S1, Dim<[usize; 2]>>,
                pencil_data: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = Self::Physical>,
                S2: Data<Elem = Self::Physical> + DataMut,
            {
                let shape = self.shape_physical();
                let dcp = self.get_decomp_from_global_shape(&shape);
                dcp.scatter_y_inplace_root(global_data, pencil_data);
            }

            fn scatter_to_x_pencil_spec<S2>(&self, pencil_data: &mut ArrayBase<S2, Dim<[usize; 2]>>)
            where
                S2: Data<Elem = Self::Spectral> + DataMut,
            {
                let shape = self.shape_spectral();
                let dcp = self.get_decomp_from_global_shape(&shape);
                dcp.scatter_x_inplace(pencil_data);
            }

            fn scatter_to_x_pencil_spec_root<S1, S2>(
                &self,
                global_data: &ArrayBase<S1, Dim<[usize; 2]>>,
                pencil_data: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = Self::Spectral>,
                S2: Data<Elem = Self::Spectral> + DataMut,
            {
                let shape = self.shape_spectral();
                let dcp = self.get_decomp_from_global_shape(&shape);
                dcp.scatter_x_inplace_root(global_data, pencil_data);
            }

            fn scatter_to_y_pencil_spec<S2>(&self, pencil_data: &mut ArrayBase<S2, Dim<[usize; 2]>>)
            where
                S2: Data<Elem = Self::Spectral> + DataMut,
            {
                let shape = self.shape_spectral();
                let dcp = self.get_decomp_from_global_shape(&shape);
                dcp.scatter_y_inplace(pencil_data);
            }

            fn scatter_to_y_pencil_spec_root<S1, S2>(
                &self,
                global_data: &ArrayBase<S1, Dim<[usize; 2]>>,
                pencil_data: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = Self::Spectral>,
                S2: Data<Elem = Self::Spectral> + DataMut,
            {
                let shape = self.shape_spectral();
                let dcp = self.get_decomp_from_global_shape(&shape);
                dcp.scatter_y_inplace_root(global_data, pencil_data);
            }

            fn transpose_x_to_y_phys<S1, S2>(
                &self,
                x_pencil: &ArrayBase<S1, Dim<[usize; 2]>>,
                y_pencil: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = Self::Physical>,
                S2: Data<Elem = Self::Physical> + DataMut,
            {
                let shape = self.shape_physical();
                let dcp = self.get_decomp_from_global_shape(&shape);
                dcp.transpose_x_to_y(x_pencil, y_pencil);
            }

            fn transpose_y_to_x_phys<S1, S2>(
                &self,
                y_pencil: &ArrayBase<S1, Dim<[usize; 2]>>,
                x_pencil: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = Self::Physical>,
                S2: Data<Elem = Self::Physical> + DataMut,
            {
                let shape = self.shape_physical();
                let dcp = self.get_decomp_from_global_shape(&shape);
                dcp.transpose_y_to_x(y_pencil, x_pencil);
            }

            fn transpose_x_to_y_spec<S1, S2>(
                &self,
                x_pencil: &ArrayBase<S1, Dim<[usize; 2]>>,
                y_pencil: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = Self::Spectral>,
                S2: Data<Elem = Self::Spectral> + DataMut,
            {
                let shape = self.shape_spectral();
                let dcp = self.get_decomp_from_global_shape(&shape);
                dcp.transpose_x_to_y(x_pencil, y_pencil);
            }

            fn transpose_y_to_x_spec<S1, S2>(
                &self,
                y_pencil: &ArrayBase<S1, Dim<[usize; 2]>>,
                x_pencil: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = Self::Spectral>,
                S2: Data<Elem = Self::Spectral> + DataMut,
            {
                let shape = self.shape_spectral();
                let dcp = self.get_decomp_from_global_shape(&shape);
                dcp.transpose_y_to_x(y_pencil, x_pencil);
            }
        }
    };
}

impl_space2_mpi!(BaseR2r, BaseR2r, A, A);
impl_space2_mpi!(BaseR2c, BaseR2r, A, Complex<A>);
impl_space2_mpi!(BaseC2c, BaseR2c, A, Complex<A>);
impl_space2_mpi!(BaseC2c, BaseC2c, Complex<A>, Complex<A>);
