//! Two-dimensional function space with mpi support
#![cfg(feature = "mpi")]
#![allow(clippy::module_name_repetitions)]
use super::{BaseSpaceMpi, Communicator, Decomp2d, DecompHandler, Equivalence, Universe};
use crate::enums::{BaseKind, TransformKind};
use crate::space::traits::{
    BaseSpaceElements, BaseSpaceFromOrtho, BaseSpaceGradient, BaseSpaceMatOpLaplacian,
    BaseSpaceMatOpStencil, BaseSpaceSize, BaseSpaceTransform,
};
use crate::traits::{
    BaseElements, BaseFromOrtho, BaseGradient, BaseMatOpLaplacian, BaseMatOpStencil, BaseSize,
    BaseTransform,
};
use crate::{BaseC2c, BaseR2c, BaseR2r, FloatNum, ScalarNum};
use ndarray::{prelude::*, Data, DataMut};
use num_complex::Complex;
use num_traits::Zero;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Clone)]
pub struct Space2Mpi<'a, B0, B1> {
    // Intermediate <-> Spectral
    pub base0: B0,
    // Phsical <-> Intermediate
    pub base1: B1,
    // Mpi Handler
    pub decomp_handler: DecompHandler<'a>,
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

macro_rules! impl_space2 {
    ($space: ident, $base0: ident, $base1: ident, $p: ty, $s: ty) => {
        impl<A> BaseSpaceSize<2> for $space<'_, $base0<A>, $base1<A>>
        where
            A: FloatNum,
        {
            fn shape_physical(&self) -> [usize; 2] {
                [self.base0.len_phys(), self.base1.len_phys()]
            }

            fn shape_spectral(&self) -> [usize; 2] {
                [self.base0.len_spec(), self.base1.len_spec()]
            }

            fn shape_spectral_ortho(&self) -> [usize; 2] {
                [self.base0.len_orth(), self.base1.len_orth()]
            }

            fn ndarray_from_shape<T: Clone + Zero>(&self, shape: [usize; 2]) -> Array2<T> {
                Array2::zeros(shape)
            }
        }

        impl<A> BaseSpaceElements<2> for $space<'_, $base0<A>, $base1<A>>
        where
            A: FloatNum,
        {
            type RealNum = A;

            /// Array of coordinates
            fn coords(&self) -> [Array1<Self::RealNum>; 2] {
                [self.coords_axis(0), self.coords_axis(1)]
            }

            /// Coordinates of grid points (in physical space)
            ///
            /// # Arguments
            ///
            /// * `axis` - usize
            fn coords_axis(&self, axis: usize) -> Array1<A> {
                if axis == 0 {
                    self.base0.coords().into()
                } else {
                    self.base1.coords().into()
                }
            }

            /// Return base key
            fn base_kind(&self, axis: usize) -> BaseKind {
                if axis == 0 {
                    self.base0.base_kind()
                } else {
                    self.base1.base_kind()
                }
            }

            /// Return transform kind
            fn transform_kind(&self, axis: usize) -> TransformKind {
                if axis == 0 {
                    self.base0.transform_kind()
                } else {
                    self.base1.transform_kind()
                }
            }
        }

        impl<A> BaseSpaceMatOpStencil for $space<'_, $base0<A>, $base1<A>>
        where
            A: FloatNum,
        {
            type NumType = A;

            /// Transformation stencil
            ///
            /// Multiplication of this matrix with a coefficient vector has
            /// the same effect as  [`BaseSpaceFromOrtho::to_ortho()`],
            /// but is less efficient.
            ///
            /// Returns identity matrix for orthogonal bases
            ///
            /// # Arguments
            ///
            /// * `axis` - usize
            fn stencil(&self, axis: usize) -> Array2<A> {
                if axis == 0 {
                    self.base0.stencil()
                } else {
                    self.base1.stencil()
                }
            }

            /// Inverse of transformation stencil
            ///
            /// Multiplication of this matrix with a coefficient vector has
            /// the same effect as  [`BaseSpaceFromOrtho::from_ortho()`],
            /// but is less efficient.
            ///
            /// Returns identity matrix for orthogonal bases
            ///
            /// # Arguments
            ///
            /// * `axis` - usize
            fn stencil_inv(&self, axis: usize) -> Array2<A> {
                if axis == 0 {
                    self.base0.stencil_inv()
                } else {
                    self.base1.stencil_inv()
                }
            }
        }

        impl<A> BaseSpaceMatOpLaplacian for $space<'_, $base0<A>, $base1<A>>
        where
            A: FloatNum,
        {
            type NumType = A;

            /// Laplacian `L`
            ///
            /// ```text
            /// L_pinv @ L = I_pinv
            /// ```
            ///
            /// # Arguments
            ///
            /// * `axis` - usize
            fn laplacian(&self, axis: usize) -> Array2<A> {
                if axis == 0 {
                    self.base0.laplacian()
                } else {
                    self.base1.laplacian()
                }
            }

            /// Pseudoinverse matrix `L_pinv` of Laplacian
            ///
            /// Returns (`L_pinv`, `I_pinv`)
            ///
            /// ```text
            /// L_pinv @ L = I_pinv
            /// ```
            ///
            /// # Arguments
            ///
            /// * `axis` - usize
            fn laplacian_pinv(&self, axis: usize) -> (Array2<A>, Array2<A>) {
                if axis == 0 {
                    self.base0.laplacian_pinv()
                } else {
                    self.base1.laplacian_pinv()
                }
            }
        }

        impl<A, T> BaseSpaceGradient<A, T, 2> for $space<'_, $base0<A>, $base1<A>>
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
                + Sub<$s, Output = T>,
        {
            /// Take gradient. Optional: Rescale result by a constant.
            ///
            /// # Arguments
            ///
            /// * `input` - *ndarray* with num type of spectral space
            /// * `deriv` - [usize; N], derivative along each axis
            /// * `scale` - [float; N], scaling factor along each axis (default [1.;n])
            fn gradient<S>(
                &self,
                input: &ArrayBase<S, Dim<[usize; 2]>>,
                deriv: [usize; 2],
                scale: Option<[A; 2]>,
            ) -> Array<T, Dim<[usize; 2]>>
            where
                S: Data<Elem = T>,
            {
                let buffer = self.base0.gradient(input, deriv[0], 0);
                let mut output = self.base1.gradient(&buffer, deriv[1], 1);
                if let Some(s) = scale {
                    let sc: T = (s[0].powi(deriv[0] as i32) * s[1].powi(deriv[1] as i32)).into();
                    for x in output.iter_mut() {
                        *x /= sc;
                    }
                }
                output
            }
            /// Take gradient. Optional: Rescale result by a constant.
            ///
            /// # Arguments
            ///
            /// * `input` - *ndarray* with num type of spectral space
            /// * `deriv` - [usize; N], derivative along each axis
            /// * `scale` - [float; N], scaling factor along each axis (default [1.;n])
            fn gradient_par<S>(
                &self,
                input: &ArrayBase<S, Dim<[usize; 2]>>,
                deriv: [usize; 2],
                scale: Option<[A; 2]>,
            ) -> Array<T, Dim<[usize; 2]>>
            where
                S: Data<Elem = T>,
            {
                let buffer = self.base0.gradient_par(input, deriv[0], 0);
                let mut output = self.base1.gradient_par(&buffer, deriv[1], 1);
                if let Some(s) = scale {
                    let sc: T = (s[0].powi(deriv[0] as i32) * s[1].powi(deriv[1] as i32)).into();
                    for x in output.iter_mut() {
                        *x /= sc;
                    }
                }
                output
            }
        }
        impl<A, T> BaseSpaceFromOrtho<A, T, 2> for $space<'_, $base0<A>, $base1<A>>
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
                + Sub<$s, Output = T>,
        {
            /// Transformation from composite and to orthonormal space (inplace).
            ///
            /// # Arguments
            ///
            /// * `input` - *ndarray* with num type of spectral space
            /// * `output` - *ndarray* with num type of spectral space
            fn to_ortho_inplace<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 2]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = T>,
                S2: Data<Elem = T> + DataMut,
            {
                let buffer = self.base0.to_ortho(input, 0);
                self.base1.to_ortho_inplace(&buffer, output, 1);
            }

            /// Transformation from orthonormal and to composite space (inplace).
            ///
            /// # Arguments
            ///
            /// * `input` - *ndarray* with num type of spectral space
            /// * `output` - *ndarray* with num type of spectral space
            fn from_ortho_inplace<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 2]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = T>,
                S2: Data<Elem = T> + DataMut,
            {
                let buffer = self.base0.from_ortho(input, 0);
                self.base1.from_ortho_inplace(&buffer, output, 1);
            }

            /// Transformation from composite and to orthonormal space (inplace).
            ///
            /// # Arguments
            ///
            /// * `input` - *ndarray* with num type of spectral space
            /// * `output` - *ndarray* with num type of spectral space
            fn to_ortho_inplace_par<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 2]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = T>,
                S2: Data<Elem = T> + DataMut,
            {
                let buffer = self.base0.to_ortho_par(input, 0);
                self.base1.to_ortho_inplace_par(&buffer, output, 1);
            }

            /// Transformation from orthonormal and to composite space (inplace).
            ///
            /// # Arguments
            ///
            /// * `input` - *ndarray* with num type of spectral space
            /// * `output` - *ndarray* with num type of spectral space
            fn from_ortho_inplace_par<S1, S2>(
                &self,
                input: &ArrayBase<S1, Dim<[usize; 2]>>,
                output: &mut ArrayBase<S2, Dim<[usize; 2]>>,
            ) where
                S1: Data<Elem = T>,
                S2: Data<Elem = T> + DataMut,
            {
                let buffer = self.base0.from_ortho_par(input, 0);
                self.base1.from_ortho_inplace_par(&buffer, output, 1);
            }
        }

        impl<A> BaseSpaceTransform<A, 2> for $space<'_, $base0<A>, $base1<A>>
        where
            A: FloatNum + ScalarNum,
            Complex<A>: ScalarNum,
        {
            type Physical = $p;

            type Spectral = $s;

            /// Transform physical -> spectral space (inplace)
            ///
            /// # Arguments
            ///
            /// * `input` - *ndarray* with num type of physical space
            /// * `output` - *ndarray* with num type of spectral space
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

            /// Transform spectral -> physical space (inplace)
            ///
            /// # Arguments
            ///
            /// * `input` - *ndarray* with num type of spectral space
            /// * `output` - *ndarray* with num type of physical space
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

            /// Transform physical -> spectral space (inplace)
            ///
            /// # Arguments
            ///
            /// * `input` - *ndarray* with num type of physical space
            /// * `output` - *ndarray* with num type of spectral space
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

            /// Transform spectral -> physical space (inplace)
            ///
            /// # Arguments
            ///
            /// * `input` - *ndarray* with num type of spectral space
            /// * `output` - *ndarray* with num type of physical space
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
impl_space2!(Space2Mpi, BaseR2r, BaseR2r, A, A);
impl_space2!(Space2Mpi, BaseR2c, BaseR2r, A, Complex<A>);
impl_space2!(Space2Mpi, BaseC2c, BaseR2c, A, Complex<A>);
impl_space2!(Space2Mpi, BaseC2c, BaseC2c, Complex<A>, Complex<A>);

macro_rules! impl_space2_mpi {
    ($base0: ident, $base1: ident, $p: ty, $s: ty) => {
        impl<A> BaseSpaceMpi<A, 2> for Space2Mpi<'_, $base0<A>, $base1<A>>
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
                        &self.base0.gradient(input, deriv[0], 0),
                        &mut buffer_ypen,
                    );
                    self.base1.gradient(&buffer_ypen, deriv[1], 1)
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
