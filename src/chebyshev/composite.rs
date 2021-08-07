//! # Composite chebyshev spaces
use super::composite_stencil::{ChebyshevStencil, Stencil};
use super::ortho::Chebyshev;
use crate::traits::BaseBasics;
use crate::traits::Differentiate;
use crate::traits::FromOrtho;
use crate::traits::LaplacianInverse;
use crate::traits::Transform;
use crate::traits::TransformKind;
use crate::traits::TransformPar;
use crate::types::FloatNum;
use ndarray::prelude::*;
use num_complex::Complex;

#[allow(clippy::module_name_repetitions)]
#[derive(Clone)]
pub struct CompositeChebyshev<A: FloatNum> {
    /// Number of coefficients in physical space
    pub n: usize,
    /// Number of coefficients in spectral space
    pub m: usize,
    /// Parent base
    pub ortho: Chebyshev<A>,
    /// Transform stencil
    pub stencil: ChebyshevStencil<A>,
    /// Transform kind (real-to-real)
    transform_kind: TransformKind,
}

impl<A: FloatNum> CompositeChebyshev<A> {
    /// Return function space of chebyshev space
    /// with *dirichlet* boundary conditions
    /// $$
    ///  \phi_k = T_k - T_{k+2}
    /// $$
    #[must_use]
    pub fn dirichlet(n: usize) -> Self {
        use super::composite_stencil::StencilChebyshev;
        let stencil = StencilChebyshev::dirichlet(n);
        Self {
            n,
            m: StencilChebyshev::<A>::get_m(n),
            stencil: ChebyshevStencil::StencilChebyshev(stencil),
            ortho: Chebyshev::<A>::new(n),
            transform_kind: TransformKind::RealToReal,
        }
    }

    /// Return function space of chebyshev space
    /// with *neumann* boundary conditions
    /// $$
    /// \phi_k = T_k - k^{2} \/ (k+2)^2 T_{k+2}
    /// $$
    #[must_use]
    pub fn neumann(n: usize) -> Self {
        use super::composite_stencil::StencilChebyshev;
        let stencil = StencilChebyshev::neumann(n);
        Self {
            n,
            m: StencilChebyshev::<A>::get_m(n),
            stencil: ChebyshevStencil::StencilChebyshev(stencil),
            ortho: Chebyshev::<A>::new(n),
            transform_kind: TransformKind::RealToReal,
        }
    }

    /// Dirichlet boundary condition basis
    /// $$
    ///     \phi_0 = 0.5 T_0 - 0.5 T_1
    /// $$
    /// $$
    ///     \phi_1 = 0.5 T_0 + 0.5 T_1
    /// $$
    #[must_use]
    pub fn dirichlet_bc(n: usize) -> Self {
        use super::composite_stencil::StencilChebyshevBoundary;
        let stencil = StencilChebyshevBoundary::dirichlet(n);
        Self {
            n,
            m: StencilChebyshevBoundary::<A>::get_m(n),
            stencil: ChebyshevStencil::StencilChebyshevBoundary(stencil),
            ortho: Chebyshev::<A>::new(n),
            transform_kind: TransformKind::RealToReal,
        }
    }

    /// Neumann boundary condition basis
    /// $$
    ///     \phi_0 = 0.5T_0 - 1/8T_1
    /// $$
    /// $$
    ///     \phi_1 = 0.5T_0 + 1/8T_1
    /// $$
    #[must_use]
    pub fn neumann_bc(n: usize) -> Self {
        use super::composite_stencil::StencilChebyshevBoundary;
        let stencil = StencilChebyshevBoundary::neumann(n);
        Self {
            n,
            m: StencilChebyshevBoundary::<A>::get_m(n),
            stencil: ChebyshevStencil::StencilChebyshevBoundary(stencil),
            ortho: Chebyshev::<A>::new(n),
            transform_kind: TransformKind::RealToReal,
        }
    }

    /// Return grid coordinates
    #[must_use]
    pub fn coords(&self) -> &Array1<A> {
        &self.ortho.x
    }
}

macro_rules! impl_from_ortho_composite_chebyshev {
    ($a: ty) => {
        impl<A: FloatNum> FromOrtho<$a> for CompositeChebyshev<A> {
            /// Return coefficents in associated composite space
            ///
            /// ```
            /// use funspace::chebyshev::CompositeChebyshev;
            /// use ndarray::prelude::*;
            /// use funspace::utils::approx_eq;
            /// use funspace::FromOrtho;
            /// let (nx, ny) = (5, 4);
            /// let mut composite_coeff = Array2::<f64>::zeros((nx - 2, ny));
            /// for (i, v) in composite_coeff.iter_mut().enumerate() {
            ///     *v = i as f64;
            /// }
            /// let cd = CompositeChebyshev::<f64>::dirichlet(nx);
            ///
            /// let expected = array![
            ///     [0., 1., 2., 3.],
            ///     [4., 5., 6., 7.],
            ///     [8., 8., 8., 8.],
            ///     [-4., -5., -6., -7.],
            ///     [-8., -9., -10., -11.],
            /// ];
            /// let parent_coeff = cd.to_ortho(&composite_coeff, 0);
            /// approx_eq(&parent_coeff, &expected);
            /// ```
            fn to_ortho<S, D>(&self, input: &ArrayBase<S, D>, axis: usize) -> Array<$a, D>
            where
                S: ndarray::Data<Elem = $a>,
                D: Dimension,
            {
                use crate::utils::array_resized_axis;
                let mut output = array_resized_axis(input, self.ortho.len_spec(), axis);
                self.to_ortho_inplace(input, &mut output, axis);
                output
            }

            /// See [`CompositeChebyshev::to_ortho`]
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
                use crate::utils::check_array_axis;
                check_array_axis(input, self.len_spec(), axis, Some("composite to_ortho"));
                check_array_axis(
                    output,
                    self.ortho.len_spec(),
                    axis,
                    Some("composite to_ortho"),
                );
                ndarray::Zip::from(input.lanes(Axis(axis)))
                    .and(output.lanes_mut(Axis(axis)))
                    .for_each(|inp, mut out| {
                        self.stencil.multiply_vec_inplace(&inp, &mut out);
                    });
            }

            /// Return coefficents in associated composite space
            ///
            /// ```
            /// use funspace::chebyshev::CompositeChebyshev;
            /// use ndarray::prelude::*;
            /// use funspace::utils::approx_eq;
            /// use funspace::FromOrtho;
            /// let (nx, ny) = (5, 4);
            /// let mut parent_coeff = Array2::<f64>::zeros((nx, ny));
            /// for (i, v) in parent_coeff.iter_mut().enumerate() {
            ///     *v = i as f64;
            /// }
            /// let cd = CompositeChebyshev::<f64>::dirichlet(nx);
            ///
            /// let expected = array![
            ///     [-8., -8., -8., -8.],
            ///     [-4., -4., -4., -4.],
            ///     [-8., -8., -8., -8.],
            /// ];
            /// let composite_coeff = cd.from_ortho(&parent_coeff, 0);
            /// approx_eq(&composite_coeff, &expected);
            /// ```
            fn from_ortho<S, D>(&self, input: &ArrayBase<S, D>, axis: usize) -> Array<$a, D>
            where
                S: ndarray::Data<Elem = $a>,
                D: Dimension,
            {
                use crate::utils::array_resized_axis;
                let mut output = array_resized_axis(input, self.len_spec(), axis);
                self.from_ortho_inplace(input, &mut output, axis);
                output
            }

            /// See [`CompositeChebyshev::from_ortho`]
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
                use crate::utils::check_array_axis;
                check_array_axis(
                    input,
                    self.ortho.len_spec(),
                    axis,
                    Some("composite from_ortho"),
                );
                check_array_axis(output, self.len_spec(), axis, Some("composite from_ortho"));
                ndarray::Zip::from(input.lanes(Axis(axis)))
                    .and(output.lanes_mut(Axis(axis)))
                    .for_each(|inp, mut out| {
                        self.stencil.solve_vec_inplace(&inp, &mut out);
                    });
            }
        }
    };
}

impl_from_ortho_composite_chebyshev!(A);
impl_from_ortho_composite_chebyshev!(Complex<A>);

impl<A: FloatNum> BaseBasics<A> for CompositeChebyshev<A> {
    /// Size in physical space
    fn len_phys(&self) -> usize {
        self.n
    }
    /// Size in spectral space
    fn len_spec(&self) -> usize {
        self.m
    }
    /// Coordinates in physical space
    fn coords(&self) -> &Array1<A> {
        &self.ortho.x
    }
    /// Returns transformation stencil
    fn mass(&self) -> Array2<A> {
        self.stencil.to_array()
    }
    /// Return transform kind
    fn get_transform_kind(&self) -> &TransformKind {
        &self.transform_kind
    }
}

impl<A: FloatNum> Transform<A, A> for CompositeChebyshev<A> {
    type Physical = A;
    type Spectral = A;

    /// # Example
    /// Forward transform along first axis
    /// ```
    /// use funspace::Transform;
    /// use funspace::chebyshev::CompositeChebyshev;
    /// use funspace::utils::approx_eq;
    /// use ndarray::prelude::*;
    /// let mut cheby = CompositeChebyshev::dirichlet(5);
    /// let mut input = array![1., 2., 3., 4., 5.];
    /// let output = cheby.forward(&mut input, 0);
    /// approx_eq(&output, &array![2., 0.70710678, 1.]);
    /// ```
    fn forward<S, D>(
        &mut self,
        input: &mut ArrayBase<S, D>,
        axis: usize,
    ) -> Array<Self::Spectral, D>
    where
        S: ndarray::Data<Elem = Self::Physical>,
        D: Dimension,
    {
        let parent_coeff = self.ortho.forward(input, axis);
        self.from_ortho(&parent_coeff, axis)
    }

    /// See [`CompositeChebyshev::forward`]
    /// ```
    /// use funspace::Transform;
    /// use funspace::chebyshev::CompositeChebyshev;
    /// use funspace::utils::approx_eq;
    /// use ndarray::prelude::*;
    /// let mut cheby = CompositeChebyshev::dirichlet(5);
    /// let mut input = array![1., 2., 3., 4., 5.];
    /// let mut output = Array1::<f64>::zeros(3);
    /// cheby.forward_inplace(&mut input, &mut output, 0);
    /// approx_eq(&output, &array![2., 0.70710678, 1.]);
    /// ```
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
        let parent_coeff = self.ortho.forward(input, axis);
        self.from_ortho_inplace(&parent_coeff, output, axis);
    }

    /// # Example
    /// Backward transform along first axis
    /// ```
    /// use funspace::Transform;
    /// use funspace::chebyshev::CompositeChebyshev;
    /// use funspace::utils::approx_eq;
    /// use ndarray::prelude::*;
    /// let mut cheby = CompositeChebyshev::dirichlet(5);
    /// let mut input = array![1., 2., 3.];
    /// let output = cheby.backward(&mut input, 0);
    /// approx_eq(&output, &array![0.,1.1716, -4., 6.8284, 0. ]);
    /// ```
    fn backward<S, D>(
        &mut self,
        input: &mut ArrayBase<S, D>,
        axis: usize,
    ) -> Array<Self::Physical, D>
    where
        S: ndarray::Data<Elem = Self::Spectral>,
        D: Dimension,
    {
        let mut parent_coeff = self.to_ortho(input, axis);
        self.ortho.backward(&mut parent_coeff, axis)
    }

    /// See [`CompositeChebyshev::backward`]
    /// ```
    /// use funspace::Transform;
    /// use funspace::chebyshev::CompositeChebyshev;
    /// use funspace::utils::approx_eq;
    /// use ndarray::prelude::*;
    /// let mut cheby = CompositeChebyshev::dirichlet(5);
    /// let mut input = array![1., 2., 3.];
    /// let  mut output = Array1::<f64>::zeros(5);
    /// cheby.backward_inplace(&mut input, &mut output, 0);
    /// approx_eq(&output, &array![0.,1.1716, -4., 6.8284, 0. ]);
    /// ```
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
        let mut parent_coeff = self.to_ortho(input, axis);
        self.ortho.backward_inplace(&mut parent_coeff, output, axis);
    }
}

impl<A: FloatNum> TransformPar<A, A> for CompositeChebyshev<A> {
    type Physical = A;
    type Spectral = A;

    /// # Example
    /// Forward transform along first axis
    /// ```
    /// use funspace::TransformPar;
    /// use funspace::chebyshev::CompositeChebyshev;
    /// use funspace::utils::approx_eq;
    /// use ndarray::prelude::*;
    /// let mut cheby = CompositeChebyshev::dirichlet(5);
    /// let mut input = array![1., 2., 3., 4., 5.];
    /// let output = cheby.forward_par(&mut input, 0);
    /// approx_eq(&output, &array![2., 0.70710678, 1.]);
    /// ```
    fn forward_par<S, D>(
        &mut self,
        input: &mut ArrayBase<S, D>,
        axis: usize,
    ) -> Array<Self::Spectral, D>
    where
        S: ndarray::Data<Elem = Self::Physical>,
        D: Dimension,
    {
        let parent_coeff = self.ortho.forward_par(input, axis);
        self.from_ortho(&parent_coeff, axis)
    }

    /// See [`CompositeChebyshev::forward_par`]
    /// ```
    /// use funspace::TransformPar;
    /// use funspace::chebyshev::CompositeChebyshev;
    /// use funspace::utils::approx_eq;
    /// use ndarray::prelude::*;
    /// let mut cheby = CompositeChebyshev::dirichlet(5);
    /// let mut input = array![1., 2., 3., 4., 5.];
    /// let mut output = Array1::<f64>::zeros(3);
    /// cheby.forward_inplace_par(&mut input, &mut output, 0);
    /// approx_eq(&output, &array![2., 0.70710678, 1.]);
    /// ```
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
        let parent_coeff = self.ortho.forward_par(input, axis);
        self.from_ortho_inplace(&parent_coeff, output, axis);
    }

    /// # Example
    /// Backward transform along first axis
    /// ```
    /// use funspace::TransformPar;
    /// use funspace::chebyshev::CompositeChebyshev;
    /// use funspace::utils::approx_eq;
    /// use ndarray::prelude::*;
    /// let mut cheby = CompositeChebyshev::dirichlet(5);
    /// let mut input = array![1., 2., 3.];
    /// let output = cheby.backward_par(&mut input, 0);
    /// approx_eq(&output, &array![0.,1.1716, -4., 6.8284, 0. ]);
    /// ```
    fn backward_par<S, D>(
        &mut self,
        input: &mut ArrayBase<S, D>,
        axis: usize,
    ) -> Array<Self::Physical, D>
    where
        S: ndarray::Data<Elem = Self::Spectral>,
        D: Dimension,
    {
        let mut parent_coeff = self.to_ortho(input, axis);
        self.ortho.backward_par(&mut parent_coeff, axis)
    }

    /// See [`CompositeChebyshev::backward_par`]
    /// ```
    /// use funspace::TransformPar;
    /// use funspace::chebyshev::CompositeChebyshev;
    /// use funspace::utils::approx_eq;
    /// use ndarray::prelude::*;
    /// let mut cheby = CompositeChebyshev::dirichlet(5);
    /// let mut input = array![1., 2., 3.];
    /// let  mut output = Array1::<f64>::zeros(5);
    /// cheby.backward_inplace_par(&mut input, &mut output, 0);
    /// approx_eq(&output, &array![0.,1.1716, -4., 6.8284, 0. ]);
    /// ```
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
        let mut parent_coeff = self.to_ortho(input, axis);
        self.ortho
            .backward_inplace_par(&mut parent_coeff, output, axis);
    }
}

macro_rules! impl_differentiate_composite_chebyshev {
    ($a: ty) => {
        impl<A: FloatNum> Differentiate<$a> for CompositeChebyshev<A> {
            /// Differentiation in spectral space
            /// ```
            /// use funspace::Differentiate;
            /// use funspace::chebyshev::CompositeChebyshev;
            /// use funspace::utils::approx_eq;
            /// use ndarray::prelude::*;
            /// let mut cheby = CompositeChebyshev::<f64>::dirichlet(5);
            /// let mut input = array![1., 2., 3.];
            /// let output = cheby.differentiate(&input, 2, 0);
            /// approx_eq(&output, &array![-88.,  -48., -144., 0., 0. ]);
            /// ```
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
                let mut parent_coeff = self.to_ortho(data, axis);
                self.ortho.differentiate_inplace(&mut parent_coeff, n_times, axis);
                parent_coeff
            }

            #[allow(unused_variables)]
            fn differentiate_inplace<S, D>(
                &self,
                data: &mut ArrayBase<S, D>,
                n_times: usize,
                axis: usize,
            ) where
                S: ndarray::Data<Elem = $a> + ndarray::DataMut,
                D: Dimension,
            {
                panic!(
                    "Method differentiate_inplace not impl for composite basis (array size would change)."
                );
            }
        }
    };
}
impl_differentiate_composite_chebyshev!(A);
impl_differentiate_composite_chebyshev!(Complex<A>);

impl<A: FloatNum> LaplacianInverse<A> for CompositeChebyshev<A> {
    /// See [`Chebyshev::laplace`]
    fn laplace(&self) -> Array2<A> {
        self.ortho.laplace()
    }
    /// See [`Chebyshev::laplace_inv`]
    fn laplace_inv(&self) -> Array2<A> {
        self.ortho.laplace_inv()
    }
    /// See [`Chebyshev::laplace_inv_eye`]
    fn laplace_inv_eye(&self) -> Array2<A> {
        self.ortho.laplace_inv_eye()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::utils::approx_eq;

    #[test]
    /// Differantiate 2d array along first and second axis
    fn test_chebdirichlet_to_ortho() {
        let (nx, ny) = (5, 4);
        let mut composite_coeff = Array2::<f64>::zeros((nx - 2, ny));

        // Axis 0
        let cheby = CompositeChebyshev::<f64>::dirichlet(nx);
        for (i, v) in composite_coeff.iter_mut().enumerate() {
            *v = i as f64;
        }
        let expected = array![
            [0., 1., 2., 3.],
            [4., 5., 6., 7.],
            [8., 8., 8., 8.],
            [-4., -5., -6., -7.],
            [-8., -9., -10., -11.],
        ];
        let parent_coeff = cheby.to_ortho(&composite_coeff, 0);
        approx_eq(&parent_coeff, &expected);

        // Axis 1
        let mut composite_coeff = Array2::<f64>::zeros((nx, ny - 2));
        let cheby = CompositeChebyshev::<f64>::dirichlet(ny);
        for (i, v) in composite_coeff.iter_mut().enumerate() {
            *v = i as f64;
        }
        let expected = array![
            [0., 1., 0., -1.],
            [2., 3., -2., -3.],
            [4., 5., -4., -5.],
            [6., 7., -6., -7.],
            [8., 9., -8., -9.],
        ];
        let parent_coeff = cheby.to_ortho(&composite_coeff, 1);
        approx_eq(&parent_coeff, &expected);
    }

    #[test]
    /// Differantiate ChebDirichlet (2d array) twice along first and second axis
    fn test_chebdirichlet_differentiate() {
        let (nx, ny) = (6, 4);
        let mut data = Array2::<f64>::zeros((nx, ny));

        // Axis 0
        let cheby = CompositeChebyshev::<f64>::dirichlet(nx + 2);
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as f64;
        }
        let expected = array![
            [-1440.0, -1548.0, -1656.0, -1764.0],
            [-5568.0, -5904.0, -6240.0, -6576.0],
            [-2688.0, -2880.0, -3072.0, -3264.0],
            [-4960.0, -5240.0, -5520.0, -5800.0],
            [-1920.0, -2040.0, -2160.0, -2280.0],
            [-3360.0, -3528.0, -3696.0, -3864.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ];
        let diff = cheby.differentiate(&data, 2, 0);
        approx_eq(&diff, &expected);

        // Axis 1
        let cheby = CompositeChebyshev::<f64>::dirichlet(ny + 2);
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as f64;
        }
        let expected = array![
            [-56.0, -312.0, -96.0, -240.0, 0.0, 0.0],
            [-184.0, -792.0, -288.0, -560.0, 0.0, 0.0],
            [-312.0, -1272.0, -480.0, -880.0, 0.0, 0.0],
            [-440.0, -1752.0, -672.0, -1200.0, 0.0, 0.0],
            [-568.0, -2232.0, -864.0, -1520.0, 0.0, 0.0],
            [-696.0, -2712.0, -1056.0, -1840.0, 0.0, 0.0],
        ];
        let diff = cheby.differentiate(&data, 2, 1);
        approx_eq(&diff, &expected);
    }
}
