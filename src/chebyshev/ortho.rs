//! # Orthogonal chebyshev space
use super::FloatNum;
use crate::Differentiate;
use crate::FromOrtho;
use crate::LaplacianInverse;
use crate::Mass;
use crate::Size;
use crate::Transform;
use crate::TransformPar;
use ndarray::prelude::*;
use ndrustfft::DctHandler;

/// # Container for chebyshev space
#[derive(Clone)]
pub struct Chebyshev<A> {
    /// Number of coefficients in physical space
    pub n: usize,
    /// Number of coefficients in spectral space ( equal to *n* in this case )
    pub m: usize,
    /// Grid coordinates of chebyshev nodes (2nd kind).
    pub x: Array1<A>,
    /// Handles discrete cosine transform
    dct_handler: DctHandler<A>,
    /// Only for internal use, defines how to correct dct to obtain
    /// chebyshev transform
    correct_dct_forward: Array1<A>,
    correct_dct_backward: Array1<A>,
}

impl<A: FloatNum> Chebyshev<A> {
    /// Creates a new Basis.
    ///
    /// # Arguments
    /// * `n` - Length of array's dimension which shall live in chebyshev space.
    ///
    /// # Panics
    /// Panics when input type cannot be cast from f64.
    ///
    /// # Examples
    /// ```
    /// use funspace::chebyshev::Chebyshev;
    /// let cheby = Chebyshev::<f64>::new(10);
    /// ```
    #[must_use]
    pub fn new(n: usize) -> Self {
        let mut correct_dct = Array1::<A>::zeros(n);
        for (i, s) in correct_dct.iter_mut().enumerate() {
            *s = A::from_f64((-1.0_f64).powf(i as f64)).unwrap();
        }
        let correct_dct_forward =
            correct_dct.mapv(|x| x * A::from_f64(1. / (n - 1) as f64).unwrap());
        let correct_dct_backward = correct_dct.mapv(|x| x / A::from_f64(2.0).unwrap());
        Self {
            n,
            m: n,
            x: Self::_nodes_2nd_kind(n),
            dct_handler: DctHandler::new(n),
            correct_dct_forward,
            correct_dct_backward,
        }
    }

    /// Chebyshev nodes of the second kind on intervall $[-1, 1]$
    fn _nodes_2nd_kind(n: usize) -> Array1<A> {
        use std::f64::consts::PI;
        let m = (n - 1) as f64;
        let mut nodes = Array1::<A>::zeros(n);
        for (k, x) in nodes.indexed_iter_mut() {
            let arg: A = A::from_f64(PI * (m - 2. * k as f64) / (2. * m)).unwrap();
            *x = -arg.sin();
        }
        nodes
    }

    /// Differentiat 1d Array *n_times* using the recurrence relation
    /// of chebyshev polynomials.
    ///
    /// Differentiation is performed on input array directly.
    ///
    /// # Panics
    /// Panics when input type cannot be cast from f64.
    ///
    /// # Example
    /// Differentiate along lane
    /// ```
    /// use funspace::chebyshev::Chebyshev;
    /// use funspace::utils::approx_eq;
    /// use ndarray::prelude::*;
    /// let mut cheby = Chebyshev::<f64>::new(4);
    /// // Differentiate twice
    /// let mut input = array![1., 2., 3., 4.];
    /// cheby.differentiate_lane(&mut input, 2);
    /// approx_eq(&input, &array![12., 96.,  0.,  0.]);
    /// // Differentiate zero times, return itself
    /// let mut input = array![1., 2., 3., 4.];
    /// cheby.differentiate_lane(&mut input, 0);
    /// approx_eq(&input, &array![1., 2., 3., 4.]);
    /// ```
    #[allow(clippy::used_underscore_binding)]
    pub fn differentiate_lane<T, S>(&self, data: &mut ArrayBase<S, Ix1>, n_times: usize)
    where
        T: FloatNum,
        S: ndarray::Data<Elem = T> + ndarray::DataMut,
    {
        let _2 = T::from_f64(2.).unwrap();
        for _ in 0..n_times {
            data[0] = data[1];
            for i in 1..data.len() - 1 {
                let _i = T::from_usize(i + 1).unwrap();
                data[i] = _2 * _i * data[i + 1];
            }
            data[self.n - 1] = T::zero();
            for i in (1..self.n - 2).rev() {
                data[i] = data[i] + data[i + 2];
            }
            data[0] = data[0] + data[2] / _2;
        }
    }
}

impl<A: FloatNum> Chebyshev<A> {
    /// Pseudoinverse matrix of chebyshev spectral
    /// differentiation matrices
    ///
    /// When preconditioned with the pseudoinverse Matrix,
    /// systems become banded and thus efficient to solve.
    ///
    /// Literature:
    /// Sahuck Oh - An Efficient Spectral Method to Solve Multi-Dimensional
    /// Linear Partial Different Equations Using Chebyshev Polynomials
    ///
    /// Output:
    /// ndarray (n x n) matrix, acts in spectral space
    fn _pinv(n: usize, deriv: usize) -> Array2<A> {
        if deriv > 2 {
            panic!("pinv does only support deriv's 1 & 2, got {}", deriv)
        }
        let mut pinv = Array2::<f64>::zeros([n, n]);
        if deriv == 1 {
            pinv[[1, 0]] = 1.;
            for i in 2..n {
                pinv[[i, i - 1]] = 1. / (2. * i as f64); // diag - 1
            }
            for i in 1..n - 2 {
                pinv[[i, i + 1]] = -1. / (2. * i as f64); // diag + 1
            }
        } else if deriv == 2 {
            pinv[[2, 0]] = 0.25;
            for i in 3..n {
                pinv[[i, i - 2]] = 1. / (4 * i * (i - 1)) as f64; // diag - 2
            }
            for i in 2..n - 2 {
                pinv[[i, i]] = -1. / (2 * (i * i - 1)) as f64; // diag 0
            }
            for i in 2..n - 4 {
                pinv[[i, i + 2]] = 1. / (4 * i * (i + 1)) as f64; // diag + 2
            }
        }
        //pinv
        pinv.mapv(|elem| A::from_f64(elem).unwrap())
    }

    /// Returns eye matrix, where the n ( = deriv) upper rows
    fn _pinv_eye(n: usize, deriv: usize) -> Array2<A> {
        let pinv_eye = Array2::<f64>::eye(n).slice(s![deriv.., ..]).to_owned();
        pinv_eye.mapv(|elem| A::from_f64(elem).unwrap())
    }
}

impl<A: FloatNum> Mass<A> for Chebyshev<A> {
    /// Return mass matrix (= eye)
    fn mass(&self) -> Array2<A> {
        Array2::<A>::eye(self.n)
    }
    /// Coordinates in physical space
    fn coords(&self) -> &Array1<A> {
        &self.x
    }
}

impl<A: FloatNum> Size for Chebyshev<A> {
    /// Size in physical space
    fn len_phys(&self) -> usize {
        self.n
    }
    /// Size in spectral space
    fn len_spec(&self) -> usize {
        self.m
    }
}

impl<A: FloatNum + std::ops::MulAssign> Transform for Chebyshev<A> {
    type Physical = A;
    type Spectral = A;

    /// # Example
    /// Forward transform along first axis
    /// ```
    /// use funspace::Transform;
    /// use funspace::chebyshev::Chebyshev;
    /// use funspace::utils::approx_eq;
    /// use ndarray::prelude::*;
    /// let mut cheby = Chebyshev::new(4);
    /// let mut input = array![1., 2., 3., 4.];
    /// let output = cheby.forward(&mut input, 0);
    /// approx_eq(&output, &array![2.5, 1.33333333, 0. , 0.16666667]);
    /// ```
    fn forward<S, D>(
        &mut self,
        input: &mut ArrayBase<S, D>,
        axis: usize,
    ) -> Array<Self::Spectral, D>
    where
        S: ndarray::Data<Elem = Self::Physical>,
        D: Dimension + ndarray::RemoveAxis,
    {
        use crate::utils::array_resized_axis;
        let mut output = array_resized_axis(input, self.m, axis);
        self.forward_inplace(input, &mut output, axis);
        output
    }

    /// See [`Chebyshev::forward`]
    #[allow(clippy::used_underscore_binding)]
    fn forward_inplace<S1, S2, D>(
        &mut self,
        input: &mut ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = Self::Physical>,
        S2: ndarray::Data<Elem = Self::Spectral> + ndarray::DataMut,
        D: Dimension + ndarray::RemoveAxis,
    {
        use crate::utils::check_array_axis;
        use ndrustfft::nddct1;
        check_array_axis(input, self.n, axis, Some("chebyshev forward"));
        check_array_axis(output, self.m, axis, Some("chebyshev forward"));
        // Cosine transform (DCT)
        nddct1(input, output, &mut self.dct_handler, axis);
        // Correct DCT
        let _05 = A::from_f64(1. / 2.).unwrap();
        for mut v in output.lanes_mut(Axis(axis)) {
            v *= &self.correct_dct_forward;
            v[0] *= _05;
            v[self.n - 1] *= _05;
        }
    }

    /// # Example
    /// Backward transform along first axis
    /// ```
    /// use funspace::Transform;
    /// use funspace::chebyshev::Chebyshev;
    /// use funspace::utils::approx_eq;
    /// use ndarray::prelude::*;
    /// let mut cheby = Chebyshev::new(4);
    /// let mut input = array![1., 2., 3., 4.];
    /// let output = cheby.backward(&mut input, 0);
    /// approx_eq(&output, &array![-2. ,  2.5, -3.5, 10.]);
    /// ```
    fn backward<S, D>(
        &mut self,
        input: &mut ArrayBase<S, D>,
        axis: usize,
    ) -> Array<Self::Physical, D>
    where
        S: ndarray::Data<Elem = Self::Spectral>,
        D: Dimension + ndarray::RemoveAxis,
    {
        use crate::utils::array_resized_axis;
        let mut output = array_resized_axis(input, self.m, axis);
        self.backward_inplace(input, &mut output, axis);
        output
    }

    /// See [`Chebyshev::backward`]
    ///
    /// # Panics
    /// Panics when input type cannot be cast from f64.
    #[allow(clippy::used_underscore_binding)]
    fn backward_inplace<S1, S2, D>(
        &mut self,
        input: &mut ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = Self::Spectral>,
        S2: ndarray::Data<Elem = Self::Physical> + ndarray::DataMut,
        D: Dimension + ndarray::RemoveAxis,
    {
        use crate::utils::check_array_axis;
        use ndrustfft::nddct1;
        check_array_axis(input, self.m, axis, Some("chebyshev backward"));
        check_array_axis(output, self.n, axis, Some("chebyshev backward"));
        // Correct
        let mut buffer = input.to_owned();
        let _2 = A::from_f64(2.).unwrap();
        for mut v in buffer.lanes_mut(Axis(axis)) {
            v *= &self.correct_dct_backward;
            v[0] *= _2;
            v[self.n - 1] *= _2;
        }
        // Cosine transform (DCT)
        nddct1(&mut buffer, output, &mut self.dct_handler, axis);
    }
}

impl<A: FloatNum + std::ops::MulAssign> TransformPar for Chebyshev<A> {
    type Physical = A;
    type Spectral = A;

    /// # Example
    /// Forward transform along first axis
    /// ```
    /// use funspace::TransformPar;
    /// use funspace::chebyshev::Chebyshev;
    /// use funspace::utils::approx_eq;
    /// use ndarray::prelude::*;
    /// let mut cheby = Chebyshev::new(4);
    /// let mut input = array![1., 2., 3., 4.];
    /// let output = cheby.forward_par(&mut input, 0);
    /// approx_eq(&output, &array![2.5, 1.33333333, 0. , 0.16666667]);
    /// ```
    fn forward_par<S, D>(
        &mut self,
        input: &mut ArrayBase<S, D>,
        axis: usize,
    ) -> Array<Self::Spectral, D>
    where
        S: ndarray::Data<Elem = Self::Physical>,
        D: Dimension + ndarray::RemoveAxis,
    {
        use crate::utils::array_resized_axis;
        let mut output = array_resized_axis(input, self.m, axis);
        self.forward_inplace_par(input, &mut output, axis);
        output
    }

    /// See [`Chebyshev::forward_par`]
    #[allow(clippy::used_underscore_binding)]
    fn forward_inplace_par<S1, S2, D>(
        &mut self,
        input: &mut ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = Self::Physical>,
        S2: ndarray::Data<Elem = Self::Spectral> + ndarray::DataMut,
        D: Dimension + ndarray::RemoveAxis,
    {
        use crate::utils::check_array_axis;
        use ndrustfft::nddct1_par;
        check_array_axis(input, self.n, axis, Some("chebyshev forward"));
        check_array_axis(output, self.m, axis, Some("chebyshev forward"));
        // Cosine transform (DCT)
        nddct1_par(input, output, &mut self.dct_handler, axis);
        // Correct DCT
        let _05 = A::from_f64(1. / 2.).unwrap();
        for mut v in output.lanes_mut(Axis(axis)) {
            v *= &self.correct_dct_forward;
            v[0] *= _05;
            v[self.n - 1] *= _05;
        }
    }

    /// # Example
    /// Backward transform along first axis
    /// ```
    /// use funspace::TransformPar;
    /// use funspace::chebyshev::Chebyshev;
    /// use funspace::utils::approx_eq;
    /// use ndarray::prelude::*;
    /// let mut cheby = Chebyshev::new(4);
    /// let mut input = array![1., 2., 3., 4.];
    /// let output = cheby.backward_par(&mut input, 0);
    /// approx_eq(&output, &array![-2. ,  2.5, -3.5, 10.]);
    /// ```
    fn backward_par<S, D>(
        &mut self,
        input: &mut ArrayBase<S, D>,
        axis: usize,
    ) -> Array<Self::Physical, D>
    where
        S: ndarray::Data<Elem = Self::Spectral>,
        D: Dimension + ndarray::RemoveAxis,
    {
        use crate::utils::array_resized_axis;
        let mut output = array_resized_axis(input, self.m, axis);
        self.backward_inplace_par(input, &mut output, axis);
        output
    }

    /// See [`Chebyshev::backward_par`]
    ///
    /// # Panics
    /// Panics when input type cannot be cast from f64.
    #[allow(clippy::used_underscore_binding)]
    fn backward_inplace_par<S1, S2, D>(
        &mut self,
        input: &mut ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = Self::Spectral>,
        S2: ndarray::Data<Elem = Self::Physical> + ndarray::DataMut,
        D: Dimension + ndarray::RemoveAxis,
    {
        use crate::utils::check_array_axis;
        use ndrustfft::nddct1_par;
        check_array_axis(input, self.m, axis, Some("chebyshev backward"));
        check_array_axis(output, self.n, axis, Some("chebyshev backward"));
        // Correct
        let mut buffer = input.to_owned();
        let _2 = A::from_f64(2.).unwrap();
        for mut v in buffer.lanes_mut(Axis(axis)) {
            v *= &self.correct_dct_backward;
            v[0] *= _2;
            v[self.n - 1] *= _2;
        }
        // Cosine transform (DCT)
        nddct1_par(&mut buffer, output, &mut self.dct_handler, axis);
    }
}

impl<A: FloatNum> Differentiate<A> for Chebyshev<A> {
    fn differentiate<S, D>(
        &self,
        data: &ArrayBase<S, D>,
        n_times: usize,
        axis: usize,
    ) -> Array<A, D>
    where
        S: ndarray::Data<Elem = A>,
        D: Dimension,
    {
        // Copy input
        let mut output = data.to_owned();
        self.differentiate_inplace(&mut output, n_times, axis);
        output
    }

    fn differentiate_inplace<S, D>(&self, data: &mut ArrayBase<S, D>, n_times: usize, axis: usize)
    where
        S: ndarray::Data<Elem = A> + ndarray::DataMut,
        D: Dimension,
    {
        use crate::utils::check_array_axis;
        check_array_axis(data, self.m, axis, Some("chebyshev differentiate"));
        ndarray::Zip::from(data.lanes_mut(Axis(axis))).for_each(|mut lane| {
            self.differentiate_lane(&mut lane, n_times);
        });
    }
}

impl<A: FloatNum> LaplacianInverse<A> for Chebyshev<A> {
    /// Pseudoinverse Laplacian of chebyshev spectral
    /// differentiation matrices
    ///
    /// Second order equations become banded
    /// when preconditioned with this matrix
    fn laplace_inv(&self) -> Array2<A> {
        Self::_pinv(self.n, 2)
    }
    /// Pseudoidentity matrix of laplacian
    fn laplace_inv_eye(&self) -> Array2<A> {
        Self::_pinv_eye(self.n, 2)
    }
}

impl<A: FloatNum> FromOrtho<A> for Chebyshev<A> {
    /// Return itself
    fn to_ortho<S, D>(&self, input: &ArrayBase<S, D>, _axis: usize) -> Array<A, D>
    where
        S: ndarray::Data<Elem = A>,
        D: Dimension,
    {
        input.to_owned()
    }

    /// Return itself
    fn to_ortho_inplace<S1, S2, D>(
        &self,
        input: &ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        _axis: usize,
    ) where
        S1: ndarray::Data<Elem = A>,
        S2: ndarray::Data<Elem = A> + ndarray::DataMut,
        D: Dimension,
    {
        output.assign(input);
    }

    /// Return itself
    fn from_ortho<S, D>(&self, input: &ArrayBase<S, D>, _axis: usize) -> Array<A, D>
    where
        S: ndarray::Data<Elem = A>,
        D: Dimension,
    {
        input.to_owned()
    }

    /// Return itself
    fn from_ortho_inplace<S1, S2, D>(
        &self,
        input: &ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        _axis: usize,
    ) where
        S1: ndarray::Data<Elem = A>,
        S2: ndarray::Data<Elem = A> + ndarray::DataMut,
        D: Dimension,
    {
        output.assign(input);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::utils::approx_eq;
    use ndarray::{Array, Dim, Ix};

    #[test]
    /// Differantiate 2d array along first and second axis
    fn test_cheby_differentiate() {
        let (nx, ny) = (6, 4);
        let mut data = Array::<f64, Dim<[Ix; 2]>>::zeros((nx, ny));

        // Axis 0
        let cheby = Chebyshev::new(nx);
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as f64;
        }
        let expected = array![
            [140.0, 149.0, 158.0, 167.0],
            [160.0, 172.0, 184.0, 196.0],
            [272.0, 288.0, 304.0, 320.0],
            [128.0, 136.0, 144.0, 152.0],
            [200.0, 210.0, 220.0, 230.0],
            [0.0, 0.0, 0.0, 0.0],
        ];
        let mut diff = cheby.differentiate(&data, 1, 0);
        approx_eq(&diff, &expected);

        // Axis 1
        let cheby = Chebyshev::new(ny);
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as f64;
        }
        let expected = array![
            [10.0, 8.0, 18.0, 0.0],
            [26.0, 24.0, 42.0, 0.0],
            [42.0, 40.0, 66.0, 0.0],
            [58.0, 56.0, 90.0, 0.0],
            [74.0, 72.0, 114.0, 0.0],
            [90.0, 88.0, 138.0, 0.0],
        ];
        diff.assign(&data);
        cheby.differentiate_inplace(&mut diff, 1, 1);
        approx_eq(&diff, &expected);
    }
}
