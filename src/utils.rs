/// Test approx equality of two arrays element-wise
///
/// # Panics
/// Panics when difference is larger than 1e-3.
#[allow(dead_code)]
pub fn approx_eq<A>(vec_a: &[A], vec_b: &[A])
where
    A: crate::types::FloatNum + std::fmt::Display,
{
    let tol = A::from_f64(1e-3).unwrap();
    for (a, b) in vec_a.iter().zip(vec_b.iter()) {
        assert!(
            ((*a - *b).abs() < tol),
            "Large difference of values, got {} expected {}.",
            b,
            a
        );
    }
}

/// Test approx equality of two arrays element-wise
///
/// # Panics
/// Panics when difference is larger than 1e-3.
pub fn approx_eq_ndarray<A, S, D>(
    result: &ndarray::ArrayBase<S, D>,
    expected: &ndarray::ArrayBase<S, D>,
) where
    A: crate::types::FloatNum + std::fmt::Display,
    S: ndarray::Data<Elem = A>,
    D: ndarray::Dimension,
{
    let dif = A::from_f64(1e-3).unwrap();
    for (a, b) in expected.iter().zip(result.iter()) {
        assert!(
            ((*a - *b).abs() < dif),
            "Large difference of values, got {} expected {}.",
            b,
            a
        );
    }
}

/// Returns a new array with same dimensionality
/// but different size *n* along the specified *axis*.
///
/// # Example
/// ```
/// use funspace::utils::array_resized_axis;
/// let array = ndarray::Array2::<f64>::zeros((5, 3));
/// let resized: ndarray::Array2<f64> = array_resized_axis(&array, 2, 1);
/// assert!(resized == ndarray::Array2::zeros((5, 2)));
/// ```
pub fn array_resized_axis<A, S, D, T>(
    input: &ndarray::ArrayBase<S, D>,
    size: usize,
    axis: usize,
) -> ndarray::Array<T, D>
where
    T: num_traits::Zero + std::clone::Clone,
    S: ndarray::Data<Elem = A>,
    D: ndarray::Dimension,
{
    // Get dim
    let mut dim = input.raw_dim();

    // Replace position in dim
    dim[axis] = size;

    // Return
    ndarray::Array::<T, D>::zeros(dim)
}
