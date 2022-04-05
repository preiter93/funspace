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
#[allow(dead_code)]
pub fn approx_eq_complex<A>(vec_a: &[num_complex::Complex<A>], vec_b: &[num_complex::Complex<A>])
where
    A: crate::types::FloatNum + std::fmt::Display,
{
    let tol = A::from_f64(1e-3).unwrap();
    for (a, b) in vec_a.iter().zip(vec_b.iter()) {
        assert!(
            ((a.re - b.re).abs() < tol || (a.im - b.im).abs() < tol),
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
    let tol = A::from_f64(1e-3).unwrap();
    for (a, b) in expected.iter().zip(result.iter()) {
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
pub fn approx_eq_complex_ndarray<A, S, D>(
    result: &ndarray::ArrayBase<S, D>,
    expected: &ndarray::ArrayBase<S, D>,
) where
    A: crate::types::FloatNum + std::fmt::Display,
    S: ndarray::Data<Elem = num_complex::Complex<A>>,
    D: ndarray::Dimension,
{
    let tol = A::from_f64(1e-3).unwrap();
    for (a, b) in expected.iter().zip(result.iter()) {
        assert!(
            ((a.re - b.re).abs() < tol || (a.im - b.im).abs() < tol),
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

/// Checks size of axis.
///
/// # Panics
/// Panics when inputs shape does not match
/// axis' size
///
/// # Example
/// ```should_panic
/// use funspace::utils::check_array_axis;
/// let array = ndarray::Array2::<f64>::zeros((5, 3));
/// check_array_axis(&array, 3, 0, "");
/// ```
pub fn check_array_axis<A, S, D>(
    input: &ndarray::ArrayBase<S, D>,
    size: usize,
    axis: usize,
    function_name: &str,
) where
    S: ndarray::Data<Elem = A>,
    D: ndarray::Dimension,
{
    // Arrays size
    let m = input.shape()[axis];

    assert!(
        input.shape()[axis] == size,
        "Size mismatch in {}, got {} expected {} along axis {}",
        function_name,
        size,
        m,
        axis
    );
}

// /// Checks size of axis.
// ///
// /// # Panics
// /// Panics when inputs shape does not match
// /// axis' size
// ///
// /// # Example
// /// ```should_panic
// /// use funspace::utils::check_array_axis;
// /// let array = ndarray::Array2::<f64>::zeros((5, 3));
// /// check_array_axis(&array, 3, 0, None);
// /// ```
// pub fn check_array_axis<A, S, D>(
//     input: &ndarray::ArrayBase<S, D>,
//     size: usize,
//     axis: usize,
//     function_name: Option<&str>,
// ) where
//     A: ndarray::LinalgScalar,
//     S: ndarray::Data<Elem = A>,
//     D: ndarray::Dimension,
// {
//     // Arrays size
//     let m = input.shape()[axis];
//
//     // Panic
//     if size != m {
//         if let Some(name) = function_name {
//             panic!(
//                 "Size mismatch in {}, got {} expected {} along axis {}",
//                 name, size, m, axis
//             );
//         } else {
//             panic!(
//                 "Size mismatch, got {} expected {} along axis {}",
//                 size, m, axis
//             );
//         };
//     }
// }
