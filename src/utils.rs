//! Collection of general functions
use super::FloatNum;
use ndarray::prelude::*;

/// Returns a new array with same dimensionality
/// but different size *n* along the specified *axis*.
///
/// # Example
/// ```
/// use funspace::utils::array_resized_axis;
/// let array = ndarray::Array2::zeros((5, 3));
/// let resized: ndarray::Array2<f64> = array_resized_axis(&array, 2, 1);
/// assert!(resized == ndarray::Array2::zeros((5, 2)));
/// ```
pub fn array_resized_axis<A, S, D>(input: &ArrayBase<S, D>, size: usize, axis: usize) -> Array<A, D>
where
    A: ndarray::LinalgScalar,
    S: ndarray::Data<Elem = A>,
    D: Dimension,
{
    // Get dim
    let mut dim = input.raw_dim();

    // Replace position in dim
    dim[axis] = size;

    // Return
    Array::<A, D>::zeros(dim)
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
/// check_array_axis(&array, 3, 0, None);
/// ```
pub fn check_array_axis<A, S, D>(
    input: &ArrayBase<S, D>,
    size: usize,
    axis: usize,
    function_name: Option<&str>,
) where
    A: ndarray::LinalgScalar,
    S: ndarray::Data<Elem = A>,
    D: Dimension,
{
    // Arrays size
    let m = input.shape()[axis];

    // Panic
    if size != m {
        if let Some(name) = function_name {
            panic!(
                "Size mismatch in {}, got {} expected {} along axis {}",
                name, size, m, axis
            );
        } else {
            panic!(
                "Size mismatch, got {} expected {} along axis {}",
                size, m, axis
            );
        };
    }
}

/// Test approx equality of two arrays element-wise
///
/// # Panics
/// Panics when difference is larger than 1e-3.
pub fn approx_eq<A, S, D>(result: &ArrayBase<S, D>, expected: &ArrayBase<S, D>)
where
    A: FloatNum + std::fmt::Display,
    S: ndarray::Data<Elem = A>,
    D: Dimension,
{
    let dif = A::from_f64(1e-3).unwrap();
    for (a, b) in expected.iter().zip(result.iter()) {
        if (*a - *b).abs() > dif {
            panic!("Large difference of values, got {} expected {}.", b, a)
        }
    }
}
