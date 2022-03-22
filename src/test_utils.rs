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
