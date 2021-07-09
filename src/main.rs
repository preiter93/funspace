use funspace::utils::approx_eq;
use funspace::utils::array_resized_axis;
use funspace::{Chebyshev, Transform};
use ndarray::prelude::*;

fn main() {
    use ndarray::prelude::*;
    let mut cheby = Chebyshev::new(4);
    let mut input = array![1., 2., 3., 4.];
    let output = cheby.forward(&mut input, 0);
    println!("{:?}", output);
    approx_eq(&output, &array![2.5, 1.33333333, 0., 0.16666667]);
}
