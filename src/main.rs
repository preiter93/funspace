use funspace::cheb_dirichlet;
// use funspace::{Chebyshev, Transform};
use funspace::Differentiate;
use funspace::Transform;
use ndarray::prelude::*;

fn main() {
    use ndarray::prelude::*;
    let mut cd = cheb_dirichlet::<f64>(5);
    let mut input = array![1., 2., 3., 4., 5.];
    let output = cd.forward(&mut input, 0);
    println!("{:?}", output);
}
