//! # Use
//! cargo run--example space2 --features="space"
#[cfg(feature = "space")]
use funspace::cheb_dirichlet;
#[cfg(feature = "space")]
use funspace::space::{traits::HasShape, traits::SpaceTransform, Space2};
#[cfg(feature = "space")]
use ndarray::Array2;

#[cfg(feature = "space")]
fn main() {
    let (nx, ny) = (16, 10);
    let space = Space2::new(&cheb_dirichlet::<f64>(nx), &cheb_dirichlet::<f64>(ny));

    // Init
    let mut v = Array2::zeros(space.shape_phys());
    for (i, vi) in v.iter_mut().enumerate() {
        *vi = i as f64;
    }
    let mut vhat = Array2::zeros(space.shape_spec());

    // transform to spectral space
    space.forward(&v, &mut vhat);
    // transform to physical space
    space.backward(&vhat, &mut v);
}

#[cfg(not(feature = "space"))]
fn main() {
    println!("Test requires space feature");
}
