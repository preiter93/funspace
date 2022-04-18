//! # Use
//! cargo mpirun --np 2 --example space_mpi --features="mpi"
#[cfg(feature = "mpi")]
use funspace::space_mpi::traits::{
    BaseSpaceMpiFromOrtho, BaseSpaceMpiGradient, BaseSpaceMpiSize, BaseSpaceMpiTransform,
};
#[cfg(feature = "mpi")]
use funspace::space_mpi::{initialize, Space2Mpi};
#[cfg(feature = "mpi")]
use funspace::{cheb_dirichlet, BaseSpaceSize, BaseSpaceTransform};
#[cfg(feature = "mpi")]
fn main() {
    let (nx, ny) = (16, 10);
    let universe = initialize().unwrap();
    let space = Space2Mpi::new(
        &cheb_dirichlet::<f64>(nx),
        &cheb_dirichlet::<f64>(ny),
        &universe,
    );
    let mut v = ndarray::Array2::<f64>::ones(space.shape_physical_mpi());
    let mut vhat = ndarray::Array2::<f64>::zeros(space.shape_spectral_mpi());
    // transform to spectral space
    space.forward_inplace_mpi(&v, &mut vhat);
    // transform back to physical space
    space.backward_inplace_mpi(&vhat, &mut v);

    // just test to ortho / from ortho transforms
    let ortho = space.to_ortho_mpi(&vhat);
    let _ = space.from_ortho_mpi(&ortho);

    // Test gradient
    let _ = space.gradient_mpi(&vhat, [2, 0], None);

    // Serial
    let mut vs = ndarray::Array2::<f64>::ones(space.shape_physical());
    let mut vhats = ndarray::Array2::<f64>::zeros(space.shape_spectral());
    space.forward_inplace(&vs, &mut vhats);
    space.backward_inplace(&vhats, &mut vs);

    // Serial and parallel results should be the same
    let dcp = space.dcp.get(&space.shape_spectral());
    let mut ii = 0;
    for i in dcp.x_pencil.dists[0].st..dcp.x_pencil.dists[0].en {
        let mut jj = 0;
        for j in dcp.x_pencil.dists[1].st..dcp.x_pencil.dists[1].en {
            assert!((vhats[[i, j]] - vhat[[ii, jj]]).abs() < 1e-3);
            // println!("{:?} {:?}", vhats[[i, j]], vhat[[ii, jj]]);
            jj += 1;
        }
        ii += 1;
    }
    // assert_eq!(v, v_mpi);
}

#[cfg(not(feature = "mpi"))]
fn main() {
    println!("Test requires mpi feature");
}
