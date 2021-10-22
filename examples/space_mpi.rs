//! # Use
//! cargo mpirun --np 2 --example space_mpi --features="mpi"
use funspace::cheb_dirichlet;
use funspace::mpi::initialize;
use funspace::mpi::BaseSpaceMpi;
use funspace::mpi::Space2;
use funspace::BaseSpace;

fn main() {
    let (nx, ny) = (16, 10);
    let universe = initialize().unwrap();
    let mut space = Space2::new(
        &cheb_dirichlet::<f64>(nx),
        &cheb_dirichlet::<f64>(ny),
        &universe,
    );
    // let dcp = DecompHandler::new(&universe);
    // let data = space.ndarray_spectral_x_pen();
    // let _ = space.to_ortho_mpi(&data);

    let mut global = space.ndarray_physical();
    for (i, v) in global.iter_mut().enumerate() {
        *v = i as f64;
    }

    // Mpi supported
    let mut y_pencil = space.ndarray_physical_y_pen();
    space.scatter_to_y_pencil_phys(&global, &mut y_pencil);
    // transform to spectral space
    let mut x_pencil = space.ndarray_spectral_x_pen();
    space.forward_inplace_mpi(&y_pencil, &mut x_pencil);
    // // transform to physical space
    // space.backward_inplace_mpi(&x_pencil, &mut y_pencil);
    // // collect
    // let mut v_mpi = space.ndarray_physical();
    // space.all_gather_from_y_pencil_phys(&y_pencil, &mut v_mpi);
    //
    // // Serial
    // let mut v = global.clone();
    // let mut vhat = space.ndarray_spectral();
    // space.forward_inplace(&v, &mut vhat);
    // space.backward_inplace(&vhat, &mut v);

    // // Serial and parallel results should be the same
    // assert_eq!(v, v_mpi);
}
