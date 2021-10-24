//! # Use
//! cargo mpirun --np 2 --example space_complex_mpi --features="mpi"
use funspace::cheb_dirichlet;
use funspace::fourier_r2c;
use funspace::mpi::initialize;
use funspace::mpi::BaseSpaceMpi;
use funspace::mpi::Space2;
use funspace::BaseSpace;

fn main() {
    let (nx, ny) = (32, 10);
    let universe = initialize().unwrap();
    let mut space = Space2::new(
        &fourier_r2c::<f64>(nx),
        &cheb_dirichlet::<f64>(ny),
        &universe,
    );

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
    // transform to physical space
    space.backward_inplace_mpi(&x_pencil, &mut y_pencil);
    // collect
    let mut v_mpi = space.ndarray_physical();
    space.all_gather_from_y_pencil_phys(&y_pencil, &mut v_mpi);

    // Serial
    let mut v = global.clone();
    let mut vhat = space.ndarray_spectral();
    space.forward_inplace(&v, &mut vhat);
    space.backward_inplace(&vhat, &mut v);

    // Serial and parallel results should be the same
    assert_eq!(v, v_mpi);

    // just test to ortho / from ortho transforms
    let ortho = space.to_ortho_mpi(&x_pencil);
    let _ = space.from_ortho_mpi(&ortho);
}
