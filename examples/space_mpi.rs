//! # Use
//! cargo mpirun --np 2 --example space_mpi --features="mpi"
#[cfg(feature = "mpi")]
use funspace::spaces_mpi::{initialize, BaseSpaceMpi, Space2};
#[cfg(feature = "mpi")]
use funspace::{cheb_dirichlet, BaseSpace};
#[cfg(feature = "mpi")]
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

    // Scatter data from too to all processor
    let mut y_pencil = space.ndarray_physical_y_pen();
    if space.get_nrank() == 0 {
        let mut global = space.ndarray_physical();
        for (i, v) in global.iter_mut().enumerate() {
            *v = i as f64;
        }
        space.scatter_to_y_pencil_phys_root(&global, &mut y_pencil);
    } else {
        space.scatter_to_y_pencil_phys(&mut y_pencil);
    }

    // transform to spectral space
    let mut x_pencil = space.ndarray_spectral_x_pen();
    space.forward_inplace_mpi(&y_pencil, &mut x_pencil);
    // transform to physical space
    space.backward_inplace_mpi(&x_pencil, &mut y_pencil);

    // collect data on all processors (expensive)
    let mut v_mpi = space.ndarray_physical();
    space.all_gather_from_y_pencil_phys(&y_pencil, &mut v_mpi);

    // Serial
    let mut v = space.ndarray_physical();
    for (i, vi) in v.iter_mut().enumerate() {
        *vi = i as f64;
    }
    let mut vhat = space.ndarray_spectral();
    space.forward_inplace(&v, &mut vhat);
    space.backward_inplace(&vhat, &mut v);

    // Serial and parallel results should be the same
    assert_eq!(v, v_mpi);

    // just test to ortho / from ortho transforms
    let ortho = space.to_ortho_mpi(&x_pencil);
    let _ = space.from_ortho_mpi(&ortho);
}

#[cfg(not(feature = "mpi"))]
fn main() {
    println!("Test requires mpi feature");
}
