# funspace

## Funspace
<img align="right" src="https://rustacean.net/assets/cuddlyferris.png" width="80">

Collection of function spaces.

A function space is made up of elements of basis functions.
Every function in the function space can be represented as a
linear combination of basis functions, represented by real/complex
coefficients (spectral space).

### Implemented function spaces:
- `Chebyshev` (Orthogonal), see [`chebyshev()`]
- `ChebDirichlet` (Composite), see [`cheb_dirichlet()`]
- `ChebNeumann` (Composite), see [`cheb_neumann()`]
- `FourierC2c` (Orthogonal), see [`fourier_c2c()`]
- `FourierR2c` (Orthogonal), see [`fourier_r2c()`]

### Transform
A transform describes a change from the physical space to the function
space. For example, a fourier transform describes a transform from
values of a function on a regular grid to coefficents of sine/cosine
polynomials. This is analogous to other function spaces. The transforms
are implemented by the [`Transfrom`] trait.

#### Example
Apply forward transform of 1d array in `cheb_dirichlet` space
```rust
use funspace::{Transform, cheb_dirichlet};
use ndarray::prelude::*;
use ndarray::Array1;
let mut cd = cheb_dirichlet::<f64>(5);
let mut input = array![1., 2., 3., 4., 5.];
let output: Array1<f64> = cd.forward(&mut input, 0);
```

### Differentiation
One key advantage representing a function with coefficents in
the function space is its ease of differentiation. Differentiation in
fourier space becomes multiplication with the wavenumbe vector.
Differentiation in Chebyshev space can be done easily by a recurrence
relation.
Each base implements a differentiation method, which must be applied on
an array of coefficents. This is defined by the [`Differentiation`] trait.

#### Example
Apply differentiation
```rust
use funspace::{Transform, Differentiate, BaseBasics, fourier_r2c};
use ndarray::prelude::*;
use ndarray::Array1;
use num_complex::Complex;
// Define base
let mut fo = fourier_r2c(8);
// Get coordinates in physical space
let x = fo.coords().clone();
let mut v = x.mapv(|xi: f64| (2. * xi).sin());
// Transform to physical space
let vhat: Array1<Complex<f64>> = fo.forward(&mut v, 0);

// Apply differentiation twice along first axis
let mut dvhat = fo.differentiate(&vhat, 2, 0);
// Transform back to spectral space
let dv: Array1<f64> = fo.backward(&mut dvhat, 0);
// Compare with correct derivative
for (exp, ist) in x
    .mapv(|xi: f64| -4. * (2. * xi).sin())
    .iter()
    .zip(dv.iter())
{
    assert!((exp - ist).abs() < 1e-5);
}
```

### Composite Bases
Bases like those of fourier polynomials or chebyshev polynomials are
considered orthonormal bases, i.e. the dot product of each individual
polynomial with any other of its set vanishes.
But other function spaces can be constructed by a linear combination
the orthonormal basis functions. This is often used
to construct bases which satisfy particular boundary conditions
like dirichlet (zero at the ends) or neumann (zero derivative at the ends).
This is usefull when solving partial differential equations. When expressed
in those composite function space, the boundary condition is automatically
satisfied. This is known as the *Galerkin* Method.

To switch from its composite form to the orthonormal form, each base implements
a [`FromOrtho`] trait, which defines the transform `to_ortho` and `from_ortho`.
If the base is already orthogonal, the input will be returned, otherwise it
is returned. Note that the dimensionality of the composite space is often
less than its orthogonal counterpart.  Therefore the output array must
not maintain the same shape (but dimensionality is conserved).

#### Example
Transform composite space `cheb_dirichlet` to its orthogonal counterpart
`chebyshev`
```rust
use funspace::{Transform, FromOrtho, BaseBasics};
use funspace::{cheb_dirichlet, chebyshev};
use std::f64::consts::PI;
use ndarray::prelude::*;
use ndarray::Array1;
use num_complex::Complex;
// Define base
let mut ch = chebyshev(8);
let mut cd = cheb_dirichlet(8);
// Get coordinates in physical space
let x = ch.coords().clone();
let mut v = x.mapv(|xi: f64| (PI / 2. * xi).cos());
// Transform to physical space
let ch_vhat: Array1<f64> = ch.forward(&mut v, 0);
let cd_vhat: Array1<f64> = cd.forward(&mut v, 0);
// Send array to orthogonal space (cheb_dirichlet
// to chebyshev in this case)
let cd_vhat_ortho = cd.to_ortho(&cd_vhat, 0);
// Both arrays are equal, because field was
// initialized with correct boundary conditions,
// i.e. dirichlet ones
for (exp, ist) in ch_vhat.iter().zip(cd_vhat_ortho.iter()) {
    assert!((exp - ist).abs() < 1e-5);
}

// However, if the physical field values do not
// satisfy dirichlet boundary conditions, they
// will be enforced by the transform to cheb_dirichle
// and ultimately the transformed values will deviate
// from a pure chebyshev transform (which does not)
// enfore the boundary conditions.
let mut v = x.mapv(|xi: f64| (PI / 2. * xi).sin());
let ch_vhat: Array1<f64> = ch.forward(&mut v, 0);
let cd_vhat: Array1<f64> = cd.forward(&mut v, 0);
let cd_vhat_ortho = cd.to_ortho(&cd_vhat, 0);
// They will deviate
println!("chebyshev     : {:?}", ch_vhat);
println!("cheb_dirichlet: {:?}", cd_vhat_ortho);
```

### Multidimensional Spaces
A collection of bases makes up a [`SpaceBase`], which defines operations
along a specfic dimension (= axis). Care must be taken when transforming
a field from the physical space to the spectral space on how the transforms
are chained in a multidimensional space. For example, `cheb_dirichlet` is a
real-to-real transform, while `fourier_r2c` defines a real-to-complex transform.
So, for a given real valued physical field, the chebyshev transform must be applied
before the fourier transform in the forward transform, and in opposite order in
the backward transform.

#### Example
Apply transform from physical to spectral in a two-dimensional space
```rust
use funspace::{fourier_r2c, cheb_dirichlet, Space2, Transform, BaseBasics};
use ndarray::prelude::*;
use num_complex::Complex;
use std::f64::consts::PI;
// Define the space and allocate arrays
let mut space = Space2::new(&[fourier_r2c(5), cheb_dirichlet(5)]);
let mut v: Array2<f64> = space.ndarray_physical();
let mut vhat: Array2<Complex<f64>> = space.ndarray_spectral();
// Set some field values
let x = space.bases[0].coords();
let y = space.bases[1].coords();
for (i,xi) in x.iter().enumerate() {
    for (j,yi) in y.iter().enumerate() {
        v[[i,j]] = xi.sin() * (PI/2.*yi).cos();
    }
}
// Transform chebyshev
let mut buffer: Array2<f64> = space.forward(&mut v, 1);
// Transform fourier
space.forward_inplace(&mut buffer, &mut vhat, 0);
```

License: MIT
