# funspace
<img align="right" src="https://rustacean.net/assets/cuddlyferris.png" width="80">

## Funspace

Collection of function spaces.

A function space is made up of elements of basis functions.
Every function in the function space can be represented as a
linear combination of basis functions, represented by real/complex
coefficients (spectral space).

### Transform
A transform describes a change from the physical space to the function
space. For example, a fourier transform describes a transform from
values of a function on a regular grid to coefficents of sine/cosine
polynomials. This concept is analogous to other function spaces.

### Differentiation
One key advantage of representation of a function with coefficents in
the function space is its ease of differentiation. Differentiation in
fourier space becomes multiplication with the wavenumbe vector.
Differentiation in Chebyshev space can be done easily by a recurrence
relation.
Each base implements a differentiation method, which must be applied on
an array of coefficents.

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
satisfied. This may be understood under *Galerkin* Method.

To switch from its composite form to the orthonormal form, each base implements
a *Parental* trait, which defines the transform *to_ortho* and *from_ortho*.
If the base is already orthogonal, the input will be returned, otherwise it
is returned. Note that the dimensionality of the composite space is often
less than its orthogonal counterpart.  Therefore the output array must
not maintain the same shape (but dimensionality is conserved).

### Implemented function spaces:
- Chebyshev (Orthogonal)
- ChebDirichlet (Composite)
- ChebNeumann (Composite)

## Example
Apply forward transform of 1d array in cheb_dirichlet space
```rust
use funspace::{Transform, cheb_dirichlet};
use ndarray::prelude::*;
let mut cd = cheb_dirichlet::<f64>(5);
let mut input = array![1., 2., 3., 4., 5.];
let output = cd.forward(&mut input, 0);
```
