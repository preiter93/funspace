use crate::enums::{BaseKind, BaseType, TransformKind};
// use num_traits::real::Real;

/// Base super trait
///
/// *A*: Real-number for coordinates
/// *T*: Scalar on which the function space can act
pub trait Base<A, T>:
    HasLength + HasType + HasCoords<A> + Differentiate<T> + ToOrtho<T> + Transform
{
}
impl<A, T, B> Base<A, T> for B where
    B: HasLength + HasType + HasCoords<A> + Differentiate<T> + ToOrtho<T> + Transform
{
}

/// Get length of base in physical and spectral space
pub trait HasLength {
    /// Size in physical space
    fn len_phys(&self) -> usize;

    /// Size in spectral space
    fn len_spec(&self) -> usize;

    /// Size of orthogonal spectral space
    fn len_ortho(&self) -> usize;
}

/// Get coordinates of a base
pub trait HasCoords<A> {
    /// Coordinates in physical space
    fn coords(&self) -> Vec<A>;
}

/// Return base specification
pub trait HasType {
    /// Return kind of base (e.g. FourierR2c, Chebyshev, ...)
    fn base_kind(&self) -> BaseKind;

    /// Return type of base (e.g. Orthogonal or Composite)
    fn base_type(&self) -> BaseType;

    /// Return kind of transform (e.g, real-to-real, real-to-complex ...)
    fn transform_kind(&self) -> TransformKind;
}

/// Transform between orthogonal <-> composite
pub trait ToOrtho<T> {
    /// Composite -> Orthogonal
    fn to_ortho(&self, comp: &[T], ortho: &mut [T]);

    /// Orthogonal -> Composite
    fn from_ortho(&self, ortho: &[T], compo: &mut [T]);
}

/// Compute derivative in spectral space
///
/// Derivatives of composite bases return coefficients
/// in the respective orthogonal base, since derivatives
/// generally do not satisfy the same boundary conditions
pub trait Differentiate<T> {
    /// Differentiate *order* times
    fn diff(&self, v: &[T], dv: &mut [T], order: usize);

    /// Differentiate *order* times (inplace)
    ///
    /// # Notes
    /// Not implemented for composite bases, where `v.len() != dv.len()`
    fn diff_inplace(&self, _v: &mut [T], _order: usize) {
        unimplemented!("Unimplemented if size change after differentiation!");
    }
}

/// Transform from physical space to spectral space and vice versa
///
/// The associated types *Physical* and *Spectral* refer
/// to the scalar types in the physical and spectral space.
/// For example, Fourier transforms from real-to-complex,
/// while Chebyshev transforms from real-to-real.
pub trait Transform {
    /// Scalar type in physical space
    type Physical;

    /// Scalar type in spectral space
    type Spectral;

    /// Physical values -> Spectral coefficients
    ///
    /// Transforms a one-dimensional slice.
    fn forward(&self, phys: &[Self::Physical], spec: &mut [Self::Spectral]);

    /// Spectral coefficients -> Physical values
    ///
    /// Transforms a one-dimensional slice.
    fn backward(&self, spec: &[Self::Spectral], phys: &mut [Self::Physical]);

    /// Only implemeted if type and size remain unchanged
    fn forward_inplace(&self, _v: &mut [Self::Spectral]) {
        unimplemented!("Unimplemented if type and/or size change after transform!");
    }

    fn backward_inplace(&self, _v: &mut [Self::Physical]) {
        unimplemented!("Unimplemented if type and/or size change after transform!");
    }
}

/// Return respective orthogonal space
pub trait HasOrthoBase<A, T, B: Base<A, T>> {
    /// Get respective orthogonal space. If space is already orthogonal,
    /// it returns an instance of itself.
    fn ortho_base(&self) -> B;
}
