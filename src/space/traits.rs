use ndarray::{ArrayBase, Data, DataMut, Dim};

/// Dimensions
pub trait HasShape<const N: usize> {
    /// Shape of physical space
    fn shape_phys(&self) -> [usize; N];

    /// Shape of spectral space
    fn shape_spec(&self) -> [usize; N];

    /// Shape of spectral space (orthogonal bases)
    fn shape_spec_ortho(&self) -> [usize; N];
}

/// Transform between orthogonal <-> composite
pub trait SpaceToOrtho<T, const N: usize> {
    /// Composite -> Orthogonal
    fn to_ortho_axis<S1, S2>(
        &self,
        comp: &ArrayBase<S1, Dim<[usize; N]>>,
        ortho: &mut ArrayBase<S2, Dim<[usize; N]>>,
        axis: usize,
    ) where
        S1: Data<Elem = T>,
        S2: Data<Elem = T> + DataMut;

    /// Orthogonal -> Composite
    fn from_ortho_axis<S1, S2>(
        &self,
        ortho: &ArrayBase<S1, Dim<[usize; N]>>,
        comp: &mut ArrayBase<S2, Dim<[usize; N]>>,
        axis: usize,
    ) where
        S1: Data<Elem = T>,
        S2: Data<Elem = T> + DataMut;

    /// Composite -> Orthogonal
    fn to_ortho<S1, S2>(
        &self,
        comp: &ArrayBase<S1, Dim<[usize; N]>>,
        ortho: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = T>,
        S2: Data<Elem = T> + DataMut;

    /// Orthogonal -> Composite
    fn from_ortho<S1, S2>(
        &self,
        ortho: &ArrayBase<S1, Dim<[usize; N]>>,
        comp: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = T>,
        S2: Data<Elem = T> + DataMut;
    // {
    //     unimplemented!()
    // }
}

/// Differentiate
pub trait SpaceDifferentiate<T, const N: usize> {
    fn diff_axis<S1, S2>(
        &self,
        v: &ArrayBase<S1, Dim<[usize; N]>>,
        dv: &mut ArrayBase<S2, Dim<[usize; N]>>,
        order: usize,
        axis: usize,
    ) where
        S1: Data<Elem = T>,
        S2: Data<Elem = T> + DataMut;
}

/// # Transformation from physical values to spectral coefficients
pub trait SpaceTransform<A, const N: usize> {
    // Number type in physical space (float or complex)
    type Physical;

    // Number type in spectral space (float or complex)
    type Spectral;

    /// Transform physical -> spectral space
    fn forward<S1, S2>(
        &self,
        phys: &ArrayBase<S1, Dim<[usize; N]>>,
        spec: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = Self::Physical>,
        S2: Data<Elem = Self::Spectral> + DataMut;

    /// Transform spectral -> physical space
    fn backward<S1, S2>(
        &self,
        spec: &ArrayBase<S1, Dim<[usize; N]>>,
        phys: &mut ArrayBase<S2, Dim<[usize; N]>>,
    ) where
        S1: Data<Elem = Self::Spectral>,
        S2: Data<Elem = Self::Physical> + DataMut;
}
