#![cfg(feature = "mpi")]
use super::decomp2d::Decomp2d;
use super::Universe;
use std::collections::HashMap;

/// Organizes multiple domain decompositions to work with different
/// array sizes. This is necessary because the dimensionality of
/// a space can change when it is transformed from physical to spectral space,
/// or from galerkin space to orthogonal space.
#[derive(Debug, Clone)]
pub struct DecompHandler<'a> {
    /// Mpi universe
    pub universe: &'a Universe,
    /// Mpi Decompositions, referenced by internal lookup value
    pub decomp: HashMap<usize, Decomp2d<'a>>,
    /// Map global array shape to lookup value
    map_shape_global: HashMap<[usize; 2], usize>,
}

impl<'a> DecompHandler<'a> {
    /// Initialize multidecomp with empty decomp hashmap.
    /// Use 'Self::insert' to add decompositions.
    #[must_use]
    pub fn new(universe: &'a Universe) -> Self {
        let decomp: HashMap<usize, Decomp2d<'a>> = HashMap::new();
        let map_shape_global: HashMap<[usize; 2], usize> = HashMap::new();
        Self {
            universe,
            decomp,
            map_shape_global,
        }
    }

    /// Add decomposition.
    /// # Panics
    /// Decomposition must have a unique domain shape.
    pub fn insert(&mut self, decomp: Decomp2d<'a>) {
        let shape_global = decomp.get_global_shape();
        self.map_shape_global.entry(shape_global).or_insert({
            let lookup_val = self.decomp.keys().len();
            self.decomp.insert(lookup_val, decomp);
            lookup_val
        });
    }

    /// Return decomposition which matches a given global arrays shape.
    ///
    /// # Panics
    /// Shape must be known to DecompHandler`. If not, create a new "Decomp2D" instance
    /// and insert in `DecompHandler`.
    #[must_use]
    pub fn get_decomp_from_global_shape(&self, shape: &[usize]) -> &Decomp2d {
        let lookup_val = self.map_shape_global.get(shape).unwrap();
        self.decomp.get(lookup_val).unwrap()
    }
}
