#![cfg(feature = "mpi")]
use super::Universe;
use rsmpi_decomp::Decomp2d;
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
    /// Map x-pencil array shape to lookup value
    map_shape_x_penc: HashMap<[usize; 2], usize>,
    /// Map x-pencil array shape to lookup value
    map_shape_y_penc: HashMap<[usize; 2], usize>,
}

impl<'a> DecompHandler<'a> {
    /// Initialize multidecomp with empty decomp hashmap.
    /// Use 'Self::insert' to add decompositions.
    pub fn new(universe: &'a Universe) -> Self {
        let decomp: HashMap<usize, Decomp2d<'a>> = HashMap::new();
        let map_shape_global: HashMap<[usize; 2], usize> = HashMap::new();
        let map_shape_x_penc: HashMap<[usize; 2], usize> = HashMap::new();
        let map_shape_y_penc: HashMap<[usize; 2], usize> = HashMap::new();
        Self {
            universe,
            decomp,
            map_shape_global,
            map_shape_x_penc,
            map_shape_y_penc,
        }
    }

    /// Add decomposition.
    /// # Panics
    /// Decomposition must have a unique domain shape.
    pub fn insert(&mut self, decomp: Decomp2d<'a>) {
        // get shapes
        let shape_global = decomp.get_global_shape();
        let shape_x_penc = decomp.get_x_pencil_shape();
        let shape_y_penc = decomp.get_y_pencil_shape();
        if self.map_shape_global.contains_key(&shape_global) {
            println!(
                "Decomposition with global shape {:?} already known to SpaceDecomp. Skip.",
                shape_global
            );
        } else {
            let lookup_val = self.decomp.keys().len();
            self.decomp.insert(lookup_val, decomp);
            // map lookup value to relevant array shapes,
            // so that we can later lookup a decomposition by
            // its shape
            self.map_shape_global.insert(shape_global, lookup_val);
            self.map_shape_x_penc.insert(shape_x_penc, lookup_val);
            self.map_shape_y_penc.insert(shape_y_penc, lookup_val);
        }
    }

    /// Return decomposition which matches a given global arrays shape.
    pub fn get_decomp_from_global(&self, shape: &[usize]) -> &Decomp2d {
        let lookup_val = self.map_shape_global.get(shape).unwrap();
        self.decomp.get(lookup_val).unwrap()
    }

    /// Return decomposition which matches a given x-pencil arrays shape.
    pub fn get_decomp_from_x_pencil(&self, shape: &[usize]) -> &Decomp2d {
        let lookup_val = self.map_shape_x_penc.get(shape).unwrap();
        self.decomp.get(lookup_val).unwrap()
    }

    /// Return decomposition which matches a given x-pencil arrays shape.
    pub fn get_decomp_from_y_pencil(&self, shape: &[usize]) -> &Decomp2d {
        let lookup_val = self.map_shape_y_penc.get(shape).unwrap();
        self.decomp.get(lookup_val).unwrap()
    }
}
