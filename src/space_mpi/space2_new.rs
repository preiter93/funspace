//! Two-dimensional function space with mpi support
#![cfg(feature = "mpi")]
use crate::space::Space2;
use std::ops::Deref;

#[derive(Clone)]
pub struct Space2Mpi<B0, B1> {
    /// Non-mpi space
    pub space: Space2<B0, B1>,
    // // Intermediate <-> Spectral
    // pub base0: B0,
    // // Phsical <-> Intermediate
    // pub base1: B1,
    // Mpi Handler
    // pub decomp_handler: DecompHandler<'a>,
}

impl<B0, B1> Space2Mpi<B0, B1>
where
    B0: Clone,
    B1: Clone,
{
    /// Create a new space
    pub fn new(base0: &B0, base1: &B1) -> Self {
        let space = Space2 {
            base0: base0.clone(),
            base1: base1.clone(),
        };
        Self { space }
    }
}

impl<B0, B1> Deref for Space2Mpi<B0, B1> {
    type Target = Space2<B0, B1>;

    fn deref(&self) -> &Self::Target {
        &self.space
    }
}
