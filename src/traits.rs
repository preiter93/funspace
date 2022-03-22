//! # Collection of traits that bases must implement
use crate::types::FloatNum;

pub trait BaseElemental {
    /// Coordinates in physical space
    fn nodes<T: FloatNum>(&self) -> &[T];
}
