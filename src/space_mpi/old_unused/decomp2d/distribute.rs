//! # Data distribution
//!
//! Save size, first/last index of current processor
//! and in a vector for all processors. The latter is necessary
//! for mpi routines.
//!
//! Furthermore, `Distribution` has functions returning counts,
//! displs for different mpi routines
#![allow(clippy::module_name_repetitions)]
use mpi_crate::Count;
use ndarray::{s, Array2, ArrayBase, Data, DataMut, Dim, Ix2};

/// Distribute Grid points to processors.
///
/// *n* specifies the number of dimensions
#[derive(Debug, Clone)]
pub struct Distribution<const N: usize> {
    // Size of data of current processor
    pub sz: [usize; N],
    // Starting index of data of current processor
    pub st: [usize; N],
    // Ending index of data of current processor
    pub en: [usize; N],
    // Size of data of all processors
    pub sz_procs: Vec<[usize; N]>,
    // Starting index of data of all processors
    pub st_procs: Vec<[usize; N]>,
    // Ending index of data of all processors
    pub en_procs: Vec<[usize; N]>,
    // Number of processors
    pub nprocs: usize,
    // Current processor id
    pub nrank: usize,
    // Axis that is contiguous, all others are split
    pub axis_contig: usize,
}

impl<const N: usize> Distribution<N> {
    /// Generate decomposition
    ///
    /// # Arguments
    /// * `n_global`: Total number of grid points [nx global, ny global]
    /// * `nprocs`: Number of processors
    /// * `nrank`: Current processor id
    /// * `axis_contig`: Axis that is contiguous, all others are split
    ///
    /// # Panics
    ///
    /// Panics if `axis_split` is not 0 or 1.
    #[must_use]
    pub fn new(n_global: [usize; N], nprocs: usize, nrank: usize, axis_contig: usize) -> Self {
        if axis_contig > 1 {
            panic!(
                "axis_contig must be 0 (first axis) or 1 (second axis), got {}.",
                axis_contig
            );
        }

        // Define default local indices, start from full domain, then update after each split
        let mut st_procs: Vec<[usize; N]> = vec![[0; N]; nprocs];
        let mut sz_procs: Vec<[usize; N]> = vec![n_global; nprocs];
        let mut en_procs: Vec<[usize; N]> = vec![n_global; nprocs];
        // Default end element is one less than size
        for array in &mut en_procs {
            for element in array.iter_mut() {
                *element -= 1;
            }
        }
        // Iterate over all non-cont axis and split them
        for axis in 0..N {
            if axis == axis_contig {
                continue;
            }
            // Distribute along the split-axis
            let (st_split, en_split, sz_split) = Self::distribute(n_global[axis], nprocs);
            // Update size of distributed axis
            for (i, j) in st_procs.iter_mut().zip(st_split.iter()) {
                i[axis] = *j;
            }
            for (i, j) in en_procs.iter_mut().zip(en_split.iter()) {
                i[axis] = *j;
            }
            for (i, j) in sz_procs.iter_mut().zip(sz_split.iter()) {
                i[axis] = *j;
            }
        }
        // Get size and start/end index of current processor
        let st = st_procs[nrank];
        let en = en_procs[nrank];
        let sz = sz_procs[nrank];

        Self {
            st,
            en,
            sz,
            st_procs,
            en_procs,
            sz_procs,
            nprocs,
            nrank,
            axis_contig,
        }
    }

    /// Distribute grid points across processors along 1-dimension
    ///
    /// # Arguments
    /// * `n_global`: Total number of grid points along the split dimension
    /// * `nprocs`: Number of processors in the split dimension
    ///
    /// # Return
    /// Vectors containing starting/ending index and size of each
    /// processor
    fn distribute(n_global: usize, nprocs: usize) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        let size = n_global / nprocs;
        let mut st = vec![0; nprocs];
        let mut en = vec![0; nprocs];
        let mut sz = vec![0; nprocs];
        // Try to distribute N points
        st[0] = 0;
        sz[0] = size;
        en[0] = size - 1;
        // Distribute the rest if necessary
        let nu = n_global - size * nprocs;
        // Define how many processors held exactly N points, the rest holds N+1
        let nl = nprocs - nu;
        // Distribute N points on the first processors
        for i in 1..nl {
            st[i] = st[i - 1] + size;
            sz[i] = size;
            en[i] = en[i - 1] + size;
        }
        // Distribute  N+1 points on the last processors
        let size = size + 1;
        for i in nl..nprocs {
            st[i] = en[i - 1] + 1;
            sz[i] = size;
            en[i] = en[i - 1] + size;
        }
        // Very last processor
        en[nprocs - 1] = n_global - 1;
        sz[nprocs - 1] = en[nprocs - 1] - st[nprocs - 1] + 1;
        (st, en, sz)
    }

    /// Return the total length of data hold by current processor
    #[must_use]
    pub fn length(&self) -> usize {
        self.sz.iter().product()
    }

    /// # Panics
    /// Panics if array shape does not conform with pencil distribution
    pub fn check_shape<A, S>(&self, data: &ArrayBase<S, Dim<[usize; N]>>)
    where
        S: Data<Elem = A>,
        Dim<[usize; N]>: ndarray::Dimension,
    {
        if data.shape() != self.sz {
            panic!(
                "Shape mismatch, got {:?} expected {:?}.",
                data.shape(),
                self.sz
            );
        }
    }

    /// Return (counts, displs) for mpi all gather varcount
    #[must_use]
    pub fn get_counts_all_gather(&self) -> (Vec<Count>, Vec<Count>) {
        let counts: Vec<Count> = self
            .sz_procs
            .iter()
            .map(|x| (x.iter().product::<usize>()) as i32)
            .collect();
        let displs: Vec<Count> = counts
            .iter()
            .scan(0, |acc, &x| {
                let tmp = *acc;
                *acc += x;
                Some(tmp)
            })
            .collect();
        (counts, displs)
    }
}

impl Distribution<2> {
    /// Return (counts, displs) for mpi all to all varcount
    /// # Arguments
    /// * `recv_dist`: Receiving distribution
    #[must_use]
    pub fn get_counts_all_to_all(&self, recv_dist: &Distribution<2>) -> (Vec<Count>, Vec<Count>) {
        let counts: Vec<Count> = recv_dist
            .sz_procs
            .iter()
            .map(|x| (x[self.axis_contig] * self.sz[recv_dist.axis_contig]) as i32)
            .collect();
        let displs: Vec<Count> = counts
            .iter()
            .scan(0, |acc, &x| {
                let tmp = *acc;
                *acc += x;
                Some(tmp)
            })
            .collect();
        (counts, displs)
    }

    /// Return processor's share of global data
    #[must_use]
    pub fn split_array<S, T>(&self, global: &ArrayBase<S, Ix2>) -> Array2<T>
    where
        S: Data<Elem = T>,
        T: Copy,
    {
        global
            .slice(s![self.st[0]..=self.en[0], self.st[1]..=self.en[1]])
            .to_owned()
    }

    /// Split global data to pencil data (inplace)
    pub fn split_array_inplace<S1, S2, T>(
        &self,
        global: &ArrayBase<S1, Ix2>,
        pencil: &mut ArrayBase<S2, Ix2>,
    ) where
        S1: Data<Elem = T>,
        S2: Data<Elem = T> + DataMut,
        T: Copy,
    {
        pencil.assign(&global.slice(s![self.st[0]..=self.en[0], self.st[1]..=self.en[1]]));
    }
}
