//! Row-major contiguous FP32 CPU tensor with checked allocation (§3.6).

pub mod ops;

#[derive(Debug, Clone, PartialEq)]
pub enum TensorError {
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },
    Rank { expected: usize, actual: usize },
    IndexOutOfBounds { index: usize, bound: usize },
    AllocationOverflow,
    AllMaskedRow { row: usize },
    InvalidArgument(&'static str),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

/// Checked element count for a shape.
pub fn numel(shape: &[usize]) -> Result<usize, TensorError> {
    shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or(TensorError::AllocationOverflow)
}

impl Tensor {
    pub fn zeros(shape: &[usize]) -> Result<Self, TensorError> {
        let n = numel(shape)?;
        Ok(Tensor { data: vec![0.0; n], shape: shape.to_vec() })
    }

    pub fn from_vec(shape: &[usize], data: Vec<f32>) -> Result<Self, TensorError> {
        let n = numel(shape)?;
        if n != data.len() {
            return Err(TensorError::ShapeMismatch {
                expected: shape.to_vec(),
                actual: vec![data.len()],
            });
        }
        Ok(Tensor { data, shape: shape.to_vec() })
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn data(&self) -> &[f32] {
        &self.data
    }
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn reshape(&self, shape: &[usize]) -> Result<Tensor, TensorError> {
        if numel(shape)? != self.data.len() {
            return Err(TensorError::ShapeMismatch {
                expected: shape.to_vec(),
                actual: self.shape.clone(),
            });
        }
        Ok(Tensor { data: self.data.clone(), shape: shape.to_vec() })
    }

    /// 2-D transpose; result is materialised contiguous (layout restored).
    pub fn transpose2d(&self) -> Result<Tensor, TensorError> {
        if self.shape.len() != 2 {
            return Err(TensorError::Rank { expected: 2, actual: self.shape.len() });
        }
        let (r, c) = (self.shape[0], self.shape[1]);
        let mut out = vec![0.0; self.data.len()];
        for i in 0..r {
            for j in 0..c {
                out[j * r + i] = self.data[i * c + j];
            }
        }
        Tensor::from_vec(&[c, r], out)
    }

    /// Gather rows of a 2-D tensor by index.
    pub fn gather_rows(&self, idx: &[usize]) -> Result<Tensor, TensorError> {
        if self.shape.len() != 2 {
            return Err(TensorError::Rank { expected: 2, actual: self.shape.len() });
        }
        let (r, c) = (self.shape[0], self.shape[1]);
        let mut out = Vec::with_capacity(idx.len().checked_mul(c).ok_or(TensorError::AllocationOverflow)?);
        for &i in idx {
            if i >= r {
                return Err(TensorError::IndexOutOfBounds { index: i, bound: r });
            }
            out.extend_from_slice(&self.data[i * c..(i + 1) * c]);
        }
        Tensor::from_vec(&[idx.len(), c], out)
    }

    pub fn all_finite(&self) -> bool {
        self.data.iter().all(|x| x.is_finite())
    }
}
