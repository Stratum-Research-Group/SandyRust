//! Reference (non-tiled) masked causal attention used as the G4 oracle (§3.3).
//! Causality uses absolute positions: key k is visible to query q iff k <= q + offset.

use crate::tensor::ops::{masked_softmax_row, matmul};
use crate::tensor::{Tensor, TensorError};

/// `q:[Tq,dh]`, `k,v:[Tk,dh]`. Padded queries output exact zero; a valid query with
/// every key masked returns `AllMaskedRow`. Returns `(output, probabilities[Tq*Tk])`.
pub fn attention_ref(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    q_valid: &[bool],
    k_valid: &[bool],
    offset: usize,
) -> Result<(Tensor, Vec<f32>), TensorError> {
    let (tq, dh) = (q.shape()[0], q.shape()[1]);
    let tk = k.shape()[0];
    if q_valid.len() != tq || k_valid.len() != tk {
        return Err(TensorError::InvalidArgument("mask length mismatch"));
    }
    let scores = matmul(q, &k.transpose2d()?)?;
    let scale = 1.0 / (dh as f32).sqrt();
    let mut probs = vec![0.0f32; tq * tk];
    for i in 0..tq {
        if !q_valid[i] {
            continue; // padded query row: bypassed, stays zero
        }
        let mut row: Vec<f32> = scores.data()[i * tk..(i + 1) * tk].iter().map(|s| s * scale).collect();
        let vis: Vec<bool> = (0..tk)
            .map(|j| k_valid[j] && j <= i.checked_add(offset).unwrap_or(usize::MAX))
            .collect();
        if !masked_softmax_row(&mut row, &vis) {
            return Err(TensorError::AllMaskedRow { row: i });
        }
        probs[i * tk..(i + 1) * tk].copy_from_slice(&row);
    }
    let p = Tensor::from_vec(&[tq, tk], probs.clone())?;
    Ok((matmul(&p, v)?, probs))
}
