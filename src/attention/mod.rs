//! Reference (non-tiled) masked causal attention used as the G4 oracle (§3.3).
//! Causality uses absolute positions: key k is visible to query q iff k <= q + offset.

use crate::tensor::ops::masked_softmax_row;
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
    if q.shape().len() != 2 {
        return Err(TensorError::Rank { expected: 2, actual: q.shape().len() });
    }
    if k.shape().len() != 2 {
        return Err(TensorError::Rank { expected: 2, actual: k.shape().len() });
    }
    if v.shape().len() != 2 {
        return Err(TensorError::Rank { expected: 2, actual: v.shape().len() });
    }
    let (tq, dh) = (q.shape()[0], q.shape()[1]);
    let (tk, dh_k) = (k.shape()[0], k.shape()[1]);
    let (tk_v, dv) = (v.shape()[0], v.shape()[1]);
    if dh != dh_k {
        return Err(TensorError::ShapeMismatch {
            expected: vec![tq, dh],
            actual: vec![tk, dh_k],
        });
    }
    if tk != tk_v {
        return Err(TensorError::ShapeMismatch {
            expected: vec![tk, dv],
            actual: vec![tk_v, dv],
        });
    }
    if q_valid.len() != tq || k_valid.len() != tk {
        return Err(TensorError::InvalidArgument("mask length mismatch"));
    }
    if dh == 0 {
        return Err(TensorError::InvalidArgument("head dim cannot be 0"));
    }

    // Check finite inputs on valid positions
    for i in 0..tq {
        if q_valid[i] {
            let row = &q.data()[i * dh..(i + 1) * dh];
            if row.iter().any(|x| !x.is_finite()) {
                return Err(TensorError::InvalidArgument("non-finite query values"));
            }
        }
    }
    for j in 0..tk {
        if k_valid[j] {
            let row_k = &k.data()[j * dh..(j + 1) * dh];
            if row_k.iter().any(|x| !x.is_finite()) {
                return Err(TensorError::InvalidArgument("non-finite key values"));
            }
            let row_v = &v.data()[j * dv..(j + 1) * dv];
            if row_v.iter().any(|x| !x.is_finite()) {
                return Err(TensorError::InvalidArgument("non-finite value values"));
            }
        }
    }

    let scale = 1.0 / (dh as f32).sqrt();
    let mut probs = vec![0.0f32; tq * tk];
    let mut out = Tensor::zeros(&[tq, dv])?;

    for i in 0..tq {
        if !q_valid[i] {
            // Padded query: stays exact zero in output and probs without multiplying V (even if V has NaN)
            continue;
        }
        let q_row = &q.data()[i * dh..(i + 1) * dh];
        let mut scores = vec![0.0f32; tk];
        for j in 0..tk {
            let k_row = &k.data()[j * dh..(j + 1) * dh];
            let mut dot = 0.0f32;
            for d in 0..dh {
                dot += q_row[d] * k_row[d];
            }
            scores[j] = dot * scale;
        }

        let vis: Vec<bool> = (0..tk)
            .map(|j| k_valid[j] && j <= i.checked_add(offset).unwrap_or(usize::MAX))
            .collect();
        if !masked_softmax_row(&mut scores, &vis) {
            return Err(TensorError::AllMaskedRow { row: i });
        }
        probs[i * tk..(i + 1) * tk].copy_from_slice(&scores);

        // Compute output for valid query row i: sum_j (probs[i, j] * v[j, :])
        let out_row = &mut out.data_mut()[i * dv..(i + 1) * dv];
        for j in 0..tk {
            let p_val = scores[j];
            if p_val > 0.0 {
                let v_row = &v.data()[j * dv..(j + 1) * dv];
                for d in 0..dv {
                    out_row[d] += p_val * v_row[d];
                }
            }
        }
    }

    Ok((out, probs))
}
