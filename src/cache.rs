//! Contiguous Key-Value Cache for autoregressive single-token decode (§8.3, Gate G7).

use crate::attention::attention_ref;
use crate::config::Config;
use crate::model::{Model, ModelError};
use crate::tensor::ops::{add, matmul, mul, rms_norm, rope, silu};
use crate::tensor::{Tensor, TensorError};

#[derive(Debug, Clone, PartialEq)]
pub enum CacheError {
    CapacityExceeded { current: usize, requested: usize, max: usize },
    BindingMismatch(&'static str),
    LayerOutOfBounds { layer: usize, total: usize },
    PositionMismatch { expected: usize, actual: usize },
    Tensor(TensorError),
}

impl From<TensorError> for CacheError {
    fn from(e: TensorError) -> Self {
        CacheError::Tensor(e)
    }
}

#[derive(Debug, Clone)]
pub struct KVCache {
    pub model_id: u64,
    pub session_id: u64,
    pub max_seq_len: usize,
    pub seq_len: usize,
    /// Per layer: key cache [max_seq_len, kv_dim]
    pub k: Vec<Tensor>,
    /// Per layer: value cache [max_seq_len, kv_dim]
    pub v: Vec<Tensor>,
}

impl KVCache {
    pub fn new(cfg: &Config, model_id: u64, session_id: u64) -> Result<Self, CacheError> {
        let kv_dim = cfg.n_kv_heads * cfg.head_dim();
        let max_seq = cfg.max_seq_len;
        let mut k = Vec::with_capacity(cfg.n_layers);
        let mut v = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            k.push(Tensor::zeros(&[max_seq, kv_dim])?);
            v.push(Tensor::zeros(&[max_seq, kv_dim])?);
        }
        Ok(KVCache {
            model_id,
            session_id,
            max_seq_len: max_seq,
            seq_len: 0,
            k,
            v,
        })
    }

    pub fn check_capacity(&self, additional: usize) -> Result<(), CacheError> {
        let total = self.seq_len.checked_add(additional).ok_or(CacheError::CapacityExceeded {
            current: self.seq_len,
            requested: additional,
            max: self.max_seq_len,
        })?;
        if total > self.max_seq_len {
            return Err(CacheError::CapacityExceeded {
                current: self.seq_len,
                requested: additional,
                max: self.max_seq_len,
            });
        }
        Ok(())
    }

    pub fn rollback(&mut self, target_len: usize) {
        if target_len <= self.seq_len {
            self.seq_len = target_len;
        }
    }

    pub fn append_step(&mut self, layer: usize, k_row: &[f32], v_row: &[f32], pos: usize) -> Result<(), CacheError> {
        if layer >= self.k.len() {
            return Err(CacheError::LayerOutOfBounds { layer, total: self.k.len() });
        }
        if pos != self.seq_len {
            return Err(CacheError::PositionMismatch { expected: self.seq_len, actual: pos });
        }
        let kv_dim = self.k[layer].shape()[1];
        if k_row.len() != kv_dim || v_row.len() != kv_dim {
            return Err(TensorError::InvalidArgument("k/v dim mismatch").into());
        }
        let dst = pos * kv_dim;
        self.k[layer].data_mut()[dst..dst + kv_dim].copy_from_slice(k_row);
        self.v[layer].data_mut()[dst..dst + kv_dim].copy_from_slice(v_row);
        Ok(())
    }

    pub fn commit_step(&mut self) {
        self.seq_len += 1;
    }
}

fn slice_cols(t: &Tensor, start: usize, len: usize) -> Result<Tensor, TensorError> {
    let (rows, cols) = (t.shape()[0], t.shape()[1]);
    let mut out = Vec::with_capacity(rows * len);
    for r in 0..rows {
        out.extend_from_slice(&t.data()[r * cols + start..r * cols + start + len]);
    }
    Tensor::from_vec(&[rows, len], out)
}

/// Execute a single incremental decoding step using the contiguous KV cache.
pub fn forward_step(model: &Model, token: u32, pos: usize, cache: &mut KVCache) -> Result<Tensor, ModelError> {
    let c = &model.cfg;
    if pos >= c.max_seq_len {
        return Err(ModelError::SequenceTooLong { len: pos + 1, max: c.max_seq_len });
    }
    if pos != cache.seq_len {
        return Err(TensorError::InvalidArgument("pos != cache.seq_len").into());
    }
    cache.check_capacity(1).map_err(|_| ModelError::SequenceTooLong { len: pos + 1, max: c.max_seq_len })?;
    if token as usize >= c.vocab_size {
        return Err(ModelError::TokenOutOfRange { token, vocab: c.vocab_size });
    }

    let mut x = model.embed.gather_rows(&[token as usize])?;
    let (dh, group) = (c.head_dim(), c.n_q_heads / c.n_kv_heads);
    let kv_dim = c.n_kv_heads * dh;
    let k_len = pos + 1;
    let q_valid = [true];
    let k_valid = vec![true; k_len];

    for (l_idx, l) in model.layers.iter().enumerate() {
        let r = rms_norm(&x, &l.rms1, c.rms_eps)?;
        let q = matmul(&r, &l.wq)?;
        let k = matmul(&r, &l.wk)?;
        let v = matmul(&r, &l.wv)?;

        // Apply RoPE at current absolute position pos
        let mut k_rope = Tensor::zeros(&[1, kv_dim])?;
        for kvh in 0..c.n_kv_heads {
            let col = slice_cols(&k, kvh * dh, dh)?;
            let col_r = rope(&col, c.rope_theta, pos)?;
            k_rope.data_mut()[kvh * dh..(kvh + 1) * dh].copy_from_slice(col_r.data());
        }

        // Store k_rope and v into cache for this layer
        cache.append_step(l_idx, k_rope.data(), v.data(), pos).map_err(|_e| TensorError::InvalidArgument("cache append"))?;

        // Extract cached slices for 0..=pos
        let cached_k_slice = slice_rows(&cache.k[l_idx], 0, k_len)?;
        let cached_v_slice = slice_rows(&cache.v[l_idx], 0, k_len)?;

        let mut merged = Tensor::zeros(&[1, c.n_q_heads * dh])?;
        for h in 0..c.n_q_heads {
            let kvh = h / group;
            let qh = rope(&slice_cols(&q, h * dh, dh)?, c.rope_theta, pos)?;
            let kh = slice_cols(&cached_k_slice, kvh * dh, dh)?;
            let vh = slice_cols(&cached_v_slice, kvh * dh, dh)?;
            let (o, _) = attention_ref(&qh, &kh, &vh, &q_valid, &k_valid, pos)?;
            merged.data_mut()[h * dh..(h + 1) * dh].copy_from_slice(o.data());
        }

        x = add(&x, &matmul(&merged, &l.wo)?)?;
        let r2 = rms_norm(&x, &l.rms2, c.rms_eps)?;
        let hidden = mul(&silu(&matmul(&r2, &l.w_gate)?), &matmul(&r2, &l.w_up)?)?;
        x = add(&x, &matmul(&hidden, &l.w_down)?)?;
    }

    cache.commit_step();
    let x = rms_norm(&x, &model.final_norm, c.rms_eps)?;
    Ok(matmul(&x, &model.embed.transpose2d()?)?)
}

fn slice_rows(t: &Tensor, start: usize, len: usize) -> Result<Tensor, TensorError> {
    let cols = t.shape()[1];
    let start_idx = start * cols;
    let end_idx = (start + len) * cols;
    let out = t.data()[start_idx..end_idx].to_vec();
    Tensor::from_vec(&[len, cols], out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_matches_full_forward_parity_g7() {
        let cfg = Config::reference_v2();
        let model = Model::init(&cfg).unwrap();
        let tokens = [256u32, 72, 101, 108, 108, 111]; // "Hello"

        // Full-sequence non-cached forward pass
        let full_logits = model.forward(&tokens, None).unwrap();
        let v = cfg.vocab_size;

        // Incremental cached single-step decode
        let mut cache = KVCache::new(&cfg, 0x1234, 1).unwrap();
        for (pos, &tok) in tokens.iter().enumerate() {
            let step_logits = forward_step(&model, tok, pos, &mut cache).unwrap();
            assert_eq!(step_logits.shape(), &[1, v]);
            let full_row = &full_logits.data()[pos * v..(pos + 1) * v];
            let step_row = step_logits.data();
            for (i, (&a, &b)) in full_row.iter().zip(step_row).enumerate() {
                let diff = (a - b).abs();
                assert!(diff <= 1e-4 + 1e-4 * a.abs(), "pos={pos}, token={i}: diff={diff}, full={a}, cached={b}");
            }
        }
    }
}
