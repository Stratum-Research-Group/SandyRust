//! Contiguous Key-Value Cache for autoregressive single-token decode (§8.3, Gate G7).

use crate::config::Config;
use crate::model::{Model, ModelError};
use crate::tensor::ops::{add, matmul, matmul_trans_b, mul, rms_norm, rope, silu};
use crate::tensor::{Tensor, TensorError};

#[derive(Debug, Clone, PartialEq)]
pub enum CacheError {
    CapacityExceeded { current: usize, requested: usize, max: usize },
    BindingMismatch(&'static str),
    InterventionMismatch { expected: [u8; 32], actual: [u8; 32] },
    SessionMismatch { expected: u64, actual: u64 },
    LayerOutOfBounds { layer: usize, total: usize },
    PositionMismatch { expected: usize, actual: usize },
    Tensor(TensorError),
}

impl From<TensorError> for CacheError {
    fn from(e: TensorError) -> Self {
        CacheError::Tensor(e)
    }
}

impl From<CacheError> for ModelError {
    fn from(e: CacheError) -> Self {
        match e {
            CacheError::Tensor(t) => ModelError::Tensor(t),
            other => ModelError::Cache(format!("{other:?}")),
        }
    }
}

#[derive(Debug, Clone)]
pub struct KVCache {
    pub model_fingerprint: [u8; 32],
    pub config_fingerprint: [u8; 32],
    pub intervention_policy_hash: [u8; 32],
    pub session_id: u64,
    pub max_seq_len: usize,
    pub seq_len: usize,
    /// Per layer: key cache [max_seq_len, kv_dim]
    pub k: Vec<Tensor>,
    /// Per layer: value cache [max_seq_len, kv_dim]
    pub v: Vec<Tensor>,
    pub bound_model_ptr: Option<usize>,
}

impl KVCache {
    pub fn new(cfg: &Config, model_fingerprint: [u8; 32], session_id: u64) -> Result<Self, CacheError> {
        cfg.validate().map_err(|_| TensorError::InvalidArgument("invalid config"))?;
        if cfg.n_q_heads == 0 || cfg.n_kv_heads == 0 {
            return Err(TensorError::InvalidArgument("n_heads == 0").into());
        }
        let kv_dim = cfg.n_kv_heads * cfg.head_dim();
        let max_seq = cfg.max_seq_len;
        let mut k = Vec::with_capacity(cfg.n_layers);
        let mut v = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            k.push(Tensor::zeros(&[max_seq, kv_dim])?);
            v.push(Tensor::zeros(&[max_seq, kv_dim])?);
        }
        Ok(KVCache {
            model_fingerprint,
            config_fingerprint: cfg.fingerprint(),
            intervention_policy_hash: [0u8; 32],
            session_id,
            max_seq_len: max_seq,
            seq_len: 0,
            k,
            v,
            bound_model_ptr: None,
        })
    }

    pub fn with_intervention_policy_hash(mut self, hash: [u8; 32]) -> Self {
        self.intervention_policy_hash = hash;
        self
    }

    /// Enforce session identity; cross-session reuse returns CacheError::SessionMismatch
    pub fn validate_session(&self, expected_session_id: u64) -> Result<(), CacheError> {
        if self.session_id != expected_session_id {
            return Err(CacheError::SessionMismatch {
                expected: expected_session_id,
                actual: self.session_id,
            });
        }
        Ok(())
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
        if pos >= self.max_seq_len {
            return Err(CacheError::CapacityExceeded { current: self.seq_len, requested: 1, max: self.max_seq_len });
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

    pub fn commit_step(&mut self) -> Result<(), CacheError> {
        if self.seq_len >= self.max_seq_len {
            return Err(CacheError::CapacityExceeded {
                current: self.seq_len,
                requested: 1,
                max: self.max_seq_len,
            });
        }
        self.seq_len += 1;
        Ok(())
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
/// Enforces model identity, transactional rollback on mid-step failure, and exact forward equivalence.
pub fn forward_step(model: &Model, token: u32, pos: usize, cache: &mut KVCache) -> Result<Tensor, ModelError> {
    forward_step_with_hooks(model, token, pos, cache, None, None)
}

/// Execute incremental decoding step enforcing session identity.
pub fn forward_step_session(
    model: &Model,
    token: u32,
    pos: usize,
    cache: &mut KVCache,
    session_id: u64,
) -> Result<Tensor, ModelError> {
    cache.validate_session(session_id)?;
    forward_step_with_hooks(model, token, pos, cache, None, None)
}

/// Execute a single incremental decoding step with optional hook context and intervention policy enforcement.
pub fn forward_step_with_hooks(
    model: &Model,
    token: u32,
    pos: usize,
    cache: &mut KVCache,
    ctx: Option<&crate::hooks::HookContext>,
    mut sink: Option<&mut crate::hooks::ActivationSink>,
) -> Result<Tensor, ModelError> {
    let c = &model.cfg;
    let model_ptr = (model as *const Model) as usize;
    if cache.bound_model_ptr != Some(model_ptr) {
        if cache.model_fingerprint != model.fingerprint() {
            return Err(CacheError::BindingMismatch("model fingerprint mismatch").into());
        }
        if cache.config_fingerprint != c.fingerprint() {
            return Err(CacheError::BindingMismatch("config fingerprint mismatch").into());
        }
        cache.bound_model_ptr = Some(model_ptr);
    }
    let active_policy_hash = ctx.map_or(crate::hooks::EMPTY_POLICY_HASH, |c| c.intervention_policy_hash());
    if cache.intervention_policy_hash != active_policy_hash {
        return Err(CacheError::InterventionMismatch {
            expected: active_policy_hash,
            actual: cache.intervention_policy_hash,
        }.into());
    }
    if pos >= c.max_seq_len || pos >= cache.max_seq_len {
        return Err(ModelError::SequenceTooLong { len: pos + 1, max: c.max_seq_len });
    }
    if pos != cache.seq_len {
        return Err(TensorError::InvalidArgument("pos != cache.seq_len").into());
    }
    cache.check_capacity(1).map_err(|_| ModelError::SequenceTooLong { len: pos + 1, max: c.max_seq_len })?;
    if token as usize >= c.vocab_size {
        return Err(ModelError::TokenOutOfRange { token, vocab: c.vocab_size });
    }

    if let Some(hook_ctx) = ctx {
        hook_ctx.validate(model)?;
    }

    let mut staged_k = Vec::with_capacity(c.n_layers);
    let mut staged_v = Vec::with_capacity(c.n_layers);

    let mut x = model.embed.gather_rows(&[token as usize])?;

    // Site 1: EmbedOut
    if let Some(hook_ctx) = ctx {
        let hook = crate::hooks::HookId::global(crate::hooks::HookSite::EmbedOut);
        if let Some(ref mut s) = sink {
            crate::hooks::apply_records(&hook, &x, crate::hooks::RecordTiming::PreEdit, &hook_ctx.records, s)?;
        }
        if !hook_ctx.edits.is_empty() {
            crate::hooks::apply_edits(&hook, &mut x, &hook_ctx.edits, pos, 1)?;
        }
        if let Some(ref mut s) = sink {
            crate::hooks::apply_records(&hook, &x, crate::hooks::RecordTiming::PostEdit, &hook_ctx.records, s)?;
        }
    }

    let (dh, group) = (c.head_dim(), c.n_q_heads / c.n_kv_heads);
    let kv_dim = c.n_kv_heads * dh;
    let k_len = pos + 1;
    let q_valid = [true];
    let k_valid = vec![true; k_len];
    let rope_freqs = if c.ext_x4_rope {
        Some(crate::attention::rope_ext::compute_frequencies(dh, c.rope_theta as f64, c.rope_scaling.as_ref()))
    } else {
        None
    };
    let attn_scale = (1.0 / (dh as f32).sqrt()) * if c.ext_x4_rope {
        crate::attention::rope_ext::yarn_temperature_scale(c.rope_scaling.as_ref())
    } else {
        1.0
    };

    let mm = |a: &Tensor, b: &Tensor| -> Result<Tensor, TensorError> {
        if c.ext_x5_gemm {
            crate::gemm::gemm(a, b)
        } else {
            matmul(a, b)
        }
    };
    let mm_tb = |a: &Tensor, b: &Tensor| -> Result<Tensor, TensorError> {
        if c.ext_x5_gemm {
            crate::gemm::gemm_trans_b(a, b)
        } else {
            matmul_trans_b(a, b)
        }
    };

    for (l_idx, l) in model.layers.iter().enumerate() {
        let layer_u16 = l_idx as u16;

        // Site 2: ResidPre
        if let Some(hook_ctx) = ctx {
            let hook = crate::hooks::HookId::layer(layer_u16, crate::hooks::HookSite::ResidPre);
            if let Some(ref mut s) = sink {
                crate::hooks::apply_records(&hook, &x, crate::hooks::RecordTiming::PreEdit, &hook_ctx.records, s)?;
            }
            if !hook_ctx.edits.is_empty() {
                crate::hooks::apply_edits(&hook, &mut x, &hook_ctx.edits, pos, 1)?;
            }
            if let Some(ref mut s) = sink {
                crate::hooks::apply_records(&hook, &x, crate::hooks::RecordTiming::PostEdit, &hook_ctx.records, s)?;
            }
        }

        let mut r = rms_norm(&x, &l.rms1, c.rms_eps)?;

        // Site 3: AttnNormOut
        if let Some(hook_ctx) = ctx {
            let hook = crate::hooks::HookId::layer(layer_u16, crate::hooks::HookSite::AttnNormOut);
            if let Some(ref mut s) = sink {
                crate::hooks::apply_records(&hook, &r, crate::hooks::RecordTiming::PreEdit, &hook_ctx.records, s)?;
            }
            if !hook_ctx.edits.is_empty() {
                crate::hooks::apply_edits(&hook, &mut r, &hook_ctx.edits, pos, 1)?;
            }
            if let Some(ref mut s) = sink {
                crate::hooks::apply_records(&hook, &r, crate::hooks::RecordTiming::PostEdit, &hook_ctx.records, s)?;
            }
        }

        let q = mm(&r, &l.wq)?;
        let k = mm(&r, &l.wk)?;
        let v = mm(&r, &l.wv)?;

        // Apply RoPE at current absolute position pos
        let mut k_rope = Tensor::zeros(&[1, kv_dim])?;
        let mut v_staged = Tensor::zeros(&[1, kv_dim])?;
        for kvh in 0..c.n_kv_heads {
            let k_col = slice_cols(&k, kvh * dh, dh)?;
            let k_normed = if let Some(ref kn) = l.k_norm {
                let gamma_kvh = crate::attention::qknorm::slice_row(kn, kvh)?;
                rms_norm(&k_col, &gamma_kvh, c.rms_eps)?
            } else {
                k_col
            };

            // Site 4b: KPreRope
            if let Some(hook_ctx) = ctx {
                let hook = crate::hooks::HookId::head(layer_u16, crate::hooks::HookSite::KPreRope, kvh as u16);
                if let Some(ref mut s) = sink {
                    crate::hooks::apply_records(&hook, &k_normed, crate::hooks::RecordTiming::PostEdit, &hook_ctx.records, s)?;
                }
            }

            let mut col_r = match &rope_freqs {
                Some(freqs) => crate::attention::rope_ext::rope_with_freqs(&k_normed, freqs, pos)?,
                None => rope(&k_normed, c.rope_theta, pos)?,
            };

            // Site 4c: K
            if let Some(hook_ctx) = ctx {
                let hook = crate::hooks::HookId::head(layer_u16, crate::hooks::HookSite::K, kvh as u16);
                if let Some(ref mut s) = sink {
                    crate::hooks::apply_records(&hook, &col_r, crate::hooks::RecordTiming::PreEdit, &hook_ctx.records, s)?;
                }
                if !hook_ctx.edits.is_empty() {
                    crate::hooks::apply_edits(&hook, &mut col_r, &hook_ctx.edits, pos, 1)?;
                }
                if let Some(ref mut s) = sink {
                    crate::hooks::apply_records(&hook, &col_r, crate::hooks::RecordTiming::PostEdit, &hook_ctx.records, s)?;
                }
            }
            k_rope.data_mut()[kvh * dh..(kvh + 1) * dh].copy_from_slice(col_r.data());

            let mut v_col = slice_cols(&v, kvh * dh, dh)?;

            // Site 4d: V
            if let Some(hook_ctx) = ctx {
                let hook = crate::hooks::HookId::head(layer_u16, crate::hooks::HookSite::V, kvh as u16);
                if let Some(ref mut s) = sink {
                    crate::hooks::apply_records(&hook, &v_col, crate::hooks::RecordTiming::PreEdit, &hook_ctx.records, s)?;
                }
                if !hook_ctx.edits.is_empty() {
                    crate::hooks::apply_edits(&hook, &mut v_col, &hook_ctx.edits, pos, 1)?;
                }
                if let Some(ref mut s) = sink {
                    crate::hooks::apply_records(&hook, &v_col, crate::hooks::RecordTiming::PostEdit, &hook_ctx.records, s)?;
                }
            }
            v_staged.data_mut()[kvh * dh..(kvh + 1) * dh].copy_from_slice(v_col.data());
        }

        staged_k.push(k_rope);
        staged_v.push(v_staged);

        // History 0..pos from cache plus current step pos
        let cached_k_prev = if pos > 0 { slice_rows(&cache.k[l_idx], 0, pos)? } else { Tensor::zeros(&[0, kv_dim])? };
        let cached_v_prev = if pos > 0 { slice_rows(&cache.v[l_idx], 0, pos)? } else { Tensor::zeros(&[0, kv_dim])? };

        let mut current_k = Vec::with_capacity(k_len * kv_dim);
        current_k.extend_from_slice(cached_k_prev.data());
        current_k.extend_from_slice(staged_k[l_idx].data());
        let current_k_tensor = Tensor::from_vec(&[k_len, kv_dim], current_k)?;

        let mut current_v = Vec::with_capacity(k_len * kv_dim);
        current_v.extend_from_slice(cached_v_prev.data());
        current_v.extend_from_slice(staged_v[l_idx].data());
        let current_v_tensor = Tensor::from_vec(&[k_len, kv_dim], current_v)?;

        let mut merged = Tensor::zeros(&[1, c.n_q_heads * dh])?;
        for h in 0..c.n_q_heads {
            let kvh = h / group;
            let q_col = slice_cols(&q, h * dh, dh)?;
            let q_normed = if let Some(ref qn) = l.q_norm {
                let gamma_h = crate::attention::qknorm::slice_row(qn, h)?;
                rms_norm(&q_col, &gamma_h, c.rms_eps)?
            } else {
                q_col
            };
            let mut qh = match &rope_freqs {
                Some(freqs) => crate::attention::rope_ext::rope_with_freqs(&q_normed, freqs, pos)?,
                None => rope(&q_normed, c.rope_theta, pos)?,
            };

            // Site 4a: Q
            if let Some(hook_ctx) = ctx {
                let hook = crate::hooks::HookId::head(layer_u16, crate::hooks::HookSite::Q, h as u16);
                if let Some(ref mut s) = sink {
                    crate::hooks::apply_records(&hook, &qh, crate::hooks::RecordTiming::PreEdit, &hook_ctx.records, s)?;
                }
                if !hook_ctx.edits.is_empty() {
                    crate::hooks::apply_edits(&hook, &mut qh, &hook_ctx.edits, pos, 1)?;
                }
                if let Some(ref mut s) = sink {
                    crate::hooks::apply_records(&hook, &qh, crate::hooks::RecordTiming::PostEdit, &hook_ctx.records, s)?;
                }
            }

            let kh = slice_cols(&current_k_tensor, kvh * dh, dh)?;
            let vh = slice_cols(&current_v_tensor, kvh * dh, dh)?;
            let (mut o, probs) = crate::attention::attention_ref_scaled(&qh, &kh, &vh, &q_valid, &k_valid, pos, attn_scale)?;

            // Site 6: AttnPattern
            if let Some(hook_ctx) = ctx {
                let hook = crate::hooks::HookId::head(layer_u16, crate::hooks::HookSite::AttnPattern, h as u16);
                if let Some(ref mut s) = sink {
                    let probs_tensor = Tensor::from_vec(&[1, k_len], probs)?;
                    crate::hooks::apply_records(&hook, &probs_tensor, crate::hooks::RecordTiming::PostEdit, &hook_ctx.records, s)?;
                }
            }

            // Site 7: HeadOut
            if let Some(hook_ctx) = ctx {
                let hook = crate::hooks::HookId::head(layer_u16, crate::hooks::HookSite::HeadOut, h as u16);
                if let Some(ref mut s) = sink {
                    crate::hooks::apply_records(&hook, &o, crate::hooks::RecordTiming::PreEdit, &hook_ctx.records, s)?;
                }
                if !hook_ctx.edits.is_empty() {
                    crate::hooks::apply_edits(&hook, &mut o, &hook_ctx.edits, pos, 1)?;
                }
                if let Some(ref mut s) = sink {
                    crate::hooks::apply_records(&hook, &o, crate::hooks::RecordTiming::PostEdit, &hook_ctx.records, s)?;
                }
            }

            merged.data_mut()[h * dh..(h + 1) * dh].copy_from_slice(o.data());
        }

        let mut attn_out = mm(&merged, &l.wo)?;

        // Site 8: AttnOut
        if let Some(hook_ctx) = ctx {
            let hook = crate::hooks::HookId::layer(layer_u16, crate::hooks::HookSite::AttnOut);
            if let Some(ref mut s) = sink {
                crate::hooks::apply_records(&hook, &attn_out, crate::hooks::RecordTiming::PreEdit, &hook_ctx.records, s)?;
            }
            if !hook_ctx.edits.is_empty() {
                crate::hooks::apply_edits(&hook, &mut attn_out, &hook_ctx.edits, pos, 1)?;
            }
            if let Some(ref mut s) = sink {
                crate::hooks::apply_records(&hook, &attn_out, crate::hooks::RecordTiming::PostEdit, &hook_ctx.records, s)?;
            }
        }

        x = add(&x, &attn_out)?;

        // Site 9: ResidMid
        if let Some(hook_ctx) = ctx {
            let hook = crate::hooks::HookId::layer(layer_u16, crate::hooks::HookSite::ResidMid);
            if let Some(ref mut s) = sink {
                crate::hooks::apply_records(&hook, &x, crate::hooks::RecordTiming::PreEdit, &hook_ctx.records, s)?;
            }
            if !hook_ctx.edits.is_empty() {
                crate::hooks::apply_edits(&hook, &mut x, &hook_ctx.edits, pos, 1)?;
            }
            if let Some(ref mut s) = sink {
                crate::hooks::apply_records(&hook, &x, crate::hooks::RecordTiming::PostEdit, &hook_ctx.records, s)?;
            }
        }

        let mut r2 = rms_norm(&x, &l.rms2, c.rms_eps)?;

        // Site 10a: MlpNormOut
        if let Some(hook_ctx) = ctx {
            let hook = crate::hooks::HookId::layer(layer_u16, crate::hooks::HookSite::MlpNormOut);
            if let Some(ref mut s) = sink {
                crate::hooks::apply_records(&hook, &r2, crate::hooks::RecordTiming::PreEdit, &hook_ctx.records, s)?;
            }
            if !hook_ctx.edits.is_empty() {
                crate::hooks::apply_edits(&hook, &mut r2, &hook_ctx.edits, pos, 1)?;
            }
            if let Some(ref mut s) = sink {
                crate::hooks::apply_records(&hook, &r2, crate::hooks::RecordTiming::PostEdit, &hook_ctx.records, s)?;
            }
        }

        let mut hidden = mul(&silu(&mm(&r2, &l.w_gate)?), &mm(&r2, &l.w_up)?)?;

        // Site 10b: MlpHidden
        if let Some(hook_ctx) = ctx {
            let hook = crate::hooks::HookId::layer(layer_u16, crate::hooks::HookSite::MlpHidden);
            if let Some(ref mut s) = sink {
                crate::hooks::apply_records(&hook, &hidden, crate::hooks::RecordTiming::PreEdit, &hook_ctx.records, s)?;
            }
            if !hook_ctx.edits.is_empty() {
                crate::hooks::apply_edits(&hook, &mut hidden, &hook_ctx.edits, pos, 1)?;
            }
            if let Some(ref mut s) = sink {
                crate::hooks::apply_records(&hook, &hidden, crate::hooks::RecordTiming::PostEdit, &hook_ctx.records, s)?;
            }
        }

        let mut mlp_out = mm(&hidden, &l.w_down)?;

        // Site 10c: MlpOut
        if let Some(hook_ctx) = ctx {
            let hook = crate::hooks::HookId::layer(layer_u16, crate::hooks::HookSite::MlpOut);
            if let Some(ref mut s) = sink {
                crate::hooks::apply_records(&hook, &mlp_out, crate::hooks::RecordTiming::PreEdit, &hook_ctx.records, s)?;
            }
            if !hook_ctx.edits.is_empty() {
                crate::hooks::apply_edits(&hook, &mut mlp_out, &hook_ctx.edits, pos, 1)?;
            }
            if let Some(ref mut s) = sink {
                crate::hooks::apply_records(&hook, &mlp_out, crate::hooks::RecordTiming::PostEdit, &hook_ctx.records, s)?;
            }
        }

        x = add(&x, &mlp_out)?;

        // Site 11: ResidPost
        if let Some(hook_ctx) = ctx {
            let hook = crate::hooks::HookId::layer(layer_u16, crate::hooks::HookSite::ResidPost);
            if let Some(ref mut s) = sink {
                crate::hooks::apply_records(&hook, &x, crate::hooks::RecordTiming::PreEdit, &hook_ctx.records, s)?;
            }
            if !hook_ctx.edits.is_empty() {
                crate::hooks::apply_edits(&hook, &mut x, &hook_ctx.edits, pos, 1)?;
            }
            if let Some(ref mut s) = sink {
                crate::hooks::apply_records(&hook, &x, crate::hooks::RecordTiming::PostEdit, &hook_ctx.records, s)?;
            }
        }
    }

    let mut final_x = rms_norm(&x, &model.final_norm, c.rms_eps)?;

    // Site 12a: FinalNormOut
    if let Some(hook_ctx) = ctx {
        let hook = crate::hooks::HookId::global(crate::hooks::HookSite::FinalNormOut);
        if let Some(ref mut s) = sink {
            crate::hooks::apply_records(&hook, &final_x, crate::hooks::RecordTiming::PreEdit, &hook_ctx.records, s)?;
        }
        if !hook_ctx.edits.is_empty() {
            crate::hooks::apply_edits(&hook, &mut final_x, &hook_ctx.edits, pos, 1)?;
        }
        if let Some(ref mut s) = sink {
            crate::hooks::apply_records(&hook, &final_x, crate::hooks::RecordTiming::PostEdit, &hook_ctx.records, s)?;
        }
    }

    let head_mat = match &model.lm_head {
        Some(h) => h,
        None => &model.embed,
    };
    let mut logits = mm_tb(&final_x, head_mat)?;

    // Site 12b: Logits
    if let Some(hook_ctx) = ctx {
        let hook = crate::hooks::HookId::global(crate::hooks::HookSite::Logits);
        if let Some(ref mut s) = sink {
            crate::hooks::apply_records(&hook, &logits, crate::hooks::RecordTiming::PreEdit, &hook_ctx.records, s)?;
        }
        if !hook_ctx.edits.is_empty() {
            crate::hooks::apply_edits(&hook, &mut logits, &hook_ctx.edits, pos, 1)?;
        }
        if let Some(ref mut s) = sink {
            crate::hooks::apply_records(&hook, &logits, crate::hooks::RecordTiming::PostEdit, &hook_ctx.records, s)?;
        }
    }

    // Commit only after complete step succeeds
    for l_idx in 0..c.n_layers {
        cache.append_step(l_idx, staged_k[l_idx].data(), staged_v[l_idx].data(), pos)
            .map_err(|_| TensorError::InvalidArgument("cache append"))?;
    }
    cache.commit_step()?;

    Ok(logits)
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

        let full_logits = model.forward(&tokens, None).unwrap();
        let v = cfg.vocab_size;

        let mut cache = KVCache::new(&cfg, model.fingerprint(), 1).unwrap();
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

    #[test]
    fn test_kv_cache_rejects_stale_model_or_config() {
        let cfg = Config::reference_v2();
        let model = Model::init(&cfg).unwrap();
        let mut cache = KVCache::new(&cfg, model.fingerprint(), 1).unwrap();

        // Mutate model weight to change fingerprint
        let mut model_changed = model.clone();
        model_changed.embed.data_mut()[0] += 1.0;
        let res = forward_step(&model_changed, 256, 0, &mut cache);
        assert!(matches!(res, Err(ModelError::Cache(_))));
        assert_eq!(cache.seq_len, 0); // Cache untouched
    }

    #[test]
    fn test_append_step_capacity_bounds() {
        let cfg = Config::reference_v2();
        let model = Model::init(&cfg).unwrap();
        let mut cache = KVCache::new(&cfg, model.fingerprint(), 1).unwrap();
        // Append beyond max_seq_len
        let kv_dim = cfg.n_kv_heads * cfg.head_dim();
        let dummy = vec![0.0f32; kv_dim];
        let res = cache.append_step(0, &dummy, &dummy, cfg.max_seq_len);
        assert!(matches!(res, Err(CacheError::CapacityExceeded { .. })));
    }

    #[test]
    fn test_n6_kv_cache_session_validation() {
        let cfg = Config::reference_v2();
        let model = Model::init(&cfg).unwrap();
        let mut cache = KVCache::new(&cfg, model.fingerprint(), 42).unwrap();

        assert!(cache.validate_session(42).is_ok());
        assert!(matches!(
            cache.validate_session(999),
            Err(CacheError::SessionMismatch { expected: 999, actual: 42 })
        ));

        let res = forward_step_session(&model, 256, 0, &mut cache, 999);
        assert!(matches!(res, Err(ModelError::Cache(_))));
    }
}
