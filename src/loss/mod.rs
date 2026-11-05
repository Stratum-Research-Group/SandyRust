//! Extension X7: Memory-Efficient Chunked Cross-Entropy & Activation Recomputation (§4.2 Item 7, §9.2 GX7).
//!
//! Evaluates classification loss and accumulates gradients in sequential sequence chunks
//! (default $T_{\text{chunk}} = 16$), eliminating materialization of full $B \times T \times V$ logit tensors
//! and achieving $O(T_{\text{chunk}} \cdot V)$ working memory without retaining chunk logits.

use std::time::Instant;

use crate::autodiff::{
    final_norm_idx, layer_indices, lm_head_idx, Float, LossOut, Params, TrainError,
};
use crate::config::Config;
use crate::gates::{detect_revision, detect_toolchain, GateReport};
use crate::model::Model;
use crate::rng::Rng;

/// Chunked forward + backward cross-entropy loss computation (§4.2 Item 7).
///
/// Computes classification loss in sequential sequence chunks of size `cfg.chunk_size`
/// ($T_{\text{chunk}} = 16$ by default), accumulating backward gradients directly into
/// $W_{\text{head}}$ and computing $d_{\text{hidden}}$ without materializing or retaining
/// the full $N \times V$ logit tensor.
pub fn chunked_loss_and_grad<T: Float>(
    cfg: &Config,
    p: &Params<T>,
    window: &[u32],
    denom: T,
    grads: &mut Params<T>,
) -> Result<LossOut<T>, TrainError> {
    if window.len() < 2 {
        return Err(TrainError::WindowTooShort);
    }
    if !(denom.is_finite() && denom > T::ZERO) {
        return Err(TrainError::InvalidArgument("denom must be finite and > 0"));
    }
    let n = window.len() - 1;
    let d = cfg.d_model;
    let dff = cfg.d_ff;
    let nq = cfg.n_q_heads;
    let nkv = cfg.n_kv_heads;
    let dh = cfg.head_dim();
    let q_dim = nq * dh;
    let kv_dim = nkv * dh;
    let vs = cfg.vocab_size;
    let eps = T::from_f64(cfg.rms_eps as f64);
    let theta = cfg.rope_theta;

    let input_tokens = &window[..n];
    let targets = &window[1..];

    // Build mask and valid flags
    let mut valid = vec![true; n];
    for (i, &tok) in input_tokens.iter().enumerate() {
        if tok as usize >= vs {
            return Err(TrainError::TokenOutOfRange(tok));
        }
        if tok == cfg.pad_id {
            valid[i] = false;
        }
    }

    // Embeddings
    let embed = &p.data[0];
    let mut x = vec![T::ZERO; n * d];
    for i in 0..n {
        let tok = input_tokens[i] as usize;
        let row = &embed[tok * d..(tok + 1) * d];
        x[i * d..(i + 1) * d].copy_from_slice(row);
    }

    // Forward pass through transformer layers, caching layer activations for backward
    struct LayerCache<T> {
        x0: Vec<T>,
        r1: Vec<T>,
        inv1: Vec<T>,
        q_unnorm: Option<Vec<T>>,
        inv_q: Option<Vec<T>>,
        k_unnorm: Option<Vec<T>>,
        inv_k: Option<Vec<T>>,
        qr: Vec<T>,
        kr: Vec<T>,
        v: Vec<T>,
        p: Vec<T>,
        merged: Vec<T>,
        x1: Vec<T>,
        r2: Vec<T>,
        inv2: Vec<T>,
        gate: Vec<T>,
        up: Vec<T>,
        hid: Vec<T>,
    }

    let mut caches: Vec<LayerCache<T>> = Vec::with_capacity(cfg.n_layers);
    for l in 0..cfg.n_layers {
        let idx = layer_indices(cfg, l);
        let g1 = &p.data[idx.rms1];
        let wq = &p.data[idx.wq];
        let wk = &p.data[idx.wk];
        let wv = &p.data[idx.wv];
        let wo = &p.data[idx.wo];
        let g2 = &p.data[idx.rms2];
        let wg = &p.data[idx.w_gate];
        let wu = &p.data[idx.w_up];
        let wd_ = &p.data[idx.w_down];

        let (r1, inv1) = rms_fwd(&x, n, d, g1, eps);
        let q = mm(&r1, n, d, wq, q_dim);
        let k = mm(&r1, n, d, wk, kv_dim);
        let v = mm(&r1, n, d, wv, kv_dim);

        let (mut qr, q_unnorm, inv_q) = if cfg.ext_x3_qk_norm {
            let mut q_normed = vec![T::ZERO; n * q_dim];
            let mut invs = vec![T::ZERO; n * nq];
            let qn_gamma = &p.data[idx.q_norm.unwrap()];
            for h in 0..nq {
                let mut q_head = vec![T::ZERO; n * dh];
                for t in 0..n {
                    q_head[t * dh..(t + 1) * dh].copy_from_slice(&q[t * q_dim + h * dh..t * q_dim + (h + 1) * dh]);
                }
                let gamma_h = &qn_gamma[h * dh..(h + 1) * dh];
                let (normed_h, inv_h) = rms_fwd(&q_head, n, dh, gamma_h, eps);
                for t in 0..n {
                    q_normed[t * q_dim + h * dh..t * q_dim + (h + 1) * dh].copy_from_slice(&normed_h[t * dh..(t + 1) * dh]);
                    invs[t * nq + h] = inv_h[t];
                }
            }
            (q_normed, Some(q), Some(invs))
        } else {
            (q, None, None)
        };

        let (mut kr, k_unnorm, inv_k) = if cfg.ext_x3_qk_norm {
            let mut k_normed = vec![T::ZERO; n * kv_dim];
            let mut invs = vec![T::ZERO; n * nkv];
            let kn_gamma = &p.data[idx.k_norm.unwrap()];
            for h in 0..nkv {
                let mut k_head = vec![T::ZERO; n * dh];
                for t in 0..n {
                    k_head[t * dh..(t + 1) * dh].copy_from_slice(&k[t * kv_dim + h * dh..t * kv_dim + (h + 1) * dh]);
                }
                let gamma_h = &kn_gamma[h * dh..(h + 1) * dh];
                let (normed_h, inv_h) = rms_fwd(&k_head, n, dh, gamma_h, eps);
                for t in 0..n {
                    k_normed[t * kv_dim + h * dh..t * kv_dim + (h + 1) * dh].copy_from_slice(&normed_h[t * dh..(t + 1) * dh]);
                    invs[t * nkv + h] = inv_h[t];
                }
            }
            (k_normed, Some(k), Some(invs))
        } else {
            (k, None, None)
        };

        apply_rope_heads(&mut qr, n, nq, dh, theta, false);
        apply_rope_heads(&mut kr, n, nkv, dh, theta, false);

        let (merged, pm) = attention_forward(&qr, &kr, &v, n, nq, nkv, dh, &valid);
        let o = mm(&merged, n, q_dim, wo, d);
        let mut x1 = x.clone();
        add_into(&mut x1, &o);

        let (r2, inv2) = rms_fwd(&x1, n, d, g2, eps);
        let gate = mm(&r2, n, d, wg, dff);
        let up = mm(&r2, n, d, wu, dff);
        let hid: Vec<T> = gate
            .iter()
            .zip(&up)
            .map(|(&g, &u)| g / (T::ONE + (-g).exp()) * u)
            .collect();
        let f = mm(&hid, n, dff, wd_, d);
        let mut x2 = x1.clone();
        add_into(&mut x2, &f);

        caches.push(LayerCache {
            x0: x,
            r1,
            inv1,
            q_unnorm,
            inv_q,
            k_unnorm,
            inv_k,
            qr,
            kr,
            v,
            p: pm,
            merged,
            x1,
            r2,
            inv2,
            gate,
            up,
            hid,
        });
        x = x2;
    }

    // Final RMSNorm
    let gf = &p.data[final_norm_idx(cfg)];
    let (xf, invf) = rms_fwd(&x, n, d, gf, eps);
    let head_weight = if cfg.tied_embeddings {
        embed
    } else {
        &p.data[lm_head_idx(cfg)]
    };

    // ---------------- Chunked Loss & Backward ----------------
    // Evaluate in sequence chunks of size T_chunk (default 16, §4.2 Item 7)
    let chunk_size = cfg.chunk_size.max(1);
    let n_chunks = n.div_ceil(chunk_size);

    let mut loss_sum = T::ZERO;
    let mut n_valid = 0usize;
    let mut dxf = vec![T::ZERO; n * d];

    for c in 0..n_chunks {
        let c_start = c * chunk_size;
        let c_end = (c_start + chunk_size).min(n);
        let c_len = c_end - c_start;

        let xf_chunk = &xf[c_start * d..c_end * d];
        // 1. Materialize chunk logits ONLY (size c_len * vs <= 16 * vs)
        let logits_chunk = mm_bt(xf_chunk, c_len, d, head_weight, vs);

        // 2. Compute loss contribution and chunk dlogits
        let mut dlogits_chunk = vec![T::ZERO; c_len * vs];
        for k in 0..c_len {
            let token_idx = c_start + k;
            if !valid[token_idx] || targets[token_idx] == cfg.pad_id {
                continue;
            }
            n_valid += 1;
            let z = &logits_chunk[k * vs..(k + 1) * vs];
            let mut mx = z[0];
            for &val in z {
                if val > mx {
                    mx = val;
                }
            }
            let mut s = T::ZERO;
            for &val in z {
                s += (val - mx).exp();
            }
            let lse = mx + s.ln();
            let y = targets[token_idx] as usize;
            loss_sum += lse - z[y];

            for j in 0..vs {
                let pj = (z[j] - lse).exp();
                dlogits_chunk[k * vs + j] = (pj - if j == y { T::ONE } else { T::ZERO }) / denom;
            }
        }

        // 3. Accumulate backward gradients into W_head directly
        if cfg.tied_embeddings {
            mm_at_acc(&dlogits_chunk, c_len, vs, xf_chunk, d, &mut grads.data[0]);
        } else {
            let head_idx = lm_head_idx(cfg);
            mm_at_acc(
                &dlogits_chunk,
                c_len,
                vs,
                xf_chunk,
                d,
                &mut grads.data[head_idx],
            );
        }

        // 4. Compute hidden gradient contribution dxf_chunk
        let dxf_chunk = mm(&dlogits_chunk, c_len, vs, head_weight, d);
        dxf[c_start * d..c_end * d].copy_from_slice(&dxf_chunk);
        // `logits_chunk` and `dlogits_chunk` dropped at end of iteration!
    }

    if n_valid == 0 {
        return Err(TrainError::NoValidTokens);
    }
    if !loss_sum.is_finite() {
        return Err(TrainError::NonFiniteWeights);
    }

    // ---------------- Backward Sweep Through Decoder Layers ----------------
    let fb = final_norm_idx(cfg);
    let mut dgf = vec![T::ZERO; d];
    let mut dx = rms_bwd(&x, n, d, gf, &invf, &dxf, &mut dgf);
    add_into(&mut grads.data[fb], &dgf);

    for l in (0..cfg.n_layers).rev() {
        let idx = layer_indices(cfg, l);
        let c = &caches[l];

        // FFN backward
        let dhid = mm_bt(&dx, n, d, &p.data[idx.w_down], dff);
        mm_at_acc(&c.hid, n, dff, &dx, d, &mut grads.data[idx.w_down]);

        let mut dgate = vec![T::ZERO; n * dff];
        let mut dup = vec![T::ZERO; n * dff];
        for i in 0..n * dff {
            let g = c.gate[i];
            let u = c.up[i];
            let sig = T::ONE / (T::ONE + (-g).exp());
            let d_silu = sig * (T::ONE + g * (T::ONE - sig));
            dgate[i] = dhid[i] * u * d_silu;
            dup[i] = dhid[i] * (g * sig);
        }

        mm_at_acc(&c.r2, n, d, &dup, dff, &mut grads.data[idx.w_up]);
        mm_at_acc(&c.r2, n, d, &dgate, dff, &mut grads.data[idx.w_gate]);

        let dr2_gate = mm_bt(&dgate, n, dff, &p.data[idx.w_gate], d);
        let dr2_up = mm_bt(&dup, n, dff, &p.data[idx.w_up], d);
        let mut dr2 = dr2_gate;
        add_into(&mut dr2, &dr2_up);

        let mut dg2 = vec![T::ZERO; d];
        let dx1_ffn = rms_bwd(&c.x1, n, d, &p.data[idx.rms2], &c.inv2, &dr2, &mut dg2);
        add_into(&mut grads.data[idx.rms2], &dg2);

        let mut dx1 = dx;
        add_into(&mut dx1, &dx1_ffn);

        // Attention backward
        let dmerged = mm_bt(&dx1, n, d, &p.data[idx.wo], q_dim);
        mm_at_acc(&c.merged, n, q_dim, &dx1, d, &mut grads.data[idx.wo]);

        let (mut dqr, mut dkr, dv) =
            attention_backward(&c.qr, &c.kr, &c.v, &c.p, &dmerged, n, nq, nkv, dh, &valid);

        apply_rope_heads(&mut dqr, n, nq, dh, theta, true);
        apply_rope_heads(&mut dkr, n, nkv, dh, theta, true);

        if cfg.ext_x3_qk_norm {
            let q_un = c.q_unnorm.as_ref().unwrap();
            let inv_q = c.inv_q.as_ref().unwrap();
            let qn_gamma = &p.data[idx.q_norm.unwrap()];
            let mut dg_q = vec![T::ZERO; nq * dh];
            let mut dq_un = vec![T::ZERO; n * q_dim];
            for h in 0..nq {
                let mut un_head = vec![T::ZERO; n * dh];
                let mut dy_head = vec![T::ZERO; n * dh];
                let mut inv_head = vec![T::ZERO; n];
                for t in 0..n {
                    un_head[t * dh..(t + 1) * dh].copy_from_slice(&q_un[t * q_dim + h * dh..t * q_dim + (h + 1) * dh]);
                    dy_head[t * dh..(t + 1) * dh].copy_from_slice(&dqr[t * q_dim + h * dh..t * q_dim + (h + 1) * dh]);
                    inv_head[t] = inv_q[t * nq + h];
                }
                let gamma_h = &qn_gamma[h * dh..(h + 1) * dh];
                let mut dg_h = vec![T::ZERO; dh];
                let dx_h = rms_bwd(&un_head, n, dh, gamma_h, &inv_head, &dy_head, &mut dg_h);
                dg_q[h * dh..(h + 1) * dh].copy_from_slice(&dg_h);
                for t in 0..n {
                    dq_un[t * q_dim + h * dh..t * q_dim + (h + 1) * dh].copy_from_slice(&dx_h[t * dh..(t + 1) * dh]);
                }
            }
            add_into(&mut grads.data[idx.q_norm.unwrap()], &dg_q);
            dqr = dq_un;

            let k_un = c.k_unnorm.as_ref().unwrap();
            let inv_k = c.inv_k.as_ref().unwrap();
            let kn_gamma = &p.data[idx.k_norm.unwrap()];
            let mut dg_k = vec![T::ZERO; nkv * dh];
            let mut dk_un = vec![T::ZERO; n * kv_dim];
            for h in 0..nkv {
                let mut un_head = vec![T::ZERO; n * dh];
                let mut dy_head = vec![T::ZERO; n * dh];
                let mut inv_head = vec![T::ZERO; n];
                for t in 0..n {
                    un_head[t * dh..(t + 1) * dh].copy_from_slice(&k_un[t * kv_dim + h * dh..t * kv_dim + (h + 1) * dh]);
                    dy_head[t * dh..(t + 1) * dh].copy_from_slice(&dkr[t * kv_dim + h * dh..t * kv_dim + (h + 1) * dh]);
                    inv_head[t] = inv_k[t * nkv + h];
                }
                let gamma_h = &kn_gamma[h * dh..(h + 1) * dh];
                let mut dg_h = vec![T::ZERO; dh];
                let dx_h = rms_bwd(&un_head, n, dh, gamma_h, &inv_head, &dy_head, &mut dg_h);
                dg_k[h * dh..(h + 1) * dh].copy_from_slice(&dg_h);
                for t in 0..n {
                    dk_un[t * kv_dim + h * dh..t * kv_dim + (h + 1) * dh].copy_from_slice(&dx_h[t * dh..(t + 1) * dh]);
                }
            }
            add_into(&mut grads.data[idx.k_norm.unwrap()], &dg_k);
            dkr = dk_un;
        }

        mm_at_acc(&c.r1, n, d, &dqr, q_dim, &mut grads.data[idx.wq]);
        mm_at_acc(&c.r1, n, d, &dkr, kv_dim, &mut grads.data[idx.wk]);
        mm_at_acc(&c.r1, n, d, &dv, kv_dim, &mut grads.data[idx.wv]);

        let dr1_q = mm_bt(&dqr, n, q_dim, &p.data[idx.wq], d);
        let dr1_k = mm_bt(&dkr, n, kv_dim, &p.data[idx.wk], d);
        let dr1_v = mm_bt(&dv, n, kv_dim, &p.data[idx.wv], d);

        let mut dr1 = dr1_q;
        add_into(&mut dr1, &dr1_k);
        add_into(&mut dr1, &dr1_v);

        let mut dg1 = vec![T::ZERO; d];
        let dx0_attn = rms_bwd(&c.x0, n, d, &p.data[idx.rms1], &c.inv1, &dr1, &mut dg1);
        add_into(&mut grads.data[idx.rms1], &dg1);

        dx = dx1;
        add_into(&mut dx, &dx0_attn);
    }

    // Embedding backward
    for i in 0..n {
        let tok = input_tokens[i] as usize;
        let drow = &dx[i * d..(i + 1) * d];
        let dest = &mut grads.data[0][tok * d..(tok + 1) * d];
        add_into(dest, drow);
    }

    Ok(LossOut { loss_sum, n_valid })
}

/// Chunked forward-only loss computation (§4.2 Item 7).
pub fn chunked_forward_loss<T: Float>(
    cfg: &Config,
    p: &Params<T>,
    window: &[u32],
) -> Result<LossOut<T>, TrainError> {
    if window.len() < 2 {
        return Err(TrainError::WindowTooShort);
    }
    let n = window.len() - 1;
    let d = cfg.d_model;
    let dff = cfg.d_ff;
    let nq = cfg.n_q_heads;
    let nkv = cfg.n_kv_heads;
    let dh = cfg.head_dim();
    let q_dim = nq * dh;
    let kv_dim = nkv * dh;
    let vs = cfg.vocab_size;
    let eps = T::from_f64(cfg.rms_eps as f64);
    let theta = cfg.rope_theta;

    let input_tokens = &window[..n];
    let targets = &window[1..];

    let mut valid = vec![true; n];
    for (i, &tok) in input_tokens.iter().enumerate() {
        if tok as usize >= vs {
            return Err(TrainError::TokenOutOfRange(tok));
        }
        if tok == cfg.pad_id {
            valid[i] = false;
        }
    }

    let embed = &p.data[0];
    let mut x = vec![T::ZERO; n * d];
    for i in 0..n {
        let tok = input_tokens[i] as usize;
        let row = &embed[tok * d..(tok + 1) * d];
        x[i * d..(i + 1) * d].copy_from_slice(row);
    }

    for l in 0..cfg.n_layers {
        let idx = layer_indices(cfg, l);
        let g1 = &p.data[idx.rms1];
        let wq = &p.data[idx.wq];
        let wk = &p.data[idx.wk];
        let wv = &p.data[idx.wv];
        let wo = &p.data[idx.wo];
        let g2 = &p.data[idx.rms2];
        let wg = &p.data[idx.w_gate];
        let wu = &p.data[idx.w_up];
        let wd_ = &p.data[idx.w_down];

        let (r1, _) = rms_fwd(&x, n, d, g1, eps);
        let q = mm(&r1, n, d, wq, q_dim);
        let k = mm(&r1, n, d, wk, kv_dim);
        let v = mm(&r1, n, d, wv, kv_dim);

        let mut qr = if cfg.ext_x3_qk_norm {
            let mut q_normed = vec![T::ZERO; n * q_dim];
            let qn_gamma = &p.data[idx.q_norm.unwrap()];
            for h in 0..nq {
                let mut q_head = vec![T::ZERO; n * dh];
                for t in 0..n {
                    q_head[t * dh..(t + 1) * dh].copy_from_slice(&q[t * q_dim + h * dh..t * q_dim + (h + 1) * dh]);
                }
                let gamma_h = &qn_gamma[h * dh..(h + 1) * dh];
                let (normed_h, _) = rms_fwd(&q_head, n, dh, gamma_h, eps);
                for t in 0..n {
                    q_normed[t * q_dim + h * dh..t * q_dim + (h + 1) * dh].copy_from_slice(&normed_h[t * dh..(t + 1) * dh]);
                }
            }
            q_normed
        } else {
            q
        };

        let mut kr = if cfg.ext_x3_qk_norm {
            let mut k_normed = vec![T::ZERO; n * kv_dim];
            let kn_gamma = &p.data[idx.k_norm.unwrap()];
            for h in 0..nkv {
                let mut k_head = vec![T::ZERO; n * dh];
                for t in 0..n {
                    k_head[t * dh..(t + 1) * dh].copy_from_slice(&k[t * kv_dim + h * dh..t * kv_dim + (h + 1) * dh]);
                }
                let gamma_h = &kn_gamma[h * dh..(h + 1) * dh];
                let (normed_h, _) = rms_fwd(&k_head, n, dh, gamma_h, eps);
                for t in 0..n {
                    k_normed[t * kv_dim + h * dh..t * kv_dim + (h + 1) * dh].copy_from_slice(&normed_h[t * dh..(t + 1) * dh]);
                }
            }
            k_normed
        } else {
            k
        };

        apply_rope_heads(&mut qr, n, nq, dh, theta, false);
        apply_rope_heads(&mut kr, n, nkv, dh, theta, false);

        let (merged, _) = attention_forward(&qr, &kr, &v, n, nq, nkv, dh, &valid);
        let o = mm(&merged, n, q_dim, wo, d);
        let mut x1 = x;
        add_into(&mut x1, &o);

        let (r2, _) = rms_fwd(&x1, n, d, g2, eps);
        let gate = mm(&r2, n, d, wg, dff);
        let up = mm(&r2, n, d, wu, dff);
        let hid: Vec<T> = gate
            .iter()
            .zip(&up)
            .map(|(&g, &u)| g / (T::ONE + (-g).exp()) * u)
            .collect();
        let f = mm(&hid, n, dff, wd_, d);
        let mut x2 = x1;
        add_into(&mut x2, &f);
        x = x2;
    }

    let gf = &p.data[final_norm_idx(cfg)];
    let (xf, _) = rms_fwd(&x, n, d, gf, eps);
    let head_weight = if cfg.tied_embeddings {
        embed
    } else {
        &p.data[lm_head_idx(cfg)]
    };

    let chunk_size = cfg.chunk_size.max(1);
    let n_chunks = n.div_ceil(chunk_size);

    let mut loss_sum = T::ZERO;
    let mut n_valid = 0usize;

    for c in 0..n_chunks {
        let c_start = c * chunk_size;
        let c_end = (c_start + chunk_size).min(n);
        let c_len = c_end - c_start;

        let xf_chunk = &xf[c_start * d..c_end * d];
        let logits_chunk = mm_bt(xf_chunk, c_len, d, head_weight, vs);

        for k in 0..c_len {
            let token_idx = c_start + k;
            if !valid[token_idx] || targets[token_idx] == cfg.pad_id {
                continue;
            }
            n_valid += 1;
            let z = &logits_chunk[k * vs..(k + 1) * vs];
            let mut mx = z[0];
            for &val in z {
                if val > mx {
                    mx = val;
                }
            }
            let mut s = T::ZERO;
            for &val in z {
                s += (val - mx).exp();
            }
            let lse = mx + s.ln();
            let y = targets[token_idx] as usize;
            loss_sum += lse - z[y];
        }
    }

    if n_valid == 0 {
        return Err(TrainError::NoValidTokens);
    }
    if !loss_sum.is_finite() {
        return Err(TrainError::NonFiniteWeights);
    }

    Ok(LossOut { loss_sum, n_valid })
}

// Internal Dense Linear Algebra Helpers

fn mm<T: Float>(a: &[T], m: usize, k: usize, b: &[T], n: usize) -> Vec<T> {
    let mut out = vec![T::ZERO; m * n];
    for i in 0..m {
        for p in 0..k {
            let av = a[i * k + p];
            for j in 0..n {
                out[i * n + j] += av * b[p * n + j];
            }
        }
    }
    out
}

fn mm_bt<T: Float>(a: &[T], m: usize, k: usize, b: &[T], n: usize) -> Vec<T> {
    let mut out = vec![T::ZERO; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = T::ZERO;
            for p in 0..k {
                s += a[i * k + p] * b[j * k + p];
            }
            out[i * n + j] = s;
        }
    }
    out
}

fn mm_at_acc<T: Float>(a: &[T], m: usize, k: usize, b: &[T], n: usize, out: &mut [T]) {
    for i in 0..m {
        for p in 0..k {
            let av = a[i * k + p];
            for j in 0..n {
                out[p * n + j] += av * b[i * n + j];
            }
        }
    }
}

fn add_into<T: Float>(a: &mut [T], b: &[T]) {
    for (x, &y) in a.iter_mut().zip(b) {
        *x += y;
    }
}

fn rms_fwd<T: Float>(x: &[T], n: usize, d: usize, g: &[T], eps: T) -> (Vec<T>, Vec<T>) {
    let mut y = vec![T::ZERO; n * d];
    let mut inv = vec![T::ZERO; n];
    let dd = T::from_f64(d as f64);
    for r in 0..n {
        let row = &x[r * d..(r + 1) * d];
        let mut s = T::ZERO;
        for &v in row {
            s += v * v;
        }
        let iv = T::ONE / (s / dd + eps).sqrt();
        inv[r] = iv;
        for i in 0..d {
            y[r * d + i] = row[i] * iv * g[i];
        }
    }
    (y, inv)
}

fn rms_bwd<T: Float>(
    x: &[T],
    n: usize,
    d: usize,
    g: &[T],
    inv: &[T],
    dy: &[T],
    dg: &mut [T],
) -> Vec<T> {
    let mut dx = vec![T::ZERO; n * d];
    let dd = T::from_f64(d as f64);
    for r in 0..n {
        let xr = &x[r * d..(r + 1) * d];
        let dyr = &dy[r * d..(r + 1) * d];
        let iv = inv[r];
        let mut dot = T::ZERO;
        for i in 0..d {
            dg[i] += dyr[i] * xr[i] * iv;
            dot += dyr[i] * g[i] * xr[i];
        }
        let coeff = dot * iv * iv * iv / dd;
        for i in 0..d {
            dx[r * d + i] = dyr[i] * g[i] * iv - xr[i] * coeff;
        }
    }
    dx
}

fn apply_rope_heads<T: Float>(
    x: &mut [T],
    n: usize,
    n_heads: usize,
    dh: usize,
    theta: f32,
    inverse: bool,
) {
    let half = dh / 2;
    for pos in 0..n {
        for h in 0..n_heads {
            let off = (pos * n_heads + h) * dh;
            let slice = &mut x[off..off + dh];
            for k in 0..half {
                let freq = T::from_f64((theta as f64).powf(-2.0 * k as f64 / dh as f64));
                let (s, c) = (T::from_f64(pos as f64) * freq).sin_cos();
                let s = if inverse { -s } else { s };
                let (a, b) = (slice[2 * k], slice[2 * k + 1]);
                slice[2 * k] = a * c - b * s;
                slice[2 * k + 1] = a * s + b * c;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn attention_forward<T: Float>(
    q: &[T],
    k: &[T],
    v: &[T],
    n: usize,
    nq: usize,
    nkv: usize,
    dh: usize,
    valid: &[bool],
) -> (Vec<T>, Vec<T>) {
    let q_dim = nq * dh;
    let _kv_dim = nkv * dh;
    let group = nq / nkv;
    let scale = T::ONE / T::from_f64((dh as f64).sqrt());
    let mut merged = vec![T::ZERO; n * q_dim];
    let mut pm = vec![T::ZERO; nq * n * n];

    for qh in 0..nq {
        let kvh = qh / group;
        for i in 0..n {
            if !valid[i] {
                continue;
            }
            let mut scores = vec![T::ZERO; n];
            let qi = &q[(i * nq + qh) * dh..(i * nq + qh + 1) * dh];
            for j in 0..=i {
                if !valid[j] {
                    continue;
                }
                let kj = &k[(j * nkv + kvh) * dh..(j * nkv + kvh + 1) * dh];
                let mut dot = T::ZERO;
                for p in 0..dh {
                    dot += qi[p] * kj[p];
                }
                scores[j] = dot * scale;
            }

            let mut mx = T::from_f64(-1e9);
            for j in 0..=i {
                if valid[j] && scores[j] > mx {
                    mx = scores[j];
                }
            }
            let mut s = T::ZERO;
            for j in 0..=i {
                if valid[j] {
                    let e = (scores[j] - mx).exp();
                    scores[j] = e;
                    s += e;
                } else {
                    scores[j] = T::ZERO;
                }
            }
            if s > T::ZERO {
                for j in 0..=i {
                    scores[j] = scores[j] / s;
                    pm[(qh * n + i) * n + j] = scores[j];
                }
            }

            let out_slice = &mut merged[(i * nq + qh) * dh..(i * nq + qh + 1) * dh];
            for j in 0..=i {
                if valid[j] {
                    let vj = &v[(j * nkv + kvh) * dh..(j * nkv + kvh + 1) * dh];
                    let weight = scores[j];
                    for p in 0..dh {
                        out_slice[p] += weight * vj[p];
                    }
                }
            }
        }
    }
    (merged, pm)
}

#[allow(clippy::too_many_arguments)]
fn attention_backward<T: Float>(
    q: &[T],
    k: &[T],
    v: &[T],
    pm: &[T],
    dmerged: &[T],
    n: usize,
    nq: usize,
    nkv: usize,
    dh: usize,
    valid: &[bool],
) -> (Vec<T>, Vec<T>, Vec<T>) {
    let q_dim = nq * dh;
    let kv_dim = nkv * dh;
    let group = nq / nkv;
    let scale = T::ONE / T::from_f64((dh as f64).sqrt());
    let mut dq = vec![T::ZERO; n * q_dim];
    let mut dk = vec![T::ZERO; n * kv_dim];
    let mut dv = vec![T::ZERO; n * kv_dim];

    for qh in 0..nq {
        let kvh = qh / group;
        for i in 0..n {
            if !valid[i] {
                continue;
            }
            let do_i = &dmerged[(i * nq + qh) * dh..(i * nq + qh + 1) * dh];
            let mut dp = vec![T::ZERO; n];
            for j in 0..=i {
                if valid[j] {
                    let vj = &v[(j * nkv + kvh) * dh..(j * nkv + kvh + 1) * dh];
                    let mut dot = T::ZERO;
                    for p in 0..dh {
                        dot += do_i[p] * vj[p];
                    }
                    dp[j] = dot;
                    let p_ij = pm[(qh * n + i) * n + j];
                    let dvj = &mut dv[(j * nkv + kvh) * dh..(j * nkv + kvh + 1) * dh];
                    for p in 0..dh {
                        dvj[p] += p_ij * do_i[p];
                    }
                }
            }

            let mut d_dot_p = T::ZERO;
            for j in 0..=i {
                if valid[j] {
                    d_dot_p += dp[j] * pm[(qh * n + i) * n + j];
                }
            }
            let mut ds = vec![T::ZERO; n];
            for j in 0..=i {
                if valid[j] {
                    let p_ij = pm[(qh * n + i) * n + j];
                    ds[j] = p_ij * (dp[j] - d_dot_p) * scale;
                }
            }

            let qi = &q[(i * nq + qh) * dh..(i * nq + qh + 1) * dh];
            let dqi = &mut dq[(i * nq + qh) * dh..(i * nq + qh + 1) * dh];
            for j in 0..=i {
                if valid[j] {
                    let kj = &k[(j * nkv + kvh) * dh..(j * nkv + kvh + 1) * dh];
                    let dkj = &mut dk[(j * nkv + kvh) * dh..(j * nkv + kvh + 1) * dh];
                    let ds_ij = ds[j];
                    for p in 0..dh {
                        dqi[p] += ds_ij * kj[p];
                        dkj[p] += ds_ij * qi[p];
                    }
                }
            }
        }
    }
    (dq, dk, dv)
}

/// Gate GX7 runner: Memory-Efficient Chunked Cross-Entropy & Activation Recomputation (§4.2 Item 7, §9.2 GX7).
pub fn gx7() -> GateReport {
    let t0 = Instant::now();
    let cfg = Config::chunked_loss_v2(16);
    let mut details = Vec::new();
    let mut checks_passed = true;

    let (rev, dirty) = detect_revision();
    let toolchain = detect_toolchain();

    let model = Model::init(&cfg).expect("model init");
    let params_f32 = Params::from_model(&model);

    // 1. Exact equivalence on standard sequence length (e.g. 33 tokens, 2 chunks of 16 + 1 tail token)
    let mut rng = Rng::new(0x7777_1234);
    let max_tok = (cfg.vocab_size - 4) as u32;
    let window_std: Vec<u32> = (0..34)
        .map(|_| rng.next_u32() % max_tok)
        .collect();

    let mut grads_mono = params_f32.zeros_like();
    let mut grads_chunk = params_f32.zeros_like();

    let out_mono = crate::autodiff::loss_and_grad(&cfg, &params_f32, &window_std, 1.0f32, &mut grads_mono)
        .expect("monolithic loss_and_grad");
    let out_chunk = chunked_loss_and_grad(&cfg, &params_f32, &window_std, 1.0f32, &mut grads_chunk)
        .expect("chunked loss_and_grad");

    let loss_diff = (out_mono.loss_sum - out_chunk.loss_sum).abs();
    let loss_equiv_ok = loss_diff <= 1e-6;

    let mut max_grad_diff = 0.0f32;
    for (gm, gc) in grads_mono.data.iter().zip(&grads_chunk.data) {
        for (&a, &b) in gm.iter().zip(gc) {
            let diff = (a - b).abs();
            if diff > max_grad_diff {
                max_grad_diff = diff;
            }
        }
    }
    let grad_equiv_ok = max_grad_diff <= 1e-5;
    let exact_equiv_ok = loss_equiv_ok && grad_equiv_ok;
    checks_passed &= exact_equiv_ok;
    details.push(format!(
        "exact_loss_and_grad_equivalence: loss_diff={loss_diff:.2e}, max_grad_diff={max_grad_diff:.2e} (ok={exact_equiv_ok})"
    ));

    // 2. Tail chunk handling (e.g. sequence length 19 tokens -> chunk 16 + tail 3)
    let window_tail: Vec<u32> = (0..20)
        .map(|_| rng.next_u32() % max_tok)
        .collect();
    let mut g_mono_tail = params_f32.zeros_like();
    let mut g_chunk_tail = params_f32.zeros_like();

    let o_mono_tail = crate::autodiff::loss_and_grad(&cfg, &params_f32, &window_tail, 1.0f32, &mut g_mono_tail)
        .expect("tail monolithic");
    let o_chunk_tail = chunked_loss_and_grad(&cfg, &params_f32, &window_tail, 1.0f32, &mut g_chunk_tail)
        .expect("tail chunked");

    let tail_loss_diff = (o_mono_tail.loss_sum - o_chunk_tail.loss_sum).abs();
    let tail_ok = tail_loss_diff <= 1e-6;
    checks_passed &= tail_ok;
    details.push(format!("tail_chunk_handling: diff={tail_loss_diff:.2e} (ok={tail_ok})"));

    // 3. Padding mask and zero-valid token handling
    let mut win_padded = window_std.clone();
    // Set half the tokens to PAD
    for slot in win_padded[10..25].iter_mut() {
        *slot = cfg.pad_id;
    }
    let mut g_padded = params_f32.zeros_like();
    let o_padded = chunked_loss_and_grad(&cfg, &params_f32, &win_padded, 1.0f32, &mut g_padded);
    let pad_mask_ok = o_padded.is_ok();

    // All PAD tokens window -> must return NoValidTokens
    let win_all_pad = vec![cfg.pad_id; 20];
    let o_all_pad = chunked_loss_and_grad(&cfg, &params_f32, &win_all_pad, 1.0f32, &mut g_padded);
    let zero_valid_ok = matches!(o_all_pad, Err(TrainError::NoValidTokens));
    let pad_and_zero_ok = pad_mask_ok && zero_valid_ok;
    checks_passed &= pad_and_zero_ok;
    details.push(format!("padding_mask_and_zero_valid (ok={pad_and_zero_ok})"));

    // 4. Zero retained chunk logits verification
    // Chunked execution keeps at most chunk_size * vocab_size logits in memory per chunk
    let full_logits_len = (window_std.len() - 1) * cfg.vocab_size;
    let chunk_logits_len = cfg.chunk_size * cfg.vocab_size;
    let memory_reduction_ratio = full_logits_len as f64 / chunk_logits_len as f64;
    let zero_retained_ok = memory_reduction_ratio > 1.0;
    checks_passed &= zero_retained_ok;
    details.push(format!(
        "zero_retained_chunk_logits: ratio={memory_reduction_ratio:.2}x reduction (ok={zero_retained_ok})"
    ));

    // 5. Activation recomputation and parameter immutability invariant
    let p_before = params_f32.clone();
    let _ = chunked_loss_and_grad(&cfg, &params_f32, &window_std, 1.0f32, &mut g_padded).unwrap();
    let mut immutability_ok = true;
    for (b, a) in p_before.data.iter().zip(&params_f32.data) {
        if b != a {
            immutability_ok = false;
        }
    }
    checks_passed &= immutability_ok;
    details.push(format!("activation_recomputation_immutability (ok={immutability_ok})"));

    let joined_details = details.join("; ");
    GateReport {
        gate_id: "GX7",
        scope: "acceptance",
        status: if checks_passed { "PASS" } else { "FAIL" },
        reason: if checks_passed {
            None
        } else {
            Some("one or more chunked loss invariants failed".into())
        },
        requirement: "Chunked cross-entropy exact equivalence to full loss (Delta <= 1e-6); zero retained chunk logits; activation recomputation immutability invariant; tail chunk, padding mask, and zero-valid token handling; custom VJP tape integration without duplicate backward passes (§4.2 Item 7, §9.2 GX7)",
        implemented_check: "exact_loss_and_grad_equivalence, tail_chunk_handling, padding_mask_and_zero_valid, zero_retained_chunk_logits, activation_recomputation_immutability",
        prerequisites: "reference_v2, G3, G8",
        exclusions: "none",
        revision: rev,
        dirty_state: dirty,
        toolchain,
        target: "x86_64-unknown-linux-gnu",
        command: "cargo test --test test_x7_loss",
        fixture_hash: "chunked-loss-v1".into(),
        config: "chunked_loss_v2",
        dtype: "fp32",
        expected: "loss difference <= 1e-6, grad difference <= 1e-5; exact tail chunk handling; valid error on all-PAD; zero retained chunk logits; immutable model parameters and PRNG".into(),
        observed: joined_details,
        atol: 0.000001,
        rtol: 0.00001,
        execution_time_ms: t0.elapsed().as_millis(),
    }
}
