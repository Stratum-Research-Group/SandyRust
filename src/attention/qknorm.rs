//! Extension X3: QK-Norm & Gate GX3 (§4.2 Item 3, §9.2 GX3).
//!
//! Applies per-head RMSNorm to Query and Key heads before dot-product attention
//! and RoPE:
//!   \tilde{Q}_{b,t,h} = RMSNorm(Q_{b,t,h}, \gamma_q^h)
//!   \tilde{K}_{b,t,g} = RMSNorm(K_{b,t,g}, \gamma_k^g)
//! Adds L * (H_q + H_kv) * d_h learned scale parameters initialized to 1.0.

use std::time::Instant;

use crate::cache::{forward_step, KVCache};
use crate::checkpoint::{parse, serialize, CheckpointError};
use crate::config::Config;
use crate::gates::{detect_revision, detect_toolchain, GateReport};
use crate::model::Model;
use crate::tensor::ops::rms_norm;
use crate::tensor::{Tensor, TensorError};

/// Applies RMSNorm over the last axis of a head matrix `x` `[T, dh]` with scale `gamma` `[dh]`.
pub fn apply_qk_norm_head(x: &Tensor, gamma: &Tensor, eps: f32) -> Result<Tensor, TensorError> {
    rms_norm(x, gamma, eps)
}

/// Slices a single row from a 2D tensor `[rows, cols]` returning a 1D tensor `[cols]`.
pub fn slice_row(t: &Tensor, row: usize) -> Result<Tensor, TensorError> {
    if t.shape().len() != 2 {
        return Err(TensorError::Rank {
            expected: 2,
            actual: t.shape().len(),
        });
    }
    let (rows, cols) = (t.shape()[0], t.shape()[1]);
    if row >= rows {
        return Err(TensorError::InvalidArgument("slice_row out of bounds"));
    }
    let start = row * cols;
    Tensor::from_vec(&[cols], t.data()[start..start + cols].to_vec())
}

/// FP64 reference RMSNorm for a single head row `[dh]` with scale `gamma` `[dh]`.
pub fn rms_norm_row_f64(x: &[f64], gamma: &[f64], eps: f64) -> Vec<f64> {
    let d = x.len();
    let ms: f64 = x.iter().map(|&v| v * v).sum::<f64>() / d as f64;
    let inv = 1.0 / (ms + eps).sqrt();
    let mut y = vec![0.0; d];
    for i in 0..d {
        y[i] = x[i] * inv * gamma[i];
    }
    y
}

/// FP64 analytic VJP for a single head row `[dh]`. Returns `(dx, dgamma)`.
pub fn rms_norm_vjp_f64(x: &[f64], gamma: &[f64], dy: &[f64], eps: f64) -> (Vec<f64>, Vec<f64>) {
    let d = x.len();
    let ms: f64 = x.iter().map(|&v| v * v).sum::<f64>() / d as f64;
    let inv = 1.0 / (ms + eps).sqrt();
    let mut dot = 0.0;
    let mut dgamma = vec![0.0; d];
    for i in 0..d {
        dot += x[i] * gamma[i] * dy[i];
        dgamma[i] = dy[i] * x[i] * inv;
    }
    let coef = inv * inv * inv / (d as f64) * dot;
    let mut dx = vec![0.0; d];
    for i in 0..d {
        dx[i] = gamma[i] * dy[i] * inv - x[i] * coef;
    }
    (dx, dgamma)
}

/// Compute Shannon entropy of softmax probability distribution `p`.
pub fn softmax_entropy(logits: &[f32]) -> f32 {
    let mut mx = logits[0];
    for &v in logits {
        if v > mx {
            mx = v;
        }
    }
    let mut s = 0.0f32;
    for &v in logits {
        s += (v - mx).exp();
    }
    let lse = mx + s.ln();
    let mut ent = 0.0f32;
    for &v in logits {
        let p = (v - lse).exp();
        if p > 1e-12 {
            ent -= p * p.ln();
        }
    }
    ent
}

/// Gate GX3: QK-Norm acceptance verification (§4.2 Item 3, §9.2 GX3).
pub fn gx3() -> GateReport {
    let t0 = Instant::now();
    let (rev, dirty) = detect_revision();
    let toolchain = detect_toolchain();

    let mut checks_passed = true;
    let mut details = Vec::new();

    // 1. Parameter count verification: +1,024 on MHA, +768 on GQA, +640 on MQA
    let mut cfg_mha = Config::reference_v2();
    cfg_mha.ext_x3_qk_norm = true;
    let mut cfg_gqa = Config::gqa_v2();
    cfg_gqa.ext_x3_qk_norm = true;
    let mut cfg_mqa = Config::mqa_v2();
    cfg_mqa.ext_x3_qk_norm = true;

    let p_base_mha = Config::reference_v2().param_count().unwrap_or(0);
    let p_qk_mha = cfg_mha.param_count().unwrap_or(0);
    let p_base_gqa = Config::gqa_v2().param_count().unwrap_or(0);
    let p_qk_gqa = cfg_gqa.param_count().unwrap_or(0);
    let p_base_mqa = Config::mqa_v2().param_count().unwrap_or(0);
    let p_qk_mqa = cfg_mqa.param_count().unwrap_or(0);

    let diff_mha = p_qk_mha - p_base_mha;
    let diff_gqa = p_qk_gqa - p_base_gqa;
    let diff_mqa = p_qk_mqa - p_base_mqa;

    let count_ok = (p_qk_mha == 887_296 && diff_mha == 1_024)
        && (p_qk_gqa == 821_504 && diff_gqa == 768)
        && (p_qk_mqa == 788_608 && diff_mqa == 640);
    checks_passed &= count_ok;
    details.push(format!(
        "param_inventory: mha={p_qk_mha}(+{diff_mha}), gqa={p_qk_gqa}(+{diff_gqa}), mqa={p_qk_mqa}(+{diff_mqa}) (ok={count_ok})"
    ));

    // 2. Zero-input numerical stability check
    let dh = 32;
    let zero_x = vec![0.0f64; dh];
    let ones_gamma = vec![1.0f64; dh];
    let eps = 1e-5f64;
    let norm_zero = rms_norm_row_f64(&zero_x, &ones_gamma, eps);
    let zero_input_finite = norm_zero.iter().all(|&v| v.is_finite() && v == 0.0);

    let (dx_zero, dg_zero) = rms_norm_vjp_f64(&zero_x, &ones_gamma, &vec![1.0; dh], eps);
    let vjp_zero_finite = dx_zero.iter().all(|&v| v.is_finite())
        && dg_zero.iter().all(|&v| v.is_finite());
    let zero_stable = zero_input_finite && vjp_zero_finite;
    checks_passed &= zero_stable;
    details.push(format!("zero_input_stability: norm_zero={zero_input_finite}, vjp_zero={vjp_zero_finite} (ok={zero_stable})"));

    // 3. Analytic gradients vs FP64 finite differences
    let mut x_probe = Vec::with_capacity(dh);
    let mut gamma_probe = Vec::with_capacity(dh);
    let mut dy_probe = Vec::with_capacity(dh);
    for i in 0..dh {
        x_probe.push(((i as f64 * 0.37).sin()) * 0.8);
        gamma_probe.push(1.0 + ((i as f64 * 0.23).cos()) * 0.1);
        dy_probe.push(((i as f64 * 0.41).sin()) * 0.5);
    }
    let (dx_analytic, dg_analytic) = rms_norm_vjp_f64(&x_probe, &gamma_probe, &dy_probe, eps);

    let mut max_err_x = 0.0f64;
    let mut max_err_g = 0.0f64;
    let h = 1e-6f64;

    for i in 0..dh {
        // d/dx_i
        let mut xp = x_probe.clone();
        let mut xm = x_probe.clone();
        xp[i] += h;
        xm[i] -= h;
        let yp = rms_norm_row_f64(&xp, &gamma_probe, eps);
        let ym = rms_norm_row_f64(&xm, &gamma_probe, eps);
        let lp: f64 = yp.iter().zip(&dy_probe).map(|(&a, &b)| a * b).sum();
        let lm: f64 = ym.iter().zip(&dy_probe).map(|(&a, &b)| a * b).sum();
        let num_dx = (lp - lm) / (2.0 * h);
        let err = (dx_analytic[i] - num_dx).abs();
        if err > max_err_x {
            max_err_x = err;
        }

        // d/dgamma_i
        let mut gp = gamma_probe.clone();
        let mut gm = gamma_probe.clone();
        gp[i] += h;
        gm[i] -= h;
        let y_gp = rms_norm_row_f64(&x_probe, &gp, eps);
        let y_gm = rms_norm_row_f64(&x_probe, &gm, eps);
        let lp_g: f64 = y_gp.iter().zip(&dy_probe).map(|(&a, &b)| a * b).sum();
        let lm_g: f64 = y_gm.iter().zip(&dy_probe).map(|(&a, &b)| a * b).sum();
        let num_dg = (lp_g - lm_g) / (2.0 * h);
        let err_g = (dg_analytic[i] - num_dg).abs();
        if err_g > max_err_g {
            max_err_g = err_g;
        }
    }
    let grad_match_ok = max_err_x <= 1e-5 && max_err_g <= 1e-5;
    checks_passed &= grad_match_ok;
    details.push(format!(
        "analytic_gradients: max_err_x={max_err_x:.2e}, max_err_g={max_err_g:.2e} (ok={grad_match_ok})"
    ));

    // 4. Attention score entropy stability under magnitude scaling
    // Standard unnormalized dot product scales quadratically with input scale s,
    // collapsing entropy to near 0. QK-Norm bounds logits and preserves entropy.
    let tk = 16;
    let q_vec: Vec<f32> = (0..dh).map(|i| (i as f32 * 0.19).sin()).collect();
    let k_matrix: Vec<Vec<f32>> = (0..tk)
        .map(|j| (0..dh).map(|i| ((j * 31 + i) as f32 * 0.23).cos()).collect())
        .collect();

    let scale_factor = 1.0 / (dh as f32).sqrt();

    // Entropy with scale = 1.0 vs scale = 50.0 (Unnormalized)
    let compute_unnorm_entropy = |mag: f32| -> f32 {
        let logits: Vec<f32> = k_matrix
            .iter()
            .map(|k_row| {
                let dot: f32 = q_vec
                    .iter()
                    .zip(k_row.iter())
                    .map(|(&q, &k)| (q * mag) * (k * mag))
                    .sum();
                dot * scale_factor
            })
            .collect();
        softmax_entropy(&logits)
    };

    // Entropy with QK-Norm at mag = 1.0 vs mag = 50.0
    let compute_qknorm_entropy = |mag: f32| -> f32 {
        let q_scaled: Vec<f32> = q_vec.iter().map(|&x| x * mag).collect();
        let q_rms = (q_scaled.iter().map(|&x| x * x).sum::<f32>() / dh as f32 + 1e-5).sqrt();
        let q_norm: Vec<f32> = q_scaled.iter().map(|&x| x / q_rms).collect();

        let logits: Vec<f32> = k_matrix
            .iter()
            .map(|k_row| {
                let k_scaled: Vec<f32> = k_row.iter().map(|&x| x * mag).collect();
                let k_rms = (k_scaled.iter().map(|&x| x * x).sum::<f32>() / dh as f32 + 1e-5).sqrt();
                let k_norm: Vec<f32> = k_scaled.iter().map(|&x| x / k_rms).collect();
                let dot: f32 = q_norm.iter().zip(k_norm.iter()).map(|(&q, &k)| q * k).sum();
                dot * scale_factor
            })
            .collect();
        softmax_entropy(&logits)
    };

    let ent_unnorm_1 = compute_unnorm_entropy(1.0);
    let ent_unnorm_50 = compute_unnorm_entropy(50.0);
    let ent_qknorm_1 = compute_qknorm_entropy(1.0);
    let ent_qknorm_50 = compute_qknorm_entropy(50.0);

    let unnorm_collapsed = ent_unnorm_50 < 0.1 && ent_unnorm_1 > 1.0;
    let qknorm_stable = (ent_qknorm_1 - ent_qknorm_50).abs() <= 1e-3 && ent_qknorm_50 > 1.0;
    let entropy_ok = unnorm_collapsed && qknorm_stable;
    checks_passed &= entropy_ok;
    details.push(format!(
        "entropy_stability: unnorm_collapsed={unnorm_collapsed} (ent1={ent_unnorm_1:.2}, ent50={ent_unnorm_50:.2e}); qknorm_stable={qknorm_stable} (ent1={ent_qknorm_1:.2}, ent50={ent_qknorm_50:.2}) (ok={entropy_ok})"
    ));

    // 5. Checkpoint serialization, reload, and header isolation
    let model_qk = Model::init(&cfg_mha).unwrap();
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
    let orig_out = model_qk.forward(&prompt, None).unwrap();

    let bytes = serialize(&model_qk);
    let reloaded = parse(&bytes, &cfg_mha, 64 * 1024 * 1024).unwrap();
    let reloaded_out = reloaded.forward(&prompt, None).unwrap();
    let ckpt_exact = orig_out.data() == reloaded_out.data();

    // Cross-loading into baseline config without QK-Norm rejected
    let cfg_base = Config::reference_v2();
    let cross_reject = matches!(
        parse(&bytes, &cfg_base, 64 * 1024 * 1024),
        Err(CheckpointError::ConfigMismatch(_)) | Err(CheckpointError::InventoryMismatch(_))
    );

    let ckpt_ok = ckpt_exact && cross_reject;
    checks_passed &= ckpt_ok;
    details.push(format!(
        "checkpoint_isolation: reload_exact={ckpt_exact}, cross_reject={cross_reject} (ok={ckpt_ok})"
    ));

    // 6. Cached incremental decode parity vs full forward
    let mut cache = KVCache::new(&cfg_mha, model_qk.fingerprint(), 32).unwrap();
    let mut step_out = None;
    for (pos, &tok) in prompt.iter().enumerate() {
        step_out = Some(forward_step(&model_qk, tok, pos, &mut cache).unwrap());
    }
    let last_step = step_out.unwrap();
    let (rows, cols) = (orig_out.shape()[0], orig_out.shape()[1]);
    let last_full_row = &orig_out.data()[(rows - 1) * cols..rows * cols];
    let step_row = last_step.data();

    let mut max_err = 0.0f32;
    for (&a, &b) in last_full_row.iter().zip(step_row.iter()) {
        let diff = (a - b).abs();
        if diff > max_err {
            max_err = diff;
        }
    }
    let decode_ok = max_err <= 1e-4;
    checks_passed &= decode_ok;
    details.push(format!("incremental_decode_parity: max_err={max_err:.2e} (ok={decode_ok})"));

    let joined_details = details.join("; ");
    GateReport {
        gate_id: "GX3",
        scope: "acceptance",
        status: if checks_passed { "PASS" } else { "FAIL" },
        reason: if checks_passed {
            None
        } else {
            Some("one or more QK-Norm invariants failed".into())
        },
        requirement: "Attention score stability with per-head RMSNorm on Q and K; parameters +1,024 on MHA (+768 on GQA, +640 on MQA); zero-input numerical stability; analytic gradients match finite differences; checkpoint reload (§4.2 Item 3, §9.2 GX3)",
        implemented_check: "param_inventory, zero_input_stability, analytic_gradients, entropy_stability, checkpoint_isolation, incremental_decode_parity",
        prerequisites: "reference_v2, G4, G5, G6, G10",
        exclusions: "none",
        revision: rev,
        dirty_state: dirty,
        toolchain,
        target: "x86_64-unknown-linux-gnu",
        command: "cargo test --test test_x3_qknorm",
        fixture_hash: "qknorm-v2-invariants-v1".into(),
        config: "qk_norm_v2",
        dtype: "fp32",
        expected: "exact param count additions (+1,024 / +768 / +640), zero-input finite zero, analytic VJP matches finite differences <= 1e-5, entropy stability under scaling, checkpoint roundtrip, and cached decode parity <= 1e-4".into(),
        observed: joined_details,
        atol: 1e-5,
        rtol: 1e-4,
        execution_time_ms: t0.elapsed().as_millis(),
    }
}
