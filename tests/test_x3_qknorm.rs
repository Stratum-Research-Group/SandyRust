//! Integration test suite for Extension X3: QK-Norm & Gate GX3 (§4.2 Item 3, §9.2 GX3).

use sandy_rust::attention::qknorm::{apply_qk_norm_head, rms_norm_row_f64, rms_norm_vjp_f64, softmax_entropy};
use sandy_rust::attention::gx3;
use sandy_rust::cache::{forward_step, KVCache};
use sandy_rust::checkpoint::{parse, serialize, CheckpointError};
use sandy_rust::config::Config;
use sandy_rust::gates;
use sandy_rust::model::Model;
use sandy_rust::tensor::Tensor;

#[test]
fn test_x3_config_validation_and_param_counts() {
    let c_mha = Config::reference_v2();
    let c_qk_mha = Config::qk_norm_v2();
    let c_gqa = Config::gqa_v2();
    let mut c_qk_gqa = Config::gqa_v2();
    c_qk_gqa.ext_x3_qk_norm = true;
    let c_mqa = Config::mqa_v2();
    let mut c_qk_mqa = Config::mqa_v2();
    c_qk_mqa.ext_x3_qk_norm = true;

    assert!(c_qk_mha.validate().is_ok());
    assert!(c_qk_gqa.validate().is_ok());
    assert!(c_qk_mqa.validate().is_ok());

    // Exact counts from spec:
    // MHA: 886,272 + 1,024 = 887,296
    // GQA: 820,736 + 768 = 821,504
    // MQA: 787,968 + 640 = 788,608
    assert_eq!(c_mha.param_count(), Some(886_272));
    assert_eq!(c_qk_mha.param_count(), Some(887_296));
    assert_eq!(c_qk_mha.param_count().unwrap() - c_mha.param_count().unwrap(), 1_024);

    assert_eq!(c_gqa.param_count(), Some(820_736));
    assert_eq!(c_qk_gqa.param_count(), Some(821_504));
    assert_eq!(c_qk_gqa.param_count().unwrap() - c_gqa.param_count().unwrap(), 768);

    assert_eq!(c_mqa.param_count(), Some(787_968));
    assert_eq!(c_qk_mqa.param_count(), Some(788_608));
    assert_eq!(c_qk_mqa.param_count().unwrap() - c_mqa.param_count().unwrap(), 640);
}

#[test]
fn test_x3_zero_input_stability() {
    // Zero input vector of dimension 32
    let d = 32;
    let x = vec![0.0f64; d];
    let gamma = vec![1.0f64; d];
    let eps = 1e-5f64;

    let y = rms_norm_row_f64(&x, &gamma, eps);
    assert_eq!(y.len(), d);
    for val in &y {
        assert_eq!(*val, 0.0);
        assert!(!val.is_nan());
        assert!(!val.is_infinite());
    }

    // Tensor version
    let x_t = Tensor::zeros(&[1, d]).unwrap();
    let gamma_t = Tensor::from_vec(&[d], vec![1.0; d]).unwrap();
    let y_t = apply_qk_norm_head(&x_t, &gamma_t, eps as f32).unwrap();
    for &val in y_t.data() {
        assert_eq!(val, 0.0);
        assert!(!val.is_nan());
        assert!(!val.is_infinite());
    }
}

#[test]
fn test_x3_vjp_finite_differences() {
    let d = 32;
    let eps = 1e-5f64;
    let h = 1e-6f64;

    // Fixed synthetic x and gamma
    let x: Vec<f64> = (0..d).map(|i| ((i as f64 + 1.0) * 0.1).sin()).collect();
    let gamma: Vec<f64> = (0..d).map(|i| 0.8 + 0.05 * (i as f64).cos()).collect();
    let dy: Vec<f64> = (0..d).map(|i| ((i as f64 + 2.0) * 0.3).cos()).collect();

    let (vjp_x, vjp_gamma) = rms_norm_vjp_f64(&x, &gamma, &dy, eps);

    // Finite differences check for x
    for i in 0..d {
        let mut x_plus = x.clone();
        let mut x_minus = x.clone();
        x_plus[i] += h;
        x_minus[i] -= h;

        let y_plus = rms_norm_row_f64(&x_plus, &gamma, eps);
        let y_minus = rms_norm_row_f64(&x_minus, &gamma, eps);

        let dot_plus: f64 = y_plus.iter().zip(&dy).map(|(a, b)| a * b).sum();
        let dot_minus: f64 = y_minus.iter().zip(&dy).map(|(a, b)| a * b).sum();
        let num_grad = (dot_plus - dot_minus) / (2.0 * h);

        let diff = (vjp_x[i] - num_grad).abs();
        assert!(
            diff < 1e-5,
            "x[{i}] vjp analytic {} vs num {}, diff {}",
            vjp_x[i],
            num_grad,
            diff
        );
    }

    // Finite differences check for gamma
    for i in 0..d {
        let mut g_plus = gamma.clone();
        let mut g_minus = gamma.clone();
        g_plus[i] += h;
        g_minus[i] -= h;

        let y_plus = rms_norm_row_f64(&x, &g_plus, eps);
        let y_minus = rms_norm_row_f64(&x, &g_minus, eps);

        let dot_plus: f64 = y_plus.iter().zip(&dy).map(|(a, b)| a * b).sum();
        let dot_minus: f64 = y_minus.iter().zip(&dy).map(|(a, b)| a * b).sum();
        let num_grad = (dot_plus - dot_minus) / (2.0 * h);

        let diff = (vjp_gamma[i] - num_grad).abs();
        assert!(
            diff < 1e-5,
            "gamma[{i}] vjp analytic {} vs num {}, diff {}",
            vjp_gamma[i],
            num_grad,
            diff
        );
    }
}

#[test]
fn test_x3_softmax_entropy_stability_under_scaling() {
    let dh = 32;
    let seq_len = 8;
    let inv_sqrt_dh = 1.0f32 / (dh as f32).sqrt();

    // Baseline vectors with normal scale
    let q: Vec<f32> = (0..dh).map(|i| (i as f32 * 0.1).sin()).collect();
    let k_seq: Vec<Vec<f32>> = (0..seq_len)
        .map(|t| (0..dh).map(|i| ((i + t) as f32 * 0.2).cos()).collect())
        .collect();

    let gamma = Tensor::from_vec(&[dh], vec![1.0; dh]).unwrap();

    // 1. Unnormalized dot product under 50x scaling
    let scale = 50.0f32;
    let logits_unnorm_base: Vec<f32> = k_seq
        .iter()
        .map(|k| q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * inv_sqrt_dh)
        .collect();
    let logits_unnorm_scaled: Vec<f32> = k_seq
        .iter()
        .map(|k| {
            let q_s: Vec<f32> = q.iter().map(|v| v * scale).collect();
            let k_s: Vec<f32> = k.iter().map(|v| v * scale).collect();
            q_s.iter().zip(&k_s).map(|(a, b)| a * b).sum::<f32>() * inv_sqrt_dh
        })
        .collect();

    let h_unnorm_base = softmax_entropy(&logits_unnorm_base);
    let h_unnorm_scaled = softmax_entropy(&logits_unnorm_scaled);
    // Unnormalized scaling shrinks entropy significantly (towards 0)
    assert!(h_unnorm_base > 1.0);
    assert!(
        (h_unnorm_base - h_unnorm_scaled).abs() > 0.5,
        "Unnormalized attention entropy must collapse under 50x scale: base {h_unnorm_base}, scaled {h_unnorm_scaled}"
    );

    // 2. QK-normalized dot product under 50x scaling
    let eps = 1e-5f32;
    let q_t = Tensor::from_vec(&[1, dh], q.clone()).unwrap();
    let q_normed = apply_qk_norm_head(&q_t, &gamma, eps).unwrap();

    let mut k_normed_seq = Vec::new();
    for k in &k_seq {
        let kt = Tensor::from_vec(&[1, dh], k.clone()).unwrap();
        k_normed_seq.push(apply_qk_norm_head(&kt, &gamma, eps).unwrap());
    }

    let logits_qk_base: Vec<f32> = k_normed_seq
        .iter()
        .map(|kt| {
            q_normed
                .data()
                .iter()
                .zip(kt.data())
                .map(|(a, b)| a * b)
                .sum::<f32>()
                * inv_sqrt_dh
        })
        .collect();

    // Now with 50x scaled inputs
    let q_t_scaled = Tensor::from_vec(&[1, dh], q.iter().map(|v| v * scale).collect()).unwrap();
    let q_normed_scaled = apply_qk_norm_head(&q_t_scaled, &gamma, eps).unwrap();

    let mut k_normed_scaled_seq = Vec::new();
    for k in &k_seq {
        let kt = Tensor::from_vec(&[1, dh], k.iter().map(|v| v * scale).collect()).unwrap();
        k_normed_scaled_seq.push(apply_qk_norm_head(&kt, &gamma, eps).unwrap());
    }

    let logits_qk_scaled: Vec<f32> = k_normed_scaled_seq
        .iter()
        .map(|kt| {
            q_normed_scaled
                .data()
                .iter()
                .zip(kt.data())
                .map(|(a, b)| a * b)
                .sum::<f32>()
                * inv_sqrt_dh
        })
        .collect();

    let h_qk_base = softmax_entropy(&logits_qk_base);
    let h_qk_scaled = softmax_entropy(&logits_qk_scaled);
    let delta_h = (h_qk_base - h_qk_scaled).abs();

    assert!(
        delta_h <= 1e-3,
        "QK-norm attention entropy must remain stable under 50x scale: delta {delta_h} > 1e-3"
    );
}

#[test]
fn test_x3_incremental_decode_parity() {
    let cfg = Config::qk_norm_v2();
    let model = Model::init(&cfg).unwrap();
    let prompt: Vec<u32> = vec![12, 45, 99, 3];

    let full_logits = model.forward(&prompt, None).unwrap();

    // Cache-based decode
    let mut cache = KVCache::new(&cfg, model.fingerprint(), 16).unwrap();
    let mut last_step = None;

    for (pos, &token) in prompt.iter().enumerate() {
        last_step = Some(forward_step(&model, token, pos, &mut cache).unwrap());
    }

    let rows = full_logits.shape()[0];
    let cols = full_logits.shape()[1];
    let full_last_row = &full_logits.data()[(rows - 1) * cols..rows * cols];
    let step_row = last_step.unwrap().data().to_vec();

    for (i, (&f, &s)) in full_last_row.iter().zip(step_row.iter()).enumerate() {
        let diff = (f - s).abs();
        assert!(
            diff < 1e-4,
            "token {i} mismatch: full {f} vs cache {s}, diff {diff}"
        );
    }
}

#[test]
fn test_x3_checkpoint_serialization_roundtrip() {
    let cfg = Config::qk_norm_v2();
    let model = Model::init(&cfg).unwrap();
    let prompt: Vec<u32> = vec![7, 22, 105];

    let orig_logits = model.forward(&prompt, None).unwrap();

    let bytes = serialize(&model);
    let reloaded = parse(&bytes, &cfg, 64 * 1024 * 1024).unwrap();
    let reloaded_logits = reloaded.forward(&prompt, None).unwrap();

    assert!(reloaded.cfg.ext_x3_qk_norm);
    assert_eq!(orig_logits.data(), reloaded_logits.data());

    // Isolation check against reference config (missing q_norm / k_norm weights)
    let cfg_ref = Config::reference_v2();
    assert!(matches!(
        parse(&bytes, &cfg_ref, 64 * 1024 * 1024),
        Err(CheckpointError::ConfigMismatch(_)) | Err(CheckpointError::InventoryMismatch(_))
    ));
}

#[test]
fn test_x3_gate_gx3_execution() {
    let rep = gx3();
    assert_eq!(rep.gate_id, "GX3");
    assert_eq!(rep.status, "PASS", "Gate GX3 must pass: {:?}", rep.reason);

    let dispatched = gates::run(&["gx3"]).unwrap();
    assert_eq!(dispatched.len(), 1);
    assert_eq!(dispatched[0].status, "PASS");
}

#[test]
fn test_x3_autodiff_loss_and_grad() {
    let mut cfg = Config::qk_norm_v2();
    cfg.chunk_size = 2;
    let model = Model::init(&cfg).unwrap();
    let p = sandy_rust::autodiff::Params::<f32>::from_model(&model);
    let mut grads_mono = p.zeros_like();
    let mut grads_chunk = p.zeros_like();
    let window = vec![12u32, 45, 99, 3, 18];

    let out_mono = sandy_rust::autodiff::loss_and_grad(&cfg, &p, &window, 1.0, &mut grads_mono)
        .expect("monolithic loss_and_grad with QK-Norm");
    let out_chunk = sandy_rust::loss::chunked_loss_and_grad(&cfg, &p, &window, 1.0, &mut grads_chunk)
        .expect("chunked loss_and_grad with QK-Norm");

    let fwd_loss = sandy_rust::loss::chunked_forward_loss(&cfg, &p, &window)
        .expect("chunked forward loss with QK-Norm");

    assert!((out_mono.loss_sum - out_chunk.loss_sum).abs() < 1e-5);
    assert!((out_chunk.loss_sum - fwd_loss.loss_sum).abs() < 1e-5);

    for (m, c) in grads_mono.data.iter().zip(&grads_chunk.data) {
        for (&a, &b) in m.iter().zip(c) {
            assert!((a - b).abs() < 1e-4, "grad mismatch: mono={a}, chunk={b}");
        }
    }
}
