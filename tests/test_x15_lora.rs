#![allow(clippy::needless_range_loop)]

//! Test suite for Gate GX15: FP32 Low-Rank Adaptation (LoRA) (§A5).

use sandy_rust::adapter::{gx15, AdapterError, AdapterTarget, LoraAdapter, LoraMatrix};
use sandy_rust::config::Config;
use sandy_rust::gates;
use sandy_rust::model::Model;
use sandy_rust::rng::Rng;
use std::fs;

#[test]
fn test_gx15_gate_passes() {
    let rep = gx15();
    assert_eq!(rep.status, "PASS", "GX15 must pass: {}", rep.observed);
}

#[test]
fn test_x15_zero_init_parity() {
    let cfg = Config::reference_v2();
    let base_model = Model::init(&cfg).unwrap();
    let base_hash = base_model.fingerprint();
    let tok_fp = [42u8; 32];

    let targets = [AdapterTarget::Wq, AdapterTarget::Wk, AdapterTarget::Wv, AdapterTarget::Wo];
    let adapter = LoraAdapter::new(&cfg, 4, 8.0, &targets, 12345, base_hash, tok_fp).unwrap();

    let merged_model = adapter.merge(&base_model).unwrap();

    let tokens = [256u32, 65, 66, 67, 257];
    let base_out = base_model.forward(&tokens, None).unwrap();
    let merged_out = merged_model.forward(&tokens, None).unwrap();

    assert_eq!(base_out, merged_out, "At step 0 (B=0), adapted model must equal base model bitwise");
}

#[test]
fn test_x15_analytic_gradients_match_finite_differences() {
    let mut rng = Rng::new(1337);
    let eps = 1e-4f32;

    for &r in &[1, 2, 4] {
        let d_in = 12;
        let d_out = 10;
        let mut mat = LoraMatrix::new(AdapterTarget::Wq, d_out, d_in, r, 4.0, &mut rng);
        for v in &mut mat.b {
            *v = rng.next_normal() * 0.2;
        }

        let x: Vec<f32> = (0..d_in).map(|_| rng.next_normal()).collect();
        let dy: Vec<f32> = (0..d_out).map(|_| rng.next_normal()).collect();

        let (d_a_ana, d_b_ana, dx_ana) = mat.backward(&x, &dy).unwrap();

        // Check dB
        for i in 0..mat.b.len() {
            let orig = mat.b[i];
            mat.b[i] = orig + eps;
            let y_p = mat.forward(&x).unwrap();
            mat.b[i] = orig - eps;
            let y_m = mat.forward(&x).unwrap();
            mat.b[i] = orig;

            let num: f32 = dy.iter().zip(&y_p).zip(&y_m).map(|((&g, &p), &m)| g * (p - m) / (2.0 * eps)).sum();
            assert!((num - d_b_ana[i]).abs() < 1e-3, "dB mismatch at index {i} for rank {r}");
        }

        // Check dA
        for i in 0..mat.a.len() {
            let orig = mat.a[i];
            mat.a[i] = orig + eps;
            let y_p = mat.forward(&x).unwrap();
            mat.a[i] = orig - eps;
            let y_m = mat.forward(&x).unwrap();
            mat.a[i] = orig;

            let num: f32 = dy.iter().zip(&y_p).zip(&y_m).map(|((&g, &p), &m)| g * (p - m) / (2.0 * eps)).sum();
            assert!((num - d_a_ana[i]).abs() < 1e-3, "dA mismatch at index {i} for rank {r}");
        }

        // Check dx
        for i in 0..x.len() {
            let mut x_p = x.clone();
            let mut x_m = x.clone();
            x_p[i] += eps;
            x_m[i] -= eps;
            let y_p = mat.forward(&x_p).unwrap();
            let y_m = mat.forward(&x_m).unwrap();

            let num: f32 = dy.iter().zip(&y_p).zip(&y_m).map(|((&g, &p), &m)| g * (p - m) / (2.0 * eps)).sum();
            assert!((num - dx_ana[i]).abs() < 1e-3, "dx mismatch at index {i} for rank {r}");
        }
    }
}

#[test]
fn test_x15_frozen_base_invariant() {
    let cfg = Config::reference_v2();
    let base_model = Model::init(&cfg).unwrap();
    let base_hash_orig = base_model.fingerprint();

    let targets = [AdapterTarget::Wgate, AdapterTarget::Wup, AdapterTarget::Wdown];
    let mut adapter = LoraAdapter::new(&cfg, 2, 4.0, &targets, 999, base_hash_orig, [0u8; 32]).unwrap();

    // Mutate adapter
    for l in &mut adapter.layers {
        for m in l.values_mut() {
            for v in &mut m.b {
                *v = 0.123;
            }
        }
    }

    // Merge onto a clone of base
    let _ = adapter.merge(&base_model).unwrap();

    // Verify original base model remains completely unchanged
    assert_eq!(base_model.fingerprint(), base_hash_orig, "Base model weights must remain frozen and unmodified");
}

#[test]
fn test_x15_merge_equivalence() {
    let cfg = Config::reference_v2();
    let base_model = Model::init(&cfg).unwrap();
    let base_hash = base_model.fingerprint();

    let targets = [AdapterTarget::Wq, AdapterTarget::Wv];
    let mut adapter = LoraAdapter::new(&cfg, 4, 8.0, &targets, 777, base_hash, [0u8; 32]).unwrap();

    for l in &mut adapter.layers {
        for m in l.values_mut() {
            for v in &mut m.b {
                *v = 0.05;
            }
        }
    }

    let merged_model = adapter.merge(&base_model).unwrap();

    // For single linear layer, verify merged output matches base + delta
    let l_base = &base_model.layers[0];
    let l_mat = &adapter.layers[0][&AdapterTarget::Wq];
    let x = vec![0.25f32; cfg.d_model];

    let x_t = sandy_rust::tensor::Tensor::from_vec(&[1, cfg.d_model], x.clone()).unwrap();
    let y_base = sandy_rust::tensor::ops::matmul(&x_t, &l_base.wq).unwrap();
    let y_delta = l_mat.forward(&x).unwrap();

    let l_merged = &merged_model.layers[0];
    let y_merged = sandy_rust::tensor::ops::matmul(&x_t, &l_merged.wq).unwrap();

    for i in 0..cfg.d_model {
        let expected = y_base.data()[i] + y_delta[i];
        let diff = (y_merged.data()[i] - expected).abs();
        assert!(diff < 1e-5, "Merged linear output must match base + delta at index {i}");
    }
}

#[test]
fn test_x15_header_and_tamper_rejection() {
    let cfg = Config::reference_v2();
    let base_model = Model::init(&cfg).unwrap();
    let base_hash = base_model.fingerprint();
    let tok_fp = [123u8; 32];

    let targets = [AdapterTarget::Wq];
    let adapter = LoraAdapter::new(&cfg, 2, 4.0, &targets, 111, base_hash, tok_fp).unwrap();

    let pid = std::process::id();
    let path_buf = std::env::temp_dir().join(format!("test_tamper_lora_{pid}.bin"));
    let path = path_buf.as_path();
    adapter.save(path).unwrap();

    // 1. Correct load succeeds
    let loaded = LoraAdapter::load(path, base_hash, tok_fp, &cfg, &targets).unwrap();
    assert_eq!(loaded.header.rank, 2);

    // 2. Base mismatch
    let wrong_base = [0x55u8; 32];
    assert!(matches!(
        LoraAdapter::load(path, wrong_base, tok_fp, &cfg, &targets),
        Err(AdapterError::BaseMismatch { ref field }) if field == "base_checkpoint_hash"
    ));

    // 3. Tokenizer mismatch
    let wrong_tok = [0x66u8; 32];
    assert!(matches!(
        LoraAdapter::load(path, base_hash, wrong_tok, &cfg, &targets),
        Err(AdapterError::BaseMismatch { ref field }) if field == "tokenizer_fingerprint"
    ));

    // 4. Corrupted payload
    let mut bytes = fs::read(path).unwrap();
    let len = bytes.len();
    bytes[len - 1] ^= 0x01; // flip 1 bit in weight payload
    let corrupt_path_buf = std::env::temp_dir().join(format!("test_corrupt_lora_{pid}.bin"));
    let corrupt_path = corrupt_path_buf.as_path();
    fs::write(corrupt_path, bytes).unwrap();

    assert!(matches!(
        LoraAdapter::load(corrupt_path, base_hash, tok_fp, &cfg, &targets),
        Err(AdapterError::BaseMismatch { ref field }) if field == "payload_hash"
    ));

    let _ = fs::remove_file(path);
    let _ = fs::remove_file(corrupt_path);
}

#[test]
fn test_x15_parameter_count_formula() {
    let cfg = Config::reference_v2();
    let base_hash = [0u8; 32];
    let tok_fp = [0u8; 32];

    // Case 1: Wq, Wv with rank 4 -> 4 layers * 2 targets * (4 * (128 + 128)) = 8 * 1024 = 8,192
    let a1 = LoraAdapter::new(&cfg, 4, 8.0, &[AdapterTarget::Wq, AdapterTarget::Wv], 1, base_hash, tok_fp).unwrap();
    assert_eq!(a1.header.parameter_count, 8192);

    // Case 2: Wgate, Wup, Wdown with rank 2:
    // Wgate: 2 * (128 + 384) = 1,024
    // Wup:   2 * (128 + 384) = 1,024
    // Wdown: 2 * (384 + 128) = 1,024
    // Per layer: 3,072. Across 4 layers: 12,288.
    let a2 = LoraAdapter::new(&cfg, 2, 4.0, &[AdapterTarget::Wgate, AdapterTarget::Wup, AdapterTarget::Wdown], 2, base_hash, tok_fp).unwrap();
    assert_eq!(a2.header.parameter_count, 12288);
}

#[test]
fn test_x15_gates_runner_integration() {
    let reports = gates::run(&["gx15"]).expect("gates::run succeeds");
    assert_eq!(reports.len(), 1);
    assert_eq!(reports[0].gate_id, "GX15");
    assert_eq!(reports[0].status, "PASS");
}

#[test]
fn test_x15_gqa_target_dims_and_merge() {
    let cfg = Config::gqa_v2();
    assert_eq!(cfg.n_q_heads, 4);
    assert_eq!(cfg.n_kv_heads, 2);
    let dh = cfg.head_dim();
    let q_dim = cfg.n_q_heads * dh;
    let kv_dim = cfg.n_kv_heads * dh;

    assert_eq!(AdapterTarget::Wq.dims(&cfg), (q_dim, cfg.d_model));
    assert_eq!(AdapterTarget::Wk.dims(&cfg), (kv_dim, cfg.d_model));
    assert_eq!(AdapterTarget::Wv.dims(&cfg), (kv_dim, cfg.d_model));
    assert_eq!(AdapterTarget::Wo.dims(&cfg), (cfg.d_model, q_dim));

    let model = Model::init(&cfg).unwrap();
    let targets = [AdapterTarget::Wq, AdapterTarget::Wk, AdapterTarget::Wv, AdapterTarget::Wo];
    let adapter = LoraAdapter::new(&cfg, 4, 8.0, &targets, 42, model.fingerprint(), [0u8; 32]).unwrap();
    let merged = adapter.merge(&model).unwrap();

    let tokens = [256u32, 65, 66, 67];
    let base_out = model.forward(&tokens, None).unwrap();
    let merged_out = merged.forward(&tokens, None).unwrap();
    assert_eq!(base_out, merged_out, "Zero-init GQA LoRA merge must bitwise match base model");
}

#[test]
fn test_x15_end_to_end_token_adaptation_decreases_loss() {
    let cfg = Config::reference_v2();
    let base_model = Model::init(&cfg).unwrap();
    let base_fp_before = base_model.fingerprint();

    // Sequence to adapt via LoRA
    let tokens = vec![256u32, 65, 66, 67, 68, 69, 70, 71, 72];
    let n_denom = (tokens.len() - 1) as f32;

    let base_params = sandy_rust::autodiff::Params::<f32>::from_model(&base_model);
    let mut initial_grads = sandy_rust::autodiff::Params::<f32>::zeros_like(&base_params);
    let initial_loss_out = sandy_rust::autodiff::loss_and_grad(&cfg, &base_params, &tokens, n_denom, &mut initial_grads).unwrap();
    let initial_loss = initial_loss_out.loss_sum / initial_loss_out.n_valid as f32;

    let targets = [
        AdapterTarget::Wq,
        AdapterTarget::Wk,
        AdapterTarget::Wv,
        AdapterTarget::Wo,
        AdapterTarget::Wgate,
        AdapterTarget::Wup,
        AdapterTarget::Wdown,
    ];
    let mut adapter = LoraAdapter::new(&cfg, 4, 8.0, &targets, 1234, base_fp_before, [0u8; 32]).unwrap();

    let lr = 0.05f32;
    for _step in 0..15 {
        let merged_model = adapter.merge(&base_model).unwrap();
        let current_params = sandy_rust::autodiff::Params::<f32>::from_model(&merged_model);
        let mut grads = sandy_rust::autodiff::Params::<f32>::zeros_like(&current_params);
        let _ = sandy_rust::autodiff::loss_and_grad(&cfg, &current_params, &tokens, n_denom, &mut grads).unwrap();

        // Accumulate gradient onto adapter
        for (l_idx, layer_adapter) in adapter.layers.iter_mut().enumerate() {
            let p_idx = sandy_rust::autodiff::layer_indices(&cfg, l_idx);
            for (&target, mat) in layer_adapter.iter_mut() {
                let grad_idx = match target {
                    AdapterTarget::Wq => p_idx.wq,
                    AdapterTarget::Wk => p_idx.wk,
                    AdapterTarget::Wv => p_idx.wv,
                    AdapterTarget::Wo => p_idx.wo,
                    AdapterTarget::Wgate => p_idx.w_gate,
                    AdapterTarget::Wup => p_idx.w_up,
                    AdapterTarget::Wdown => p_idx.w_down,
                };
                let dw = &grads.data[grad_idx];
                let s = mat.scale();
                let d_in = mat.d_in;
                let d_out = mat.d_out;
                let r = mat.r;

                // dB[i, k] = s * sum_j (dW[j * d_out + i] * A[k * d_in + j])
                let mut db = vec![0.0f32; d_out * r];
                for i in 0..d_out {
                    for k in 0..r {
                        let mut sum = 0.0f32;
                        for j in 0..d_in {
                            sum += dw[j * d_out + i] * mat.a[k * d_in + j];
                        }
                        db[i * r + k] = s * sum;
                    }
                }

                // dA[k, j] = s * sum_i (dW[j * d_out + i] * B[i * r + k])
                let mut da = vec![0.0f32; r * d_in];
                for k in 0..r {
                    for j in 0..d_in {
                        let mut sum = 0.0f32;
                        for i in 0..d_out {
                            sum += dw[j * d_out + i] * mat.b[i * r + k];
                        }
                        da[k * d_in + j] = s * sum;
                    }
                }

                for (v, &g) in mat.b.iter_mut().zip(&db) {
                    *v -= lr * g;
                }
                for (v, &g) in mat.a.iter_mut().zip(&da) {
                    *v -= lr * g;
                }
            }
        }
    }

    let final_merged = adapter.merge(&base_model).unwrap();
    let final_params = sandy_rust::autodiff::Params::<f32>::from_model(&final_merged);
    let mut final_grads = sandy_rust::autodiff::Params::<f32>::zeros_like(&final_params);
    let final_loss_out = sandy_rust::autodiff::loss_and_grad(&cfg, &final_params, &tokens, n_denom, &mut final_grads).unwrap();
    let final_loss = final_loss_out.loss_sum / final_loss_out.n_valid as f32;

    assert!(
        final_loss < initial_loss,
        "LoRA adaptation must strictly decrease loss: initial {initial_loss} vs final {final_loss}"
    );
    assert_eq!(
        base_model.fingerprint(),
        base_fp_before,
        "Base model must remain strictly frozen and unmodified throughout LoRA adaptation"
    );
}

