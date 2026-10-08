//! Integration tests for Extension X1: Grouped-Query Attention (GQA / MQA) and Gate GX1.

use sandy_rust::attention::{attention_gqa, gx1, repeat_kv_heads};
use sandy_rust::cache::KVCache;
use sandy_rust::checkpoint::{parse, serialize, CheckpointError, DEFAULT_MAX_BYTES};
use sandy_rust::config::{Config, ConfigError};
use sandy_rust::gates;
use sandy_rust::model::Model;
use sandy_rust::tensor::Tensor;

#[test]
fn test_x1_config_validation_and_rejections() {
    // Baseline reference: ext_x1_gqa=false, n_kv_heads=4
    let ref_cfg = Config::reference_v2();
    assert!(!ref_cfg.ext_x1_gqa);
    assert_eq!(ref_cfg.n_q_heads, 4);
    assert_eq!(ref_cfg.n_kv_heads, 4);
    assert!(ref_cfg.validate().is_ok());

    // Without extension enabled, n_kv_heads != n_q_heads is rejected
    let mut bad_cfg = Config::reference_v2();
    bad_cfg.n_kv_heads = 2;
    assert_eq!(
        bad_cfg.validate(),
        Err(ConfigError::ExtensionNotEnabled(
            "n_kv_heads != n_q_heads is extension X1 (GQA/MQA), disabled in baseline"
        ))
    );

    // GQA profile: n_kv_heads = 2
    let gqa_cfg = Config::gqa_v2();
    assert!(gqa_cfg.ext_x1_gqa);
    assert_eq!(gqa_cfg.n_q_heads, 4);
    assert_eq!(gqa_cfg.n_kv_heads, 2);
    assert!(gqa_cfg.validate().is_ok());

    // MQA profile: n_kv_heads = 1
    let mqa_cfg = Config::mqa_v2();
    assert!(mqa_cfg.ext_x1_gqa);
    assert_eq!(mqa_cfg.n_q_heads, 4);
    assert_eq!(mqa_cfg.n_kv_heads, 1);
    assert!(mqa_cfg.validate().is_ok());

    // Indivisible head configuration rejected
    let mut indivisible = Config::gqa_v2();
    indivisible.n_kv_heads = 3;
    assert_eq!(
        indivisible.validate(),
        Err(ConfigError::Divisibility("n_q_heads % n_kv_heads"))
    );
}

#[test]
fn test_x1_parameter_counts_exact() {
    let mha = Config::reference_v2();
    let gqa = Config::gqa_v2();
    let mqa = Config::mqa_v2();

    // Invariants from spec §2.3 and §4.2 Item 1:
    assert_eq!(mha.param_count(), Some(886_272));
    assert_eq!(gqa.param_count(), Some(820_736));
    assert_eq!(mqa.param_count(), Some(787_968));

    // Also verify physical Model tensor allocations match formula
    let model_mha = Model::init(&mha).unwrap();
    let model_gqa = Model::init(&gqa).unwrap();
    let model_mqa = Model::init(&mqa).unwrap();

    assert_eq!(model_mha.param_count(), 886_272);
    assert_eq!(model_gqa.param_count(), 820_736);
    assert_eq!(model_mqa.param_count(), 787_968);
}

#[test]
fn test_x1_repeat_kv_forward_equivalence() {
    let t_len = 6;
    let dh = 32;
    let h_q = 4;

    for &h_kv in &[2, 1] {
        let group = h_q / h_kv;
        let mut q = Tensor::zeros(&[t_len, h_q * dh]).unwrap();
        let mut k = Tensor::zeros(&[t_len, h_kv * dh]).unwrap();
        let mut v = Tensor::zeros(&[t_len, h_kv * dh]).unwrap();

        for (i, x) in q.data_mut().iter_mut().enumerate() {
            *x = ((i as f32 * 0.19).cos()) * 0.4;
        }
        for (i, x) in k.data_mut().iter_mut().enumerate() {
            *x = ((i as f32 * 0.27).sin()) * 0.4;
        }
        for (i, x) in v.data_mut().iter_mut().enumerate() {
            *x = ((i as f32 * 0.13).cos()) * 0.4;
        }

        let q_valid = vec![true; t_len];
        let k_valid = vec![true; t_len];

        let (gqa_out, _) = attention_gqa(&q, &k, &v, &q_valid, &k_valid, 0, h_q, h_kv).unwrap();
        let rep_k = repeat_kv_heads(&k, group, dh).unwrap();
        let rep_v = repeat_kv_heads(&v, group, dh).unwrap();
        let (mha_rep_out, _) = attention_gqa(&q, &rep_k, &rep_v, &q_valid, &k_valid, 0, h_q, h_q).unwrap();

        for (a, b) in gqa_out.data().iter().zip(mha_rep_out.data().iter()) {
            assert!((a - b).abs() <= 1e-6, "GQA and repeat-KV MHA oracle must match");
        }
    }
}

#[test]
fn test_x1_physical_cache_allocation() {
    let mha = Config::reference_v2();
    let gqa = Config::gqa_v2();
    let mqa = Config::mqa_v2();

    let model_mha = Model::init(&mha).unwrap();
    let model_gqa = Model::init(&gqa).unwrap();
    let model_mqa = Model::init(&mqa).unwrap();

    let cache_mha = KVCache::new(&mha, model_mha.fingerprint(), 1).unwrap();
    let cache_gqa = KVCache::new(&gqa, model_gqa.fingerprint(), 2).unwrap();
    let cache_mqa = KVCache::new(&mqa, model_mqa.fingerprint(), 3).unwrap();

    // Shape per layer: [max_seq_len, H_kv * dh]
    assert_eq!(cache_mha.k[0].shape(), &[64, 4 * 32]);
    assert_eq!(cache_gqa.k[0].shape(), &[64, 2 * 32]);
    assert_eq!(cache_mqa.k[0].shape(), &[64, 32]);

    assert_eq!(cache_gqa.k[0].len(), cache_mha.k[0].len() / 2);
    assert_eq!(cache_mqa.k[0].len(), cache_mha.k[0].len() / 4);
}

#[test]
fn test_x1_cached_incremental_decode_parity() {
    let prompt: Vec<u32> = vec![12, 45, 99, 103, 7];

    for cfg in [Config::gqa_v2(), Config::mqa_v2()] {
        let model = Model::init(&cfg).unwrap();
        let full_out = model.forward(&prompt, None).unwrap();

        let mut cache = KVCache::new(&cfg, model.fingerprint(), 42).unwrap();
        let mut last_step = None;
        for (pos, &tok) in prompt.iter().enumerate() {
            last_step = Some(sandy_rust::cache::forward_step(&model, tok, pos, &mut cache).unwrap());
        }
        let step_out = last_step.unwrap();

        let (rows, cols) = (full_out.shape()[0], full_out.shape()[1]);
        let last_full_row = &full_out.data()[(rows - 1) * cols..rows * cols];
        let step_row = step_out.data();

        for (&a, &b) in last_full_row.iter().zip(step_row.iter()) {
            assert!(
                (a - b).abs() <= 1e-4,
                "Incremental cached decode must match full un-cached forward within 1e-4"
            );
        }
    }
}

#[test]
fn test_x1_autodiff_gradient_accumulation_and_step() {
    let train_tokens: Vec<u32> = vec![1, 5, 10, 15, 20];

    for cfg in [Config::gqa_v2(), Config::mqa_v2()] {
        let model = Model::init(&cfg).unwrap();
        let mut params = sandy_rust::autodiff::Params::<f32>::from_model(&model);
        let mut grads = params.zeros_like();
        let loss_out = sandy_rust::autodiff::loss_and_grad(&cfg, &params, &train_tokens, 1.0f32, &mut grads).unwrap();
        let loss = loss_out.loss_sum;

        assert!(loss.is_finite() && loss > 0.0);
        assert!(grads.all_finite(), "GQA backward gradients must be finite");

        let initial_wk = params.data[3].clone(); // layer 0 wk
        let mut opt = sandy_rust::optimizer::AdamW::new(&params);
        let updated = opt.update(&mut params, &grads, 1e-3).unwrap();
        assert!(updated);

        assert_ne!(
            params.data[3],
            initial_wk,
            "Wk weights must update during training step"
        );
    }
}

#[test]
fn test_x1_checkpoint_isolation_and_reload() {
    let gqa_cfg = Config::gqa_v2();
    let mha_cfg = Config::reference_v2();
    let mqa_cfg = Config::mqa_v2();

    let model = Model::init(&gqa_cfg).unwrap();
    let bytes = serialize(&model);

    // Mismatched H_kv architectures must be rejected
    assert!(matches!(
        parse(&bytes, &mha_cfg, DEFAULT_MAX_BYTES),
        Err(CheckpointError::ArchMismatch)
    ));
    assert!(matches!(
        parse(&bytes, &mqa_cfg, DEFAULT_MAX_BYTES),
        Err(CheckpointError::ArchMismatch)
    ));

    // Matching architecture reloads successfully with bitwise identical weights
    let reloaded = parse(&bytes, &gqa_cfg, DEFAULT_MAX_BYTES).unwrap();
    assert_eq!(reloaded.fingerprint(), model.fingerprint());

    let test_prompt = vec![10, 20, 30];
    let out_orig = model.forward(&test_prompt, None).unwrap();
    let out_reloaded = reloaded.forward(&test_prompt, None).unwrap();
    assert_eq!(out_orig.data(), out_reloaded.data());
}

#[test]
fn test_gx1_gate_passes() {
    let rep = gx1();
    assert_eq!(rep.gate_id, "GX1");
    assert_eq!(rep.scope, "acceptance");
    assert_eq!(rep.status, "PASS");
    assert!(rep.reason.is_none());

    // Also run via gate dispatcher
    let reps = gates::run(&["gx1"]).unwrap();
    assert_eq!(reps.len(), 1);
    assert_eq!(reps[0].status, "PASS");
}
