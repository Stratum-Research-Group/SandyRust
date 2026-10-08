//! Integration test suite for Extension X2: Untied Embeddings & Gate GX2 (§4.2 Item 2, §9.2 GX2).

use sandy_rust::autodiff::{gradcheck, loss_and_grad, Params};
use sandy_rust::cache::{forward_step, KVCache};
use sandy_rust::checkpoint::{parse, serialize, CheckpointError};
use sandy_rust::config::{Config, ConfigError};
use sandy_rust::embedding::gx2;
use sandy_rust::gates;
use sandy_rust::model::Model;
use sandy_rust::optimizer::{AdamW, Schedule};

#[test]
fn test_x2_config_validation_and_rejections() {
    let mut c = Config::reference_v2();
    assert!(c.validate().is_ok());

    // Without extension flag, tied_embeddings=false is rejected (Gate G6)
    c.tied_embeddings = false;
    assert!(matches!(c.validate(), Err(ConfigError::ExtensionNotEnabled(_))));

    // With extension flag, tied_embeddings=false is accepted
    c.ext_x2_untied_embeddings = true;
    assert!(c.validate().is_ok());

    // Helper constructors validate cleanly
    assert!(Config::untied_v2().validate().is_ok());
    assert!(Config::untied_gqa_v2().validate().is_ok());
    assert!(Config::untied_mqa_v2().validate().is_ok());
}

#[test]
fn test_x2_parameter_counts_exact() {
    let c_tied_mha = Config::reference_v2();
    let c_untied_mha = Config::untied_v2();
    let c_tied_gqa = Config::gqa_v2();
    let c_untied_gqa = Config::untied_gqa_v2();
    let c_tied_mqa = Config::mqa_v2();
    let c_untied_mqa = Config::untied_mqa_v2();

    assert_eq!(c_tied_mha.param_count(), Some(886_272));
    assert_eq!(c_untied_mha.param_count(), Some(919_424));
    assert_eq!(c_tied_gqa.param_count(), Some(820_736));
    assert_eq!(c_untied_gqa.param_count(), Some(853_888));
    assert_eq!(c_tied_mqa.param_count(), Some(787_968));
    assert_eq!(c_untied_mqa.param_count(), Some(821_120));

    let delta = 259 * 128; // vocab_size * d_model
    assert_eq!(c_untied_mha.param_count().unwrap() - c_tied_mha.param_count().unwrap(), delta);
    assert_eq!(c_untied_gqa.param_count().unwrap() - c_tied_gqa.param_count().unwrap(), delta);
    assert_eq!(c_untied_mqa.param_count().unwrap() - c_tied_mqa.param_count().unwrap(), delta);
}

#[test]
fn test_x2_buffer_separation_and_identity() {
    let cfg = Config::untied_v2();
    let model = Model::init(&cfg).unwrap();

    let head = model.lm_head.as_ref().expect("lm_head must be present on untied model");
    assert_ne!(
        model.embed.data().as_ptr(),
        head.data().as_ptr(),
        "Embedding and LM head buffers must be completely distinct physical allocations"
    );

    let ts = model.tensors();
    let names: Vec<&str> = ts.iter().map(|(n, _)| n.as_str()).collect();
    assert!(names.contains(&"embed"));
    assert!(names.contains(&"lm_head"));
    assert_eq!(names[0], "embed");
    assert_eq!(names[names.len() - 1], "lm_head");

    let tied_model = Model::init(&Config::reference_v2()).unwrap();
    assert!(tied_model.lm_head.is_none());
}

#[test]
fn test_x2_parameter_perturbation_independence() {
    let cfg = Config::untied_v2();
    let mut model = Model::init(&cfg).unwrap();

    let orig_head_data = model.lm_head.as_ref().unwrap().data().to_vec();
    let orig_embed_data = model.embed.data().to_vec();

    // Perturb embedding table
    model.embed.data_mut()[10] += 5.0;
    assert_eq!(
        model.lm_head.as_ref().unwrap().data(),
        orig_head_data.as_slice(),
        "Mutating embed table must NOT alter lm_head"
    );

    // Perturb lm_head
    model.lm_head.as_mut().unwrap().data_mut()[20] += 7.0;
    assert_ne!(
        model.embed.data(),
        orig_embed_data.as_slice(),
        "Embedding contains its earlier perturbation"
    );
    assert_eq!(
        model.embed.data()[0..10],
        orig_embed_data[0..10],
        "Mutating lm_head must NOT alter embed table"
    );
}

#[test]
fn test_x2_autodiff_gradients_and_moments() {
    let cfg = Config::untied_v2();
    let model = Model::init(&cfg).unwrap();
    let prompt: Vec<u32> = vec![5, 6, 7, 8];

    let mut params = Params::<f32>::from_model(&model);
    let mut grads = params.zeros_like();

    let loss_out = loss_and_grad(&cfg, &params, &prompt, 1.0, &mut grads).unwrap();
    assert!(loss_out.loss_sum.is_finite());

    let d = cfg.d_model;
    let head_idx = 2 + 9 * cfg.n_layers;

    // Token 100 was not in prompt -> dE row 100 must be zero
    let unused_token = 100usize;
    for &g in &grads.data[0][unused_token * d..(unused_token + 1) * d] {
        assert_eq!(g, 0.0);
    }

    // Token 5 was in prompt -> dE row 5 must have non-zero gradient
    let used_token = prompt[0] as usize;
    let used_sum: f32 = grads.data[0][used_token * d..(used_token + 1) * d].iter().map(|x| x.abs()).sum();
    assert!(used_sum > 1e-6);

    // lm_head must have accumulated non-zero gradient
    let head_sum: f32 = grads.data[head_idx].iter().map(|x| x.abs()).sum();
    assert!(head_sum > 1e-6);

    // Finite difference check
    let eps_list = [1e-4];
    let params64 = Params::<f64>::from_model(&model);
    let gc = gradcheck(&cfg, &params64, &prompt, &eps_list, 1e-3, 1e-3, 2).unwrap();
    assert_eq!(gc.failures, 0);
    assert!(gc.max_abs_err <= 1e-3, "max_abs_err={:.2e}", gc.max_abs_err);

    // AdamW dedicated moments
    let mut opt = AdamW::new(&params);
    let sched = Schedule {
        eta_max: 1e-3,
        eta_min: 1e-5,
        warmup: 1,
        max_steps: 5,
    };
    let updated = opt.update(&mut params, &grads, sched.eta(1)).unwrap();
    assert!(updated);
    assert_eq!(opt.m.len(), params.data.len());
    assert_eq!(opt.v.len(), params.data.len());
    assert!(opt.m[0].iter().any(|&x| x != 0.0));
    assert!(opt.m[head_idx].iter().any(|&x| x != 0.0));
}

#[test]
fn test_x2_checkpoint_isolation_and_reload() {
    let cfg_untied = Config::untied_v2();
    let cfg_tied = Config::reference_v2();

    let model_untied = Model::init(&cfg_untied).unwrap();
    let prompt = vec![1, 2, 3, 4, 5];
    let orig_logits = model_untied.forward(&prompt, None).unwrap();

    let bytes = serialize(&model_untied);
    let reloaded = parse(&bytes, &cfg_untied, 64 * 1024 * 1024).unwrap();
    let reloaded_logits = reloaded.forward(&prompt, None).unwrap();
    assert_eq!(orig_logits.data(), reloaded_logits.data());

    // Cross-loading rejection
    assert!(matches!(
        parse(&bytes, &cfg_tied, 64 * 1024 * 1024),
        Err(CheckpointError::ArchMismatch)
    ));

    let model_tied = Model::init(&cfg_tied).unwrap();
    let tied_bytes = serialize(&model_tied);
    assert!(matches!(
        parse(&tied_bytes, &cfg_untied, 64 * 1024 * 1024),
        Err(CheckpointError::ArchMismatch)
    ));
}

#[test]
fn test_x2_cached_incremental_decode_parity() {
    let cfg = Config::untied_v2();
    let model = Model::init(&cfg).unwrap();
    let prompt: Vec<u32> = (10..18).collect();

    let full_logits = model.forward(&prompt, None).unwrap();
    let mut cache = KVCache::new(&cfg, model.fingerprint(), 32).unwrap();

    let mut last_step = None;
    for (pos, &tok) in prompt.iter().enumerate() {
        last_step = Some(forward_step(&model, tok, pos, &mut cache).unwrap());
    }

    let rows = full_logits.shape()[0];
    let cols = full_logits.shape()[1];
    let full_last_row = &full_logits.data()[(rows - 1) * cols..rows * cols];
    let step_row = last_step.unwrap().data().to_vec();

    for (i, (&a, &b)) in full_last_row.iter().zip(step_row.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(diff <= 1e-4, "Mismatch at col {i}: full={a}, step={b}, diff={diff}");
    }
}

#[test]
fn test_gx2_gate_passes() {
    let rep = gx2();
    assert_eq!(rep.gate_id, "GX2");
    assert_eq!(rep.status, "PASS", "Gate GX2 failed: {:?}", rep.reason);

    let dispatched = gates::run(&["gx2"]).unwrap();
    assert_eq!(dispatched.len(), 1);
    assert_eq!(dispatched[0].status, "PASS");
}
