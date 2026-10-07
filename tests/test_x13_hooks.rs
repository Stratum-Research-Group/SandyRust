//! Test suite for Gate GX13: Activation Hooks & Bounded Steering (§A2.8).

use sandy_rust::cache::{forward_step_with_hooks, KVCache};
use sandy_rust::config::Config;
use sandy_rust::hooks::{
    gx13, AblateKind, ActivationSink, DirectionHeader, EditOp, HookContext, HookError, HookId,
    HookSite, RecordTiming, SteerPosition, DEFAULT_MAX_EXPORT_BYTES,
};
use sandy_rust::model::{Model, ModelError};

#[test]
fn test_gx13_gate_passes() {
    let rep = gx13();
    assert_eq!(rep.status, "PASS", "GX13 must pass: {}", rep.observed);
}

#[test]
fn test_gx13_item1_disabled_parity() {
    let cfg_base = Config::reference_v2();
    let m = Model::init(&cfg_base).unwrap();
    let toks = [256u32, 65, 66, 67, 257];

    let mut cfg_ext = cfg_base.clone();
    cfg_ext.ext_x13_hooks = false;
    assert_eq!(cfg_base.fingerprint(), cfg_ext.fingerprint());

    let out_plain = m.forward(&toks, None).unwrap();
    let out_hook_none = m.forward_with_hooks(&toks, None, None, None).unwrap();
    assert_eq!(out_plain, out_hook_none);
}

#[test]
fn test_gx13_item2_record_parity_all_sites() {
    let cfg = Config::reference_v2();
    let m = Model::init(&cfg).unwrap();
    let toks = [256u32, 72, 101, 108, 108, 111, 257];
    let base_out = m.forward(&toks, None).unwrap();

    let all_sites = [
        HookSite::EmbedOut,
        HookSite::ResidPre,
        HookSite::AttnNormOut,
        HookSite::Q,
        HookSite::KPreRope,
        HookSite::K,
        HookSite::V,
        HookSite::AttnPattern,
        HookSite::HeadOut,
        HookSite::AttnOut,
        HookSite::ResidMid,
        HookSite::MlpNormOut,
        HookSite::MlpHidden,
        HookSite::MlpOut,
        HookSite::ResidPost,
        HookSite::FinalNormOut,
        HookSite::Logits,
    ];

    let mut ctx = HookContext::new();
    for site in all_sites {
        let hook = if site.is_layer_site() {
            if matches!(
                site,
                HookSite::Q
                    | HookSite::HeadOut
                    | HookSite::AttnPattern
                    | HookSite::K
                    | HookSite::KPreRope
                    | HookSite::V
            ) {
                HookId::head(0, site, 0)
            } else {
                HookId::layer(0, site)
            }
        } else {
            HookId::global(site)
        };
        ctx = ctx.with_record(hook, RecordTiming::PostEdit);
    }

    let mut sink = ActivationSink::new(DEFAULT_MAX_EXPORT_BYTES).unwrap();
    let rec_out = m.forward_with_hooks(&toks, None, Some(&ctx), Some(&mut sink)).unwrap();

    // Record mode cannot alter forward output
    assert_eq!(rec_out, base_out);
    // Every registered site produced an activation entry
    assert_eq!(sink.records.len(), all_sites.len());
}

#[test]
fn test_gx13_item3_zero_direction_parity() {
    let cfg = Config::reference_v2();
    let m = Model::init(&cfg).unwrap();
    let toks = [256u32, 10, 20, 30, 257];
    let base_out = m.forward(&toks, None).unwrap();

    // alpha = 0.0
    let steer_zero_alpha = EditOp::Steer {
        hook: HookId::layer(0, HookSite::ResidPre),
        direction: vec![1.0f32; 128],
        alpha: 0.0,
        positions: SteerPosition::AllPositions,
        direction_id: "test".to_string(),
    };
    let ctx_a = HookContext::new().with_edit(steer_zero_alpha);
    let out_a = m.forward_with_hooks(&toks, None, Some(&ctx_a), None).unwrap();
    assert_eq!(out_a, base_out);

    // zero vector
    let steer_zero_vec = EditOp::Steer {
        hook: HookId::layer(1, HookSite::ResidMid),
        direction: vec![0.0f32; 128],
        alpha: 5.0,
        positions: SteerPosition::AllPositions,
        direction_id: "test_zero_vec".to_string(),
    };
    let ctx_b = HookContext::new().with_edit(steer_zero_vec);
    let out_b = m.forward_with_hooks(&toks, None, Some(&ctx_b), None).unwrap();
    assert_eq!(out_b, base_out);
}

#[test]
fn test_gx13_item4_ablate_and_patch() {
    let cfg = Config::reference_v2();
    let m = Model::init(&cfg).unwrap();
    let toks = [256u32, 80, 81, 82, 257];
    let base_out = m.forward(&toks, None).unwrap();

    // 1. Patch parity: record ResidPost of layer 1, then patch it in another forward run
    let mut sink = ActivationSink::new(DEFAULT_MAX_EXPORT_BYTES).unwrap();
    let rec_ctx = HookContext::new().with_record(HookId::layer(1, HookSite::ResidPost), RecordTiming::PostEdit);
    let _ = m.forward_with_hooks(&toks, None, Some(&rec_ctx), Some(&mut sink)).unwrap();
    let recorded = sink.get(&HookId::layer(1, HookSite::ResidPost), RecordTiming::PostEdit).unwrap().clone();

    let patch_ctx = HookContext::new().with_edit(EditOp::Patch {
        hook: HookId::layer(1, HookSite::ResidPost),
        source: recorded,
    });
    let patch_out = m.forward_with_hooks(&toks, None, Some(&patch_ctx), None).unwrap();
    assert_eq!(patch_out, base_out);

    // 2. Ablate zero: zeroing HeadOut 0 changes output
    let ablate_ctx = HookContext::new().with_edit(EditOp::Ablate {
        hook: HookId::head(0, HookSite::HeadOut, 0),
        kind: AblateKind::Zero,
    });
    let ablate_out = m.forward_with_hooks(&toks, None, Some(&ablate_ctx), None).unwrap();
    assert_ne!(ablate_out, base_out);
    assert!(ablate_out.all_finite());
}

#[test]
fn test_gx13_item5_bounds_and_safety_checks() {
    let cfg = Config::reference_v2();
    let m = Model::init(&cfg).unwrap();
    let toks = [256u32, 1, 2, 3];

    // Oversize export cap
    let mut tiny_sink = ActivationSink::new(8).unwrap();
    let rec_ctx = HookContext::new().with_record(HookId::global(HookSite::EmbedOut), RecordTiming::PostEdit);
    let err = m.forward_with_hooks(&toks, None, Some(&rec_ctx), Some(&mut tiny_sink));
    assert!(matches!(err, Err(ModelError::Hook(HookError::ExportTooLarge { .. }))));

    // Non-finite alpha
    let bad_alpha_ctx = HookContext::new().with_edit(EditOp::Steer {
        hook: HookId::layer(0, HookSite::ResidPre),
        direction: vec![1.0; 128],
        alpha: f32::NAN,
        positions: SteerPosition::AllPositions,
        direction_id: "nan".to_string(),
    });
    assert!(matches!(
        m.forward_with_hooks(&toks, None, Some(&bad_alpha_ctx), None),
        Err(ModelError::Hook(HookError::InvalidSteering { .. }))
    ));

    // Alpha exceeds alpha_max (default 8.0)
    let big_alpha_ctx = HookContext::new().with_edit(EditOp::Steer {
        hook: HookId::layer(0, HookSite::ResidPre),
        direction: vec![1.0; 128],
        alpha: 10.0,
        positions: SteerPosition::AllPositions,
        direction_id: "big".to_string(),
    });
    assert!(matches!(
        m.forward_with_hooks(&toks, None, Some(&big_alpha_ctx), None),
        Err(ModelError::Hook(HookError::InvalidSteering { .. }))
    ));

    // Direction wrong dimension
    let bad_dim_ctx = HookContext::new().with_edit(EditOp::Steer {
        hook: HookId::layer(0, HookSite::ResidPre),
        direction: vec![1.0; 64], // expected 128
        alpha: 1.0,
        positions: SteerPosition::AllPositions,
        direction_id: "bad_dim".to_string(),
    });
    assert!(matches!(
        m.forward_with_hooks(&toks, None, Some(&bad_dim_ctx), None),
        Err(ModelError::Hook(HookError::InvalidSteering { .. }))
    ));

    // Non-steerable site
    let non_steer_ctx = HookContext::new().with_edit(EditOp::Steer {
        hook: HookId::layer(0, HookSite::AttnNormOut), // only ResidPre/Mid/Post steerable
        direction: vec![1.0; 128],
        alpha: 1.0,
        positions: SteerPosition::AllPositions,
        direction_id: "non_steer".to_string(),
    });
    assert!(matches!(
        m.forward_with_hooks(&toks, None, Some(&non_steer_ctx), None),
        Err(ModelError::Hook(HookError::InvalidSteering { .. }))
    ));

    // Out of bounds layer
    let bad_layer_ctx = HookContext::new().with_edit(EditOp::Steer {
        hook: HookId::layer(10, HookSite::ResidPre),
        direction: vec![1.0; 128],
        alpha: 1.0,
        positions: SteerPosition::AllPositions,
        direction_id: "bad_layer".to_string(),
    });
    assert!(matches!(
        m.forward_with_hooks(&toks, None, Some(&bad_layer_ctx), None),
        Err(ModelError::Hook(HookError::UnknownSite { .. }))
    ));
}

#[test]
fn test_gx13_item6_and_item7_cache_parity_and_identity() {
    let cfg = Config::reference_v2();
    let m = Model::init(&cfg).unwrap();
    let prompt = [256u32, 65, 66, 67];

    let direction = vec![0.05f32; 128];
    let steer_op = EditOp::Steer {
        hook: HookId::layer(0, HookSite::ResidPre),
        direction,
        alpha: 1.5,
        positions: SteerPosition::AllPositions,
        direction_id: "test_dir".to_string(),
    };
    let ctx = HookContext::new().with_edit(steer_op);
    let policy_hash = ctx.intervention_policy_hash();
    assert_ne!(policy_hash, [0u8; 32]);

    // Full forward under steering
    let full_out = m.forward_with_hooks(&prompt, None, Some(&ctx), None).unwrap();

    // Incremental cached decode under same policy
    let mut cache = KVCache::new(&cfg, m.fingerprint(), 1)
        .unwrap()
        .with_intervention_policy_hash(policy_hash);

    let mut cached_logits = Vec::new();
    for (pos, &tok) in prompt.iter().enumerate() {
        let step_logit = forward_step_with_hooks(&m, tok, pos, &mut cache, Some(&ctx), None).unwrap();
        cached_logits.push(step_logit);
    }

    // Terminal logits must match full forward within numerical tolerance
    let last_cached = &cached_logits.last().unwrap();
    let v = cfg.vocab_size;
    let full_last = &full_out.data()[(prompt.len() - 1) * v..prompt.len() * v];
    let max_diff = last_cached
        .data()
        .iter()
        .zip(full_last.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1e-4, "Full vs cached under steering max_diff = {}", max_diff);

    // Item 7: Reusing cache under changed policy must return InterventionMismatch
    let different_policy = HookContext::new().with_edit(EditOp::Steer {
        hook: HookId::layer(1, HookSite::ResidPre),
        direction: vec![0.1f32; 128],
        alpha: 2.0,
        positions: SteerPosition::AllPositions,
        direction_id: "diff_dir".to_string(),
    });
    let err = forward_step_with_hooks(&m, 68, prompt.len(), &mut cache, Some(&different_policy), None);
    assert!(matches!(err, Err(ModelError::Cache(_))));
}

#[test]
fn test_gx13_item8_direction_metadata_validation() {
    let d_model = 128;
    let n_layers = 4;
    let ckpt_hash = [0xAAu8; 32];
    let tok_fp = [0xBBu8; 32];
    let v = vec![0.5f32; d_model];
    let norm = (0.5f32 * 0.5f32 * d_model as f32).sqrt();

    let header = DirectionHeader {
        schema_version: 1,
        direction_id: "dir_test".to_string(),
        dim: d_model,
        layer: 1,
        site: HookSite::ResidPre,
        model_instance_id: "inst_1".to_string(),
        checkpoint_hash: ckpt_hash,
        tokenizer_fingerprint: tok_fp,
        extraction_method: "difference_in_means".to_string(),
        created_from_dataset_hash: [0xCCu8; 32],
        norm,
        notes: "test vector".to_string(),
    };

    // Valid header passes
    assert!(header.validate(d_model, n_layers, ckpt_hash, tok_fp, &v).is_ok());

    // Mismatched checkpoint hash rejected
    let mut bad_ckpt = header.clone();
    bad_ckpt.checkpoint_hash = [0x00u8; 32];
    assert!(matches!(
        bad_ckpt.validate(d_model, n_layers, ckpt_hash, tok_fp, &v),
        Err(HookError::DirectionMismatch { field }) if field == "checkpoint_hash"
    ));

    // Mismatched tokenizer fingerprint rejected
    let mut bad_tok = header.clone();
    bad_tok.tokenizer_fingerprint = [0x00u8; 32];
    assert!(matches!(
        bad_tok.validate(d_model, n_layers, ckpt_hash, tok_fp, &v),
        Err(HookError::DirectionMismatch { field }) if field == "tokenizer_fingerprint"
    ));

    // Mismatched layer rejected
    let mut bad_layer = header.clone();
    bad_layer.layer = 5; // exceeds n_layers = 4
    assert!(matches!(
        bad_layer.validate(d_model, n_layers, ckpt_hash, tok_fp, &v),
        Err(HookError::DirectionMismatch { field }) if field == "layer"
    ));

    // Non-finite direction rejected
    let mut nan_v = v.clone();
    nan_v[0] = f32::NAN;
    assert!(matches!(
        header.validate(d_model, n_layers, ckpt_hash, tok_fp, &nan_v),
        Err(HookError::InvalidSteering { .. })
    ));
}
