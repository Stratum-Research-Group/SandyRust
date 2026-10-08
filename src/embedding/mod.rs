//! Extension X2: Untied Embeddings & Gate GX2 (§4.2 Item 2, §9.2 GX2).
//!
//! Decouples token embedding $E \in \mathbb{R}^{V \times D}$ from output unembedding
//! projection $W_{\text{head}} \in \mathbb{R}^{V \times D}$.
//! Establishes distinct physical memory buffers, distinct stable tensor IDs,
//! separate gradients ($dE$ via lookup scatter-add, $dW_{\text{head}} = X^T \cdot dz$),
//! and dedicated independent AdamW optimizer moments ($m_E, v_E$ and $m_W, v_W$).

use std::time::Instant;

use crate::autodiff::{gradcheck, loss_and_grad, Params};
use crate::cache::{forward_step, KVCache};
use crate::checkpoint::{parse, serialize, CheckpointError};
use crate::config::Config;
use crate::gates::{detect_revision, detect_toolchain, GateReport};
use crate::model::Model;
use crate::optimizer::{AdamW, Schedule};

/// Gate GX2: Untied Embeddings acceptance verification (§4.2 Item 2, §9.2 GX2).
pub fn gx2() -> GateReport {
    let t0 = Instant::now();
    let (rev, dirty) = detect_revision();
    let toolchain = detect_toolchain();

    let mut checks_passed = true;
    let mut details = Vec::new();

    // 1. Parameter count assertions across MHA, GQA, and MQA
    let cfg_tied_mha = Config::reference_v2();
    let cfg_untied_mha = Config::untied_v2();
    let cfg_untied_gqa = Config::untied_gqa_v2();
    let cfg_untied_mqa = Config::untied_mqa_v2();

    let p_tied_mha = cfg_tied_mha.param_count().unwrap_or(0);
    let p_untied_mha = cfg_untied_mha.param_count().unwrap_or(0);
    let p_untied_gqa = cfg_untied_gqa.param_count().unwrap_or(0);
    let p_untied_mqa = cfg_untied_mqa.param_count().unwrap_or(0);

    let count_ok = (p_tied_mha == 886_272)
        && (p_untied_mha == 919_424)
        && (p_untied_gqa == 853_888)
        && (p_untied_mqa == 821_120);
    checks_passed &= count_ok;
    details.push(format!(
        "param_counts: tied_mha={p_tied_mha}, untied_mha={p_untied_mha}, untied_gqa={p_untied_gqa}, untied_mqa={p_untied_mqa} (ok={count_ok})"
    ));

    // 2. Physical buffer separation (id_E != id_Whead)
    let model_untied = Model::init(&cfg_untied_mha).unwrap();
    let model_tied = Model::init(&cfg_tied_mha).unwrap();

    let head_tensor = model_untied.lm_head.as_ref().unwrap();
    let ptr_embed = model_untied.embed.data().as_ptr();
    let ptr_head = head_tensor.data().as_ptr();
    let buffer_distinct = ptr_embed != ptr_head;
    let tied_head_none = model_tied.lm_head.is_none();
    let shape_match = model_untied.embed.shape() == head_tensor.shape()
        && head_tensor.shape() == [cfg_untied_mha.vocab_size, cfg_untied_mha.d_model];

    let inv = model_untied.tensors();
    let has_embed_name = inv.iter().any(|(name, _)| name == "embed");
    let has_lm_head_name = inv.iter().any(|(name, _)| name == "lm_head");

    let buffer_ok = buffer_distinct && tied_head_none && shape_match && has_embed_name && has_lm_head_name;
    checks_passed &= buffer_ok;
    details.push(format!(
        "buffer_separation: distinct_ptrs={buffer_distinct}, tied_is_none={tied_head_none}, shapes_match={shape_match} (ok={buffer_ok})"
    ));

    // 3. Parameter perturbation verification
    let mut model_mut = model_untied.clone();
    let orig_head_sample = model_mut.lm_head.as_ref().unwrap().data()[0];
    let orig_embed_sample = model_mut.embed.data()[0];

    // Perturb E
    model_mut.embed.data_mut()[0] += 1.5;
    let head_unperturbed = model_mut.lm_head.as_ref().unwrap().data()[0] == orig_head_sample;

    // Perturb W_head
    model_mut.lm_head.as_mut().unwrap().data_mut()[0] += 2.5;
    let embed_unperturbed = model_mut.embed.data()[0] == orig_embed_sample + 1.5;

    let perturb_ok = head_unperturbed && embed_unperturbed;
    checks_passed &= perturb_ok;
    details.push(format!(
        "perturbation_independence: head_unperturbed={head_unperturbed}, embed_unperturbed={embed_unperturbed} (ok={perturb_ok})"
    ));

    // 4. Separate gradient accumulation & finite difference check
    let prompt: Vec<u32> = vec![10, 11, 12, 13, 14, 15];
    let mut params = Params::<f32>::from_model(&model_untied);
    let mut grads = params.zeros_like();

    let loss_out = loss_and_grad(&cfg_untied_mha, &params, &prompt, 1.0, &mut grads).unwrap();
    let loss_finite = loss_out.loss_sum.is_finite() && loss_out.loss_sum > 0.0;

    let d_model = cfg_untied_mha.d_model;
    let head_idx = 2 + 9 * cfg_untied_mha.n_layers;

    // Verify unused token in E has exactly zero gradient
    let unused_token = 200usize;
    assert!(!prompt.contains(&(unused_token as u32)));
    let unused_embed_grad_zero = grads.data[0][unused_token * d_model..(unused_token + 1) * d_model]
        .iter()
        .all(|&g| g == 0.0);

    // Verify used token in E has non-zero gradient
    let used_token = prompt[0] as usize;
    let used_embed_grad_nonzero = grads.data[0][used_token * d_model..(used_token + 1) * d_model]
        .iter()
        .any(|&g: &f32| g.abs() > 1e-8);

    // Verify W_head has non-zero gradients
    let head_grad_nonzero = grads.data[head_idx].iter().any(|&g: &f32| g.abs() > 1e-8);

    // Analytic vs numerical gradient verification on untied model
    let eps_list = [1e-4];
    let params64 = Params::<f64>::from_model(&model_untied);
    let gc_rep = gradcheck(&cfg_untied_mha, &params64, &prompt, &eps_list, 1e-3, 1e-3, 2).unwrap();
    let grad_check_ok = gc_rep.failures == 0 && gc_rep.max_abs_err <= 1e-3;

    let grad_ok = loss_finite
        && unused_embed_grad_zero
        && used_embed_grad_nonzero
        && head_grad_nonzero
        && grad_check_ok;
    checks_passed &= grad_ok;
    details.push(format!(
        "gradients: loss_finite={loss_finite}, unused_zero={unused_embed_grad_zero}, used_nonzero={used_embed_grad_nonzero}, head_nonzero={head_grad_nonzero}, gc_err={:.2e} (ok={grad_ok})",
        gc_rep.max_abs_err
    ));

    // 5. Dedicated optimizer moments in AdamW
    let mut opt = AdamW::new(&params);
    let sched = Schedule {
        eta_max: 1e-3,
        eta_min: 1e-5,
        warmup: 1,
        max_steps: 10,
    };
    let step_committed = opt.update(&mut params, &grads, sched.eta(1)).unwrap();
    let opt_has_separate_m = opt.m.len() == params.data.len();
    let opt_has_separate_v = opt.v.len() == params.data.len();
    let m_e_nonzero = opt.m[0].iter().any(|&x| x != 0.0);
    let m_head_nonzero = opt.m[head_idx].iter().any(|&x| x != 0.0);
    let opt_ok = step_committed && opt_has_separate_m && opt_has_separate_v && m_e_nonzero && m_head_nonzero;
    checks_passed &= opt_ok;
    details.push(format!(
        "optimizer_moments: committed={step_committed}, separate_m={opt_has_separate_m}, m_e={m_e_nonzero}, m_head={m_head_nonzero} (ok={opt_ok})"
    ));

    // 6. Checkpoint reload & cross-architecture rejection
    let ckpt_bytes = serialize(&model_untied);
    let reloaded_model = parse(&ckpt_bytes, &cfg_untied_mha, 64 * 1024 * 1024).unwrap();

    let out_orig = model_untied.forward(&prompt, None).unwrap();
    let out_reloaded = reloaded_model.forward(&prompt, None).unwrap();
    let ckpt_exact_match = out_orig.data() == out_reloaded.data();

    // Cross-loading rejection: untied bytes loaded into tied cfg -> ArchMismatch
    let cross_untied_to_tied = parse(&ckpt_bytes, &cfg_tied_mha, 64 * 1024 * 1024);
    let reject_untied_to_tied = matches!(cross_untied_to_tied, Err(CheckpointError::ArchMismatch));

    // Cross-loading rejection: tied bytes loaded into untied cfg -> ArchMismatch
    let tied_ckpt_bytes = serialize(&model_tied);
    let cross_tied_to_untied = parse(&tied_ckpt_bytes, &cfg_untied_mha, 64 * 1024 * 1024);
    let reject_tied_to_untied = matches!(cross_tied_to_untied, Err(CheckpointError::ArchMismatch));

    let ckpt_ok = ckpt_exact_match && reject_untied_to_tied && reject_tied_to_untied;
    checks_passed &= ckpt_ok;
    details.push(format!(
        "checkpoint: reload_exact={ckpt_exact_match}, reject_to_tied={reject_untied_to_tied}, reject_to_untied={reject_tied_to_untied} (ok={ckpt_ok})"
    ));

    // 7. Cached incremental decode parity vs full sequence forward
    let mut cache = KVCache::new(&cfg_untied_mha, model_untied.fingerprint(), 100).unwrap();
    let mut step_out = None;
    for (pos, &tok) in prompt.iter().enumerate() {
        step_out = Some(forward_step(&model_untied, tok, pos, &mut cache).unwrap());
    }
    let last_step = step_out.unwrap();
    let (rows, cols) = (out_orig.shape()[0], out_orig.shape()[1]);
    let last_full_row = &out_orig.data()[(rows - 1) * cols..rows * cols];
    let step_row = last_step.data();

    let mut max_decode_err = 0.0f32;
    for (&a, &b) in last_full_row.iter().zip(step_row.iter()) {
        let err = (a - b).abs();
        if err > max_decode_err {
            max_decode_err = err;
        }
    }
    let decode_ok = max_decode_err <= 1e-4;
    checks_passed &= decode_ok;
    details.push(format!(
        "incremental_decode_parity: max_err={max_decode_err:.2e} (ok={decode_ok})"
    ));

    let joined_details = details.join("; ");
    GateReport {
        gate_id: "GX2",
        scope: "integrity",
        status: if checks_passed { "PASS" } else { "FAIL" },
        reason: if checks_passed {
            None
        } else {
            Some("one or more untied embedding invariants failed".into())
        },
        requirement: "Untied embeddings buffer separation, perturbation invariance, separate gradient accumulation, optimizer moments, and checkpoint reload (§4.2 Item 2, §9.2 GX2)",
        implemented_check: "param_counts_exact, buffer_distinct, perturbation_independent, separate_gradients, dedicated_moments, checkpoint_arch_mismatch_rejection, kv_cache_incremental_parity",
        prerequisites: "reference_v2, G6, G7, G8, G10",
        exclusions: "none",
        revision: rev,
        dirty_state: dirty,
        toolchain,
        target: "x86_64-unknown-linux-gnu",
        command: "cargo test --test test_x2_untied",
        fixture_hash: "untied-v2-invariants-v1".into(),
        config: "untied_v2",
        dtype: "fp32",
        expected: "exact param counts (919,424 / 853,888 / 821,120), distinct buffers, independent perturbation, separate gradients, dedicated moments, and checkpoint isolation".into(),
        observed: joined_details,
        atol: 1e-4,
        rtol: 1e-4,
        execution_time_ms: t0.elapsed().as_millis(),
    }
}
