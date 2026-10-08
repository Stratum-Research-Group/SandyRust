//! Tests for Extension X7: Memory-Efficient Chunked Cross-Entropy & Activation Recomputation (§4.2 Item 7, §9.2 GX7).

use sandy_rust::autodiff::{loss_and_grad, Params, TrainError};
use sandy_rust::config::Config;
use sandy_rust::loss::{chunked_forward_loss, chunked_loss_and_grad, gx7};
use sandy_rust::model::Model;
use sandy_rust::rng::Rng;

#[test]
fn test_x7_config_validation_and_fingerprint() {
    let base = Config::reference_v2();
    assert!(!base.ext_x7_chunked_loss);

    let chunk_cfg = Config::chunked_loss_v2(16);
    assert!(chunk_cfg.ext_x7_chunked_loss);
    assert_eq!(chunk_cfg.chunk_size, 16);
    assert!(chunk_cfg.validate().is_ok());

    // Baseline parameter count and fingerprint unchanged
    assert_eq!(base.param_count(), Some(886_272));
    assert_ne!(base.fingerprint(), chunk_cfg.fingerprint());

    // Validation rejection of zero chunk size
    let mut invalid = chunk_cfg.clone();
    invalid.chunk_size = 0;
    assert!(invalid.validate().is_err());
}

#[test]
fn test_x7_exact_loss_and_grad_parity_standard_window() {
    let cfg = Config::chunked_loss_v2(16);
    let model = Model::init(&cfg).expect("model init");
    let params = Params::from_model(&model);

    let mut rng = Rng::new(0x42);
    let max_tok = (cfg.vocab_size - 4) as u32;
    let window: Vec<u32> = (0..33)
        .map(|_| rng.next_u32() % max_tok)
        .collect();

    let mut g_mono = params.zeros_like();
    let mut g_chunk = params.zeros_like();

    // Base monolithic configuration
    let base_cfg = Config::reference_v2();
    let out_mono = loss_and_grad(&base_cfg, &params, &window, 1.0f32, &mut g_mono)
        .expect("monolithic loss_and_grad");
    let out_chunk = chunked_loss_and_grad(&cfg, &params, &window, 1.0f32, &mut g_chunk)
        .expect("chunked loss_and_grad");

    assert_eq!(out_mono.n_valid, out_chunk.n_valid);
    let loss_diff = (out_mono.loss_sum - out_chunk.loss_sum).abs();
    assert!(
        loss_diff <= 1e-6,
        "Loss difference too large: {loss_diff:.2e}"
    );

    let mut max_grad_diff = 0.0f32;
    for (m_slice, c_slice) in g_mono.data.iter().zip(&g_chunk.data) {
        for (&gm, &gc) in m_slice.iter().zip(c_slice) {
            let diff = (gm - gc).abs();
            if diff > max_grad_diff {
                max_grad_diff = diff;
            }
        }
    }
    assert!(
        max_grad_diff <= 1e-5,
        "Max gradient difference too large: {max_grad_diff:.2e}"
    );

    // Also verify forward-only loss parity
    let fwd_mono = sandy_rust::autodiff::forward_loss(&base_cfg, &params, &window).unwrap();
    let fwd_chunk = chunked_forward_loss(&cfg, &params, &window).unwrap();
    assert!((fwd_mono.loss_sum - fwd_chunk.loss_sum).abs() <= 1e-6);
}

#[test]
fn test_x7_tail_chunk_handling() {
    let cfg = Config::chunked_loss_v2(16);
    let model = Model::init(&cfg).expect("model init");
    let params = Params::from_model(&model);
    let base_cfg = Config::reference_v2();

    let mut rng = Rng::new(0x1337);
    let max_tok = (cfg.vocab_size - 4) as u32;
    // Non-multiple sequence lengths (tail chunks)
    for len in [3, 7, 17, 23, 31, 35, 49] {
        let window: Vec<u32> = (0..len)
            .map(|_| rng.next_u32() % max_tok)
            .collect();

        let mut g_mono = params.zeros_like();
        let mut g_chunk = params.zeros_like();

        let out_mono = loss_and_grad(&base_cfg, &params, &window, 1.0f32, &mut g_mono).unwrap();
        let out_chunk = chunked_loss_and_grad(&cfg, &params, &window, 1.0f32, &mut g_chunk).unwrap();

        assert_eq!(out_mono.n_valid, out_chunk.n_valid);
        let loss_diff = (out_mono.loss_sum - out_chunk.loss_sum).abs();
        assert!(
            loss_diff <= 1e-6,
            "Length {len} loss diff {loss_diff:.2e} > 1e-6"
        );
    }
}

#[test]
fn test_x7_different_chunk_sizes() {
    let model = Model::init(&Config::reference_v2()).expect("model init");
    let params = Params::from_model(&model);
    let base_cfg = Config::reference_v2();

    let mut rng = Rng::new(0x9999);
    let max_tok = (base_cfg.vocab_size - 4) as u32;
    let window: Vec<u32> = (0..25)
        .map(|_| rng.next_u32() % max_tok)
        .collect();

    let mut g_ref = params.zeros_like();
    let ref_out = loss_and_grad(&base_cfg, &params, &window, 1.0f32, &mut g_ref).unwrap();

    // Verify chunk sizes: 1, 2, 4, 8, 16, 32
    for chunk_size in [1, 2, 4, 8, 16, 32] {
        let cfg = Config::chunked_loss_v2(chunk_size);
        let mut g_test = params.zeros_like();
        let test_out = chunked_loss_and_grad(&cfg, &params, &window, 1.0f32, &mut g_test).unwrap();

        assert_eq!(ref_out.n_valid, test_out.n_valid);
        let diff = (ref_out.loss_sum - test_out.loss_sum).abs();
        assert!(
            diff <= 1e-6,
            "Chunk size {chunk_size} diff {diff:.2e} exceeds tolerance"
        );
    }
}

#[test]
fn test_x7_padding_masks_and_zero_valid() {
    let cfg = Config::chunked_loss_v2(8);
    let model = Model::init(&cfg).expect("model init");
    let params = Params::from_model(&model);

    // Partially padded window
    let mut win_padded = vec![10u32; 20];
    for slot in win_padded[5..12].iter_mut() {
        *slot = cfg.pad_id;
    }
    let mut g = params.zeros_like();
    let res = chunked_loss_and_grad(&cfg, &params, &win_padded, 1.0f32, &mut g);
    assert!(res.is_ok());

    // All-padded window -> must return NoValidTokens
    let win_all_pad = vec![cfg.pad_id; 20];
    let res_all_pad = chunked_loss_and_grad(&cfg, &params, &win_all_pad, 1.0f32, &mut g);
    assert_eq!(res_all_pad, Err(TrainError::NoValidTokens));
}

#[test]
fn test_x7_untied_embeddings_compatibility() {
    let mut cfg = Config::chunked_loss_v2(16);
    cfg.tied_embeddings = false;
    cfg.ext_x2_untied_embeddings = true;

    let model = Model::init(&cfg).expect("model init");
    let params = Params::from_model(&model);

    let window = vec![10u32, 20, 30, 40, 50, 60, 70, 80, 90, 100];
    let mut g_mono = params.zeros_like();
    let mut g_chunk = params.zeros_like();

    let mut base_untied = Config::untied_v2();
    base_untied.max_seq_len = cfg.max_seq_len;

    let o_mono = loss_and_grad(&base_untied, &params, &window, 1.0f32, &mut g_mono).unwrap();
    let o_chunk = chunked_loss_and_grad(&cfg, &params, &window, 1.0f32, &mut g_chunk).unwrap();

    let diff = (o_mono.loss_sum - o_chunk.loss_sum).abs();
    assert!(diff <= 1e-6, "Untied loss diff: {diff:.2e}");
}

#[test]
fn test_x7_activation_recomputation_immutability() {
    let cfg = Config::chunked_loss_v2(16);
    let model = Model::init(&cfg).expect("model init");
    let params = Params::from_model(&model);
    let fp_before = model.fingerprint();

    let window = vec![5u32, 10, 15, 20, 25, 30];
    let mut g = params.zeros_like();
    let _ = chunked_loss_and_grad(&cfg, &params, &window, 1.0f32, &mut g).unwrap();

    let fp_after = model.fingerprint();
    assert_eq!(fp_before, fp_after, "Model parameters mutated during chunked loss");
}

#[test]
fn test_x7_gate_gx7_execution() {
    let rep = gx7();
    assert_eq!(rep.gate_id, "GX7");
    assert_eq!(rep.scope, "acceptance");
    assert_eq!(rep.status, "PASS", "Gate GX7 failed: {:?}", rep.reason);
}
