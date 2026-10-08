//! Test Suite for Extension X8: FlashAttention-2 / Custom Tiled Attention & Gate GX8.

use sandy_rust::attention::flash::{
    attention_ref_with_segments, flash_attention, flash_attention_backward,
    flash_attention_multihead_backward, flash_attention_multihead_forward, gx8, AttentionError,
    AttentionMetadata, validate_attention_metadata, validate_x8_attention_metadata,
};
use sandy_rust::attention::attention_ref;
use sandy_rust::config::{Config, ConfigError};
use sandy_rust::tensor::Tensor;

#[test]
fn test_x8_config_validation_and_fingerprint() {
    let base = Config::reference_v2();
    assert!(!base.ext_x8_flash_attn);
    assert_eq!(base.flash_br, 16);
    assert_eq!(base.flash_bc, 16);

    let flash_cfg = Config::flash_attn_v2();
    assert!(flash_cfg.ext_x8_flash_attn);
    assert_eq!(flash_cfg.flash_br, 16);
    assert_eq!(flash_cfg.flash_bc, 16);
    assert_ne!(base.fingerprint(), flash_cfg.fingerprint());

    let mut invalid_br = flash_cfg.clone();
    invalid_br.flash_br = 0;
    assert_eq!(
        invalid_br.validate().unwrap_err(),
        ConfigError::OutOfRange("flash_br and flash_bc must be > 0")
    );

    let mut invalid_bc = flash_cfg;
    invalid_bc.flash_bc = 0;
    assert_eq!(
        invalid_bc.validate().unwrap_err(),
        ConfigError::OutOfRange("flash_br and flash_bc must be > 0")
    );
}

#[test]
fn test_x8_exact_forward_parity_various_offsets() {
    let mut rng_state: u64 = 0x1234_5678;
    let mut next_f32 = || -> f32 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        ((rng_state as f32) / (u64::MAX as f32) - 0.5) * 0.5
    };

    let (tq, tk, dh) = (32, 48, 16);
    let q_vec: Vec<f32> = (0..tq * dh).map(|_| next_f32()).collect();
    let k_vec: Vec<f32> = (0..tk * dh).map(|_| next_f32()).collect();
    let v_vec: Vec<f32> = (0..tk * dh).map(|_| next_f32()).collect();
    let q = Tensor::from_vec(&[tq, dh], q_vec).unwrap();
    let k = Tensor::from_vec(&[tk, dh], k_vec).unwrap();
    let v = Tensor::from_vec(&[tk, dh], v_vec).unwrap();
    let q_valid = vec![true; tq];
    let k_valid = vec![true; tk];

    for &offset in &[0usize, 1, 3, 7, 16] {
        let (out_ref, _) = attention_ref(&q, &k, &v, &q_valid, &k_valid, offset).unwrap();
        let (out_flash, _) = flash_attention(&q, &k, &v, &q_valid, &k_valid, offset, None, 16, 16).unwrap();

        for i in 0..out_ref.data().len() {
            let diff = (out_ref.data()[i] - out_flash.data()[i]).abs();
            assert!(
                diff <= 1e-5,
                "Offset {offset} mismatch at idx {i}: ref={}, flash={}, diff={diff}",
                out_ref.data()[i],
                out_flash.data()[i]
            );
        }
    }
}

#[test]
fn test_x8_padded_query_invariants() {
    let (tq, tk, dh) = (16, 16, 16);
    let q = Tensor::from_vec(&[tq, dh], vec![0.3f32; tq * dh]).unwrap();
    let k = Tensor::from_vec(&[tk, dh], vec![0.2f32; tk * dh]).unwrap();
    let v = Tensor::from_vec(&[tk, dh], vec![1.5f32; tk * dh]).unwrap();

    let mut q_valid = vec![true; tq];
    q_valid[2] = false;
    q_valid[5] = false;
    q_valid[11] = false;
    let k_valid = vec![true; tk];

    let (out, lse) = flash_attention(&q, &k, &v, &q_valid, &k_valid, 0, None, 16, 16).unwrap();

    for &padded_row in &[2, 5, 11] {
        let row = &out.data()[padded_row * dh..(padded_row + 1) * dh];
        assert!(row.iter().all(|&x| x == 0.0), "Padded row {padded_row} not exactly zero in forward output");
        assert_eq!(lse[padded_row], f32::NEG_INFINITY, "Padded row {padded_row} LSE not NEG_INFINITY");
    }

    let do_attn = Tensor::from_vec(&[tq, dh], vec![1.0f32; tq * dh]).unwrap();
    let (dq, _, _) = flash_attention_backward(&q, &k, &v, &out, &do_attn, &lse, &q_valid, &k_valid, 0, None, 16, 16).unwrap();

    for &padded_row in &[2, 5, 11] {
        let row = &dq.data()[padded_row * dh..(padded_row + 1) * dh];
        assert!(row.iter().all(|&x| x == 0.0), "Padded row {padded_row} not exactly zero in dQ");
    }
}

#[test]
fn test_x8_tail_block_clamping_and_odd_shapes() {
    let mut rng_state: u64 = 0xCAFE_BABE;
    let mut next_f32 = || -> f32 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        ((rng_state as f32) / (u64::MAX as f32) - 0.5) * 0.4
    };

    // Test prime / non-block-multiple lengths
    let shapes = [(13, 29), (23, 37), (7, 19), (33, 49)];
    let dh = 16;

    for (tq, tk) in shapes {
        let q = Tensor::from_vec(&[tq, dh], (0..tq * dh).map(|_| next_f32()).collect()).unwrap();
        let k = Tensor::from_vec(&[tk, dh], (0..tk * dh).map(|_| next_f32()).collect()).unwrap();
        let v = Tensor::from_vec(&[tk, dh], (0..tk * dh).map(|_| next_f32()).collect()).unwrap();
        let q_valid = vec![true; tq];
        let k_valid = vec![true; tk];

        let (out_ref, _) = attention_ref(&q, &k, &v, &q_valid, &k_valid, 2).unwrap();
        let (out_flash, _) = flash_attention(&q, &k, &v, &q_valid, &k_valid, 2, None, 16, 16).unwrap();

        for i in 0..out_ref.data().len() {
            let diff = (out_ref.data()[i] - out_flash.data()[i]).abs();
            assert!(
                diff <= 1e-5,
                "Tail block mismatch for ({tq}, {tk}) at idx {i}: diff={diff}"
            );
        }
    }
}

#[test]
fn test_x8_segment_id_matching() {
    let (tq, tk, dh) = (16, 32, 16);
    let mut rng_state: u64 = 0xFEED_FACE;
    let mut next_f32 = || -> f32 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        ((rng_state as f32) / (u64::MAX as f32) - 0.5) * 0.4
    };

    let q = Tensor::from_vec(&[tq, dh], (0..tq * dh).map(|_| next_f32()).collect()).unwrap();
    let k = Tensor::from_vec(&[tk, dh], (0..tk * dh).map(|_| next_f32()).collect()).unwrap();
    let v = Tensor::from_vec(&[tk, dh], (0..tk * dh).map(|_| next_f32()).collect()).unwrap();
    let q_valid = vec![true; tq];
    let k_valid = vec![true; tk];

    // Segment 0 for tokens 0..16, segment 1 for tokens 16..32
    let segment_ids: Vec<u32> = (0..tk).map(|idx| if idx < 16 { 0 } else { 1 }).collect();

    let (out_ref, _) = attention_ref_with_segments(&q, &k, &v, &q_valid, &k_valid, 4, Some(&segment_ids)).unwrap();
    let (out_flash, _) = flash_attention(&q, &k, &v, &q_valid, &k_valid, 4, Some(&segment_ids), 16, 16).unwrap();

    for i in 0..out_ref.data().len() {
        let diff = (out_ref.data()[i] - out_flash.data()[i]).abs();
        assert!(
            diff <= 1e-5,
            "Segment mismatch at idx {i}: ref={}, flash={}, diff={diff}",
            out_ref.data()[i],
            out_flash.data()[i]
        );
    }
}

#[test]
fn test_x8_backward_vjp_finite_differences() {
    let (tq, tk, dh) = (4, 4, 4);
    let mut rng_state: u64 = 0xABCD_EF01;
    let mut next_f32 = || -> f32 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        ((rng_state as f32) / (u64::MAX as f32) - 0.5) * 0.3
    };

    let q = Tensor::from_vec(&[tq, dh], (0..tq * dh).map(|_| next_f32()).collect()).unwrap();
    let k = Tensor::from_vec(&[tk, dh], (0..tk * dh).map(|_| next_f32()).collect()).unwrap();
    let v = Tensor::from_vec(&[tk, dh], (0..tk * dh).map(|_| next_f32()).collect()).unwrap();
    let q_valid = vec![true; tq];
    let k_valid = vec![true; tk];

    let (out, lse) = flash_attention(&q, &k, &v, &q_valid, &k_valid, 0, None, 4, 4).unwrap();
    let do_attn = Tensor::from_vec(&[tq, dh], (0..tq * dh).map(|_| next_f32()).collect()).unwrap();
    let (dq, dk, dv) = flash_attention_backward(&q, &k, &v, &out, &do_attn, &lse, &q_valid, &k_valid, 0, None, 4, 4).unwrap();

    let eps = 1e-3f32;

    // Check dQ components
    for idx in 0..tq * dh {
        let mut q_plus = q.clone();
        q_plus.data_mut()[idx] += eps;
        let (out_plus, _) = flash_attention(&q_plus, &k, &v, &q_valid, &k_valid, 0, None, 4, 4).unwrap();

        let mut q_minus = q.clone();
        q_minus.data_mut()[idx] -= eps;
        let (out_minus, _) = flash_attention(&q_minus, &k, &v, &q_valid, &k_valid, 0, None, 4, 4).unwrap();

        let mut dot_plus = 0.0f32;
        let mut dot_minus = 0.0f32;
        for i in 0..tq * dh {
            dot_plus += out_plus.data()[i] * do_attn.data()[i];
            dot_minus += out_minus.data()[i] * do_attn.data()[i];
        }
        let num_grad = (dot_plus - dot_minus) / (2.0 * eps);
        let ana_grad = dq.data()[idx];
        assert!(
            (num_grad - ana_grad).abs() <= 1e-3,
            "dQ mismatch at {idx}: num={num_grad}, ana={ana_grad}"
        );
    }

    // Check dK components
    for idx in 0..tk * dh {
        let mut k_plus = k.clone();
        k_plus.data_mut()[idx] += eps;
        let (out_plus, _) = flash_attention(&q, &k_plus, &v, &q_valid, &k_valid, 0, None, 4, 4).unwrap();

        let mut k_minus = k.clone();
        k_minus.data_mut()[idx] -= eps;
        let (out_minus, _) = flash_attention(&q, &k_minus, &v, &q_valid, &k_valid, 0, None, 4, 4).unwrap();

        let mut dot_plus = 0.0f32;
        let mut dot_minus = 0.0f32;
        for i in 0..tq * dh {
            dot_plus += out_plus.data()[i] * do_attn.data()[i];
            dot_minus += out_minus.data()[i] * do_attn.data()[i];
        }
        let num_grad = (dot_plus - dot_minus) / (2.0 * eps);
        let ana_grad = dk.data()[idx];
        assert!(
            (num_grad - ana_grad).abs() <= 1e-3,
            "dK mismatch at {idx}: num={num_grad}, ana={ana_grad}"
        );
    }

    // Check dV components
    for idx in 0..tk * dh {
        let mut v_plus = v.clone();
        v_plus.data_mut()[idx] += eps;
        let (out_plus, _) = flash_attention(&q, &k, &v_plus, &q_valid, &k_valid, 0, None, 4, 4).unwrap();

        let mut v_minus = v.clone();
        v_minus.data_mut()[idx] -= eps;
        let (out_minus, _) = flash_attention(&q, &k, &v_minus, &q_valid, &k_valid, 0, None, 4, 4).unwrap();

        let mut dot_plus = 0.0f32;
        let mut dot_minus = 0.0f32;
        for i in 0..tq * dh {
            dot_plus += out_plus.data()[i] * do_attn.data()[i];
            dot_minus += out_minus.data()[i] * do_attn.data()[i];
        }
        let num_grad = (dot_plus - dot_minus) / (2.0 * eps);
        let ana_grad = dv.data()[idx];
        assert!(
            (num_grad - ana_grad).abs() <= 1e-3,
            "dV mismatch at {idx}: num={num_grad}, ana={ana_grad}"
        );
    }
}

#[test]
fn test_x8_multihead_and_gqa_accumulation() {
    let (b, hq, hkv, tq, tk, dh, dv) = (1, 4, 2, 8, 8, 8, 8);
    let q = vec![0.1f32; b * hq * tq * dh];
    let k = vec![0.1f32; b * hkv * tk * dh];
    let v = vec![0.1f32; b * hkv * tk * dv];
    let q_valid = vec![true; tq];
    let k_valid = vec![true; tk];

    let mut out = vec![0.0f32; b * hq * tq * dv];
    let mut lse = vec![0.0f32; b * hq * tq];

    flash_attention_multihead_forward(
        &q, &k, &v, b, hq, hkv, tq, tk, dh, dv, &q_valid, &k_valid, 0, None, 8, 8, &mut out, &mut lse,
    ).unwrap();

    let do_attn = vec![1.0f32; b * hq * tq * dv];
    let mut dq = vec![0.0f32; b * hq * tq * dh];
    let mut dk = vec![0.0f32; b * hkv * tk * dh];
    let mut dv_out = vec![0.0f32; b * hkv * tk * dv];

    flash_attention_multihead_backward(
        &q, &k, &v, &out, &do_attn, &lse, b, hq, hkv, tq, tk, dh, dv, &q_valid, &k_valid, 0, None, 8, 8, &mut dq, &mut dk, &mut dv_out,
    ).unwrap();

    // With 2 query heads per KV head and identical inputs, dK and dV are accumulated from both query heads
    assert!(dk.iter().all(|&x| x.is_finite() && x != 0.0));
    assert!(dv_out.iter().all(|&x| x.is_finite() && x != 0.0));
}

#[test]
fn test_x8_metadata_validators() {
    let valid_meta = AttentionMetadata {
        query_pos: None,
        causal_offset: 0,
        q_pad_mask: Some(&[true, true]),
        k_pad_mask: Some(&[true, true]),
        segment_ids: Some(&[0, 0]),
    };
    assert!(validate_attention_metadata(&valid_meta, 1, 2, 2).is_ok());

    // Zero extent
    assert_eq!(
        validate_attention_metadata(&valid_meta, 0, 2, 2).unwrap_err(),
        AttentionError::ZeroExtent
    );

    // Decode query length mismatch
    let decode_invalid_qlen = AttentionMetadata {
        query_pos: Some(0),
        causal_offset: 0,
        q_pad_mask: None,
        k_pad_mask: None,
        segment_ids: None,
    };
    assert_eq!(
        validate_attention_metadata(&decode_invalid_qlen, 1, 2, 4).unwrap_err(),
        AttentionError::DecodeQueryLenMismatch { q_len: 2 }
    );

    // Decode position out of bounds
    let decode_oob = AttentionMetadata {
        query_pos: Some(5),
        causal_offset: 0,
        q_pad_mask: None,
        k_pad_mask: None,
        segment_ids: None,
    };
    assert_eq!(
        validate_attention_metadata(&decode_oob, 1, 1, 4).unwrap_err(),
        AttentionError::DecodePositionOutOfBounds { pos: 5, k_len: 4 }
    );

    // Prefill extent mismatch
    let prefill_mismatch = AttentionMetadata {
        query_pos: None,
        causal_offset: 0,
        q_pad_mask: None,
        k_pad_mask: None,
        segment_ids: None,
    };
    assert_eq!(
        validate_attention_metadata(&prefill_mismatch, 1, 2, 4).unwrap_err(),
        AttentionError::PrefillExtentMismatch { q_len: 2, k_len: 4 }
    );

    // X8 validator with valid causal offset
    let x8_meta = AttentionMetadata {
        query_pos: None,
        causal_offset: 2,
        q_pad_mask: None,
        k_pad_mask: None,
        segment_ids: Some(&[0, 0, 1, 1]),
    };
    assert!(validate_x8_attention_metadata(&x8_meta, 1, 2, 4).is_ok());

    // X8 validator offset overflow
    let x8_meta_invalid_offset = AttentionMetadata {
        query_pos: None,
        causal_offset: 3,
        q_pad_mask: None,
        k_pad_mask: None,
        segment_ids: None,
    };
    assert_eq!(
        validate_x8_attention_metadata(&x8_meta_invalid_offset, 1, 2, 4).unwrap_err(),
        AttentionError::InvalidCausalOffset { offset: 3 }
    );
}

#[test]
fn test_x8_gate_gx8_execution() {
    let report = gx8();
    assert_eq!(report.gate_id, "GX8");
    assert_eq!(report.status, "PASS", "Gate GX8 failed: {:?}", report.observed);
}
