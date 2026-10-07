use sandy_rust::attention::attention_ref;
use sandy_rust::tensor::ops::*;
use sandy_rust::tensor::{Tensor, TensorError};

fn t(shape: &[usize], d: Vec<f32>) -> Tensor {
    Tensor::from_vec(shape, d).unwrap()
}

#[test]
fn matmul_errors_and_odd_shapes() {
    let a = t(&[1, 3], vec![1., 2., 3.]);
    let b = t(&[3, 1], vec![4., 5., 6.]);
    assert_eq!(matmul(&a, &b).unwrap().data(), [32.0]);
    assert!(matches!(matmul(&a, &a), Err(TensorError::ShapeMismatch { .. })));
    assert!(matches!(matmul(&t(&[3], vec![0.; 3]), &b), Err(TensorError::Rank { .. })));
}

#[test]
fn gather_bounds_and_transpose() {
    let m = t(&[2, 3], vec![1., 2., 3., 4., 5., 6.]);
    assert!(matches!(m.gather_rows(&[2]), Err(TensorError::IndexOutOfBounds { .. })));
    assert_eq!(m.transpose2d().unwrap().data(), [1., 4., 2., 5., 3., 6.]);
    assert!(matches!(
        Tensor::zeros(&[usize::MAX, 2]),
        Err(TensorError::AllocationOverflow)
    ));
}

#[test]
fn rope_position_zero_is_identity_and_offset_consistent() {
    let x = t(&[1, 4], vec![0.5, -1.0, 2.0, 0.25]);
    assert_eq!(rope(&x, 10000.0, 0).unwrap().data(), x.data());
    // Rotating row at offset 3 equals row 3 of a longer sequence.
    let seq = t(&[4, 4], (0..16).map(|i| i as f32 * 0.1).collect());
    let full = rope(&seq, 10000.0, 0).unwrap();
    let last = t(&[1, 4], seq.data()[12..16].to_vec());
    let one = rope(&last, 10000.0, 3).unwrap();
    for (a, b) in one.data().iter().zip(&full.data()[12..16]) {
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn silu_known_values() {
    let y = silu(&t(&[2], vec![0.0, 1.0]));
    assert_eq!(y.data()[0], 0.0);
    assert!((y.data()[1] - 0.731_058_6).abs() < 1e-6);
}

#[test]
fn padded_query_with_nan_v_outputs_zero() {
    let q = t(&[2, 2], vec![1., 0., 0., 1.]);
    let k = t(&[2, 2], vec![1., 0., 0., 1.]);
    // v has finite values in row 0, and NaN in row 1
    let v = t(&[2, 2], vec![3., 4., f32::NAN, f32::NAN]);
    // row 0 is valid query, row 1 is padded query
    let (out, p) = attention_ref(&q, &k, &v, &[true, false], &[true, false], 0).unwrap();
    // row 0 gets row 0 of V
    assert_eq!(out.data()[0], 3.0);
    assert_eq!(out.data()[1], 4.0);
    // row 1 was padded, so its output MUST be exact 0.0 without multiplying NaN V
    assert_eq!(out.data()[2], 0.0);
    assert_eq!(out.data()[3], 0.0);
    assert_eq!(p[2], 0.0);
    assert_eq!(p[3], 0.0);
}

#[test]
fn attention_rank_validation() {
    let q = t(&[2], vec![1., 0.]);
    let k = t(&[2, 2], vec![1., 0., 0., 1.]);
    let v = k.clone();
    assert!(matches!(
        attention_ref(&q, &k, &v, &[true], &[true, true], 0),
        Err(TensorError::Rank { .. })
    ));
}

#[test]
fn batched_matmul_and_reductions() {
    let a = t(&[2, 2, 2], vec![1., 2., 3., 4., 5., 6., 7., 8.]);
    let b = t(&[2, 2, 2], vec![1., 0., 0., 1., 2., 0., 0., 2.]);
    let c = batched_matmul(&a, &b).unwrap();
    assert_eq!(c.data()[..4], [1., 2., 3., 4.]);
    assert_eq!(c.data()[4..], [10., 12., 14., 16.]);

    let lse = logsumexp(&[0.0, 0.0]);
    assert!((lse - 2.0f32.ln()).abs() < 1e-6);

    let m = t(&[2, 3], vec![1., 2., 3., 4., 5., 6.]);
    let s0 = sum_axis(&m, 0).unwrap();
    assert_eq!(s0.data(), [5., 7., 9.]);
    let s1 = sum_axis(&m, 1).unwrap();
    assert_eq!(s1.data(), [6., 15.]);
}

#[test]
fn test_masked_softmax_and_attention_mask_length_mismatch() {
    let mut scores = [1.0, 2.0, 3.0];
    let valid_short = [true, false];
    assert!(!masked_softmax_row(&mut scores, &valid_short));

    let q = t(&[2, 2], vec![1., 0., 0., 1.]);
    let k = t(&[2, 2], vec![1., 0., 0., 1.]);
    let v = t(&[2, 2], vec![1., 0., 0., 1.]);
    let err = attention_ref(&q, &k, &v, &[true], &[true, true], 0);
    assert!(matches!(err, Err(TensorError::InvalidArgument("mask length mismatch"))));
}

#[test]
fn test_train_forward_matches_model_forward() {
    let cfg = sandy_rust::config::Config::reference_v2();
    let model = sandy_rust::model::Model::init(&cfg).unwrap();
    let tokens = [256u32, 65, 66, 67, 257];

    let logits_model = model.forward(&tokens, None).unwrap();
    let p = sandy_rust::autodiff::Params::<f32>::from_model(&model);
    let out_autodiff = sandy_rust::autodiff::forward_loss(&cfg, &p, &tokens).unwrap();

    let t = tokens.len() - 1;
    let v = cfg.vocab_size;
    let mut expected_loss_sum = 0.0f32;
    for i in 0..t {
        let target = tokens[i + 1] as usize;
        let row = &logits_model.data()[i * v..(i + 1) * v];
        let lse = logsumexp(row);
        let nll = lse - row[target];
        expected_loss_sum += nll;
    }
    assert!((out_autodiff.loss_sum - expected_loss_sum).abs() < 1e-4);
    assert_eq!(out_autodiff.n_valid, t);
}
