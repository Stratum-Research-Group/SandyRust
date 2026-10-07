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
fn padded_query_counterexample_is_zero() {
    // Padded query with an allowed key and V=(7,0): output must be exactly zero.
    let q = t(&[1, 2], vec![1., 1.]);
    let k = t(&[1, 2], vec![1., 1.]);
    let v = t(&[1, 2], vec![7., 0.]);
    let (out, p) = attention_ref(&q, &k, &v, &[false], &[true], 0).unwrap();
    assert_eq!(out.data(), [0.0, 0.0]);
    assert_eq!(p, [0.0]);
}

#[test]
fn fully_masked_valid_query_errors_and_offset_works() {
    let q = t(&[1, 2], vec![1., 0.]);
    let k = t(&[2, 2], vec![1., 0., 0., 1.]);
    let v = k.clone();
    assert!(matches!(
        attention_ref(&q, &k, &v, &[true], &[false, false], 0),
        Err(TensorError::AllMaskedRow { .. })
    ));
    // offset=1 lets the single query see both keys (decode-style).
    let (_, p) = attention_ref(&q, &k, &v, &[true], &[true, true], 1).unwrap();
    assert!(p[0] > 0.0 && p[1] > 0.0 && (p[0] + p[1] - 1.0).abs() < 1e-6);
}
