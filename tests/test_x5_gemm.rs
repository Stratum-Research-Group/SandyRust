//! Integration test suite for Extension X5: Optimized CPU GEMM Kernels & Gate GX5 (§4.2 Item 5, §9.2 GX5).

use sandy_rust::config::Config;
use sandy_rust::gates;
use sandy_rust::gemm::{
    batched_gemm, check_parity, gemm, gemm_parallel, gemm_strided, gemm_trans_a, gemm_trans_b, gx5,
};
use sandy_rust::tensor::ops::{batched_matmul, matmul};
use sandy_rust::tensor::Tensor;

#[test]
fn test_x5_config_validation() {
    let c_base = Config::reference_v2();
    let c_gemm = Config::gemm_opt_v2();

    assert!(c_base.validate().is_ok());
    assert!(c_gemm.validate().is_ok());

    assert!(!c_base.ext_x5_gemm);
    assert!(c_gemm.ext_x5_gemm);

    // Fingerprint changes when extension is active
    assert_ne!(c_base.fingerprint(), c_gemm.fingerprint());
}

#[test]
fn test_x5_numerical_parity_various_shapes() {
    let test_cases = [
        (1, 1, 1),
        (2, 2, 2),
        (3, 5, 7),
        (8, 16, 8),
        (17, 31, 23),
        (32, 64, 48),
        (63, 65, 33),
        (64, 128, 64),
        (128, 64, 128),
    ];

    for (m, k, n) in test_cases {
        let a_data: Vec<f32> = (0..m * k)
            .map(|i| ((i as f32 * 0.17 + 0.1).sin()) * 1.5)
            .collect();
        let b_data: Vec<f32> = (0..k * n)
            .map(|i| ((i as f32 * 0.23 + 0.2).cos()) * 1.5)
            .collect();

        let ta = Tensor::from_vec(&[m, k], a_data).unwrap();
        let tb = Tensor::from_vec(&[k, n], b_data).unwrap();

        let ref_out = matmul(&ta, &tb).unwrap();
        let opt_out = gemm(&ta, &tb).unwrap();

        let (ok, max_diff) = check_parity(opt_out.data(), ref_out.data(), 1e-5, 1e-4);
        assert!(
            ok,
            "Parity failed on shape ({m}, {k}, {n}): max_diff = {max_diff}"
        );
    }
}

#[test]
fn test_x5_strided_and_transposed_operations() {
    let m = 23;
    let k = 37;
    let n = 19;

    // 1. Transpose A: A^T is [m, k], so A is stored as [k, m] row-major
    let a_t_data: Vec<f32> = (0..k * m).map(|i| (i as f32 * 0.13).sin()).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.19).cos()).collect();

    let mut a_transposed = vec![0.0f32; m * k];
    for r in 0..k {
        for c in 0..m {
            a_transposed[c * k + r] = a_t_data[r * m + c];
        }
    }
    let ta_explicit = Tensor::from_vec(&[m, k], a_transposed).unwrap();
    let tb = Tensor::from_vec(&[k, n], b_data.clone()).unwrap();
    let ref_out_ta = matmul(&ta_explicit, &tb).unwrap();

    let ta_raw = Tensor::from_vec(&[k, m], a_t_data).unwrap();
    let opt_out_ta = gemm_trans_a(&ta_raw, &tb).unwrap();

    let (ok_ta, diff_ta) = check_parity(opt_out_ta.data(), ref_out_ta.data(), 1e-5, 1e-4);
    assert!(ok_ta, "Transpose A parity failed: max_diff = {diff_ta}");

    // 2. Transpose B: B^T is [k, n], so B is stored as [n, k] row-major
    let a_norm_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.29).sin()).collect();
    let b_t_data: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.31).cos()).collect();

    let mut b_transposed = vec![0.0f32; k * n];
    for r in 0..n {
        for c in 0..k {
            b_transposed[c * n + r] = b_t_data[r * k + c];
        }
    }
    let tb_explicit = Tensor::from_vec(&[k, n], b_transposed).unwrap();
    let ta_norm = Tensor::from_vec(&[m, k], a_norm_data).unwrap();
    let ref_out_tb = matmul(&ta_norm, &tb_explicit).unwrap();

    let tb_raw = Tensor::from_vec(&[n, k], b_t_data).unwrap();
    let opt_out_tb = gemm_trans_b(&ta_norm, &tb_raw).unwrap();

    let (ok_tb, diff_tb) = check_parity(opt_out_tb.data(), ref_out_tb.data(), 1e-5, 1e-4);
    assert!(ok_tb, "Transpose B parity failed: max_diff = {diff_tb}");
}

#[test]
fn test_x5_batched_gemm() {
    for &b_count in &[1, 2, 4] {
        let bm = 15;
        let bk = 25;
        let bn = 35;

        let a_data: Vec<f32> = (0..b_count * bm * bk)
            .map(|i| (i as f32 * 0.17).sin())
            .collect();
        let b_data: Vec<f32> = (0..b_count * bk * bn)
            .map(|i| (i as f32 * 0.23).cos())
            .collect();

        let ta = Tensor::from_vec(&[b_count, bm, bk], a_data).unwrap();
        let tb = Tensor::from_vec(&[b_count, bk, bn], b_data).unwrap();

        let ref_batched = batched_matmul(&ta, &tb).unwrap();
        let opt_batched = batched_gemm(&ta, &tb).unwrap();

        let (ok, max_diff) = check_parity(opt_batched.data(), ref_batched.data(), 1e-5, 1e-4);
        assert!(
            ok,
            "Batched GEMM failed on batch {b_count}: max_diff = {max_diff}"
        );
    }
}

#[test]
fn test_x5_multithread_partitioning_and_determinism() {
    let m = 73;
    let k = 59;
    let n = 61;

    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.21).sin()).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.27).cos()).collect();

    let mut c_single = vec![0.0f32; m * n];
    gemm_strided(m, n, k, 1.0, &a_data, k, 1, &b_data, n, 1, 0.0, &mut c_single, n, 1);

    for threads in [1, 2, 4, 8] {
        let mut c_multi = vec![0.0f32; m * n];
        gemm_parallel(m, n, k, 1.0, &a_data, k, 1, &b_data, n, 1, 0.0, &mut c_multi, n, 1, threads);

        assert_eq!(
            c_single, c_multi,
            "Bitwise determinism failed with {threads} threads"
        );
    }
}

#[test]
fn test_x5_autodiff_and_gradient_parity() {
    // Linear layer forward and backward:
    // Y = X * W
    // dX = dY * W^T
    // dW = X^T * dY
    let batch = 8;
    let d_in = 32;
    let d_out = 48;

    let x_data: Vec<f32> = (0..batch * d_in).map(|i| (i as f32 * 0.11).sin()).collect();
    let w_data: Vec<f32> = (0..d_in * d_out).map(|i| (i as f32 * 0.17).cos()).collect();
    let dy_data: Vec<f32> = (0..batch * d_out).map(|i| (i as f32 * 0.23).sin()).collect();

    let x = Tensor::from_vec(&[batch, d_in], x_data).unwrap();
    let w = Tensor::from_vec(&[d_in, d_out], w_data).unwrap();
    let dy = Tensor::from_vec(&[batch, d_out], dy_data).unwrap();

    // Reference forward and backward
    let y_ref = matmul(&x, &w).unwrap();
    let mut w_t = vec![0.0f32; d_out * d_in];
    for r in 0..d_in {
        for c in 0..d_out {
            w_t[c * d_in + r] = w.data()[r * d_out + c];
        }
    }
    let w_t_tensor = Tensor::from_vec(&[d_out, d_in], w_t).unwrap();
    let dx_ref = matmul(&dy, &w_t_tensor).unwrap();

    let mut x_t = vec![0.0f32; d_in * batch];
    for r in 0..batch {
        for c in 0..d_in {
            x_t[c * batch + r] = x.data()[r * d_in + c];
        }
    }
    let x_t_tensor = Tensor::from_vec(&[d_in, batch], x_t).unwrap();
    let dw_ref = matmul(&x_t_tensor, &dy).unwrap();

    // Optimized GEMM forward and backward
    let y_opt = gemm(&x, &w).unwrap();
    let dx_opt = gemm_trans_b(&dy, &w).unwrap();
    let dw_opt = gemm_trans_a(&x, &dy).unwrap();

    let (ok_y, diff_y) = check_parity(y_opt.data(), y_ref.data(), 1e-5, 1e-4);
    assert!(ok_y, "Forward parity failed: {diff_y}");

    let (ok_dx, diff_dx) = check_parity(dx_opt.data(), dx_ref.data(), 1e-5, 1e-4);
    assert!(ok_dx, "Backward dX parity failed: {diff_dx}");

    let (ok_dw, diff_dw) = check_parity(dw_opt.data(), dw_ref.data(), 1e-5, 1e-4);
    assert!(ok_dw, "Backward dW parity failed: {diff_dw}");
}

#[test]
fn test_x5_gate_gx5_execution() {
    let rep = gx5();
    assert_eq!(rep.gate_id, "GX5");
    assert_eq!(rep.status, "PASS", "Gate GX5 must pass: {:?}", rep.reason);

    let dispatched = gates::run(&["gx5"]).unwrap();
    assert_eq!(dispatched.len(), 1);
    assert_eq!(dispatched[0].status, "PASS");
}
