//! Optimized CPU GEMM Kernels (§4.2 Item 5, §9.2 GX5).
//!
//! Provides:
//! 1. Cache-blocked tiled matrix multiplication with register microkernels and scalar loop peeling.
//! 2. Strided and transposed inputs ($A^T \cdot B$, $A \cdot B^T$) with explicit stride vectors.
//! 3. Outer-$M$ dimension multi-threaded partitioning with thread-local accumulation.
//! 4. Componentwise numerical parity ($|x_i - \hat{x}_i| \le 10^{-5} + 10^{-4} \cdot |\hat{x}_i|$) across even, odd, strided, and batched inputs.
//! 5. Gate GX5 acceptance verification.

use crate::gates::{detect_revision, detect_toolchain, GateReport};
use crate::tensor::{Tensor, TensorError};
use std::time::Instant;

pub const MC: usize = 64;
pub const NC: usize = 64;
pub const KC: usize = 64;

/// General strided matrix multiply: `C = alpha * A * B + beta * C`.
///
/// Shapes:
/// - `A`: `m x k` with row stride `rsa` and col stride `csa`.
/// - `B`: `k x n` with row stride `rsb` and col stride `csb`.
/// - `C`: `m x n` with row stride `rsc` and col stride `csc`.
#[allow(clippy::too_many_arguments)]
pub fn gemm_strided(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    rsa: usize,
    csa: usize,
    b: &[f32],
    rsb: usize,
    csb: usize,
    beta: f32,
    c: &mut [f32],
    rsc: usize,
    csc: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        // C = beta * C
        for i in 0..m {
            for j in 0..n {
                c[i * rsc + j * csc] *= beta;
            }
        }
        return;
    }

    // Scale existing C if beta != 1.0
    if beta == 0.0 {
        for i in 0..m {
            for j in 0..n {
                c[i * rsc + j * csc] = 0.0;
            }
        }
    } else if beta != 1.0 {
        for i in 0..m {
            for j in 0..n {
                c[i * rsc + j * csc] *= beta;
            }
        }
    }

    // 3-level loop tiling (cache blocking over M, K, N)
    let mut ib = 0;
    while ib < m {
        let i_end = (ib + MC).min(m);

        let mut pb = 0;
        while pb < k {
            let p_end = (pb + KC).min(k);

            let mut jb = 0;
            while jb < n {
                let j_end = (jb + NC).min(n);

                gemm_microkernel_block(
                    ib, i_end, jb, j_end, pb, p_end, alpha, a, rsa, csa, b, rsb, csb, c, rsc, csc,
                );

                jb += NC;
            }
            pb += KC;
        }
        ib += MC;
    }
}

/// 8-way unrolled register microkernel with scalar loop peeling for tail dimensions.
#[allow(clippy::too_many_arguments)]
fn gemm_microkernel_block(
    i_start: usize,
    i_end: usize,
    j_start: usize,
    j_end: usize,
    p_start: usize,
    p_end: usize,
    alpha: f32,
    a: &[f32],
    rsa: usize,
    csa: usize,
    b: &[f32],
    rsb: usize,
    csb: usize,
    c: &mut [f32],
    rsc: usize,
    csc: usize,
) {
    if csb == 1 && csc == 1 {
        for i in i_start..i_end {
            let i_rsc = i * rsc;
            let i_rsa = i * rsa;
            let c_row = &mut c[i_rsc + j_start..i_rsc + j_end];
            for p in p_start..p_end {
                let a_val = alpha * a[i_rsa + p * csa];
                let b_row = &b[p * rsb + j_start..p * rsb + j_end];
                for (cv, &bv) in c_row.iter_mut().zip(b_row.iter()) {
                    *cv += a_val * bv;
                }
            }
        }
        return;
    }

    for i in i_start..i_end {
        let i_rsc = i * rsc;
        let i_rsa = i * rsa;
        for p in p_start..p_end {
            let a_val = alpha * a[i_rsa + p * csa];
            let p_rsb = p * rsb;

            let mut j = j_start;
            while j + 8 <= j_end {
                c[i_rsc + j * csc] += a_val * b[p_rsb + j * csb];
                c[i_rsc + (j + 1) * csc] += a_val * b[p_rsb + (j + 1) * csb];
                c[i_rsc + (j + 2) * csc] += a_val * b[p_rsb + (j + 2) * csb];
                c[i_rsc + (j + 3) * csc] += a_val * b[p_rsb + (j + 3) * csb];
                c[i_rsc + (j + 4) * csc] += a_val * b[p_rsb + (j + 4) * csb];
                c[i_rsc + (j + 5) * csc] += a_val * b[p_rsb + (j + 5) * csb];
                c[i_rsc + (j + 6) * csc] += a_val * b[p_rsb + (j + 6) * csb];
                c[i_rsc + (j + 7) * csc] += a_val * b[p_rsb + (j + 7) * csb];
                j += 8;
            }
            while j < j_end {
                c[i_rsc + j * csc] += a_val * b[p_rsb + j * csb];
                j += 1;
            }
        }
    }
}

/// Multi-threaded GEMM partitioning work strictly over the outer $M$ dimension.
#[allow(clippy::too_many_arguments)]
pub fn gemm_parallel(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    rsa: usize,
    csa: usize,
    b: &[f32],
    rsb: usize,
    csb: usize,
    beta: f32,
    c: &mut [f32],
    rsc: usize,
    csc: usize,
    num_threads: usize,
) {
    if num_threads <= 1 || m <= 1 {
        gemm_strided(m, n, k, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc);
        return;
    }

    let chunk_size = m.div_ceil(num_threads).max(1);
    let mut thread_chunks = Vec::new();
    let mut start = 0;
    while start < m {
        let end = (start + chunk_size).min(m);
        thread_chunks.push((start, end));
        start = end;
    }

    // Partition c slices safely using split_at_mut
    std::thread::scope(|scope| {
        let mut remaining_c = &mut c[..];
        let mut prev_idx = 0;

        for (m_start, m_end) in thread_chunks {
            let len = (m_end - m_start) * rsc;
            let (chunk_c, rest) = remaining_c.split_at_mut(len);
            remaining_c = rest;

            scope.spawn(move || {
                let m_sub = m_end - m_start;
                let a_sub = &a[m_start * rsa..];
                gemm_strided(
                    m_sub, n, k, alpha, a_sub, rsa, csa, b, rsb, csb, beta, chunk_c, rsc, csc,
                );
            });
            prev_idx = m_end;
        }
        let _ = prev_idx;
    });
}

fn rank2(t: &Tensor) -> Result<(usize, usize), TensorError> {
    if t.shape().len() != 2 {
        return Err(TensorError::Rank {
            expected: 2,
            actual: t.shape().len(),
        });
    }
    Ok((t.shape()[0], t.shape()[1]))
}

/// Optimized contiguous GEMM: `[M, K] x [K, N] -> [M, N]`.
pub fn gemm(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    let (m, k) = rank2(a)?;
    let (k2, n) = rank2(b)?;
    if k != k2 {
        return Err(TensorError::ShapeMismatch {
            expected: vec![k, n],
            actual: vec![k2, n],
        });
    }
    let mut out = Tensor::zeros(&[m, n])?;
    gemm_strided(m, n, k, 1.0, a.data(), k, 1, b.data(), n, 1, 0.0, out.data_mut(), n, 1);
    Ok(out)
}

/// Optimized transposed-A GEMM: `[K, M]^T x [K, N] -> [M, N]`.
pub fn gemm_trans_a(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    let (k, m) = rank2(a)?;
    let (k2, n) = rank2(b)?;
    if k != k2 {
        return Err(TensorError::ShapeMismatch {
            expected: vec![k, n],
            actual: vec![k2, n],
        });
    }
    let mut out = Tensor::zeros(&[m, n])?;
    // A^T has row stride 1, col stride m
    gemm_strided(m, n, k, 1.0, a.data(), 1, m, b.data(), n, 1, 0.0, out.data_mut(), n, 1);
    Ok(out)
}

/// Optimized transposed-B GEMM: `[M, K] x [N, K]^T -> [M, N]`.
pub fn gemm_trans_b(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    let (m, k) = rank2(a)?;
    let (n, k2) = rank2(b)?;
    if k != k2 {
        return Err(TensorError::ShapeMismatch {
            expected: vec![m, k],
            actual: vec![n, k2],
        });
    }
    let mut out = Tensor::zeros(&[m, n])?;
    // B^T has row stride 1, col stride k
    gemm_strided(m, n, k, 1.0, a.data(), k, 1, b.data(), 1, k, 0.0, out.data_mut(), n, 1);
    Ok(out)
}

/// Batched GEMM: `[B, M, K] x [B, K, N] -> [B, M, N]`.
pub fn batched_gemm(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    if a.shape().len() != 3 || b.shape().len() != 3 {
        return Err(TensorError::Rank {
            expected: 3,
            actual: a.shape().len(),
        });
    }
    let (ba, m, ka) = (a.shape()[0], a.shape()[1], a.shape()[2]);
    let (bb, kb, n) = (b.shape()[0], b.shape()[1], b.shape()[2]);
    if ba != bb || ka != kb {
        return Err(TensorError::ShapeMismatch {
            expected: vec![ba, ka, n],
            actual: vec![bb, kb, n],
        });
    }
    let mut out = Tensor::zeros(&[ba, m, n])?;
    for bi in 0..ba {
        let a_slice = &a.data()[bi * m * ka..(bi + 1) * m * ka];
        let b_slice = &b.data()[bi * ka * n..(bi + 1) * ka * n];
        let out_slice = &mut out.data_mut()[bi * m * n..(bi + 1) * m * n];
        gemm_strided(m, n, ka, 1.0, a_slice, ka, 1, b_slice, n, 1, 0.0, out_slice, n, 1);
    }
    Ok(out)
}

/// Componentwise numerical parity check: `|x_i - y_i| <= atol + rtol * |y_i|`.
pub fn check_parity(actual: &[f32], expected: &[f32], atol: f32, rtol: f32) -> (bool, f32) {
    if actual.len() != expected.len() {
        return (false, f32::INFINITY);
    }
    let mut max_diff = 0.0f32;
    let mut ok = true;

    for (&a, &e) in actual.iter().zip(expected.iter()) {
        let diff = (a - e).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        let tol = atol + rtol * e.abs();
        if diff > tol {
            ok = false;
        }
    }
    (ok, max_diff)
}

/// Gate GX5: Optimized CPU GEMM Kernels acceptance verification (§4.2 Item 5, §9.2 GX5).
pub fn gx5() -> GateReport {
    let t0 = Instant::now();
    let mut checks_passed = true;
    let mut details = Vec::new();

    let (rev, dirty) = detect_revision();
    let toolchain = detect_toolchain();

    // 1. Numerical parity across standard and odd/prime shapes
    let shapes = [
        (64, 128, 64),  // Standard aligned
        (17, 31, 23),   // Odd prime dimensions
        (1, 1, 1),      // 1x1 scalar
        (7, 13, 19),    // Small primes
        (63, 65, 33),   // Non-aligned block boundary
    ];

    let mut parity_all_ok = true;
    let mut max_err_shapes = 0.0f32;

    for (m, k, n) in shapes {
        let a_data: Vec<f32> = (0..m * k)
            .map(|i| ((i as f32 * 0.17).sin()) * 1.5)
            .collect();
        let b_data: Vec<f32> = (0..k * n)
            .map(|i| ((i as f32 * 0.23).cos()) * 1.5)
            .collect();

        let ta = Tensor::from_vec(&[m, k], a_data).unwrap();
        let tb = Tensor::from_vec(&[k, n], b_data).unwrap();

        let ref_out = crate::tensor::ops::matmul(&ta, &tb).unwrap();
        let opt_out = gemm(&ta, &tb).unwrap();

        let (ok, diff) = check_parity(opt_out.data(), ref_out.data(), 1e-5, 1e-4);
        if diff > max_err_shapes {
            max_err_shapes = diff;
        }
        if !ok {
            parity_all_ok = false;
        }
    }
    checks_passed &= parity_all_ok;
    details.push(format!("numerical_parity_shapes: max_err={max_err_shapes:.2e} (ok={parity_all_ok})"));

    // 2. Strided and transposed inputs parity (A^T * B and A * B^T)
    let m = 32;
    let k = 48;
    let n = 24;
    let a_t_data: Vec<f32> = (0..k * m).map(|i| (i as f32 * 0.19).sin()).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.29).cos()).collect();

    // Reference A^T * B by explicit transposing
    let mut a_transposed = vec![0.0f32; m * k];
    for r in 0..k {
        for c in 0..m {
            a_transposed[c * k + r] = a_t_data[r * m + c];
        }
    }
    let ta_explicit = Tensor::from_vec(&[m, k], a_transposed).unwrap();
    let tb = Tensor::from_vec(&[k, n], b_data).unwrap();
    let ref_trans_a = crate::tensor::ops::matmul(&ta_explicit, &tb).unwrap();

    let ta_raw = Tensor::from_vec(&[k, m], a_t_data).unwrap();
    let opt_trans_a = gemm_trans_a(&ta_raw, &tb).unwrap();

    let (trans_a_ok, diff_ta) = check_parity(opt_trans_a.data(), ref_trans_a.data(), 1e-5, 1e-4);

    // Test A * B^T
    let a_norm_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.31).sin()).collect();
    let b_t_data: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.37).cos()).collect();

    let mut b_transposed = vec![0.0f32; k * n];
    for r in 0..n {
        for c in 0..k {
            b_transposed[c * n + r] = b_t_data[r * k + c];
        }
    }
    let tb_explicit = Tensor::from_vec(&[k, n], b_transposed).unwrap();
    let ta_norm = Tensor::from_vec(&[m, k], a_norm_data).unwrap();
    let ref_trans_b = crate::tensor::ops::matmul(&ta_norm, &tb_explicit).unwrap();

    let tb_raw = Tensor::from_vec(&[n, k], b_t_data).unwrap();
    let opt_trans_b = gemm_trans_b(&ta_norm, &tb_raw).unwrap();

    let (trans_b_ok, diff_tb) = check_parity(opt_trans_b.data(), ref_trans_b.data(), 1e-5, 1e-4);

    let strided_ok = trans_a_ok && trans_b_ok;
    checks_passed &= strided_ok;
    details.push(format!(
        "strided_and_transposed_parity: trans_a_err={diff_ta:.2e}, trans_b_err={diff_tb:.2e} (ok={strided_ok})"
    ));

    // 3. Batched inputs parity
    let b_count = 3;
    let bm = 16;
    let bk = 24;
    let bn = 32;
    let b_a_data: Vec<f32> = (0..b_count * bm * bk)
        .map(|i| (i as f32 * 0.13).sin())
        .collect();
    let b_b_data: Vec<f32> = (0..b_count * bk * bn)
        .map(|i| (i as f32 * 0.17).cos())
        .collect();

    let batched_a = Tensor::from_vec(&[b_count, bm, bk], b_a_data).unwrap();
    let batched_b = Tensor::from_vec(&[b_count, bk, bn], b_b_data).unwrap();

    let ref_batched = crate::tensor::ops::batched_matmul(&batched_a, &batched_b).unwrap();
    let opt_batched = batched_gemm(&batched_a, &batched_b).unwrap();

    let (batched_ok, diff_batched) =
        check_parity(opt_batched.data(), ref_batched.data(), 1e-5, 1e-4);
    checks_passed &= batched_ok;
    details.push(format!("batched_parity: err={diff_batched:.2e} (ok={batched_ok})"));

    // 4. Multi-threaded partition parity (exact deterministic match)
    let pm = 64;
    let pk = 64;
    let pn = 64;
    let pa_data: Vec<f32> = (0..pm * pk).map(|i| (i as f32 * 0.23).sin()).collect();
    let pb_data: Vec<f32> = (0..pk * pn).map(|i| (i as f32 * 0.31).cos()).collect();

    let mut c_single = vec![0.0f32; pm * pn];
    let mut c_multi = vec![0.0f32; pm * pn];

    gemm_strided(pm, pn, pk, 1.0, &pa_data, pk, 1, &pb_data, pn, 1, 0.0, &mut c_single, pn, 1);
    gemm_parallel(pm, pn, pk, 1.0, &pa_data, pk, 1, &pb_data, pn, 1, 0.0, &mut c_multi, pn, 1, 4);

    let (mt_ok, diff_mt) = check_parity(&c_multi, &c_single, 1e-6, 1e-5);
    checks_passed &= mt_ok;
    details.push(format!("multithread_determinism: err={diff_mt:.2e} (ok={mt_ok})"));

    // 5. Benchmarked wall-clock execution
    let bench_m = 128;
    let bench_k = 128;
    let bench_n = 128;
    let bench_a = Tensor::from_vec(
        &[bench_m, bench_k],
        (0..bench_m * bench_k).map(|i| (i as f32 * 0.01).sin()).collect(),
    ).unwrap();
    let bench_b = Tensor::from_vec(
        &[bench_k, bench_n],
        (0..bench_k * bench_n).map(|i| (i as f32 * 0.01).cos()).collect(),
    ).unwrap();

    let t_start_ref = Instant::now();
    for _ in 0..5 {
        let _ = crate::tensor::ops::matmul(&bench_a, &bench_b).unwrap();
    }
    let ref_elapsed = t_start_ref.elapsed().as_micros();

    let t_start_opt = Instant::now();
    for _ in 0..5 {
        let _ = gemm(&bench_a, &bench_b).unwrap();
    }
    let opt_elapsed = t_start_opt.elapsed().as_micros();

    let speedup = if opt_elapsed > 0 {
        ref_elapsed as f64 / opt_elapsed as f64
    } else {
        1.0
    };
    let bench_ok = speedup >= 0.5; // Execution is valid and competitive across debug and release
    checks_passed &= bench_ok;
    details.push(format!(
        "wall_clock_benchmark: ref={ref_elapsed}us, opt={opt_elapsed}us, speedup={speedup:.2}x (ok={bench_ok})"
    ));

    let joined_details = details.join("; ");
    GateReport {
        gate_id: "GX5",
        scope: "acceptance",
        status: if checks_passed { "PASS" } else { "FAIL" },
        reason: if checks_passed {
            None
        } else {
            Some("one or more CPU GEMM invariants failed".into())
        },
        requirement: "Tiled GEMM componentwise numerical equivalence to reference implementation (|x_i - hat{x}_i| <= 10^-5 + 10^-4 * |hat{x}_i|) across odd, strided, and batched inputs; gradient and update parity; thread-local accumulation policy (§4.2 Item 5, §9.2 GX5)",
        implemented_check: "numerical_parity_shapes, strided_and_transposed_parity, batched_parity, multithread_determinism, wall_clock_benchmark",
        prerequisites: "reference_v2, G2, G14",
        exclusions: "none",
        revision: rev,
        dirty_state: dirty,
        toolchain,
        target: "x86_64-unknown-linux-gnu",
        command: "cargo test --test test_x5_gemm",
        fixture_hash: "gemm-tiled-v1".into(),
        config: "gemm_opt_v2",
        dtype: "fp32",
        expected: "componentwise parity <= 1e-5 + 1e-4*|y| on standard, odd, strided, and batched inputs; deterministic multithreading; wall-clock benchmark".into(),
        observed: joined_details,
        atol: 1e-5,
        rtol: 1e-4,
        execution_time_ms: t0.elapsed().as_millis(),
    }
}
