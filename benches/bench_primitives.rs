//! Zero-dependency microbenchmarks for core primitives (§2.2, §3.2, §3.3).

use sandy_rust::attention::flash::flash_attention;
use sandy_rust::attention::attention_ref;
use sandy_rust::gemm::gemm;
use sandy_rust::tensor::ops::{matmul, rms_norm, rope};
use sandy_rust::tensor::Tensor;
use std::time::Instant;

fn bench_matmul() {
    let m = 64;
    let k = 64;
    let n = 64;
    let a_data = vec![0.1f32; m * k];
    let b_data = vec![0.2f32; k * n];
    let ta = Tensor::from_vec(&[m, k], a_data).unwrap();
    let tb = Tensor::from_vec(&[k, n], b_data).unwrap();

    // Warmup
    for _ in 0..10 {
        let _ = matmul(&ta, &tb).unwrap();
        let _ = gemm(&ta, &tb).unwrap();
    }

    let iters = 500;

    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = matmul(&ta, &tb).unwrap();
    }
    let ref_elapsed = t0.elapsed();

    let t1 = Instant::now();
    for _ in 0..iters {
        let _ = gemm(&ta, &tb).unwrap();
    }
    let opt_elapsed = t1.elapsed();

    let ref_us = ref_elapsed.as_micros() as f64 / iters as f64;
    let opt_us = opt_elapsed.as_micros() as f64 / iters as f64;
    let speedup = ref_us / opt_us.max(0.001);

    println!(
        "Benchmark GEMM [64x64x64]: reference = {:.2} us, tiled = {:.2} us (speedup: {:.2}x)",
        ref_us, opt_us, speedup
    );
}

fn bench_attention() {
    let t = 64;
    let dh = 32;
    let q_data = vec![0.05f32; t * dh];
    let k_data = vec![0.05f32; t * dh];
    let v_data = vec![0.05f32; t * dh];
    let q = Tensor::from_vec(&[t, dh], q_data).unwrap();
    let k = Tensor::from_vec(&[t, dh], k_data).unwrap();
    let v = Tensor::from_vec(&[t, dh], v_data).unwrap();
    let valid = vec![true; t];

    // Warmup
    for _ in 0..10 {
        let _ = attention_ref(&q, &k, &v, &valid, &valid, 0).unwrap();
        let _ = flash_attention(&q, &k, &v, &valid, &valid, 0, None, 16, 16).unwrap();
    }

    let iters = 200;

    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = attention_ref(&q, &k, &v, &valid, &valid, 0).unwrap();
    }
    let ref_elapsed = t0.elapsed();

    let t1 = Instant::now();
    for _ in 0..iters {
        let _ = flash_attention(&q, &k, &v, &valid, &valid, 0, None, 16, 16).unwrap();
    }
    let opt_elapsed = t1.elapsed();

    let ref_us = ref_elapsed.as_micros() as f64 / iters as f64;
    let opt_us = opt_elapsed.as_micros() as f64 / iters as f64;
    let speedup = ref_us / opt_us.max(0.001);

    println!(
        "Benchmark Attention [T=64, dh=32]: reference = {:.2} us, FlashAttention-2 = {:.2} us (speedup: {:.2}x)",
        ref_us, opt_us, speedup
    );
}

fn bench_rmsnorm_and_rope() {
    let t = 64;
    let d = 256;
    let x_data = vec![0.01f32; t * d];
    let gamma = vec![1.0f32; d];
    let tx = Tensor::from_vec(&[t, d], x_data).unwrap();
    let tg = Tensor::from_vec(&[d], gamma).unwrap();

    let iters = 1000;

    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = rms_norm(&tx, &tg, 1e-5).unwrap();
    }
    let norm_us = t0.elapsed().as_micros() as f64 / iters as f64;

    let tr_in = Tensor::from_vec(&[t, 32], vec![0.02f32; t * 32]).unwrap();
    let t1 = Instant::now();
    for _ in 0..iters {
        let _ = rope(&tr_in, 10000.0, 0).unwrap();
    }
    let rope_us = t1.elapsed().as_micros() as f64 / iters as f64;

    println!(
        "Benchmark RMSNorm [64x256]: {:.2} us | RoPE [64x32]: {:.2} us",
        norm_us, rope_us
    );
}

fn main() {
    println!("=== SandyRust Microbenchmarks (Pure-Rust std) ===");
    bench_matmul();
    bench_attention();
    bench_rmsnorm_and_rope();
    println!("==================================================");
}

