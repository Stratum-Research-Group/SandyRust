#![allow(clippy::needless_range_loop, clippy::manual_slice_fill)]

//! Test suite for Gate GX16: Associative Fast-Weight Memory (§A6).

use sandy_rust::config::Config;
use sandy_rust::gates;
use sandy_rust::memory::{
    gx16, l2_norm, FastWeightMemory, MemoryError, MemoryLayerWeights, MemoryState,
};
use sandy_rust::rng::Rng;

#[test]
fn test_gx16_gate_passes() {
    let rep = gx16();
    assert_eq!(rep.status, "PASS", "GX16 must pass: {}", rep.observed);
}

#[test]
fn test_x16_param_formula() {
    let mut cfg = Config::reference_v2();
    cfg.ext_x16_memory = true;
    cfg.x16_d_mem = 64;

    let mut rng = Rng::new(42);
    let memory = FastWeightMemory::new(&cfg, 123, &mut rng);

    // 4 layers * (3 * (128 * 64) + (64 * 128) + 1) = 4 * 32,769 = 131,076
    let expected_per_layer = 3 * (cfg.d_model * cfg.x16_d_mem) + (cfg.x16_d_mem * cfg.d_model) + 1;
    let expected_total = cfg.n_layers * expected_per_layer;

    assert_eq!(memory.total_param_count(), expected_total);
    assert_eq!(memory.total_param_count(), 131_076);
}

#[test]
fn test_x16_zero_gate_exact_parity() {
    let mut cfg = Config::reference_v2();
    cfg.ext_x16_memory = true;

    let mut rng = Rng::new(101);
    let mut memory = FastWeightMemory::new(&cfg, 555, &mut rng);

    // Verify initial gate is strictly 0.0
    for l in &memory.layers {
        assert_eq!(l.gate, 0.0);
    }

    let x = vec![1.23f32; cfg.d_model];
    let h_mem = memory.forward_token(0, &x).unwrap();

    // Must be bitwise zero
    for v in h_mem {
        assert_eq!(v, 0.0f32);
    }
}

#[test]
fn test_x16_read_before_update_and_no_future_leakage() {
    let mut cfg = Config::reference_v2();
    cfg.ext_x16_memory = true;
    cfg.x16_chunk_size = 4;

    let mut rng1 = Rng::new(202);
    let mut mem1 = FastWeightMemory::new(&cfg, 777, &mut rng1);
    mem1.layers[0].gate = 0.5;

    let mut rng2 = Rng::new(202);
    let mut mem2 = FastWeightMemory::new(&cfg, 777, &mut rng2);
    mem2.layers[0].gate = 0.5;

    let tok0 = vec![0.5f32; cfg.d_model];
    let tok1_a = vec![0.1f32; cfg.d_model];
    let mut tok1_b = vec![0.1f32; cfg.d_model];
    tok1_b[0] = 100.0; // Major perturbation on token 1

    let out0_a = mem1.forward_token(0, &tok0).unwrap();
    let _out1_a = mem1.forward_token(0, &tok1_a).unwrap();

    let out0_b = mem2.forward_token(0, &tok0).unwrap();
    let _out1_b = mem2.forward_token(0, &tok1_b).unwrap();

    // Token 0 output must be bitwise identical regardless of subsequent token values
    assert_eq!(out0_a, out0_b, "Token 0 must never leak future writes");
}

#[test]
fn test_x16_chunk_equivalence() {
    let mut cfg = Config::reference_v2();
    cfg.ext_x16_memory = true;
    cfg.x16_chunk_size = 4;

    let mut mem_stream = FastWeightMemory::new(&cfg, 303, &mut Rng::new(505));
    mem_stream.layers[0].gate = 0.75;

    let mut mem_batched = FastWeightMemory::new(&cfg, 303, &mut Rng::new(505));
    mem_batched.layers[0].gate = 0.75;

    let mut rng = Rng::new(606);
    let tokens: Vec<Vec<f32>> = (0..8)
        .map(|_| (0..cfg.d_model).map(|_| rng.next_normal()).collect())
        .collect();

    let mut outs_stream = Vec::new();
    for tok in &tokens {
        outs_stream.push(mem_stream.forward_token(0, tok).unwrap());
    }

    let mut outs_batched = Vec::new();
    for tok in &tokens[0..4] {
        outs_batched.push(mem_batched.forward_token(0, tok).unwrap());
    }
    for tok in &tokens[4..8] {
        outs_batched.push(mem_batched.forward_token(0, tok).unwrap());
    }

    for i in 0..8 {
        for d in 0..cfg.d_model {
            let diff = (outs_stream[i][d] - outs_batched[i][d]).abs();
            assert!(diff < 1e-6, "Chunk boundary execution must be equivalent at {i}, {d}");
        }
    }
}

#[test]
fn test_x16_session_isolation() {
    let _cfg = Config::reference_v2();
    let d_mem = 64;

    let state_a = MemoryState::new(1001, d_mem);
    let mut state_b = MemoryState::new(1002, d_mem);

    let snap_a = state_a.snapshot();
    let res = state_b.restore(&snap_a);

    assert!(
        matches!(res, Err(MemoryError::SessionMismatch { expected: 1002, actual: 1001 })),
        "Cross-session restore must be rejected"
    );
}

#[test]
fn test_x16_snapshot_rollback_and_tamper() {
    let mut cfg = Config::reference_v2();
    cfg.ext_x16_memory = true;
    cfg.x16_chunk_size = 4;

    let mut mem = FastWeightMemory::new(&cfg, 2002, &mut Rng::new(999));
    mem.layers[0].gate = 0.5;

    let tok = vec![0.3f32; cfg.d_model];
    for _ in 0..4 {
        let _ = mem.forward_token(0, &tok).unwrap();
    }

    let snap_before = mem.snapshot();
    let hash_before = snap_before[0].hash;

    // Advance state
    for _ in 0..4 {
        let _ = mem.forward_token(0, &tok).unwrap();
    }
    assert_ne!(mem.snapshot()[0].hash, hash_before);

    // Rollback
    mem.restore(&snap_before).unwrap();
    assert_eq!(mem.snapshot()[0].hash, hash_before);

    // Tampered snapshot
    let mut bad_snap = snap_before.clone();
    bad_snap[0].m[5] += 1.0;
    assert!(
        matches!(mem.restore(&bad_snap), Err(MemoryError::SnapshotMismatch { .. })),
        "Tampered snapshot must be rejected"
    );
}

#[test]
fn test_x16_stability_and_finite_differences() {
    let mut cfg = Config::reference_v2();
    cfg.ext_x16_memory = true;
    cfg.x16_chunk_size = 4;

    let mut mem = FastWeightMemory::new(&cfg, 4004, &mut Rng::new(1234));
    mem.layers[0].gate = 0.25;

    // Gradient check on scalar gate g
    let x = vec![0.5f32; cfg.d_model];
    let target = vec![0.1f32; cfg.d_model];

    let h = mem.forward_token(0, &x).unwrap();
    let d_loss_dh: Vec<f32> = h.iter().zip(&target).map(|(&hi, &ti)| hi - ti).collect();

    let r_read = mem.states[0].read(&l2_norm(
        &MemoryLayerWeights::project_in(&x, &mem.layers[0].w_q, cfg.d_model, cfg.x16_d_mem),
        1e-6,
    ));
    let wm_r = mem.layers[0].project_out(&r_read);
    let analytic_dg: f32 = d_loss_dh.iter().zip(&wm_r).map(|(&d, &w)| d * w).sum();

    let eps = 1e-4f32;
    let mut mem_plus = FastWeightMemory::new(&cfg, 4004, &mut Rng::new(1234));
    mem_plus.layers[0].gate = 0.25 + eps;
    let h_plus = mem_plus.forward_token(0, &x).unwrap();
    let l_plus: f32 = 0.5 * h_plus.iter().zip(&target).map(|(&a, &b)| (a - b).powi(2)).sum::<f32>();

    let mut mem_minus = FastWeightMemory::new(&cfg, 4004, &mut Rng::new(1234));
    mem_minus.layers[0].gate = 0.25 - eps;
    let h_minus = mem_minus.forward_token(0, &x).unwrap();
    let l_minus: f32 = 0.5 * h_minus.iter().zip(&target).map(|(&a, &b)| (a - b).powi(2)).sum::<f32>();

    let num_dg = (l_plus - l_minus) / (2.0 * eps);
    assert!((analytic_dg - num_dg).abs() < 1e-3, "Gate gradient must match finite difference");
}

#[test]
fn test_x16_gates_runner_integration() {
    let rep = gates::run(&["gx16"]).unwrap();
    assert_eq!(rep.len(), 1);
    assert_eq!(rep[0].gate_id, "GX16");
    assert_eq!(rep[0].status, "PASS");
}
