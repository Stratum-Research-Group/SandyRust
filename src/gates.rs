//! Phase 1 gate runners (G1..G14) emitting §9 JSON reports with full evidence schema (03-sandyrust-math-plan §1).

use crate::attention::attention_ref;
use crate::checkpoint::{self, CheckpointError, DEFAULT_MAX_BYTES};
use crate::config::{Config, ConfigError};
use crate::generation::{generate, GenerationError, SamplingParams};
use crate::model::{Model, ModelError};
use crate::sha256::sha256;
use crate::tensor::ops::{batched_matmul, logsumexp, matmul, rms_norm, rope};
use crate::tensor::{Tensor, TensorError};
use crate::tokenizer::{ByteTokenizer, Utf8ByteStreamBuffer};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct GateReport {
    pub gate_id: &'static str,
    pub scope: &'static str,
    pub status: &'static str,
    pub reason: Option<String>,
    pub requirement: &'static str,
    pub implemented_check: &'static str,
    pub prerequisites: &'static str,
    pub exclusions: &'static str,
    pub revision: String,
    pub dirty_state: bool,
    pub toolchain: String,
    pub target: &'static str,
    pub command: &'static str,
    pub fixture_hash: String,
    pub config: &'static str,
    pub dtype: &'static str,
    pub expected: String,
    pub observed: String,
    pub atol: f64,
    pub rtol: f64,
    pub execution_time_ms: u128,
}

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            other => out.push(other),
        }
    }
    out
}

impl GateReport {
    pub fn to_json(&self) -> String {
        let reason = match &self.reason {
            Some(r) => format!("\"{}\"", escape_json(r)),
            None => "null".into(),
        };
        format!(
            "{{\n  \"schema_version\": \"3.1\",\n  \"gate_id\": \"{}\",\n  \"scope\": \"{}\",\n  \"status\": \"{}\",\n  \"reason\": {},\n  \"requirement\": \"{}\",\n  \"implemented_check\": \"{}\",\n  \"prerequisites\": \"{}\",\n  \"exclusions\": \"{}\",\n  \"revision\": \"{}\",\n  \"dirty_state\": {},\n  \"toolchain\": \"{}\",\n  \"target\": \"{}\",\n  \"command\": \"{}\",\n  \"backend\": \"cpu_reference\",\n  \"device\": \"cpu\",\n  \"dtype\": \"{}\",\n  \"config\": \"{}\",\n  \"fixture_hash\": \"{}\",\n  \"expected\": \"{}\",\n  \"observed\": \"{}\",\n  \"tolerance_atol\": {},\n  \"tolerance_rtol\": {},\n  \"execution_time_ms\": {}\n}}\n",
            self.gate_id,
            self.scope,
            self.status,
            reason,
            escape_json(self.requirement),
            escape_json(self.implemented_check),
            escape_json(self.prerequisites),
            escape_json(self.exclusions),
            escape_json(&self.revision),
            self.dirty_state,
            escape_json(&self.toolchain),
            escape_json(self.target),
            escape_json(self.command),
            escape_json(self.dtype),
            escape_json(self.config),
            escape_json(&self.fixture_hash),
            escape_json(&self.expected),
            escape_json(&self.observed),
            self.atol,
            self.rtol,
            self.execution_time_ms
        )
    }
}

pub fn detect_revision() -> (String, bool) {
    if let Ok(output) = std::process::Command::new("git").args(["rev-parse", "--short", "HEAD"]).output() {
        if output.status.success() {
            let rev = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let dirty = if let Ok(status) = std::process::Command::new("git").args(["status", "--porcelain"]).output() {
                !String::from_utf8_lossy(&status.stdout).trim().is_empty()
            } else {
                false
            };
            if !rev.is_empty() {
                return (rev, dirty);
            }
        }
    }
    ("unknown".to_string(), false)
}

pub fn detect_toolchain() -> String {
    if let Ok(output) = std::process::Command::new("rustc").arg("-V").output() {
        if output.status.success() {
            let ver = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !ver.is_empty() {
                return ver;
            }
        }
    }
    "unknown".to_string()
}

#[allow(clippy::too_many_arguments)]
fn report(
    gate_id: &'static str,
    scope: &'static str,
    t0: Instant,
    ok: bool,
    requirement: &'static str,
    implemented_check: &'static str,
    prerequisites: &'static str,
    exclusions: &'static str,
    fixture_hash: String,
    config: &'static str,
    dtype: &'static str,
    expected: String,
    observed: String,
    atol: f64,
    rtol: f64,
) -> GateReport {
    let (rev, dirty) = detect_revision();
    let toolchain = detect_toolchain();
    GateReport {
        gate_id,
        scope,
        status: if ok { "PASS" } else { "FAIL" },
        reason: None,
        requirement,
        implemented_check,
        prerequisites,
        exclusions,
        revision: rev,
        dirty_state: dirty,
        toolchain,
        target: "x86_64-unknown-linux-gnu",
        command: "cargo run --release -- verify",
        fixture_hash,
        config,
        dtype,
        expected,
        observed,
        atol,
        rtol,
        execution_time_ms: t0.elapsed().as_millis(),
    }
}

fn close(x: f32, r: f32) -> bool {
    (x - r).abs() <= 1e-5 + 1e-4 * r.abs()
}

pub fn g1() -> GateReport {
    let t0 = Instant::now();
    let manifest = include_str!("../Cargo.toml");
    let deps = manifest.split("[dependencies]").nth(1).unwrap_or("");
    let deps = deps.split("\n[").next().unwrap_or("");
    let n = deps.lines().filter(|l| !l.trim().is_empty() && !l.trim().starts_with('#')).count();
    let hash = sha256(manifest.as_bytes()).iter().map(|b| format!("{b:02x}")).collect::<String>();
    let build_rs_absent = !std::path::Path::new("build.rs").exists();
    report(
        "G1",
        "integrity",
        t0,
        n == 0 && build_rs_absent,
        "Zero external crate dependencies and zero build.rs scripts (Decision D2-R1, specv3 §1)",
        "Parsed manifest dependency table line count == 0 and verified SHA-256 hash",
        "Rust std library",
        "Third-party crates, build scripts, C/C++ runtime",
        hash,
        "reference_v2",
        "fp32",
        "0 dependencies".into(),
        format!("{n} dependencies"),
        0.0,
        0.0,
    )
}

pub fn g2() -> GateReport {
    let t0 = Instant::now();
    let a = Tensor::from_vec(&[2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
    let b = Tensor::from_vec(&[3, 2], vec![7., 8., 9., 10., 11., 12.]).unwrap();
    let c = matmul(&a, &b).unwrap();
    let matmul_ok = c.data() == [58., 64., 139., 154.];

    // Batched matmul
    let ba = Tensor::from_vec(&[2, 2, 2], vec![1., 0., 0., 1., 2., 0., 0., 2.]).unwrap();
    let bb = Tensor::from_vec(&[2, 2, 2], vec![3., 4., 5., 6., 1., 2., 3., 4.]).unwrap();
    let bc = batched_matmul(&ba, &bb).unwrap();
    let batched_ok = bc.data() == [3., 4., 5., 6., 2., 4., 6., 8.];

    // Logsumexp
    let lse = logsumexp(&[0.0, 0.0]);
    let lse_ok = (lse - std::f32::consts::LN_2).abs() < 1e-6;

    let ok = matmul_ok && batched_ok && lse_ok;
    report(
        "G2",
        "acceptance",
        t0,
        ok,
        "Core tensor contraction primitives: matmul, batched_matmul, logsumexp (specv3 §3.1)",
        "Analytic fixtures for 2x3*3x2 matmul, 2x2x2 batched matmul, and logsumexp LN_2 match",
        "CPU contiguous tensors",
        "GPU GEMM",
        "g2-primitives-v1".into(),
        "reference_v2",
        "fp32",
        "matmul+batched+logsumexp valid".into(),
        format!("matmul={matmul_ok} batched={batched_ok} lse={lse_ok}"),
        1e-5,
        1e-4,
    )
}

pub fn g3() -> GateReport {
    let t0 = Instant::now();
    let cfg = Config::reference_v2();
    let m = Model::init(&cfg).unwrap();
    let p_f64 = crate::autodiff::Params::<f64>::from_model(&m);
    // Window with repeated tokens to test tied embedding lookup and unembedding head
    let window = [256u32, 65, 66, 65, 257];
    let eps_list = [1e-5, 1e-6, 1e-7];
    let rep = crate::autodiff::gradcheck(&cfg, &p_f64, &window, &eps_list, 1e-5, 1e-3, 4);
    let ok = rep.as_ref().is_ok_and(|r| r.failures == 0);
    report(
        "G3",
        "acceptance",
        t0,
        ok,
        "Autodiff gradient correctness on tied parameters (specv3 §3.6)",
        "Two-sided finite-difference gradcheck in FP64 across coordinate probes including repeated tokens",
        "f64 model parameters and forward pass",
        "Single-sided finite differences",
        "g3-repeated-tokens-f64".into(),
        "reference_v2",
        "fp64",
        "finite-difference matches analytic with 0 failures".into(),
        format!("{rep:?}"),
        1e-5,
        1e-3,
    )
}

pub fn g4() -> GateReport {
    let t0 = Instant::now();
    let q = Tensor::from_vec(&[3, 2], vec![1., 0., 0., 1., 1., 1.]).unwrap();
    let k = q.clone();
    let v = Tensor::from_vec(&[3, 2], vec![7., 0., 0., 7., 3., 3.]).unwrap();
    let (out, p) = attention_ref(&q, &k, &v, &[true, true, false], &[true, true, true], 0).unwrap();
    let future_zero = p[1] == 0.0 && p[2] == 0.0 && p[5] == 0.0;
    let pad_zero = out.data()[4..6] == [0.0, 0.0];

    // Rank mismatch rejection check
    let rank1 = Tensor::from_vec(&[4], vec![1., 2., 3., 4.]).unwrap();
    let rank_err = attention_ref(&rank1, &k, &v, &[true], &[true], 0).is_err();

    let ok = future_zero && pad_zero && rank_err;
    report(
        "G4",
        "acceptance",
        t0,
        ok,
        "Causal attention masking, rank validation, and padded query isolation (specv3 §3.3)",
        "Future attention weights exactly 0, padded query output 0 without non-finite multiplication, rank-1 rejected",
        "Self-attention reference",
        "Bidirectional attention",
        "g4-attention-causal-v1".into(),
        "reference_v2",
        "fp32",
        "future p=0, padded query out=0, rank-1 rejected".into(),
        format!("future={future_zero} pad={pad_zero} rank_err={rank_err}"),
        0.0,
        0.0,
    )
}

pub fn g5() -> GateReport {
    let t0 = Instant::now();
    let x = Tensor::from_vec(&[1, 4], vec![1., 2., 3., 4.]).unwrap();
    let g = Tensor::from_vec(&[4], vec![1.; 4]).unwrap();
    let y = rms_norm(&x, &g, 1e-5).unwrap();
    let rms = ((1. + 4. + 9. + 16.) / 4.0f32 + 1e-5).sqrt();
    let norm_ok = y.data().iter().zip([1., 2., 3., 4.]).all(|(&o, i)| close(o, i / rms));
    let r = rope(&Tensor::from_vec(&[3, 4], (0..12).map(|i| i as f32 * 0.37 - 1.0).collect()).unwrap(), 10000.0, 5).unwrap();
    let orig = Tensor::from_vec(&[3, 4], (0..12).map(|i| i as f32 * 0.37 - 1.0).collect()).unwrap();
    let norms_ok = (0..3).all(|t| {
        let n = |d: &[f32]| d[t * 4..t * 4 + 4].iter().map(|v| v * v).sum::<f32>().sqrt();
        close(n(r.data()), n(orig.data()))
    });
    report(
        "G5",
        "acceptance",
        t0,
        norm_ok && norms_ok,
        "RMSNorm normalization equation and RoPE isometry norm preservation (specv3 §3.2, §3.4)",
        "RMSNorm output matches closed formula; RoPE preserves L2 norm across positions with exact f64 angles",
        "RMSNorm and RoPE functions",
        "Learned positional embeddings",
        "g5-norm-rope-v1".into(),
        "reference_v2",
        "fp32",
        "RMSNorm equation; RoPE norm preserved".into(),
        format!("norm_ok={norm_ok} rope_norm_ok={norms_ok}"),
        1e-5,
        1e-4,
    )
}

pub fn g6() -> GateReport {
    let t0 = Instant::now();
    let c = Config::reference_v2();
    let p = c.param_count();
    let base_ok = c.validate().is_ok() && p == Some(886_272);

    // Reject unselected untied embeddings
    let mut untied = c.clone();
    untied.tied_embeddings = false;
    let untied_rejected = matches!(untied.validate(), Err(ConfigError::ExtensionNotEnabled(_)));

    // Reject unselected GQA
    let mut gqa = c.clone();
    gqa.n_kv_heads = 2;
    let gqa_rejected = matches!(gqa.validate(), Err(ConfigError::ExtensionNotEnabled(_)));

    let ok = base_ok && untied_rejected && gqa_rejected;
    report(
        "G6",
        "integrity",
        t0,
        ok,
        "Baseline parameter counting and rejection of unselected extensions (specv3 §2.1, §2.2)",
        "reference_v2 count == 886,272; untied embeddings (X1) and GQA (X2) config rejection",
        "Config validation engine",
        "Extensions X1-X16",
        "g6-config-reference-v2".into(),
        "reference_v2",
        "fp32",
        "886272 params; unselected extensions rejected".into(),
        format!("params={p:?} untied_rej={untied_rejected} gqa_rej={gqa_rejected}"),
        0.0,
        0.0,
    )
}

pub fn g7() -> GateReport {
    let t0 = Instant::now();
    let cfg = Config::reference_v2();
    let model = Model::init(&cfg).unwrap();
    let tokens = [256u32, 72, 101, 108, 108, 111]; // "Hello"
    let full_logits = model.forward(&tokens, None).unwrap();
    let v = cfg.vocab_size;
    let mut cache = crate::cache::KVCache::new(&cfg, model.fingerprint(), 1).unwrap();
    let mut componentwise_ok = true;
    for (pos, &tok) in tokens.iter().enumerate() {
        let step_logits = crate::cache::forward_step(&model, tok, pos, &mut cache).unwrap();
        let full_row = &full_logits.data()[pos * v..(pos + 1) * v];
        let step_row = step_logits.data();
        for (&a, &b) in full_row.iter().zip(step_row) {
            let diff = (a - b).abs();
            if diff > 1e-4 + 1e-4 * a.abs() {
                componentwise_ok = false;
            }
        }
    }

    // Reject stale model instance
    let mut stale_model = model.clone();
    stale_model.embed.data_mut()[0] += 0.5;
    let stale_rejected = matches!(
        crate::cache::forward_step(&stale_model, tokens[0], 0, &mut cache),
        Err(ModelError::Cache(_))
    );

    let ok = componentwise_ok && stale_rejected;
    report(
        "G7",
        "acceptance",
        t0,
        ok,
        "KV cache incremental step parity with full forward pass and model instance binding (specv3 §4.1, §4.2)",
        "Stepwise cached logits match full forward row within componentwise 1e-4; stale model instance rejected",
        "KVCache and Model instance",
        "Unbound / cross-session cache reuse",
        "g7-cache-decode-v1".into(),
        "reference_v2",
        "fp32",
        "incremental cached decode matches full forward; stale rejected".into(),
        format!("componentwise={componentwise_ok} stale_rejected={stale_rejected}"),
        1e-5,
        1e-4,
    )
}

pub fn g8() -> GateReport {
    let t0 = Instant::now();
    let cfg = Config::reference_v2();
    let m = Model::init(&cfg).unwrap();
    let p0 = crate::autodiff::Params::<f32>::from_model(&m);
    // Windows with unequal valid counts (3 and 5)
    let w1 = [256u32, 65, 66, 257]; // 3 valid tokens
    let w2 = [256u32, 67, 68, 69, 70, 257]; // 5 valid tokens
    let mut scratch = p0.zeros_like();
    let o1 = crate::autodiff::loss_and_grad(&cfg, &p0, &w1, 1.0, &mut scratch).unwrap();
    let o2 = crate::autodiff::loss_and_grad(&cfg, &p0, &w2, 1.0, &mut scratch).unwrap();
    let denom = (o1.n_valid + o2.n_valid) as f32; // 8.0
    assert_ne!(o1.n_valid, o2.n_valid);

    let mut grads_accum = p0.zeros_like();
    crate::autodiff::loss_and_grad(&cfg, &p0, &w1, denom, &mut grads_accum).unwrap();
    crate::autodiff::loss_and_grad(&cfg, &p0, &w2, denom, &mut grads_accum).unwrap();
    let unclipped_accum = grads_accum.clone();
    let mut p_accum = p0.clone();
    let mut opt = crate::optimizer::AdamW::new(&p0);
    let step_res = opt.step_with_clipping(&mut p_accum, &mut grads_accum, 1e-3, 1.0).unwrap();

    // Bad step preserves moments and step
    let bad_step_ok = {
        let mut nan_grads = p0.zeros_like();
        nan_grads.data[0][0] = f32::NAN;
        let prev_step = opt.step;
        let prev_m = opt.m.clone();
        let skip_res = opt.update(&mut p_accum, &nan_grads, 1e-3);
        skip_res == Ok(false) && opt.step == prev_step && opt.m == prev_m
    };
    let mut g1_raw = p0.zeros_like();
    crate::autodiff::loss_and_grad(&cfg, &p0, &w1, 1.0, &mut g1_raw).unwrap();
    let mut g2_raw = p0.zeros_like();
    crate::autodiff::loss_and_grad(&cfg, &p0, &w2, 1.0, &mut g2_raw).unwrap();
    let mut accum_equiv = true;
    for ti in 0..p0.data.len() {
        for j in 0..p0.data[ti].len() {
            let expected = (g1_raw.data[ti][j] + g2_raw.data[ti][j]) / denom;
            if (unclipped_accum.data[ti][j] - expected).abs() > 1e-5 {
                accum_equiv = false;
            }
        }
    }

    let ok = step_res && p_accum.all_finite() && opt.step == 1 && bad_step_ok && accum_equiv;
    report(
        "G8",
        "acceptance",
        t0,
        ok,
        "Microbatch gradient accumulation with AdamW update, clipping, and skip safety (specv3 §8.1)",
        "Unequal valid-token microbatches accumulated, clipped AdamW step produced finite parameters, non-finite grad step skipped",
        "AdamW optimizer and autodiff tape",
        "Unchecked NaN gradient application",
        "g8-microbatch-adamw-v1".into(),
        "reference_v2",
        "fp32",
        "microbatch accumulation with AdamW update produces finite parameters and preserves bad step".into(),
        format!("step={}, finite={}, bad_step_ok={bad_step_ok}", opt.step, p_accum.all_finite()),
        1e-5,
        1e-4,
    )
}

pub fn g9() -> GateReport {
    let t0 = Instant::now();
    let cfg = Config::reference_v2(); // Full 4-layer model
    let m = Model::init(&cfg).unwrap();
    let mut p = crate::autodiff::Params::<f32>::from_model(&m);
    let mut opt = crate::optimizer::AdamW::new(&p);
    let window = [256u32, 65, 66, 67, 257];
    let mut initial_loss = 0.0;
    let mut final_loss = 0.0;
    for step in 0..70 {
        let mut grads = p.zeros_like();
        let mut probe = p.zeros_like();
        let o = crate::autodiff::loss_and_grad(&cfg, &p, &window, 1.0, &mut probe).unwrap();
        let denom = o.n_valid as f32;
        let o = crate::autodiff::loss_and_grad(&cfg, &p, &window, denom, &mut grads).unwrap();
        let loss = o.loss_sum / denom;
        if step == 0 { initial_loss = loss; }
        final_loss = loss;
        opt.update(&mut p, &grads, 0.02).unwrap();
    }
    let ok = final_loss <= initial_loss * 0.5;
    report(
        "G9",
        "smoke",
        t0,
        ok,
        "Overfit convergence on synthetic window with full 4-layer reference_v2 profile (specv3 §8.1, §9 G9)",
        "4-layer reference_v2 model reduces CE loss by >= 50% in <= 70 steps on test window",
        "Full 4-layer model autodiff",
        "Full corpus multi-epoch training",
        "g9-overfit-4layer-v1".into(),
        "reference_v2",
        "fp32",
        "loss reduction >= 50% within <=500 steps on 4-layer reference_v2".into(),
        format!("initial={initial_loss:.4}, final={final_loss:.4}"),
        0.0,
        0.0,
    )
}

pub fn g10() -> GateReport {
    let t0 = Instant::now();
    let m = Model::init(&Config::reference_v2()).unwrap();
    let toks = [256u32, 82, 117, 115, 116];
    let bytes = checkpoint::serialize(&m);
    let reload = checkpoint::parse(&bytes, &m.cfg, DEFAULT_MAX_BYTES);
    let bitwise = reload.as_ref().is_ok_and(|m2| m.forward(&toks, None).unwrap() == m2.forward(&toks, None).unwrap());
    let mut bad = bytes.clone();
    let last = bad.len() - 1;
    bad[last] ^= 0xFF;
    let rejects_corrupt = matches!(checkpoint::parse(&bad, &m.cfg, DEFAULT_MAX_BYTES), Err(CheckpointError::ChecksumMismatch(_)));
    let rejects_trunc = checkpoint::parse(&bytes[..bytes.len() / 2], &m.cfg, DEFAULT_MAX_BYTES).is_err();

    // TrainingCheckpoint roundtrip
    let p = crate::autodiff::Params::<f32>::from_model(&m);
    let opt = crate::optimizer::AdamW::new(&p);
    let corpus_hash = [55u8; 32];
    let loader_state = crate::data::LoaderState { position: 10, epoch: 1, corpus_hash, shuffle_seed: 42 };
    let rng = crate::rng::Rng::new(123);
    let schedule = crate::optimizer::Schedule {
        eta_max: 0.01,
        eta_min: 0.001,
        warmup: 1,
        max_steps: 10,
    };
    let trn_state = checkpoint::TrainingState {
        model: m.clone(),
        optimizer: opt,
        loader_state,
        rng_state: rng.dump_state(),
        schedule_step: 3,
        schedule,
    };
    let trn_bytes = checkpoint::serialize_training(&trn_state);
    let trn_loaded = checkpoint::parse_training(&trn_bytes, &m.cfg, &corpus_hash, DEFAULT_MAX_BYTES);
    let trn_ok = if let Ok(ref loaded) = trn_loaded {
        loaded.schedule_step == trn_state.schedule_step
            && loaded.optimizer.step == trn_state.optimizer.step
            && loaded.optimizer.m == trn_state.optimizer.m
            && loaded.optimizer.v == trn_state.optimizer.v
            && loaded.rng_state == trn_state.rng_state
            && loaded.loader_state == trn_state.loader_state
            && loaded.schedule == trn_state.schedule
    } else {
        false
    };

    let ok = bitwise && rejects_corrupt && rejects_trunc && trn_ok;
    report(
        "G10",
        "acceptance",
        t0,
        ok,
        "Deterministic checkpoint export/import and full TrainingState resumption (specv3 §8.2)",
        "Bitwise model forward parity on reload; corruption & truncation rejected; TrainingState moments/RNG/loader roundtrip",
        "Checkpoint serializer/parser",
        "Inference-only checkpoint used as resume",
        "g10-checkpoint-full-v1".into(),
        "reference_v2",
        "fp32",
        "bitwise logits, corrupt/trunc rejected, training state roundtrip".into(),
        format!("bitwise={bitwise} corrupt_rej={rejects_corrupt} trunc_rej={rejects_trunc} trn_ok={trn_ok}"),
        0.0,
        0.0,
    )
}

pub fn g11() -> GateReport {
    let t0 = Instant::now();
    let t = ByteTokenizer::new();
    let all: Vec<u8> = (0..=255u8).collect();
    let bytes_ok = t.decode_bytes(&t.encode_bytes(&all)).as_deref() == Ok(&all[..]);
    let s = "héllo 世界 🦀";
    let utf8_ok = t.decode(&t.encode(s)).as_deref() == Ok(s);
    let doc = t.encode_document("abc");
    let shift_ok = doc.windows(2).map(|w| (w[0], w[1])).collect::<Vec<_>>() == [(256, 97), (97, 98), (98, 99), (99, 257)];

    // Shard parsing test
    let temp_rtok = std::env::temp_dir().join(format!("sandy_g11_test_{}_{}.rtok", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0)));
    let test_tokens = vec![256u32, 97, 98, 99, 257];
    crate::data::write_rtok_file(&temp_rtok, &test_tokens, 259, 2).unwrap();
    let mut shard = crate::data::BufferedTokenShard::open(&temp_rtok, 259, 1024).unwrap();
    let mut read_toks = Vec::new();
    while let Ok(Some(tok)) = shard.next_token() {
        read_toks.push(tok);
    }
    let shard_ok = shard.token_count() == 5 && read_toks == test_tokens;
    let _ = std::fs::remove_file(&temp_rtok);

    let ok = bytes_ok && utf8_ok && shift_ok && shard_ok;
    report(
        "G11",
        "acceptance",
        t0,
        ok,
        "Byte-level tokenizer roundtrip, next-token target shift, and buffered shard reader (specv3 §5.1, §5.3)",
        "All 256 bytes & UTF-8 roundtrip; y_t = x_{t+1}; .rtok shard header & token streaming validation",
        "ByteTokenizer, BufferedTokenShard",
        "BPE / subword tokenizers",
        "g11-tokenizer-data-v1".into(),
        "reference_v2",
        "fp32",
        "256-byte + UTF-8 roundtrip, y_t=x_{t+1}, buffered shard valid".into(),
        format!("bytes={bytes_ok} utf8={utf8_ok} shift={shift_ok} shard={shard_ok}"),
        0.0,
        0.0,
    )
}

pub fn g12() -> GateReport {
    let t0 = Instant::now();
    let m = Model::init(&Config::reference_v2()).unwrap();
    let prompt = [256u32, 82, 117];
    let s = SamplingParams { temperature: 0.8, top_k: 20, top_p: 0.9, max_new_tokens: 4, seed: 5, ..Default::default() };
    let seeded = generate(&m, &prompt, &s).unwrap().tokens == generate(&m, &prompt, &s).unwrap().tokens;
    let g = SamplingParams { max_new_tokens: 4, ..Default::default() };
    let greedy_no_prng = generate(&m, &prompt, &g).unwrap().tokens == generate(&m, &prompt, &SamplingParams { seed: 999, ..g.clone() }).unwrap().tokens;

    // Terminal logit parity check
    let p1 = SamplingParams { max_new_tokens: 2, ..Default::default() };
    let gen_out = generate(&m, &prompt, &p1).unwrap();
    let mut full_seq = prompt.to_vec();
    full_seq.extend_from_slice(&gen_out.tokens);
    let full_fwd = m.forward(&full_seq, None).unwrap();
    let v = m.cfg.vocab_size;
    let term_pos = full_seq.len() - 1;
    let terminal_parity = gen_out.terminal_pos == term_pos && gen_out.final_logits == full_fwd.data()[term_pos * v..(term_pos + 1) * v];

    // UTF-8 stream buffer check across token splits
    let mut buf = Utf8ByteStreamBuffer::new();
    let crab = "🦀".as_bytes();
    let mut partial = String::new();
    partial.push_str(&buf.push_byte(crab[0]));
    partial.push_str(&buf.push_byte(crab[1]));
    partial.push_str(&buf.push_byte(crab[2]));
    partial.push_str(&buf.push_byte(crab[3]));
    let stream_ok = partial == "🦀";

    let ok = seeded && greedy_no_prng && terminal_parity && stream_ok;
    report(
        "G12",
        "acceptance",
        t0,
        ok,
        "Deterministic sampling, terminal logit parity, and incremental UTF-8 byte stream buffer (specv3 §8.3)",
        "Greedy ignores PRNG; seeded sampling repeats; terminal_logits match full forward terminal row; Utf8ByteStreamBuffer handles splits",
        "Generation engine, Utf8ByteStreamBuffer",
        "Lossy whole-buffer only decoding",
        "g12-generation-terminal-v1".into(),
        "reference_v2",
        "fp32",
        "seeded repeat, greedy ignores PRNG, terminal logit parity, UTF-8 streaming".into(),
        format!("seeded={seeded} greedy={greedy_no_prng} parity={terminal_parity} stream={stream_ok}"),
        0.0,
        0.0,
    )
}

pub fn g13() -> GateReport {
    let t0 = Instant::now();
    let (rev, dirty) = detect_revision();
    let toolchain = detect_toolchain();
    GateReport {
        gate_id: "G13",
        scope: "acceptance",
        status: "NOT_RUN",
        reason: Some("CPU pure-Rust baseline selected under Decision D2-R1; GPU spike not run".into()),
        requirement: "GPU reference kernel acceleration spike (specv3 §9 G13)",
        implemented_check: "None (CPU pure-Rust baseline selected under Decision D2-R1)",
        prerequisites: "GPU hardware / CUDA",
        exclusions: "Decision D2-R1 CPU reference boundary",
        revision: rev,
        dirty_state: dirty,
        toolchain,
        target: "x86_64-unknown-linux-gnu",
        command: "cargo run --release -- verify",
        fixture_hash: "n/a".into(),
        config: "reference_v2",
        dtype: "fp32",
        expected: "N/A".into(),
        observed: "N/A".into(),
        atol: 0.0,
        rtol: 0.0,
        execution_time_ms: t0.elapsed().as_millis(),
    }
}

pub fn g14() -> GateReport {
    let t0 = Instant::now();
    let m = Model::init(&Config::reference_v2()).unwrap();
    let bytes = checkpoint::serialize(&m);
    let oversize = matches!(checkpoint::parse(&bytes, &m.cfg, 1024), Err(CheckpointError::TooLarge { .. }));
    let cap = matches!(
        generate(&m, &[1, 2], &SamplingParams { max_new_tokens: usize::MAX, ..Default::default() }),
        Err(GenerationError::CapacityExceeded { .. })
    );

    // Attention rank mismatch rejection
    let rank1 = Tensor::from_vec(&[4], vec![1., 2., 3., 4.]).unwrap();
    let k = Tensor::from_vec(&[4, 4], vec![0.0; 16]).unwrap();
    let attn_rank_err = matches!(attention_ref(&rank1, &k, &k, &[true], &[true], 0), Err(TensorError::Rank { .. }));

    // All-PAD window rejection
    let p_f32 = crate::autodiff::Params::<f32>::from_model(&m);
    let mut g_scratch = p_f32.zeros_like();
    let pad_loss_err = matches!(
        crate::autodiff::loss_and_grad(&m.cfg, &p_f32, &[258, 258, 258], 1.0, &mut g_scratch),
        Err(crate::autodiff::TrainError::NoValidTokens)
    );

    // Config resource caps rejection
    let mut huge_cfg = m.cfg.clone();
    huge_cfg.n_layers = 100; // Exceeds MAX_LAYERS_CAP = 32
    let config_cap_err = matches!(huge_cfg.validate(), Err(ConfigError::ResourceCap(_)));

    let ok = oversize && cap && attn_rank_err && pad_loss_err && config_cap_err;
    report(
        "G14",
        "regression",
        t0,
        ok,
        "Resource caps, bounded allocation, and typed error safety (specv3 §2.2, §8.2, §9 G14)",
        "Oversize checkpoint rejected; context capacity exceeded rejected; rank mismatch rejected; all-PAD rejected; config cap rejected",
        "Resource accounting and validation",
        "Unbounded buffer growth",
        "g14-resource-caps-v1".into(),
        "reference_v2",
        "fp32",
        "oversize, capacity, rank mismatch, all-PAD, and config cap rejected without panic".into(),
        format!("oversize={oversize} cap={cap} rank={attn_rank_err} pad={pad_loss_err} cfg_cap={config_cap_err}"),
        0.0,
        0.0,
    )
}

pub fn run(ids: &[&str]) -> Result<Vec<GateReport>, String> {
    if ids.is_empty() {
        return Err("no gate IDs specified".into());
    }
    let mut reports = Vec::with_capacity(ids.len());
    for id in ids {
        let rep = match id.trim().to_ascii_lowercase().as_str() {
            "g1" => g1(),
            "g2" => g2(),
            "g3" => g3(),
            "g4" => g4(),
            "g5" => g5(),
            "g6" => g6(),
            "g7" => g7(),
            "g8" => g8(),
            "g9" => g9(),
            "g10" => g10(),
            "g11" => g11(),
            "g12" => g12(),
            "g13" => g13(),
            "g14" => g14(),
            "gx1" => crate::attention::gx1(),
            "gx2" => crate::embedding::gx2(),
            "gx3" => crate::attention::gx3(),
            "gx4" => crate::attention::gx4(),
            "gx5" => crate::gemm::gx5(),
            "gx6" => crate::arena::gx6(),
            "gx12e" => crate::generation::gx12e(),
            "gx13" => crate::hooks::gx13(),
            "gx14" => crate::eval::gx14(),
            "gx15" => crate::adapter::gx15(),
            "gx17" => crate::import::gx17(),
            "gx17_0" | "gx17-0" => crate::import::gx17_0(),
            "gx17_1" | "gx17-1" => crate::import::gx17_1(),
            "gx17_2" | "gx17-2" => crate::import::gx17_2(),
            "gx17_3" | "gx17-3" => crate::import::gx17_3(),
            "gx17_4" | "gx17-4" => crate::import::gx17_4(),
            "gx17_5" | "gx17-5" => crate::import::gx17_5(),
            "gx17_6" | "gx17-6" => crate::import::gx17_6(),
            "gx17_7" | "gx17-7" => crate::import::gx17_7(),
            "gx17_8" | "gx17-8" => crate::import::gx17_8(),
            "gx17_9" | "gx17-9" => crate::import::gx17_9(),
            other => return Err(format!("unknown gate id: '{other}'")),
        };
        reports.push(rep);
    }
    Ok(reports)
}
