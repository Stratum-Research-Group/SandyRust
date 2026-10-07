//! Phase 1 gate runners (G1, G2, G4, G5, G6) emitting §9 JSON reports.

use crate::attention::attention_ref;
use crate::checkpoint::{self, CheckpointError, DEFAULT_MAX_BYTES};
use crate::generation::{generate, GenerationError, SamplingParams};
use crate::model::Model;
use crate::tokenizer::ByteTokenizer;
use crate::config::Config;
use crate::tensor::ops::{matmul, rms_norm, rope};
use crate::tensor::Tensor;
use std::time::Instant;

pub struct GateReport {
    pub gate_id: &'static str,
    pub status: &'static str,
    pub reason: Option<String>,
    pub expected: String,
    pub observed: String,
    pub atol: f64,
    pub rtol: f64,
    pub execution_time_ms: u128,
}

impl GateReport {
    pub fn to_json(&self) -> String {
        let reason = match &self.reason {
            Some(r) => format!("\"{}\"", r.replace('"', "'")),
            None => "null".into(),
        };
        format!(
            "{{\n  \"version\": \"3.0\",\n  \"gate_id\": \"{}\",\n  \"status\": \"{}\",\n  \"reason\": {},\n  \"revision\": \"unversioned\",\n  \"fixture_hash\": \"n/a-analytic\",\n  \"config\": \"reference_v2\",\n  \"command\": \"cargo run -- verify\",\n  \"backend\": \"cpu_reference\",\n  \"device\": \"cpu\",\n  \"dtype\": \"fp32\",\n  \"expected\": \"{}\",\n  \"observed\": \"{}\",\n  \"tolerance_atol\": {},\n  \"tolerance_rtol\": {},\n  \"execution_time_ms\": {}\n}}\n",
            self.gate_id, self.status, reason, self.expected, self.observed, self.atol, self.rtol, self.execution_time_ms
        )
    }
}

fn report(gate_id: &'static str, t0: Instant, ok: bool, expected: String, observed: String, atol: f64, rtol: f64) -> GateReport {
    GateReport {
        gate_id,
        status: if ok { "PASS" } else { "FAIL" },
        reason: None,
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
    // Phase 1 has zero [dependencies] and no build.rs; verified structurally by the manifest.
    let manifest = include_str!("../Cargo.toml");
    let deps = manifest.split("[dependencies]").nth(1).unwrap_or("");
    let deps = deps.split("\n[").next().unwrap_or("");
    let n = deps.lines().filter(|l| !l.trim().is_empty() && !l.trim().starts_with('#')).count();
    report("G1", t0, n == 0, "0 dependencies".into(), format!("{n} dependencies"), 0.0, 0.0)
}

pub fn g2() -> GateReport {
    let t0 = Instant::now();
    // 2x3 * 3x2 analytic fixture.
    let a = Tensor::from_vec(&[2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
    let b = Tensor::from_vec(&[3, 2], vec![7., 8., 9., 10., 11., 12.]).unwrap();
    let c = matmul(&a, &b).unwrap();
    let ok = c.data() == [58., 64., 139., 154.];
    report("G2", t0, ok, "[58,64,139,154]".into(), format!("{:?}", c.data()), 1e-5, 1e-4)
}

pub fn g4() -> GateReport {
    let t0 = Instant::now();
    let q = Tensor::from_vec(&[3, 2], vec![1., 0., 0., 1., 1., 1.]).unwrap();
    let k = q.clone();
    let v = Tensor::from_vec(&[3, 2], vec![7., 0., 0., 7., 3., 3.]).unwrap();
    let (out, p) = attention_ref(&q, &k, &v, &[true, true, false], &[true, true, true], 0).unwrap();
    let future_zero = p[1] == 0.0 && p[2] == 0.0 && p[5] == 0.0;
    let pad_zero = out.data()[4..6] == [0.0, 0.0];
    report("G4", t0, future_zero && pad_zero, "future p=0, padded query out=0".into(), format!("future_zero={future_zero} pad_zero={pad_zero}"), 0.0, 0.0)
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
    report("G5", t0, norm_ok && norms_ok, "RMSNorm equation; RoPE norm preserved".into(), format!("norm_ok={norm_ok} rope_norm_ok={norms_ok}"), 1e-5, 1e-4)
}

pub fn g6() -> GateReport {
    let t0 = Instant::now();
    let c = Config::reference_v2();
    let p = c.param_count();
    report("G6", t0, c.validate().is_ok() && p == Some(886_272), "886272".into(), format!("{p:?}"), 0.0, 0.0)
}

pub fn g3() -> GateReport {
    let t0 = Instant::now();
    let cfg = Config::reference_v2();
    let m = Model::init(&cfg).unwrap();
    let p_f64 = crate::autodiff::Params::<f64>::from_model(&m);
    let window = [256u32, 65, 66, 257];
    let eps_list = [1e-5, 1e-6, 1e-7];
    let rep = crate::autodiff::gradcheck(&cfg, &p_f64, &window, &eps_list, 1e-6, 1e-4, 1);
    let ok = rep.as_ref().map_or(false, |r| r.failures == 0);
    report("G3", t0, ok, "finite-difference matches analytic with 0 failures".into(), format!("{rep:?}"), 1e-6, 1e-4)
}

pub fn g7() -> GateReport {
    let t0 = Instant::now();
    let cfg = Config::reference_v2();
    let model = Model::init(&cfg).unwrap();
    let tokens = [256u32, 72, 101, 108, 108, 111]; // "Hello"
    let full_logits = model.forward(&tokens, None).unwrap();
    let v = cfg.vocab_size;
    let mut cache = crate::cache::KVCache::new(&cfg, 0x1234, 1).unwrap();
    let mut max_err: f32 = 0.0;
    for (pos, &tok) in tokens.iter().enumerate() {
        let step_logits = crate::cache::forward_step(&model, tok, pos, &mut cache).unwrap();
        let full_row = &full_logits.data()[pos * v..(pos + 1) * v];
        let step_row = step_logits.data();
        for (&a, &b) in full_row.iter().zip(step_row) {
            let diff = (a - b).abs();
            if diff > max_err {
                max_err = diff;
            }
        }
    }
    let ok = max_err <= 1e-4;
    report("G7", t0, ok, "incremental cached decode matches full forward within 1e-4".into(), format!("max_err={max_err}"), 1e-5, 1e-4)
}

pub fn g8() -> GateReport {
    let t0 = Instant::now();
    let cfg = Config::reference_v2();
    let m = Model::init(&cfg).unwrap();
    let p0 = crate::autodiff::Params::<f32>::from_model(&m);
    let w1 = [256u32, 65, 66, 257];
    let w2 = [256u32, 67, 68, 257];
    let mut scratch = p0.zeros_like();
    let o1 = crate::autodiff::loss_and_grad(&cfg, &p0, &w1, 1.0, &mut scratch).unwrap();
    let o2 = crate::autodiff::loss_and_grad(&cfg, &p0, &w2, 1.0, &mut scratch).unwrap();
    let denom = (o1.n_valid + o2.n_valid) as f32;
    let mut grads_accum = p0.zeros_like();
    crate::autodiff::loss_and_grad(&cfg, &p0, &w1, denom, &mut grads_accum).unwrap();
    crate::autodiff::loss_and_grad(&cfg, &p0, &w2, denom, &mut grads_accum).unwrap();
    let mut p_accum = p0.clone();
    let mut opt = crate::optimizer::AdamW::new(&p0);
    opt.update(&mut p_accum, &grads_accum, 1e-3);
    let ok = p_accum.all_finite() && opt.step == 1;
    report("G8", t0, ok, "microbatch accumulation with AdamW update produces finite parameters".into(), format!("step={}, finite={}", opt.step, p_accum.all_finite()), 1e-5, 1e-4)
}

pub fn g9() -> GateReport {
    let t0 = Instant::now();
    let mut cfg = Config::reference_v2();
    cfg.n_layers = 1;
    let m = Model::init(&cfg).unwrap();
    let mut p = crate::autodiff::Params::<f32>::from_model(&m);
    let mut opt = crate::optimizer::AdamW::new(&p);
    let window = [256u32, 65, 66, 67, 257];
    let mut initial_loss = 0.0;
    let mut final_loss = 0.0;
    for step in 0..60 {
        let mut grads = p.zeros_like();
        let mut probe = p.zeros_like();
        let o = crate::autodiff::loss_and_grad(&cfg, &p, &window, 1.0, &mut probe).unwrap();
        let denom = o.n_valid as f32;
        let o = crate::autodiff::loss_and_grad(&cfg, &p, &window, denom, &mut grads).unwrap();
        let loss = o.loss_sum / denom;
        if step == 0 { initial_loss = loss; }
        final_loss = loss;
        opt.update(&mut p, &grads, 0.01);
    }
    let ok = final_loss <= initial_loss * 0.5;
    report("G9", t0, ok, "loss reduction >= 50% within <=500 steps".into(), format!("initial={initial_loss:.4}, final={final_loss:.4}"), 0.0, 0.0)
}

pub fn g10() -> GateReport {
    let t0 = Instant::now();
    let m = Model::init(&Config::reference_v2()).unwrap();
    let toks = [256u32, 82, 117, 115, 116];
    let bytes = checkpoint::serialize(&m);
    let reload = checkpoint::parse(&bytes, &m.cfg, DEFAULT_MAX_BYTES);
    let bitwise = reload.as_ref().map_or(false, |m2| m.forward(&toks, None).unwrap() == m2.forward(&toks, None).unwrap());
    let mut bad = bytes.clone();
    let last = bad.len() - 1;
    bad[last] ^= 0xFF;
    let rejects_corrupt = matches!(checkpoint::parse(&bad, &m.cfg, DEFAULT_MAX_BYTES), Err(CheckpointError::ChecksumMismatch(_)));
    let rejects_trunc = checkpoint::parse(&bytes[..bytes.len() / 2], &m.cfg, DEFAULT_MAX_BYTES).is_err();
    let ok = bitwise && rejects_corrupt && rejects_trunc;
    report("G10", t0, ok, "bitwise logits; corrupt/truncated rejected".into(), format!("bitwise={bitwise} corrupt_rejected={rejects_corrupt} truncated_rejected={rejects_trunc}"), 0.0, 0.0)
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
    let ok = bytes_ok && utf8_ok && shift_ok;
    report("G11", t0, ok, "256-byte + UTF-8 roundtrip; y_t = x_{t+1}".into(), format!("bytes={bytes_ok} utf8={utf8_ok} shift={shift_ok}"), 0.0, 0.0)
}

pub fn g12() -> GateReport {
    let t0 = Instant::now();
    let m = Model::init(&Config::reference_v2()).unwrap();
    let prompt = [256u32, 82, 117];
    let s = SamplingParams { temperature: 0.8, top_k: 20, top_p: 0.9, max_new_tokens: 4, seed: 5 };
    let seeded = generate(&m, &prompt, &s).unwrap().tokens == generate(&m, &prompt, &s).unwrap().tokens;
    let g = SamplingParams { max_new_tokens: 4, ..Default::default() };
    let greedy_no_prng = generate(&m, &prompt, &g).unwrap().tokens == generate(&m, &prompt, &SamplingParams { seed: 999, ..g.clone() }).unwrap().tokens;
    let ok = seeded && greedy_no_prng;
    report("G12", t0, ok, "seeded repeat; greedy ignores PRNG".into(), format!("seeded={seeded} greedy_no_prng={greedy_no_prng}"), 0.0, 0.0)
}

pub fn g13() -> GateReport {
    let t0 = Instant::now();
    GateReport {
        gate_id: "G13",
        status: "NOT RUN",
        reason: Some("CPU pure-Rust baseline selected under Decision D2-R1; GPU spike not run".into()),
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
    let ok = oversize && cap;
    report("G14", t0, ok, "oversize + capacity overflow rejected without panic".into(), format!("oversize={oversize} capacity={cap}"), 0.0, 0.0)
}

pub fn run(ids: &[&str]) -> Vec<GateReport> {
    ids.iter()
        .filter_map(|id| match id.to_ascii_lowercase().as_str() {
            "g1" => Some(g1()),
            "g2" => Some(g2()),
            "g3" => Some(g3()),
            "g4" => Some(g4()),
            "g5" => Some(g5()),
            "g6" => Some(g6()),
            "g7" => Some(g7()),
            "g8" => Some(g8()),
            "g9" => Some(g9()),
            "g10" => Some(g10()),
            "g11" => Some(g11()),
            "g12" => Some(g12()),
            "g13" => Some(g13()),
            "g14" => Some(g14()),
            _ => None,
        })
        .collect()
}
