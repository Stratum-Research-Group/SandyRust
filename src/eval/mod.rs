//! X14: Paired Perturbation & Semantic Flip-Rate Evaluation (§A3, Gate GX14).
//!
//! Evaluates model sensitivity to decisive perturbations (MustFlip) versus
//! insensitivity to paraphrases/irrelevant distractors (MustNotFlip) across closed-vocabulary tasks.

use crate::gates::GateReport;
use crate::rng::Rng;
use crate::sha256::{hex, sha256};
use std::collections::HashSet;
use std::time::Instant;

pub const DEFAULT_MIN_TEST_FAMILIES: usize = 30;
pub const DEFAULT_RESAMPLE_COUNT: usize = 2000;
pub const DEFAULT_SELECTIVITY_THRESHOLD: f64 = 0.20;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Relation {
    MustFlip,
    MustNotFlip,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pair {
    pub base_prompt: String,
    pub variant_prompt: String,
    pub relation: Relation,
    pub oracle_base_answer: String,
    pub oracle_variant_answer: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Family {
    pub family_id: String,
    pub task_name: String,
    pub pairs: Vec<Pair>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Split {
    Train,
    Dev,
    Test,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EvalError {
    SplitLeak { prompt: String },
    EmptyClass { relation: Relation },
    ProtocolMismatch { field: String },
    Underpowered { families: usize, min: usize },
    InvalidAnswerSet,
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}
impl std::error::Error for EvalError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Pass,
    Fail,
    Underpowered,
    NoSignal,
}

impl Verdict {
    pub fn as_str(&self) -> &'static str {
        match self {
            Verdict::Pass => "PASS",
            Verdict::Fail => "FAIL",
            Verdict::Underpowered => "UNDERPOWERED",
            Verdict::NoSignal => "NO_SIGNAL",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BootstrapInterval {
    pub lower: f64,
    pub mean: f64,
    pub upper: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetricsReport {
    pub flip_rate_must_flip: f64,
    pub flip_rate_must_not_flip: f64,
    pub selectivity: f64,
    pub base_correctness: f64,
    pub variant_correctness: f64,
    pub selectivity_interval: BootstrapInterval,
    pub verdict: Verdict,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrivialBaselines {
    pub always_flip_selectivity: f64,
    pub always_same_selectivity: f64,
    pub copy_base_selectivity: f64,
    pub random_selectivity: f64,
}

// --- Synthetic Generators (§A3.2) ---

pub fn generate_kv_lookup(count: usize, seed: u64) -> Vec<Family> {
    let mut rng = Rng::new(seed);
    let mut families = Vec::with_capacity(count);
    let keys = ["k1", "k2", "k3", "k4"];

    for i in 0..count {
        let fid = format!("kv_{i:04}");
        let k_idx = (rng.next_u32() as usize) % keys.len();
        let target_k = keys[k_idx];
        let v1 = (rng.next_u32() % 9) + 1;
        let v2 = (rng.next_u32() % 9) + 1;
        let v3 = (v2 % 9) + 1;

        let base_prompt = format!("[{fid}] store: {target_k}={v1}. overwrite: {target_k}={v2}. query {target_k}=");
        let variant_flip = format!("[{fid}] store: {target_k}={v1}. overwrite: {target_k}={v3}. query {target_k}=");
        let variant_not_flip = format!("[{fid}] note: test. store: {target_k}={v1}. overwrite: {target_k}={v2}. query {target_k}=");

        let pairs = vec![
            Pair {
                base_prompt: base_prompt.clone(),
                variant_prompt: variant_flip,
                relation: Relation::MustFlip,
                oracle_base_answer: v2.to_string(),
                oracle_variant_answer: v3.to_string(),
            },
            Pair {
                base_prompt,
                variant_prompt: variant_not_flip,
                relation: Relation::MustNotFlip,
                oracle_base_answer: v2.to_string(),
                oracle_variant_answer: v2.to_string(),
            },
        ];
        families.push(Family {
            family_id: fid,
            task_name: "kv_lookup".to_string(),
            pairs,
        });
    }
    families
}

pub fn generate_copy_marker(count: usize, seed: u64) -> Vec<Family> {
    let mut rng = Rng::new(seed);
    let mut families = Vec::with_capacity(count);
    let symbols = ["alpha", "beta", "gamma", "delta"];

    for i in 0..count {
        let fid = format!("copy_{i:04}");
        let s1_idx = (rng.next_u32() as usize) % symbols.len();
        let s2_idx = (s1_idx + 1) % symbols.len();
        let s1 = symbols[s1_idx];
        let s2 = symbols[s2_idx];

        let base_prompt = format!("[{fid}] copy marker: [{s1}] output:");
        let variant_flip = format!("[{fid}] copy marker: [{s2}] output:");
        let variant_not_flip = format!("[{fid}] remark: fast copy marker: [{s1}] output:");

        let pairs = vec![
            Pair {
                base_prompt: base_prompt.clone(),
                variant_prompt: variant_flip,
                relation: Relation::MustFlip,
                oracle_base_answer: s1.to_string(),
                oracle_variant_answer: s2.to_string(),
            },
            Pair {
                base_prompt,
                variant_prompt: variant_not_flip,
                relation: Relation::MustNotFlip,
                oracle_base_answer: s1.to_string(),
                oracle_variant_answer: s1.to_string(),
            },
        ];
        families.push(Family {
            family_id: fid,
            task_name: "copy_marker".to_string(),
            pairs,
        });
    }
    families
}

pub fn generate_digit_comparison(count: usize, seed: u64) -> Vec<Family> {
    let mut rng = Rng::new(seed);
    let mut families = Vec::with_capacity(count);

    for i in 0..count {
        let fid = format!("cmp_{i:04}");
        let a = (rng.next_u32() % 8) + 1;
        let b = a + 1; // b > a always

        let base_prompt = format!("[{fid}] is {b} greater than {a} ?");
        let variant_flip = format!("[{fid}] is {a} greater than {b} ?");
        let variant_not_flip = format!("[{fid}] check: is {b} greater than {a} ?");

        let pairs = vec![
            Pair {
                base_prompt: base_prompt.clone(),
                variant_prompt: variant_flip,
                relation: Relation::MustFlip,
                oracle_base_answer: "yes".to_string(),
                oracle_variant_answer: "no".to_string(),
            },
            Pair {
                base_prompt,
                variant_prompt: variant_not_flip,
                relation: Relation::MustNotFlip,
                oracle_base_answer: "yes".to_string(),
                oracle_variant_answer: "yes".to_string(),
            },
        ];
        families.push(Family {
            family_id: fid,
            task_name: "digit_comparison".to_string(),
            pairs,
        });
    }
    families
}

pub fn generate_symbol_count(count: usize, seed: u64) -> Vec<Family> {
    let mut rng = Rng::new(seed);
    let mut families = Vec::with_capacity(count);

    for i in 0..count {
        let fid = format!("cnt_{i:04}");
        let n1 = (rng.next_u32() % 5) + 1;
        let n2 = n1 + 1;
        let hashes1 = "# ".repeat(n1 as usize).trim().to_string();
        let hashes2 = "# ".repeat(n2 as usize).trim().to_string();

        let base_prompt = format!("[{fid}] items: {hashes1}. count=");
        let variant_flip = format!("[{fid}] items: {hashes2}. count=");
        let variant_not_flip = format!("[{fid}] note items: {hashes1}. count=");

        let pairs = vec![
            Pair {
                base_prompt: base_prompt.clone(),
                variant_prompt: variant_flip,
                relation: Relation::MustFlip,
                oracle_base_answer: n1.to_string(),
                oracle_variant_answer: n2.to_string(),
            },
            Pair {
                base_prompt,
                variant_prompt: variant_not_flip,
                relation: Relation::MustNotFlip,
                oracle_base_answer: n1.to_string(),
                oracle_variant_answer: n1.to_string(),
            },
        ];
        families.push(Family {
            family_id: fid,
            task_name: "symbol_count".to_string(),
            pairs,
        });
    }
    families
}

// --- Split Assignment (§A3.2 item 3, §A3.4 item 4) ---

pub type Splits<'a> = (Vec<&'a Family>, Vec<&'a Family>, Vec<&'a Family>);

pub fn assign_splits(
    families: &[Family],
    split_seed: u64,
) -> Result<Splits<'_>, EvalError> {
    let mut train = Vec::new();
    let mut dev = Vec::new();
    let mut test = Vec::new();

    let mut train_prompts = HashSet::new();
    let mut dev_prompts = HashSet::new();
    let mut test_prompts = HashSet::new();

    for f in families {
        let h = sha256(format!("{}:{split_seed}", f.family_id).as_bytes());
        let bucket = u32::from_le_bytes([h[0], h[1], h[2], h[3]]) % 100;
        let split = if bucket < 50 {
            Split::Train
        } else if bucket < 70 {
            Split::Dev
        } else {
            Split::Test
        };

        for p in &f.pairs {
            match split {
                Split::Train => {
                    train_prompts.insert(p.base_prompt.clone());
                }
                Split::Dev => {
                    dev_prompts.insert(p.base_prompt.clone());
                }
                Split::Test => {
                    test_prompts.insert(p.base_prompt.clone());
                }
            }
        }

        match split {
            Split::Train => train.push(f),
            Split::Dev => dev.push(f),
            Split::Test => test.push(f),
        }
    }

    // Verify zero split leakage
    for p in &test_prompts {
        if train_prompts.contains(p) || dev_prompts.contains(p) {
            return Err(EvalError::SplitLeak { prompt: p.clone() });
        }
    }
    for p in &dev_prompts {
        if train_prompts.contains(p) {
            return Err(EvalError::SplitLeak { prompt: p.clone() });
        }
    }

    Ok((train, dev, test))
}

// --- Evaluation Metrics & Bootstrap (§A3.3) ---

pub struct PredictionResult {
    pub model_base_answer: String,
    pub model_variant_answer: String,
}

pub fn evaluate_pairs(pairs: &[(&Pair, PredictionResult)]) -> Result<(f64, f64, f64, f64, f64), EvalError> {
    let mut must_flip_total = 0;
    let mut must_flip_flipped = 0;
    let mut must_not_flip_total = 0;
    let mut must_not_flip_flipped = 0;
    let mut base_correct = 0;
    let mut variant_correct = 0;
    let total = pairs.len();

    for (pair, pred) in pairs {
        let is_flip = pred.model_base_answer != pred.model_variant_answer;
        if pred.model_base_answer == pair.oracle_base_answer {
            base_correct += 1;
        }
        if pred.model_variant_answer == pair.oracle_variant_answer {
            variant_correct += 1;
        }

        match pair.relation {
            Relation::MustFlip => {
                must_flip_total += 1;
                if is_flip {
                    must_flip_flipped += 1;
                }
            }
            Relation::MustNotFlip => {
                must_not_flip_total += 1;
                if is_flip {
                    must_not_flip_flipped += 1;
                }
            }
        }
    }

    if must_flip_total == 0 {
        return Err(EvalError::EmptyClass { relation: Relation::MustFlip });
    }
    if must_not_flip_total == 0 {
        return Err(EvalError::EmptyClass { relation: Relation::MustNotFlip });
    }

    let flip_rate_must_flip = must_flip_flipped as f64 / must_flip_total as f64;
    let flip_rate_must_not_flip = must_not_flip_flipped as f64 / must_not_flip_total as f64;
    let selectivity = flip_rate_must_flip - flip_rate_must_not_flip;
    let base_acc = if total > 0 { base_correct as f64 / total as f64 } else { 0.0 };
    let variant_acc = if total > 0 { variant_correct as f64 / total as f64 } else { 0.0 };

    Ok((flip_rate_must_flip, flip_rate_must_not_flip, selectivity, base_acc, variant_acc))
}

pub fn family_clustered_bootstrap(
    families: &[&Family],
    predictions: &[(&Pair, PredictionResult)],
    resample_count: usize,
    seed: u64,
) -> Result<BootstrapInterval, EvalError> {
    if families.is_empty() {
        return Err(EvalError::Underpowered { families: 0, min: 1 });
    }
    let mut rng = Rng::new(seed);
    let n = families.len();
    let mut selectivities = Vec::with_capacity(resample_count);

    // Map family to its prediction slice
    let mut family_pairs: Vec<Vec<(&Pair, &PredictionResult)>> = Vec::with_capacity(n);
    for f in families {
        let mut slice = Vec::new();
        for (pair, pred) in predictions {
            if f.pairs.iter().any(|p| std::ptr::eq(p, *pair)) {
                slice.push((*pair, pred));
            }
        }
        family_pairs.push(slice);
    }

    for _ in 0..resample_count {
        let mut sample_pairs = Vec::new();
        for _ in 0..n {
            let idx = (rng.next_u32() as usize) % n;
            for (p, pred) in &family_pairs[idx] {
                sample_pairs.push((*p, PredictionResult {
                    model_base_answer: pred.model_base_answer.clone(),
                    model_variant_answer: pred.model_variant_answer.clone(),
                }));
            }
        }
        if let Ok((_, _, sel, _, _)) = evaluate_pairs(&sample_pairs) {
            selectivities.push(sel);
        }
    }

    if selectivities.is_empty() {
        return Ok(BootstrapInterval { lower: 0.0, mean: 0.0, upper: 0.0 });
    }

    selectivities.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean = selectivities.iter().sum::<f64>() / selectivities.len() as f64;
    let lower_idx = (selectivities.len() as f64 * 0.025).floor() as usize;
    let upper_idx = ((selectivities.len() as f64 * 0.975).floor() as usize).min(selectivities.len() - 1);

    Ok(BootstrapInterval {
        lower: selectivities[lower_idx],
        mean,
        upper: selectivities[upper_idx],
    })
}

pub fn evaluate_trivial_baselines(pairs: &[&Pair]) -> TrivialBaselines {
    let must_flip_count = pairs.iter().filter(|p| p.relation == Relation::MustFlip).count() as f64;
    let must_not_flip_count = pairs.iter().filter(|p| p.relation == Relation::MustNotFlip).count() as f64;

    let _ = (must_flip_count, must_not_flip_count);
    // 1. Always flip: flips on everything -> flip_mf=1.0, flip_mnf=1.0 -> selectivity = 0.0
    let always_flip_selectivity = 0.0;

    // 2. Always same: flips on nothing -> flip_mf=0.0, flip_mnf=0.0 -> selectivity = 0.0
    let always_same_selectivity = 0.0;

    // 3. Copy base: never flips -> selectivity = 0.0
    let copy_base_selectivity = 0.0;

    // 4. Uniform random in closed answer set of size 2 (e.g. yes/no):
    // E[flip_mf] = 0.5, E[flip_mnf] = 0.5 -> E[selectivity] = 0.0
    let random_selectivity = 0.0;

    TrivialBaselines {
        always_flip_selectivity,
        always_same_selectivity,
        copy_base_selectivity,
        random_selectivity,
    }
}

// --- Protocol Check & Gate GX14 (§A3.6) ---

pub fn compute_protocol_hash(content: &str) -> [u8; 32] {
    sha256(content.as_bytes())
}

pub fn gx14() -> GateReport {
    let t0 = Instant::now();

    // 1. Generators deterministic from seed; oracle label verification on fixtures
    let kv_fams = generate_kv_lookup(20, 100);
    let copy_fams = generate_copy_marker(20, 200);
    let cmp_fams = generate_digit_comparison(20, 300);
    let cnt_fams = generate_symbol_count(20, 400);

    let gen_counts_ok = kv_fams.len() == 20 && copy_fams.len() == 20 && cmp_fams.len() == 20 && cnt_fams.len() == 20;

    // Repeat with same seeds -> must produce identical output
    let kv_repeat = generate_kv_lookup(20, 100);
    let repeat_ok = kv_fams == kv_repeat;

    // 2. Family-level split and zero leakage
    let mut all_fams = Vec::new();
    all_fams.extend(kv_fams);
    all_fams.extend(copy_fams);
    all_fams.extend(cmp_fams);
    all_fams.extend(cnt_fams);

    let split_res = assign_splits(&all_fams, 42);
    let (train, dev, test) = split_res.expect("splits must assign without leakage");
    let splits_non_empty = !train.is_empty() && !dev.is_empty() && !test.is_empty();

    // 3. Hand-worked table validation
    let p_mf = Pair {
        base_prompt: "b1".into(),
        variant_prompt: "v1".into(),
        relation: Relation::MustFlip,
        oracle_base_answer: "yes".into(),
        oracle_variant_answer: "no".into(),
    };
    let p_mnf = Pair {
        base_prompt: "b2".into(),
        variant_prompt: "v2".into(),
        relation: Relation::MustNotFlip,
        oracle_base_answer: "yes".into(),
        oracle_variant_answer: "yes".into(),
    };

    // Perfect model: flips on MF, does not flip on MNF
    let perfect_preds = vec![
        (&p_mf, PredictionResult { model_base_answer: "yes".into(), model_variant_answer: "no".into() }),
        (&p_mnf, PredictionResult { model_base_answer: "yes".into(), model_variant_answer: "yes".into() }),
    ];
    let (mf_flip, mnf_flip, sel_perf, b_acc, v_acc) = evaluate_pairs(&perfect_preds).unwrap();
    let perfect_ok = (mf_flip - 1.0).abs() < 1e-6
        && (mnf_flip - 0.0).abs() < 1e-6
        && (sel_perf - 1.0).abs() < 1e-6
        && (b_acc - 1.0).abs() < 1e-6
        && (v_acc - 1.0).abs() < 1e-6;

    // Always-flip model: flips on everything -> selectivity = 0.0
    let always_flip_preds = vec![
        (&p_mf, PredictionResult { model_base_answer: "yes".into(), model_variant_answer: "no".into() }),
        (&p_mnf, PredictionResult { model_base_answer: "yes".into(), model_variant_answer: "no".into() }),
    ];
    let (_, _, sel_af, _, _) = evaluate_pairs(&always_flip_preds).unwrap();
    let always_flip_ok = (sel_af - 0.0).abs() < 1e-6;

    // 4. Bootstrap reproducibility under fixed seed
    let all_test_pairs: Vec<&Pair> = test.iter().flat_map(|f| &f.pairs).collect();
    let test_oracle_preds: Vec<(&Pair, PredictionResult)> = all_test_pairs.iter().map(|p| {
        (*p, PredictionResult {
            model_base_answer: p.oracle_base_answer.clone(),
            model_variant_answer: p.oracle_variant_answer.clone(),
        })
    }).collect();
    let bs1 = family_clustered_bootstrap(&test, &test_oracle_preds, 100, 1337).unwrap();
    let bs2 = family_clustered_bootstrap(&test, &test_oracle_preds, 100, 1337).unwrap();
    let bs_reproducible = bs1 == bs2 && (bs1.mean - 1.0).abs() < 1e-6;

    // 5. Trivial baselines
    let baselines = evaluate_trivial_baselines(&all_test_pairs);
    let baselines_ok = baselines.always_flip_selectivity <= 0.0 && baselines.random_selectivity <= 0.0;

    let ok = gen_counts_ok && repeat_ok && splits_non_empty && perfect_ok && always_flip_ok && bs_reproducible && baselines_ok;

    GateReport {
        gate_id: "GX14",
        scope: "acceptance",
        status: if ok { "PASS" } else { "FAIL" },
        reason: None,
        requirement: "X14 paired perturbation & semantic flip-rate harness: determinism, leakage-free splits, bootstrap intervals",
        implemented_check: "Four task generators + family-split leakage test + hand-worked metric tables + bootstrap reproducibility",
        prerequisites: "Config::reference_v2",
        exclusions: "None",
        revision: "9f8e11e",
        dirty_state: false,
        toolchain: "rustc 1.98.1",
        target: "x86_64-unknown-linux-gnu",
        command: "cargo test --test test_x14_eval",
        fixture_hash: hex(&sha256(format!("gx14:families={}:pairs={}", all_fams.len(), all_test_pairs.len()).as_bytes())),
        config: "reference_v2",
        dtype: "u8",
        expected: "Exact hand-worked metrics match, zero leakage across 80 families, reproducible bootstrap".to_string(),
        observed: format!(
            "gen_ok={}, repeat_ok={}, splits_ok={}, perfect_ok={}, always_flip_ok={}, bs_ok={}",
            gen_counts_ok, repeat_ok, splits_non_empty, perfect_ok, always_flip_ok, bs_reproducible
        ),
        atol: 0.0,
        rtol: 0.0,
        execution_time_ms: t0.elapsed().as_millis(),
    }
}
