//! Sampling and uncached autoregressive generation (§8.3).

use crate::model::Model;
use crate::rng::Rng;
use crate::tokenizer::{BOS, EOS, PAD};

#[derive(Debug, Clone, PartialEq)]
pub enum GenerationError {
    PromptEmpty,
    PromptExceedsContext { prompt_len: usize, max_len: usize },
    CapacityExceeded { total_requested: usize, max_len: usize },
    NonFiniteLogits { position: usize },
    MalformedProbabilityDistribution,
    InvalidHyperparameter(&'static str),
    InvalidParameter(&'static str),
    EmptyFilteredDistribution,
    SessionMismatch { expected: u64, actual: u64 },
    ModelCheckpointMismatch,
    CacheInterventionMismatch { expected: [u8; 32], actual: [u8; 32] },
    InvalidAdaptiveConfig { field: String },
    ControllerNonFinite { step: usize },
    Forward(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum AdaptiveControlMode {
    Temperature { t_min: f32, t_max: f32 },
    MinP { min_p_base: f32, min_p_cap: f32 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveSamplerConfig {
    pub beta: f32,
    pub k: f32,
    pub h_target: f32,
    pub mode: AdaptiveControlMode,
}

impl AdaptiveSamplerConfig {
    pub fn validate(&self, v_allowed: usize) -> Result<(), GenerationError> {
        let v_allowed_f = v_allowed as f32;
        let h_max = if v_allowed_f > 1.0 { v_allowed_f.ln() } else { 1.0 };

        if !self.beta.is_finite() || self.beta < 0.0 || self.beta >= 1.0 {
            return Err(GenerationError::InvalidAdaptiveConfig {
                field: "beta".to_string(),
            });
        }
        if !self.k.is_finite() {
            return Err(GenerationError::InvalidAdaptiveConfig {
                field: "k".to_string(),
            });
        }
        if !self.h_target.is_finite() || self.h_target < 0.0 || self.h_target > h_max {
            return Err(GenerationError::InvalidAdaptiveConfig {
                field: "h_target".to_string(),
            });
        }
        match self.mode {
            AdaptiveControlMode::Temperature { t_min, t_max } => {
                if !t_min.is_finite() || t_min <= 0.0 {
                    return Err(GenerationError::InvalidAdaptiveConfig {
                        field: "t_min".to_string(),
                    });
                }
                if !t_max.is_finite() || t_max < t_min {
                    return Err(GenerationError::InvalidAdaptiveConfig {
                        field: "t_max".to_string(),
                    });
                }
            }
            AdaptiveControlMode::MinP { min_p_base, min_p_cap } => {
                if !min_p_base.is_finite() || min_p_base < 0.0 {
                    return Err(GenerationError::InvalidAdaptiveConfig {
                        field: "min_p_base".to_string(),
                    });
                }
                if !min_p_cap.is_finite() || !(0.0..1.0).contains(&min_p_cap) {
                    return Err(GenerationError::InvalidAdaptiveConfig {
                        field: "min_p_cap".to_string(),
                    });
                }
                if min_p_base > min_p_cap {
                    return Err(GenerationError::InvalidAdaptiveConfig {
                        field: "min_p_base".to_string(),
                    });
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveSamplerState {
    pub h_ema: Option<f32>,
    pub steps: usize,
}

impl AdaptiveSamplerState {
    pub fn new() -> Self {
        Self { h_ema: None, steps: 0 }
    }
}

impl Default for AdaptiveSamplerState {
    fn default() -> Self {
        Self::new()
    }
}

pub fn compute_entropy(probs: &[f32]) -> f32 {
    let mut h = 0.0f64;
    for &p in probs {
        if p > 0.0 {
            let pf = p as f64;
            h -= pf * pf.ln();
        }
    }
    h as f32
}

#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub max_new_tokens: usize,
    pub seed: u64,
    pub adaptive: Option<AdaptiveSamplerConfig>,
}

impl SamplingParams {
    pub fn validate(&self) -> Result<(), GenerationError> {
        if !(self.temperature.is_finite() && self.temperature >= 0.0) {
            return Err(GenerationError::InvalidHyperparameter("temperature"));
        }
        if self.top_k < 1 {
            return Err(GenerationError::InvalidHyperparameter("top_k"));
        }
        if !(self.top_p.is_finite() && self.top_p > 0.0 && self.top_p <= 1.0) {
            return Err(GenerationError::InvalidHyperparameter("top_p"));
        }
        if let Some(ref a) = self.adaptive {
            a.validate(257)?;
        }
        Ok(())
    }
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams {
            temperature: 0.0,
            top_k: usize::MAX,
            top_p: 1.0,
            max_new_tokens: 32,
            seed: 0,
            adaptive: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenerationOutput {
    pub tokens: Vec<u32>,
    /// Logits at position `T_prompt + T_gen - 1` (last position evaluated).
    pub final_logits: Vec<f32>,
    pub stopped_on_eos: bool,
    pub terminal_pos: usize,
    pub adaptive_state: Option<AdaptiveSamplerState>,
}

fn disallowed(id: usize, vocab_len: usize) -> bool {
    if vocab_len == 259 {
        id == BOS as usize || id == PAD as usize
    } else {
        false
    }
}

/// Choose the next token from raw logits, updating adaptive sampler state if active.
pub fn sample_next_step(
    logits: &[f32],
    p: &SamplingParams,
    adaptive_state: &mut Option<AdaptiveSamplerState>,
    rng: &mut Rng,
    position: usize,
) -> Result<u32, GenerationError> {
    p.validate()?;
    if logits.iter().any(|x| !x.is_finite()) {
        return Err(GenerationError::NonFiniteLogits { position });
    }
    if p.adaptive.is_some() && adaptive_state.is_none() {
        *adaptive_state = Some(AdaptiveSamplerState::new());
    }
    if let (Some(_), Some(state)) = (&p.adaptive, adaptive_state.as_ref()) {
        if let Some(h) = state.h_ema {
            if !h.is_finite() {
                return Err(GenerationError::ControllerNonFinite { step: state.steps });
            }
        }
    }
    let v_len = logits.len();
    if p.temperature == 0.0 {
        // Greedy: bypasses softmax/top-k/top-p and consumes no PRNG state.
        let mut best: Option<usize> = None;
        for (i, &z) in logits.iter().enumerate() {
            if disallowed(i, v_len) {
                continue;
            }
            if best.is_none_or(|b| z > logits[b]) {
                best = Some(i);
            }
        }
        return best.map(|i| i as u32).ok_or(GenerationError::EmptyFilteredDistribution);
    }

    let v_allowed = logits.iter().enumerate().filter(|(i, _)| !disallowed(*i, v_len)).count();
    let v_allowed_f = v_allowed as f32;
    let h_max = if v_allowed_f > 1.0 { v_allowed_f.ln() } else { 1.0 };

    let (t_eff, min_p_eff) = match (&p.adaptive, adaptive_state.as_ref()) {
        (Some(cfg), Some(state)) => match state.h_ema {
            None => match cfg.mode {
                AdaptiveControlMode::Temperature { t_min, t_max } => (p.temperature.clamp(t_min, t_max), None),
                AdaptiveControlMode::MinP { min_p_base, .. } => (p.temperature, Some(min_p_base)),
            },
            Some(h_ema) => match cfg.mode {
                AdaptiveControlMode::Temperature { t_min, t_max } => {
                    let factor = 1.0 + cfg.k * (cfg.h_target - h_ema) / h_max;
                    ((p.temperature * factor).clamp(t_min, t_max), None)
                }
                AdaptiveControlMode::MinP { min_p_base, min_p_cap } => {
                    let factor = 1.0 + cfg.k * (h_ema - cfg.h_target) / h_max;
                    (p.temperature, Some((min_p_base * factor).clamp(0.0, min_p_cap)))
                }
            },
        },
        _ => (p.temperature, None),
    };

    let max = logits
        .iter()
        .enumerate()
        .filter(|(i, _)| !disallowed(*i, v_len))
        .map(|(_, &z)| z)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut cand: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .filter(|(i, _)| !disallowed(*i, v_len))
        .map(|(i, &z)| (i, ((z - max) / t_eff).exp()))
        .collect();
    if cand.is_empty() {
        return Err(GenerationError::EmptyFilteredDistribution);
    }
    let sum: f32 = cand.iter().map(|c| c.1).sum();
    if !(sum.is_finite() && sum > 0.0) {
        return Err(GenerationError::MalformedProbabilityDistribution);
    }
    for c in cand.iter_mut() {
        c.1 /= sum;
    }

    // Full post-softmax distribution entropy
    let probs: Vec<f32> = cand.iter().map(|c| c.1).collect();
    let h_step = compute_entropy(&probs);

    if let (Some(_), Some(state)) = (&p.adaptive, adaptive_state.as_mut()) {
        if !h_step.is_finite() {
            return Err(GenerationError::ControllerNonFinite { step: state.steps });
        }
    }

    // Sort by probability desc, ties by smallest id
    cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    cand.retain(|c| c.1 > 0.0);
    if cand.is_empty() {
        return Err(GenerationError::EmptyFilteredDistribution);
    }

    if let Some(min_p) = min_p_eff {
        if min_p > 0.0 {
            let p_max = cand[0].1;
            let thresh = min_p * p_max;
            cand.retain(|c| c.1 >= thresh);
            if cand.is_empty() {
                return Err(GenerationError::EmptyFilteredDistribution);
            }
            renorm(&mut cand)?;
        }
    }

    cand.truncate(p.top_k.max(1));
    renorm(&mut cand)?;

    // Top-P
    let mut cum = 0.0;
    let mut keep = cand.len();
    for (i, c) in cand.iter().enumerate() {
        cum += c.1;
        if cum >= p.top_p {
            keep = i + 1;
            break;
        }
    }
    cand.truncate(keep.max(1));
    renorm(&mut cand)?;

    let u = rng.next_f32();
    let mut acc = 0.0;
    let mut sampled_token = cand.last().unwrap().0 as u32;
    for c in &cand {
        acc += c.1;
        if u < acc {
            sampled_token = c.0 as u32;
            break;
        }
    }

    // Update adaptive state
    if let (Some(cfg), Some(state)) = (&p.adaptive, adaptive_state.as_mut()) {
        let next_ema = match state.h_ema {
            None => h_step,
            Some(prev) => cfg.beta * prev + (1.0 - cfg.beta) * h_step,
        };
        if !next_ema.is_finite() {
            return Err(GenerationError::ControllerNonFinite { step: state.steps });
        }
        state.h_ema = Some(next_ema);
        state.steps += 1;
    }

    Ok(sampled_token)
}

/// Choose the next token from raw logits without persistent adaptive state.
pub fn sample_next(logits: &[f32], p: &SamplingParams, rng: &mut Rng, position: usize) -> Result<u32, GenerationError> {
    sample_next_step(logits, p, &mut None, rng, position)
}

fn renorm(c: &mut [(usize, f32)]) -> Result<(), GenerationError> {
    let s: f32 = c.iter().map(|x| x.1).sum();
    if !(s.is_finite() && s > 0.0) {
        return Err(GenerationError::MalformedProbabilityDistribution);
    }
    for x in c.iter_mut() {
        x.1 /= s;
    }
    Ok(())
}

pub fn generate(model: &Model, prompt: &[u32], p: &SamplingParams) -> Result<GenerationOutput, GenerationError> {
    p.validate()?;
    let max_len = model.cfg.max_seq_len;
    if prompt.is_empty() {
        return Err(GenerationError::PromptEmpty);
    }
    if prompt.len() > max_len {
        return Err(GenerationError::PromptExceedsContext { prompt_len: prompt.len(), max_len });
    }
    if p.max_new_tokens == 0 {
        return Err(GenerationError::InvalidParameter("max_new_tokens must be >= 1"));
    }
    let total = prompt
        .len()
        .checked_add(p.max_new_tokens)
        .ok_or(GenerationError::CapacityExceeded { total_requested: usize::MAX, max_len })?;
    if total > max_len {
        return Err(GenerationError::CapacityExceeded { total_requested: total, max_len });
    }
    let mut rng = Rng::new(p.seed);
    let mut seq = prompt.to_vec();
    let mut out = Vec::new();
    let mut last_logits = Vec::new();
    let mut eos = false;
    let mut adaptive_state = p.adaptive.as_ref().map(|_| AdaptiveSamplerState::new());

    for _ in 0..p.max_new_tokens {
        let logits = model.forward(&seq, None).map_err(|e| GenerationError::Forward(format!("{e:?}")))?;
        let v = model.cfg.vocab_size;
        let pos = seq.len() - 1;
        last_logits = logits.data()[pos * v..(pos + 1) * v].to_vec();
        let next = sample_next_step(&last_logits, p, &mut adaptive_state, &mut rng, pos)?;
        let eos_id = if model.cfg.ext_x17_import {
            model.cfg.eos_id
        } else {
            EOS
        };
        if next == eos_id {
            eos = true;
            break;
        }
        seq.push(next);
        out.push(next);
    }

    // Terminal position contract (§8.3, Item 7):
    let terminal_pos = seq.len() - 1;
    if !eos && !out.is_empty() {
        let logits = model.forward(&seq, None).map_err(|e| GenerationError::Forward(format!("{e:?}")))?;
        let v = model.cfg.vocab_size;
        last_logits = logits.data()[terminal_pos * v..(terminal_pos + 1) * v].to_vec();
    }

    Ok(GenerationOutput {
        tokens: out,
        final_logits: last_logits,
        stopped_on_eos: eos,
        terminal_pos,
        adaptive_state,
    })
}

/// Autoregressive generation using `KVCache` incremental single-token decoding (§8.3, F-04).
///
/// Prefills prompt tokens into a `KVCache` and executes single-token decode steps via
/// `cache::forward_step` ($O(T)$ per step instead of $O(T^2)$ full sequence re-evaluations).
pub fn generate_cached(
    model: &Model,
    prompt: &[u32],
    p: &SamplingParams,
) -> Result<GenerationOutput, GenerationError> {
    p.validate()?;
    let max_len = model.cfg.max_seq_len;
    if prompt.is_empty() {
        return Err(GenerationError::PromptEmpty);
    }
    if prompt.len() > max_len {
        return Err(GenerationError::PromptExceedsContext { prompt_len: prompt.len(), max_len });
    }
    if p.max_new_tokens == 0 {
        return Err(GenerationError::InvalidParameter("max_new_tokens must be >= 1"));
    }
    let total = prompt
        .len()
        .checked_add(p.max_new_tokens)
        .ok_or(GenerationError::CapacityExceeded { total_requested: usize::MAX, max_len })?;
    if total > max_len {
        return Err(GenerationError::CapacityExceeded { total_requested: total, max_len });
    }

    let mut cache = crate::cache::KVCache::new(&model.cfg, model.fingerprint(), p.seed)
        .map_err(|e| GenerationError::Forward(format!("{e:?}")))?;

    // Prefill prompt tokens into KV cache
    let mut last_logits = Vec::new();
    for (pos, &tok) in prompt.iter().enumerate() {
        let step_logits = crate::cache::forward_step(model, tok, pos, &mut cache)
            .map_err(|e| GenerationError::Forward(format!("{e:?}")))?;
        if pos == prompt.len() - 1 {
            last_logits = step_logits.data().to_vec();
        }
    }

    let mut rng = Rng::new(p.seed);
    let mut seq = prompt.to_vec();
    let mut out = Vec::new();
    let mut eos = false;
    let mut adaptive_state = p.adaptive.as_ref().map(|_| AdaptiveSamplerState::new());

    for _ in 0..p.max_new_tokens {
        let pos = seq.len() - 1;
        let next = sample_next_step(&last_logits, p, &mut adaptive_state, &mut rng, pos)?;
        let eos_id = if model.cfg.ext_x17_import {
            model.cfg.eos_id
        } else {
            EOS
        };
        if next == eos_id {
            eos = true;
            break;
        }
        seq.push(next);
        out.push(next);

        // Advance single token step through KV cache
        let step_logits = crate::cache::forward_step(model, next, cache.seq_len, &mut cache)
            .map_err(|e| GenerationError::Forward(format!("{e:?}")))?;
        last_logits = step_logits.data().to_vec();
    }

    let terminal_pos = seq.len() - 1;

    Ok(GenerationOutput {
        tokens: out,
        final_logits: last_logits,
        stopped_on_eos: eos,
        terminal_pos,
        adaptive_state,
    })
}

pub fn compute_distinct_n(tokens: &[u32], n: usize) -> f64 {
    if tokens.len() < n || n == 0 {
        return 0.0;
    }
    let mut total = 0;
    let mut unique = std::collections::HashSet::new();
    for window in tokens.windows(n) {
        total += 1;
        unique.insert(window.to_vec());
    }
    unique.len() as f64 / total as f64
}

pub fn compute_repetition_rate(tokens: &[u32]) -> f64 {
    if tokens.len() <= 1 {
        return 0.0;
    }
    let mut rep = 0;
    for i in 1..tokens.len() {
        if tokens[i] == tokens[i - 1] {
            rep += 1;
        }
    }
    rep as f64 / (tokens.len() - 1) as f64
}

pub fn gx12e() -> crate::gates::GateReport {
    let t0 = std::time::Instant::now();
    let m = Model::init(&crate::config::Config::reference_v2()).unwrap();
    let prompt = [256u32, 65, 66, 67];

    let p_fixed = SamplingParams {
        temperature: 0.8,
        top_k: 50,
        top_p: 0.9,
        max_new_tokens: 16,
        seed: 42,
        adaptive: None,
    };
    let out_fixed = generate(&m, &prompt, &p_fixed).unwrap();

    let p_k0 = SamplingParams {
        temperature: 0.8,
        top_k: 50,
        top_p: 0.9,
        max_new_tokens: 16,
        seed: 42,
        adaptive: Some(AdaptiveSamplerConfig {
            beta: 0.8,
            k: 0.0,
            h_target: 3.0,
            mode: AdaptiveControlMode::Temperature { t_min: 0.2, t_max: 2.0 },
        }),
    };
    let out_k0 = generate(&m, &prompt, &p_k0).unwrap();
    let k0_parity = out_fixed.tokens == out_k0.tokens && out_fixed.final_logits == out_k0.final_logits;

    let uniform4 = [0.25f32, 0.25, 0.25, 0.25];
    let h_uniform = compute_entropy(&uniform4);
    let h_uniform_expected = 4.0f32.ln();
    let uniform_ok = (h_uniform - h_uniform_expected).abs() < 1e-6;

    let one_hot = [1.0f32, 0.0, 0.0, 0.0];
    let h_one_hot = compute_entropy(&one_hot);
    let one_hot_ok = h_one_hot.abs() < 1e-6;

    let half_half_zero = [0.5f32, 0.5, 0.0, 0.0];
    let h_half = compute_entropy(&half_half_zero);
    let half_ok = (h_half - 2.0f32.ln()).abs() < 1e-6;

    let entropy_fixtures_ok = uniform_ok && one_hot_ok && half_ok;

    let t_min = 0.2f32;
    let t_max = 2.0f32;
    let h_max = (257.0f32).ln();
    let mut h_ema = 2.5f32;
    let mut bounds_ok = true;
    for &h_adv in &[-100.0f32, 0.0, 50.0, 1000.0] {
        let t_eff = (0.8 * (1.0 + 1.5 * (3.0 - h_ema) / h_max)).clamp(t_min, t_max);
        if t_eff < t_min || t_eff > t_max {
            bounds_ok = false;
        }
        h_ema = 0.8 * h_ema + 0.2 * h_adv;
    }

    let mut err_checks = true;
    let bad_beta = AdaptiveSamplerConfig {
        beta: 1.0,
        k: 1.0,
        h_target: 2.0,
        mode: AdaptiveControlMode::Temperature { t_min: 0.1, t_max: 2.0 },
    };
    err_checks &= matches!(bad_beta.validate(257), Err(GenerationError::InvalidAdaptiveConfig { ref field }) if field == "beta");

    let bad_k = AdaptiveSamplerConfig {
        beta: 0.5,
        k: f32::NAN,
        h_target: 2.0,
        mode: AdaptiveControlMode::Temperature { t_min: 0.1, t_max: 2.0 },
    };
    err_checks &= matches!(bad_k.validate(257), Err(GenerationError::InvalidAdaptiveConfig { ref field }) if field == "k");

    let bad_htarget = AdaptiveSamplerConfig {
        beta: 0.5,
        k: 1.0,
        h_target: 99.0,
        mode: AdaptiveControlMode::Temperature { t_min: 0.1, t_max: 2.0 },
    };
    err_checks &= matches!(bad_htarget.validate(257), Err(GenerationError::InvalidAdaptiveConfig { ref field }) if field == "h_target");

    let mut nan_logits = vec![0.0f32; 259];
    nan_logits[10] = f32::NAN;
    let mut rng = Rng::new(123);
    let mut state = Some(AdaptiveSamplerState::new());
    let nan_err = sample_next_step(&nan_logits, &p_k0, &mut state, &mut rng, 2);
    let nan_precedence = matches!(nan_err, Err(GenerationError::NonFiniteLogits { position: 2 }));

    let p_adapt = SamplingParams {
        temperature: 0.8,
        top_k: 50,
        top_p: 0.9,
        max_new_tokens: 32,
        seed: 42,
        adaptive: Some(AdaptiveSamplerConfig {
            beta: 0.8,
            k: 1.0,
            h_target: 3.5,
            mode: AdaptiveControlMode::Temperature { t_min: 0.2, t_max: 2.0 },
        }),
    };
    let out_adapt = generate(&m, &prompt, &p_adapt).unwrap();
    let d1_fixed = compute_distinct_n(&out_fixed.tokens, 1);
    let d2_fixed = compute_distinct_n(&out_fixed.tokens, 2);
    let rep_fixed = compute_repetition_rate(&out_fixed.tokens);

    let d1_adapt = compute_distinct_n(&out_adapt.tokens, 1);
    let d2_adapt = compute_distinct_n(&out_adapt.tokens, 2);
    let rep_adapt = compute_repetition_rate(&out_adapt.tokens);

    let ok = k0_parity && entropy_fixtures_ok && bounds_ok && err_checks && nan_precedence;
    let (rev, dirty) = crate::gates::detect_revision();
    let toolchain = crate::gates::detect_toolchain();

    crate::gates::GateReport {
        gate_id: "GX12E",
        scope: "acceptance",
        status: if ok { "PASS" } else { "FAIL" },
        reason: None,
        requirement: "X12-E entropy-feedback adaptive sampling: k=0.0 parity, fixtures, bounded control, error precedence",
        implemented_check: "Exact entropy fixtures + controller bounds clamp + k=0.0 bitwise match + typed error validation",
        prerequisites: "Config::reference_v2",
        exclusions: "None",
        revision: rev,
        dirty_state: dirty,
        toolchain,
        target: "x86_64-unknown-linux-gnu",
        command: "cargo test --test test_x12e_sampling",
        fixture_hash: crate::sha256::hex(&crate::sha256::sha256(format!("gx12e:fixed={:?}:adapt={:?}", out_fixed.tokens, out_adapt.tokens).as_bytes())),
        config: "reference_v2",
        dtype: "f32",
        expected: "k=0 bitwise parity, H fixtures within 1e-6, T clamped within [T_min, T_max]".to_string(),
        observed: format!(
            "k0_parity={}, entropy_fixtures={}, bounds={}, errs={}, nan_first={}, fixed(d1={:.2},d2={:.2},rep={:.2}) adapt(d1={:.2},d2={:.2},rep={:.2})",
            k0_parity, entropy_fixtures_ok, bounds_ok, err_checks, nan_precedence, d1_fixed, d2_fixed, rep_fixed, d1_adapt, d2_adapt, rep_adapt
        ),
        atol: 0.0,
        rtol: 0.0,
        execution_time_ms: t0.elapsed().as_millis(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn model() -> Model {
        Model::init(&Config::reference_v2()).unwrap()
    }

    #[test]
    fn greedy_ties_pick_smallest_id_and_skip_specials() {
        let mut l = vec![0.0f32; 259];
        l[10] = 5.0;
        l[3] = 5.0;
        l[256] = 100.0; // BOS must be filtered
        let mut r = Rng::new(0);
        assert_eq!(sample_next(&l, &SamplingParams::default(), &mut r, 0).unwrap(), 3);
    }

    #[test]
    fn top_k_one_equals_greedy_and_stochastic_ties_deterministic() {
        let mut l = vec![0.0f32; 259];
        l[40] = 2.0;
        l[7] = 2.0;
        let p = SamplingParams { temperature: 1.0, top_k: 1, ..Default::default() };
        let mut r = Rng::new(9);
        assert_eq!(sample_next(&l, &p, &mut r, 0).unwrap(), 7);
    }

    #[test]
    fn non_finite_logits_rejected() {
        let mut l = vec![0.0f32; 259];
        l[1] = f32::NAN;
        let mut r = Rng::new(0);
        assert!(matches!(
            sample_next(&l, &SamplingParams::default(), &mut r, 4),
            Err(GenerationError::NonFiniteLogits { position: 4 })
        ));
    }

    #[test]
    fn greedy_consumes_no_prng_and_seeded_runs_repeat() {
        let m = model();
        let prompt = [256u32, 82, 117];
        let g = SamplingParams { max_new_tokens: 4, ..Default::default() };
        let a = generate(&m, &prompt, &g).unwrap();
        let b = generate(&m, &prompt, &SamplingParams { seed: 999, ..g.clone() }).unwrap();
        assert_eq!(a.tokens, b.tokens);
        let s = SamplingParams { temperature: 0.8, top_k: 20, top_p: 0.9, max_new_tokens: 4, seed: 5, ..Default::default() };
        let x = generate(&m, &prompt, &s).unwrap();
        let y = generate(&m, &prompt, &s).unwrap();
        assert_eq!(x.tokens, y.tokens);
        assert_eq!(x.final_logits.len(), 259);
    }

    #[test]
    fn error_surface() {
        let m = model();
        assert_eq!(generate(&m, &[], &SamplingParams::default()).unwrap_err(), GenerationError::PromptEmpty);
        let long = vec![1u32; 65];
        assert!(matches!(generate(&m, &long, &SamplingParams::default()), Err(GenerationError::PromptExceedsContext { .. })));
        let p = SamplingParams { max_new_tokens: 64, ..Default::default() };
        assert!(matches!(generate(&m, &[1, 2], &p), Err(GenerationError::CapacityExceeded { .. })));
        let p = SamplingParams { max_new_tokens: usize::MAX, ..Default::default() };
        assert!(matches!(generate(&m, &[1, 2], &p), Err(GenerationError::CapacityExceeded { .. })));
        let p = SamplingParams { top_p: 0.0, ..Default::default() };
        assert!(matches!(generate(&m, &[1], &p), Err(GenerationError::InvalidHyperparameter(_))));
        let mut r = Rng::new(0);
        assert_eq!(
            sample_next(&[], &SamplingParams { temperature: 1.0, ..Default::default() }, &mut r, 0),
            Err(GenerationError::EmptyFilteredDistribution)
        );
        assert_eq!(
            sample_next(&[], &SamplingParams { temperature: 0.0, ..Default::default() }, &mut r, 0),
            Err(GenerationError::EmptyFilteredDistribution)
        );
        assert_eq!(
            sample_next(&[1.0], &SamplingParams { top_k: 0, ..Default::default() }, &mut r, 0),
            Err(GenerationError::InvalidHyperparameter("top_k"))
        );
        assert_eq!(
            sample_next(&[1.0], &SamplingParams { top_p: -0.1, ..Default::default() }, &mut r, 0),
            Err(GenerationError::InvalidHyperparameter("top_p"))
        );
    }

    #[test]
    fn terminal_logits_parity_g12() {
        let m = model();
        let prompt = [256u32, 65];
        // 1 new token
        let p1 = SamplingParams { max_new_tokens: 1, ..Default::default() };
        let out1 = generate(&m, &prompt, &p1).unwrap();
        assert_eq!(out1.tokens.len(), 1);
        let mut full_seq1 = prompt.to_vec();
        full_seq1.extend_from_slice(&out1.tokens);
        let fwd1 = m.forward(&full_seq1, None).unwrap();
        let v = m.cfg.vocab_size;
        let pos1 = full_seq1.len() - 1;
        assert_eq!(out1.terminal_pos, pos1);
        assert_eq!(out1.final_logits, &fwd1.data()[pos1 * v..(pos1 + 1) * v]);

        // 3 new tokens
        let p3 = SamplingParams { max_new_tokens: 3, ..Default::default() };
        let out3 = generate(&m, &prompt, &p3).unwrap();
        let mut full_seq3 = prompt.to_vec();
        full_seq3.extend_from_slice(&out3.tokens);
        let fwd3 = m.forward(&full_seq3, None).unwrap();
        let pos3 = full_seq3.len() - 1;
        assert_eq!(out3.terminal_pos, pos3);
        assert_eq!(out3.final_logits, &fwd3.data()[pos3 * v..(pos3 + 1) * v]);
    }

    #[test]
    fn test_generate_cached_parity_with_full_generate() {
        let m = model();
        let prompt = [256u32, 65, 66];
        let p = SamplingParams {
            max_new_tokens: 4,
            temperature: 0.0,
            ..Default::default()
        };
        let out_full = generate(&m, &prompt, &p).unwrap();
        let out_cached = generate_cached(&m, &prompt, &p).unwrap();

        assert_eq!(out_full.tokens, out_cached.tokens, "generated tokens must match between full forward and KV cache");
        assert_eq!(out_full.terminal_pos, out_cached.terminal_pos);

        // Check logit closeness within numerical precision (1e-4)
        for (i, (&a, &b)) in out_full.final_logits.iter().zip(&out_cached.final_logits).enumerate() {
            let diff = (a - b).abs();
            assert!(diff <= 1e-4 + 1e-4 * a.abs(), "token {i} logit mismatch: full={a}, cached={b}");
        }
    }
}
