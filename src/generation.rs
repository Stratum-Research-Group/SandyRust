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
    Forward(String),
}

#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub max_new_tokens: usize,
    pub seed: u64,
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
        Ok(())
    }
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams { temperature: 0.0, top_k: usize::MAX, top_p: 1.0, max_new_tokens: 32, seed: 0 }
    }
}

#[derive(Debug, Clone)]
pub struct GenerationOutput {
    pub tokens: Vec<u32>,
    /// Logits at position `T_prompt + T_gen - 1` (last position evaluated).
    pub final_logits: Vec<f32>,
    pub stopped_on_eos: bool,
    pub terminal_pos: usize,
}

fn disallowed(id: usize) -> bool {
    id == BOS as usize || id == PAD as usize
}

/// Choose the next token from raw logits. Ties break to the smallest token id.
pub fn sample_next(logits: &[f32], p: &SamplingParams, rng: &mut Rng, position: usize) -> Result<u32, GenerationError> {
    p.validate()?;
    if logits.iter().any(|x| !x.is_finite()) {
        return Err(GenerationError::NonFiniteLogits { position });
    }
    if p.temperature == 0.0 {
        // Greedy: bypasses softmax/top-k/top-p and consumes no PRNG state.
        let mut best: Option<usize> = None;
        for (i, &z) in logits.iter().enumerate() {
            if disallowed(i) {
                continue;
            }
            if best.is_none_or(|b| z > logits[b]) {
                best = Some(i);
            }
        }
        return best.map(|i| i as u32).ok_or(GenerationError::EmptyFilteredDistribution);
    }
    let max = logits
        .iter()
        .enumerate()
        .filter(|(i, _)| !disallowed(*i))
        .map(|(_, &z)| z)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut cand: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .filter(|(i, _)| !disallowed(*i))
        .map(|(i, &z)| (i, ((z - max) / p.temperature).exp()))
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
    // Sort by probability desc, ties by smallest id.
    cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    cand.retain(|c| c.1 > 0.0);
    if cand.is_empty() {
        return Err(GenerationError::EmptyFilteredDistribution);
    }
    cand.truncate(p.top_k.max(1));
    renorm(&mut cand)?;
    // Top-P: smallest prefix with cumulative >= P (at least one survivor).
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
    for c in &cand {
        acc += c.1;
        if u < acc {
            return Ok(c.0 as u32);
        }
    }
    Ok(cand.last().unwrap().0 as u32) // floating-point rounding remainder only
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
    for _ in 0..p.max_new_tokens {
        let logits = model.forward(&seq, None).map_err(|e| GenerationError::Forward(format!("{e:?}")))?;
        let v = model.cfg.vocab_size;
        let pos = seq.len() - 1;
        last_logits = logits.data()[pos * v..(pos + 1) * v].to_vec();
        let next = sample_next(&last_logits, p, &mut rng, pos)?;
        if next == EOS {
            eos = true;
            break;
        }
        seq.push(next);
        out.push(next);
    }

    // Terminal position contract (§8.3, Item 7):
    // If tokens were generated and stopped due to budget (not EOS), evaluate the terminal position.
    let terminal_pos = seq.len() - 1;
    if !eos && !out.is_empty() {
        let logits = model.forward(&seq, None).map_err(|e| GenerationError::Forward(format!("{e:?}")))?;
        let v = model.cfg.vocab_size;
        last_logits = logits.data()[terminal_pos * v..(terminal_pos + 1) * v].to_vec();
    }

    Ok(GenerationOutput { tokens: out, final_logits: last_logits, stopped_on_eos: eos, terminal_pos })
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
        let s = SamplingParams { temperature: 0.8, top_k: 20, top_p: 0.9, max_new_tokens: 4, seed: 5 };
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
}
