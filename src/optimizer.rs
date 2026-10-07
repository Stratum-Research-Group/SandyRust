//! AdamW (decoupled weight decay), global-norm clipping and the LR schedule (§8.1).

use crate::autodiff::{Params, TrainError};

pub const BETA1: f32 = 0.9;
pub const BETA2: f32 = 0.999;
pub const ADAM_EPS: f32 = 1e-8;
pub const WEIGHT_DECAY: f32 = 0.01;
pub const CLIP_NORM: f32 = 1.0;

#[derive(Debug, Clone, PartialEq)]
pub struct Schedule {
    pub eta_max: f32,
    pub eta_min: f32,
    pub warmup: u64,
    pub max_steps: u64,
}

impl Schedule {
    pub fn validate(&self) -> Result<(), TrainError> {
        if !(self.eta_min.is_finite() && self.eta_max.is_finite() && self.eta_min >= 0.0 && self.eta_max > 0.0 && self.eta_min <= self.eta_max) {
            return Err(TrainError::InvalidSchedule("eta must be finite with 0 <= eta_min <= eta_max, eta_max > 0"));
        }
        if self.max_steps < 1 || self.max_steps < self.warmup {
            return Err(TrainError::InvalidSchedule("require max_steps >= 1 and max_steps >= warmup"));
        }
        Ok(())
    }

    /// Learning rate for successful-update index `t >= 1`.
    pub fn eta(&self, t: u64) -> f32 {
        if t > self.max_steps {
            return self.eta_min;
        }
        if self.warmup > 0 && t <= self.warmup {
            return (self.eta_max as f64 * t as f64 / self.warmup as f64) as f32;
        }
        let denom = (self.max_steps - self.warmup) as f64; // > 0 here
        let prog = (t - self.warmup) as f64 / denom;
        let (lo, hi) = (self.eta_min as f64, self.eta_max as f64);
        (lo + 0.5 * (hi - lo) * (1.0 + (std::f64::consts::PI * prog).cos())) as f32
    }
}

#[derive(Debug, Clone)]
pub struct AdamW {
    pub m: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    pub step: u64,
}

impl AdamW {
    pub fn new(p: &Params<f32>) -> Self {
        AdamW { m: p.zeros_like().data, v: p.zeros_like().data, step: 0 }
    }

    /// One update at learning rate `eta`. Weight decay applies to 2-D tensors only.
    pub fn update(&mut self, params: &mut Params<f32>, grads: &Params<f32>, eta: f32) {
        self.step += 1;
        let t = self.step.min(i32::MAX as u64) as i32;
        let bc1 = (1.0 - (BETA1 as f64).powi(t)) as f32;
        let bc2 = (1.0 - (BETA2 as f64).powi(t)) as f32;
        for ti in 0..params.data.len() {
            let decay = params.shapes[ti].len() == 2;
            for i in 0..params.data[ti].len() {
                let g = grads.data[ti][i];
                let m = BETA1 * self.m[ti][i] + (1.0 - BETA1) * g;
                let v = BETA2 * self.v[ti][i] + (1.0 - BETA2) * g * g;
                self.m[ti][i] = m;
                self.v[ti][i] = v;
                let w = params.data[ti][i];
                let upd = (m / bc1) / ((v / bc2).sqrt() + ADAM_EPS) + if decay { WEIGHT_DECAY * w } else { 0.0 };
                params.data[ti][i] = w - eta * upd;
            }
        }
    }
}

/// Clip the global L2 norm to `max_norm`; returns the pre-clip norm.
pub fn clip_global_norm(g: &mut Params<f32>, max_norm: f32) -> f64 {
    let sq: f64 = g.data.iter().flat_map(|d| d.iter()).map(|&x| (x as f64) * (x as f64)).sum();
    let norm = sq.sqrt();
    if norm > max_norm as f64 {
        let s = (max_norm as f64 / norm) as f32;
        for d in &mut g.data {
            for x in d.iter_mut() {
                *x *= s;
            }
        }
    }
    norm
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(w: u64, m: u64) -> Schedule {
        Schedule { eta_max: 1e-3, eta_min: 1e-4, warmup: w, max_steps: m }
    }

    #[test]
    fn schedule_domains_and_shape() {
        assert!(s(10, 100).validate().is_ok());
        assert!(Schedule { eta_max: f32::NAN, ..s(1, 2) }.validate().is_err());
        assert!(Schedule { eta_min: 2e-3, ..s(1, 2) }.validate().is_err());
        assert!(s(5, 4).validate().is_err());
        assert!(s(0, 0).validate().is_err());
        let sc = s(10, 100);
        assert!((sc.eta(5) - 5e-4).abs() < 1e-9);
        assert!((sc.eta(10) - 1e-3).abs() < 1e-9);
        assert!((sc.eta(100) - 1e-4).abs() < 1e-8);
        assert_eq!(sc.eta(101), 1e-4);
        assert!(s(0, 10).eta(1) < 1e-3 && s(0, 10).eta(1) > 1e-4); // zero warmup starts in decay
        assert!((s(10, 10).eta(10) - 1e-3).abs() < 1e-9); // warmup == max: pure warmup
    }

    #[test]
    fn adamw_decays_only_matrices_and_clips() {
        let mut p = Params::<f32> { shapes: vec![vec![2], vec![1, 2]], data: vec![vec![1.0, 1.0], vec![1.0, 1.0]] };
        let zero = p.zeros_like();
        let mut o = AdamW::new(&p);
        o.update(&mut p, &zero, 0.1);
        assert_eq!(p.data[0], [1.0, 1.0]); // 1-D: no decay, zero grad => unchanged
        assert!(p.data[1][0] < 1.0); // 2-D: decayed
        let mut g = Params::<f32> { shapes: vec![vec![2]], data: vec![vec![3.0, 4.0]] };
        let n = clip_global_norm(&mut g, 1.0);
        assert!((n - 5.0).abs() < 1e-6);
        assert!((g.data[0][0] - 0.6).abs() < 1e-6 && (g.data[0][1] - 0.8).abs() < 1e-6);
    }
}
