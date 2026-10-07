//! 4-layer decoder stack (reference_v2): forward pass, seeded init, tensor inventory.

use crate::attention::attention_ref;
use crate::config::{Config, ConfigError};
use crate::rng::Rng;
use crate::tensor::ops::{add, matmul, mul, rms_norm, rope, silu};
use crate::tensor::{Tensor, TensorError};

#[derive(Debug, Clone, PartialEq)]
pub enum ModelError {
    Tensor(TensorError),
    Config(ConfigError),
    EmptyInput,
    SequenceTooLong { len: usize, max: usize },
    TokenOutOfRange { token: u32, vocab: usize },
    Inventory(&'static str),
    Cache(String),
}

impl From<TensorError> for ModelError {
    fn from(e: TensorError) -> Self {
        ModelError::Tensor(e)
    }
}
impl From<ConfigError> for ModelError {
    fn from(e: ConfigError) -> Self {
        ModelError::Config(e)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    pub rms1: Tensor,
    pub wq: Tensor,
    pub wk: Tensor,
    pub wv: Tensor,
    pub wo: Tensor,
    pub rms2: Tensor,
    pub w_gate: Tensor,
    pub w_up: Tensor,
    pub w_down: Tensor,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Model {
    pub cfg: Config,
    pub embed: Tensor,
    pub layers: Vec<Layer>,
    pub final_norm: Tensor,
}

/// Ordered `(name, shape)` inventory for a configuration (single source of truth for checkpoints).
pub fn inventory(cfg: &Config) -> Vec<(String, Vec<usize>)> {
    let (d, dh) = (cfg.d_model, cfg.head_dim());
    let (q, kv) = (cfg.n_q_heads * dh, cfg.n_kv_heads * dh);
    let mut v = vec![("embed".to_string(), vec![cfg.vocab_size, d])];
    for i in 0..cfg.n_layers {
        let p = |n: &str| format!("layers.{i}.{n}");
        v.push((p("rms1"), vec![d]));
        v.push((p("wq"), vec![d, q]));
        v.push((p("wk"), vec![d, kv]));
        v.push((p("wv"), vec![d, kv]));
        v.push((p("wo"), vec![q, d]));
        v.push((p("rms2"), vec![d]));
        v.push((p("w_gate"), vec![d, cfg.d_ff]));
        v.push((p("w_up"), vec![d, cfg.d_ff]));
        v.push((p("w_down"), vec![cfg.d_ff, d]));
    }
    v.push(("final_norm".to_string(), vec![d]));
    v
}

fn normal_tensor(shape: &[usize], std: f32, rng: &mut Rng) -> Result<Tensor, TensorError> {
    let mut t = Tensor::zeros(shape)?;
    for x in t.data_mut() {
        *x = rng.next_normal() * std;
    }
    Ok(t)
}

fn ones(shape: &[usize]) -> Result<Tensor, TensorError> {
    Tensor::from_vec(shape, vec![1.0; shape.iter().product()])
}

fn slice_cols(t: &Tensor, start: usize, len: usize) -> Result<Tensor, TensorError> {
    let (rows, cols) = (t.shape()[0], t.shape()[1]);
    let mut out = Vec::with_capacity(rows * len);
    for r in 0..rows {
        out.extend_from_slice(&t.data()[r * cols + start..r * cols + start + len]);
    }
    Tensor::from_vec(&[rows, len], out)
}

impl Model {
    /// Seeded random initialisation per §3.5.
    pub fn init(cfg: &Config) -> Result<Self, ModelError> {
        cfg.validate()?;
        let mut rng = Rng::new(cfg.seed);
        let resid = 1.0 / (2.0 * cfg.n_layers as f32).sqrt();
        let mut tensors = Vec::new();
        for (name, shape) in inventory(cfg) {
            let leaf = name.rsplit('.').next().unwrap_or(&name);
            let t = match leaf {
                "rms1" | "rms2" | "final_norm" => ones(&shape)?,
                "wo" | "w_down" => normal_tensor(&shape, 0.02 * resid, &mut rng)?,
                _ => normal_tensor(&shape, 0.02, &mut rng)?,
            };
            tensors.push(t);
        }
        Self::from_tensors(cfg, tensors)
    }

    /// Build from tensors in `inventory(cfg)` order.
    pub fn from_tensors(cfg: &Config, tensors: Vec<Tensor>) -> Result<Self, ModelError> {
        cfg.validate()?;
        let inv = inventory(cfg);
        if tensors.len() != inv.len() {
            return Err(ModelError::Inventory("tensor count"));
        }
        for (t, (_, shape)) in tensors.iter().zip(&inv) {
            if t.shape() != shape.as_slice() {
                return Err(ModelError::Inventory("tensor shape"));
            }
        }
        let mut it = tensors.into_iter();
        let embed = it.next().unwrap();
        let mut layers = Vec::new();
        for _ in 0..cfg.n_layers {
            let mut n = || it.next().unwrap();
            layers.push(Layer {
                rms1: n(), wq: n(), wk: n(), wv: n(), wo: n(), rms2: n(), w_gate: n(), w_up: n(), w_down: n(),
            });
        }
        let final_norm = it.next().unwrap();
        Ok(Model { cfg: cfg.clone(), embed, layers, final_norm })
    }

    pub fn from_params(cfg: &Config, p: &crate::autodiff::Params<f32>) -> Result<Self, ModelError> {
        p.to_model(cfg)
    }

    /// Tensors in inventory order.
    pub fn tensors(&self) -> Vec<(String, &Tensor)> {
        let names = inventory(&self.cfg);
        let mut refs: Vec<&Tensor> = vec![&self.embed];
        for l in &self.layers {
            refs.extend([&l.rms1, &l.wq, &l.wk, &l.wv, &l.wo, &l.rms2, &l.w_gate, &l.w_up, &l.w_down]);
        }
        refs.push(&self.final_norm);
        names.into_iter().map(|(n, _)| n).zip(refs).collect()
    }

    pub fn param_count(&self) -> usize {
        self.tensors().iter().map(|(_, t)| t.len()).sum()
    }

    /// Cryptographic SHA-256 fingerprint of config and all model weights.
    pub fn fingerprint(&self) -> [u8; 32] {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.cfg.fingerprint());
        for (name, t) in self.tensors() {
            bytes.extend_from_slice(name.as_bytes());
            for x in t.data() {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
        }
        crate::sha256::sha256(&bytes)
    }

    /// Full forward for one sequence: tokens `[T]` -> logits `[T, V]`.
    /// `valid[i]=false` marks padded positions (zero attention output for those queries).
    pub fn forward(&self, tokens: &[u32], valid: Option<&[bool]>) -> Result<Tensor, ModelError> {
        let c = &self.cfg;
        let t = tokens.len();
        if t == 0 {
            return Err(ModelError::EmptyInput);
        }
        if t > c.max_seq_len {
            return Err(ModelError::SequenceTooLong { len: t, max: c.max_seq_len });
        }
        for &tok in tokens {
            if tok as usize >= c.vocab_size {
                return Err(ModelError::TokenOutOfRange { token: tok, vocab: c.vocab_size });
            }
        }
        let all = vec![true; t];
        let valid = valid.unwrap_or(&all);
        if valid.len() != t {
            return Err(TensorError::InvalidArgument("valid mask length").into());
        }
        let idx: Vec<usize> = tokens.iter().map(|&x| x as usize).collect();
        let mut x = self.embed.gather_rows(&idx)?;
        let (dh, group) = (c.head_dim(), c.n_q_heads / c.n_kv_heads);
        for l in &self.layers {
            let r = rms_norm(&x, &l.rms1, c.rms_eps)?;
            let (q, k, v) = (matmul(&r, &l.wq)?, matmul(&r, &l.wk)?, matmul(&r, &l.wv)?);
            let mut merged = Tensor::zeros(&[t, c.n_q_heads * dh])?;
            for h in 0..c.n_q_heads {
                let kvh = h / group;
                let qh = rope(&slice_cols(&q, h * dh, dh)?, c.rope_theta, 0)?;
                let kh = rope(&slice_cols(&k, kvh * dh, dh)?, c.rope_theta, 0)?;
                let vh = slice_cols(&v, kvh * dh, dh)?;
                let (o, _) = attention_ref(&qh, &kh, &vh, valid, valid, 0)?;
                for row in 0..t {
                    let dst = row * c.n_q_heads * dh + h * dh;
                    merged.data_mut()[dst..dst + dh].copy_from_slice(&o.data()[row * dh..(row + 1) * dh]);
                }
            }
            x = add(&x, &matmul(&merged, &l.wo)?)?;
            let r2 = rms_norm(&x, &l.rms2, c.rms_eps)?;
            let hidden = mul(&silu(&matmul(&r2, &l.w_gate)?), &matmul(&r2, &l.w_up)?)?;
            x = add(&x, &matmul(&hidden, &l.w_down)?)?;
        }
        let x = rms_norm(&x, &self.final_norm, c.rms_eps)?;
        Ok(matmul(&x, &self.embed.transpose2d()?)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_matches_inventory_and_is_deterministic() {
        let cfg = Config::reference_v2();
        let m = Model::init(&cfg).unwrap();
        assert_eq!(m.param_count(), 886_272);
        let m2 = Model::init(&cfg).unwrap();
        assert_eq!(m.embed, m2.embed);
        assert!(m.tensors().iter().all(|(_, t)| t.all_finite()));
    }

    #[test]
    fn forward_shape_finite_and_causal() {
        let m = Model::init(&Config::reference_v2()).unwrap();
        let a = [256u32, 72, 105, 33, 10];
        let mut b = a;
        b[4] = 99; // change only the last token
        let la = m.forward(&a, None).unwrap();
        let lb = m.forward(&b, None).unwrap();
        assert_eq!(la.shape(), [5, 259]);
        assert!(la.all_finite());
        // earlier positions unaffected by a future token
        assert_eq!(&la.data()[..4 * 259], &lb.data()[..4 * 259]);
        assert_ne!(&la.data()[4 * 259..], &lb.data()[4 * 259..]);
    }

    #[test]
    fn forward_rejects_bad_input() {
        let m = Model::init(&Config::reference_v2()).unwrap();
        assert_eq!(m.forward(&[], None).unwrap_err(), ModelError::EmptyInput);
        assert!(matches!(m.forward(&[259], None), Err(ModelError::TokenOutOfRange { .. })));
        assert!(matches!(m.forward(&vec![1; 65], None), Err(ModelError::SequenceTooLong { .. })));
    }
}
