//! Reverse-mode autodiff for the reference decoder (D3-A), generic over `f32` (runtime)
//! and `f64` (scoped Gate-G3 diagnostic path, §3.6).
//!
//! Design note: the decoder is a fixed graph, so the backward pass is an explicit
//! layer-by-layer VJP sweep over a per-step activation cache (the "tape"). The cache is
//! built per call, never mutated during backward, and dropped afterwards (no cycles, no
//! reuse across steps). Gradients are validated against central finite differences by G3.

use crate::config::Config;
use crate::model::{Model, ModelError};
use crate::tensor::Tensor;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

pub trait Float:
    Copy + PartialOrd + std::fmt::Debug + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Neg<Output = Self> + AddAssign
{
    const ZERO: Self;
    const ONE: Self;
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sin_cos(self) -> (Self, Self);
    fn is_finite(self) -> bool;
}

macro_rules! impl_float {
    ($t:ty) => {
        impl Float for $t {
            const ZERO: $t = 0.0;
            const ONE: $t = 1.0;
            fn from_f64(x: f64) -> $t { x as $t }
            fn to_f64(self) -> f64 { self as f64 }
            fn sqrt(self) -> $t { <$t>::sqrt(self) }
            fn exp(self) -> $t { <$t>::exp(self) }
            fn ln(self) -> $t { <$t>::ln(self) }
            fn sin_cos(self) -> ($t, $t) { <$t>::sin_cos(self) }
            fn is_finite(self) -> bool { <$t>::is_finite(self) }
        }
    };
}
impl_float!(f32);
impl_float!(f64);

#[derive(Debug, Clone, PartialEq)]
pub enum TrainError {
    Model(String),
    TokenOutOfRange(u32),
    WindowTooShort,
    AllMaskedRow(usize),
    NoValidTokens,
    NonFiniteWeights,
    InvalidSchedule(&'static str),
    InvalidArgument(&'static str),
    Checkpoint(String),
    Data(&'static str),
}

impl From<ModelError> for TrainError {
    fn from(e: ModelError) -> Self {
        TrainError::Model(format!("{e:?}"))
    }
}

/// Flat parameter store in `model::inventory` order.
#[derive(Debug, Clone)]
pub struct Params<T> {
    pub shapes: Vec<Vec<usize>>,
    pub data: Vec<Vec<T>>,
}

impl<T: Float> Params<T> {
    pub fn from_model(m: &Model) -> Self {
        let ts = m.tensors();
        Params {
            shapes: ts.iter().map(|(_, t)| t.shape().to_vec()).collect(),
            data: ts.iter().map(|(_, t)| t.data().iter().map(|&x| T::from_f64(x as f64)).collect()).collect(),
        }
    }
    pub fn zeros_like(&self) -> Self {
        Params { shapes: self.shapes.clone(), data: self.data.iter().map(|d| vec![T::ZERO; d.len()]).collect() }
    }
    pub fn zero(&mut self) {
        for d in &mut self.data {
            for x in d.iter_mut() {
                *x = T::ZERO;
            }
        }
    }
    pub fn all_finite(&self) -> bool {
        self.data.iter().all(|d| d.iter().all(|x| x.is_finite()))
    }
    pub fn numel(&self) -> usize {
        self.data.iter().map(Vec::len).sum()
    }
}

impl Params<f32> {
    pub fn to_model(&self, cfg: &Config) -> Result<Model, ModelError> {
        let ts = self
            .shapes
            .iter()
            .zip(&self.data)
            .map(|(s, d)| Tensor::from_vec(s, d.clone()).map_err(ModelError::from))
            .collect::<Result<Vec<_>, _>>()?;
        Model::from_tensors(cfg, ts)
    }
}

// ---------- dense helpers (row-major) ----------

fn mm<T: Float>(a: &[T], m: usize, k: usize, b: &[T], n: usize) -> Vec<T> {
    let mut out = vec![T::ZERO; m * n];
    for i in 0..m {
        for p in 0..k {
            let av = a[i * k + p];
            for j in 0..n {
                out[i * n + j] += av * b[p * n + j];
            }
        }
    }
    out
}

/// `a[m,k] * b[n,k]^T -> [m,n]`
fn mm_bt<T: Float>(a: &[T], m: usize, k: usize, b: &[T], n: usize) -> Vec<T> {
    let mut out = vec![T::ZERO; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = T::ZERO;
            for p in 0..k {
                s += a[i * k + p] * b[j * k + p];
            }
            out[i * n + j] = s;
        }
    }
    out
}

/// `out[k,n] += a[m,k]^T * b[m,n]`
fn mm_at_acc<T: Float>(a: &[T], m: usize, k: usize, b: &[T], n: usize, out: &mut [T]) {
    for i in 0..m {
        for p in 0..k {
            let av = a[i * k + p];
            for j in 0..n {
                out[p * n + j] += av * b[i * n + j];
            }
        }
    }
}

fn add_into<T: Float>(a: &mut [T], b: &[T]) {
    for (x, &y) in a.iter_mut().zip(b) {
        *x += y;
    }
}

fn rms_fwd<T: Float>(x: &[T], n: usize, d: usize, g: &[T], eps: T) -> (Vec<T>, Vec<T>) {
    let mut y = vec![T::ZERO; n * d];
    let mut inv = vec![T::ZERO; n];
    let dd = T::from_f64(d as f64);
    for r in 0..n {
        let row = &x[r * d..(r + 1) * d];
        let mut s = T::ZERO;
        for &v in row {
            s += v * v;
        }
        let iv = T::ONE / (s / dd + eps).sqrt();
        inv[r] = iv;
        for i in 0..d {
            y[r * d + i] = row[i] * iv * g[i];
        }
    }
    (y, inv)
}

fn rms_bwd<T: Float>(x: &[T], n: usize, d: usize, g: &[T], inv: &[T], dy: &[T], dg: &mut [T]) -> Vec<T> {
    let mut dx = vec![T::ZERO; n * d];
    let dd = T::from_f64(d as f64);
    for r in 0..n {
        let (xr, dyr, iv) = (&x[r * d..(r + 1) * d], &dy[r * d..(r + 1) * d], inv[r]);
        let mut dot = T::ZERO;
        for i in 0..d {
            dot += xr[i] * g[i] * dyr[i];
            dg[i] += dyr[i] * xr[i] * iv;
        }
        let coef = iv * iv * iv / dd * dot;
        for i in 0..d {
            dx[r * d + i] = g[i] * dyr[i] * iv - xr[i] * coef;
        }
    }
    dx
}

/// In-place adjacent-pair RoPE over `rows` x `dh` (position = row index). `inverse` rotates by -angle (VJP).
fn rope_apply<T: Float>(x: &mut [T], rows: usize, dh: usize, theta: f32, inverse: bool) {
    for pos in 0..rows {
        for k in 0..dh / 2 {
            let freq = T::from_f64((theta as f64).powf(-2.0 * k as f64 / dh as f64));
            let (s, c) = (T::from_f64(pos as f64) * freq).sin_cos();
            let s = if inverse { -s } else { s };
            let (a, b) = (x[pos * dh + 2 * k], x[pos * dh + 2 * k + 1]);
            x[pos * dh + 2 * k] = a * c - b * s;
            x[pos * dh + 2 * k + 1] = a * s + b * c;
        }
    }
}

struct LayerCache<T> {
    x0: Vec<T>,
    r1: Vec<T>,
    inv1: Vec<T>,
    qr: Vec<T>,
    kr: Vec<T>,
    v: Vec<T>,
    p: Vec<T>,
    merged: Vec<T>,
    x1: Vec<T>,
    r2: Vec<T>,
    inv2: Vec<T>,
    gate: Vec<T>,
    up: Vec<T>,
    hid: Vec<T>,
}

pub struct LossOut<T> {
    pub loss_sum: T,
    pub n_valid: usize,
}

/// Forward + backward for one window of `n+1` tokens (`x = w[..n]`, `y = w[1..]`).
/// Accumulates `d(loss_sum / denom)` into `grads`. Padded inputs are masked as queries/keys;
/// targets equal to PAD are excluded from the loss.
pub fn loss_and_grad<T: Float>(
    cfg: &Config,
    p: &Params<T>,
    window: &[u32],
    denom: T,
    grads: &mut Params<T>,
) -> Result<LossOut<T>, TrainError> {
    if window.len() < 2 {
        return Err(TrainError::WindowTooShort);
    }
    let n = window.len() - 1;
    let (tokens, targets) = (&window[..n], &window[1..]);
    let (d, vs, dh, hq, hkv, dff) = (cfg.d_model, cfg.vocab_size, cfg.head_dim(), cfg.n_q_heads, cfg.n_kv_heads, cfg.d_ff);
    let (qd, kd, group) = (hq * dh, hkv * dh, hq / hkv);
    for &t in window {
        if t as usize >= vs {
            return Err(TrainError::TokenOutOfRange(t));
        }
    }
    let valid: Vec<bool> = tokens.iter().map(|&t| t != cfg.pad_id).collect();
    let eps = T::from_f64(cfg.rms_eps as f64);
    let scale = T::ONE / T::from_f64(dh as f64).sqrt();
    let embed = &p.data[0];

    // ---------------- forward ----------------
    let mut x = vec![T::ZERO; n * d];
    for t in 0..n {
        let id = tokens[t] as usize;
        x[t * d..(t + 1) * d].copy_from_slice(&embed[id * d..(id + 1) * d]);
    }
    let mut caches: Vec<LayerCache<T>> = Vec::with_capacity(cfg.n_layers);
    for l in 0..cfg.n_layers {
        let b = 1 + 9 * l;
        let (g1, wq, wk, wv, wo, g2, wg, wu, wd_) =
            (&p.data[b], &p.data[b + 1], &p.data[b + 2], &p.data[b + 3], &p.data[b + 4], &p.data[b + 5], &p.data[b + 6], &p.data[b + 7], &p.data[b + 8]);
        let (r1, inv1) = rms_fwd(&x, n, d, g1, eps);
        let q = mm(&r1, n, d, wq, qd);
        let k = mm(&r1, n, d, wk, kd);
        let v = mm(&r1, n, d, wv, kd);
        let mut qr = vec![T::ZERO; hq * n * dh];
        for h in 0..hq {
            for t in 0..n {
                qr[h * n * dh + t * dh..h * n * dh + (t + 1) * dh].copy_from_slice(&q[t * qd + h * dh..t * qd + (h + 1) * dh]);
            }
            rope_apply(&mut qr[h * n * dh..(h + 1) * n * dh], n, dh, cfg.rope_theta, false);
        }
        let mut kr = vec![T::ZERO; hkv * n * dh];
        for h in 0..hkv {
            for t in 0..n {
                kr[h * n * dh + t * dh..h * n * dh + (t + 1) * dh].copy_from_slice(&k[t * kd + h * dh..t * kd + (h + 1) * dh]);
            }
            rope_apply(&mut kr[h * n * dh..(h + 1) * n * dh], n, dh, cfg.rope_theta, false);
        }
        let mut pm = vec![T::ZERO; hq * n * n];
        let mut merged = vec![T::ZERO; n * qd];
        for h in 0..hq {
            let kvh = h / group;
            let qh = &qr[h * n * dh..(h + 1) * n * dh];
            let kh = &kr[kvh * n * dh..(kvh + 1) * n * dh];
            for i in 0..n {
                if !valid[i] {
                    continue; // padded query: bypassed, output stays exactly zero
                }
                let row = &mut pm[(h * n + i) * n..(h * n + i + 1) * n];
                let mut mx: Option<T> = None;
                for j in 0..=i {
                    if !valid[j] {
                        continue;
                    }
                    let mut s = T::ZERO;
                    for c in 0..dh {
                        s += qh[i * dh + c] * kh[j * dh + c];
                    }
                    let s = s * scale;
                    row[j] = s;
                    if mx.map_or(true, |m| s > m) {
                        mx = Some(s);
                    }
                }
                let mx = mx.ok_or(TrainError::AllMaskedRow(i))?;
                let mut sum = T::ZERO;
                for j in 0..=i {
                    if valid[j] {
                        row[j] = (row[j] - mx).exp();
                        sum += row[j];
                    }
                }
                for j in 0..=i {
                    if valid[j] {
                        row[j] = row[j] / sum;
                    }
                }
                for j in 0..=i {
                    if valid[j] {
                        let pij = row[j];
                        for c in 0..dh {
                            merged[i * qd + h * dh + c] += pij * v[j * kd + kvh * dh + c];
                        }
                    }
                }
            }
        }
        let a = mm(&merged, n, qd, wo, d);
        let mut x1 = x.clone();
        add_into(&mut x1, &a);
        let (r2, inv2) = rms_fwd(&x1, n, d, g2, eps);
        let gate = mm(&r2, n, d, wg, dff);
        let up = mm(&r2, n, d, wu, dff);
        let hid: Vec<T> = gate.iter().zip(&up).map(|(&g, &u)| g / (T::ONE + (-g).exp()) * u).collect();
        let f = mm(&hid, n, dff, wd_, d);
        let mut x2 = x1.clone();
        add_into(&mut x2, &f);
        caches.push(LayerCache { x0: x, r1, inv1, qr, kr, v, p: pm, merged, x1, r2, inv2, gate, up, hid });
        x = x2;
    }
    let gf = &p.data[1 + 9 * cfg.n_layers];
    let (xf, invf) = rms_fwd(&x, n, d, gf, eps);
    let logits = mm_bt(&xf, n, d, embed, vs);

    // ---------------- loss ----------------
    let mut loss_sum = T::ZERO;
    let mut n_valid = 0usize;
    let mut dlogits = vec![T::ZERO; n * vs];
    for i in 0..n {
        if !valid[i] || targets[i] == cfg.pad_id {
            continue;
        }
        n_valid += 1;
        let z = &logits[i * vs..(i + 1) * vs];
        let mut mx = z[0];
        for &v in z {
            if v > mx {
                mx = v;
            }
        }
        let mut s = T::ZERO;
        for &v in z {
            s += (v - mx).exp();
        }
        let lse = mx + s.ln();
        let y = targets[i] as usize;
        loss_sum += lse - z[y];
        for j in 0..vs {
            let pj = (z[j] - lse).exp();
            dlogits[i * vs + j] = (pj - if j == y { T::ONE } else { T::ZERO }) / denom;
        }
    }

    // ---------------- backward ----------------
    let dxf = mm(&dlogits, n, vs, embed, d);
    mm_at_acc(&dlogits, n, vs, &xf, d, &mut grads.data[0]); // head path of tied E
    let fb = 1 + 9 * cfg.n_layers;
    let mut dgf = vec![T::ZERO; d];
    let mut dx = rms_bwd(&x, n, d, gf, &invf, &dxf, &mut dgf);
    add_into(&mut grads.data[fb], &dgf);

    for l in (0..cfg.n_layers).rev() {
        let b = 1 + 9 * l;
        let c = &caches[l];
        // FFN
        let dhid = mm_bt(&dx, n, d, &p.data[b + 8], dff);
        mm_at_acc(&c.hid, n, dff, &dx, d, &mut grads.data[b + 8]);
        let mut dgate = vec![T::ZERO; n * dff];
        let mut dup = vec![T::ZERO; n * dff];
        for i in 0..n * dff {
            let g = c.gate[i];
            let sg = T::ONE / (T::ONE + (-g).exp());
            dup[i] = dhid[i] * g * sg;
            dgate[i] = dhid[i] * c.up[i] * sg * (T::ONE + g * (T::ONE - sg));
        }
        let mut dr2 = mm_bt(&dgate, n, dff, &p.data[b + 6], d);
        add_into(&mut dr2, &mm_bt(&dup, n, dff, &p.data[b + 7], d));
        mm_at_acc(&c.r2, n, d, &dgate, dff, &mut grads.data[b + 6]);
        mm_at_acc(&c.r2, n, d, &dup, dff, &mut grads.data[b + 7]);
        let mut dg2 = vec![T::ZERO; d];
        let mut dx1 = dx;
        add_into(&mut dx1, &rms_bwd(&c.x1, n, d, &p.data[b + 5], &c.inv2, &dr2, &mut dg2));
        add_into(&mut grads.data[b + 5], &dg2);
        // attention
        let dmerged = mm_bt(&dx1, n, d, &p.data[b + 4], qd);
        mm_at_acc(&c.merged, n, qd, &dx1, d, &mut grads.data[b + 4]);
        let mut dqr = vec![T::ZERO; hq * n * dh];
        let mut dkr = vec![T::ZERO; hkv * n * dh];
        let mut dv = vec![T::ZERO; n * kd];
        for h in 0..hq {
            let kvh = h / group;
            let qh = &c.qr[h * n * dh..(h + 1) * n * dh];
            let kh = &c.kr[kvh * n * dh..(kvh + 1) * n * dh];
            for i in 0..n {
                if !valid[i] {
                    continue; // dQ=0, dS=0 for padded queries
                }
                let prow = &c.p[(h * n + i) * n..(h * n + i + 1) * n];
                let mut dp = vec![T::ZERO; i + 1];
                let mut rowdot = T::ZERO;
                for j in 0..=i {
                    if !valid[j] {
                        continue;
                    }
                    let mut s = T::ZERO;
                    for cc in 0..dh {
                        s += dmerged[i * qd + h * dh + cc] * c.v[j * kd + kvh * dh + cc];
                        dv[j * kd + kvh * dh + cc] += prow[j] * dmerged[i * qd + h * dh + cc];
                    }
                    dp[j] = s;
                    rowdot += prow[j] * s;
                }
                for j in 0..=i {
                    if !valid[j] {
                        continue;
                    }
                    let ds = prow[j] * (dp[j] - rowdot) * scale;
                    for cc in 0..dh {
                        dqr[h * n * dh + i * dh + cc] += ds * kh[j * dh + cc];
                        dkr[kvh * n * dh + j * dh + cc] += ds * qh[i * dh + cc];
                    }
                }
            }
        }
        let mut dq = vec![T::ZERO; n * qd];
        for h in 0..hq {
            rope_apply(&mut dqr[h * n * dh..(h + 1) * n * dh], n, dh, cfg.rope_theta, true);
            for t in 0..n {
                dq[t * qd + h * dh..t * qd + (h + 1) * dh].copy_from_slice(&dqr[h * n * dh + t * dh..h * n * dh + (t + 1) * dh]);
            }
        }
        let mut dk = vec![T::ZERO; n * kd];
        for h in 0..hkv {
            rope_apply(&mut dkr[h * n * dh..(h + 1) * n * dh], n, dh, cfg.rope_theta, true);
            for t in 0..n {
                dk[t * kd + h * dh..t * kd + (h + 1) * dh].copy_from_slice(&dkr[h * n * dh + t * dh..h * n * dh + (t + 1) * dh]);
            }
        }
        let mut dr1 = mm_bt(&dq, n, qd, &p.data[b + 1], d);
        add_into(&mut dr1, &mm_bt(&dk, n, kd, &p.data[b + 2], d));
        add_into(&mut dr1, &mm_bt(&dv, n, kd, &p.data[b + 3], d));
        mm_at_acc(&c.r1, n, d, &dq, qd, &mut grads.data[b + 1]);
        mm_at_acc(&c.r1, n, d, &dk, kd, &mut grads.data[b + 2]);
        mm_at_acc(&c.r1, n, d, &dv, kd, &mut grads.data[b + 3]);
        let mut dg1 = vec![T::ZERO; d];
        let mut dx0 = dx1;
        add_into(&mut dx0, &rms_bwd(&c.x0, n, d, &p.data[b], &c.inv1, &dr1, &mut dg1));
        add_into(&mut grads.data[b], &dg1);
        dx = dx0;
    }
    // embedding lookup path (scatter-add) accumulates into the same shared E gradient
    for t in 0..n {
        let id = tokens[t] as usize;
        for i in 0..d {
            grads.data[0][id * d + i] += dx[t * d + i];
        }
    }
    Ok(LossOut { loss_sum, n_valid })
}

#[derive(Debug, Clone)]
pub struct GradCheckReport {
    pub checked: usize,
    pub failures: usize,
    pub max_abs_err: f64,
    pub max_rel_err: f64,
}

/// Central finite-difference check of every parameter tensor (sampled indices) across an epsilon sweep.
pub fn gradcheck(
    cfg: &Config,
    params: &Params<f64>,
    window: &[u32],
    eps_list: &[f64],
    atol: f64,
    rtol: f64,
    samples_per_tensor: usize,
) -> Result<GradCheckReport, TrainError> {
    let mut scratch = params.zeros_like();
    let probe = loss_and_grad(cfg, params, window, 1.0f64, &mut scratch)?;
    if probe.n_valid == 0 {
        return Err(TrainError::NoValidTokens);
    }
    let denom = probe.n_valid as f64;
    let mut analytic = params.zeros_like();
    loss_and_grad(cfg, params, window, denom, &mut analytic)?;
    let mut rep = GradCheckReport { checked: 0, failures: 0, max_abs_err: 0.0, max_rel_err: 0.0 };
    let mut p = params.clone();
    for ti in 0..p.data.len() {
        let len = p.data[ti].len();
        for s in 0..samples_per_tensor {
            let idx = (s * 2_654_435_761usize + ti * 40_503 + 17) % len;
            for &eps in eps_list {
                let orig = p.data[ti][idx];
                let eval = |delta: f64, p: &mut Params<f64>| -> Result<f64, TrainError> {
                    p.data[ti][idx] = orig + delta;
                    let mut g = p.zeros_like();
                    let o = loss_and_grad(cfg, p, window, denom, &mut g)?;
                    Ok(o.loss_sum / denom)
                };
                let lp = eval(eps, &mut p)?;
                let lm = eval(-eps, &mut p)?;
                p.data[ti][idx] = orig;
                let numeric = (lp - lm) / (2.0 * eps);
                let a = analytic.data[ti][idx];
                let err = (a - numeric).abs();
                rep.checked += 1;
                rep.max_abs_err = rep.max_abs_err.max(err);
                rep.max_rel_err = rep.max_rel_err.max(err / a.abs().max(numeric.abs()).max(1e-12));
                if err > atol + rtol * a.abs().max(numeric.abs()) {
                    rep.failures += 1;
                }
            }
        }
    }
    Ok(rep)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loss_and_gradcheck() {
        let mut cfg = Config::reference_v2();
        cfg.n_layers = 1;
        let m = Model::init(&cfg).unwrap();
        let p_f64 = Params::<f64>::from_model(&m);
        let window = [256u32, 65, 66, 257];
        let rep = gradcheck(&cfg, &p_f64, &window, &[1e-5], 1e-5, 1e-3, 1).unwrap();
        assert_eq!(rep.failures, 0);
    }

    #[test]
    fn test_synthetic_overfit_50_percent() {
        let mut cfg = Config::reference_v2();
        cfg.n_layers = 1;
        let m = Model::init(&cfg).unwrap();
        let mut p = Params::<f32>::from_model(&m);
        let mut opt = crate::optimizer::AdamW::new(&p);
        let window = [256u32, 65, 66, 67, 257];
        let mut initial_loss = 0.0;
        let mut final_loss = 0.0;
        for step in 0..60 {
            let mut grads = p.zeros_like();
            let mut probe = p.zeros_like();
            let o = loss_and_grad(&cfg, &p, &window, 1.0, &mut probe).unwrap();
            let denom = o.n_valid as f32;
            let o = loss_and_grad(&cfg, &p, &window, denom, &mut grads).unwrap();
            let loss = o.loss_sum / denom;
            if step == 0 { initial_loss = loss; }
            final_loss = loss;
            opt.update(&mut p, &grads, 0.01);
        }
        assert!(final_loss <= initial_loss * 0.5, "initial={initial_loss}, final={final_loss}");
    }
}
