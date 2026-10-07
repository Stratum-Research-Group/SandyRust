//! Forward primitives: matmul, add, mul, silu, rms_norm, rope (§3.1–3.4).

use super::{Tensor, TensorError};

fn rank2(t: &Tensor) -> Result<(usize, usize), TensorError> {
    if t.shape().len() != 2 {
        return Err(TensorError::Rank { expected: 2, actual: t.shape().len() });
    }
    Ok((t.shape()[0], t.shape()[1]))
}

/// `[m,k] x [k,n] -> [m,n]`.
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    let (m, k) = rank2(a)?;
    let (k2, n) = rank2(b)?;
    if k != k2 {
        return Err(TensorError::ShapeMismatch { expected: vec![k, n], actual: vec![k2, n] });
    }
    let mut out = Tensor::zeros(&[m, n])?;
    let (ad, bd) = (a.data(), b.data());
    let od = out.data_mut();
    for i in 0..m {
        for p in 0..k {
            let av = ad[i * k + p];
            for j in 0..n {
                od[i * n + j] += av * bd[p * n + j];
            }
        }
    }
    Ok(out)
}

fn zip(a: &Tensor, b: &Tensor, f: impl Fn(f32, f32) -> f32) -> Result<Tensor, TensorError> {
    if a.shape() != b.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: a.shape().to_vec(),
            actual: b.shape().to_vec(),
        });
    }
    let data = a.data().iter().zip(b.data()).map(|(&x, &y)| f(x, y)).collect();
    Tensor::from_vec(a.shape(), data)
}

pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    zip(a, b, |x, y| x + y)
}
pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    zip(a, b, |x, y| x * y)
}

pub fn silu(x: &Tensor) -> Tensor {
    let data = x.data().iter().map(|&z| z / (1.0 + (-z).exp())).collect();
    Tensor::from_vec(x.shape(), data).expect("same shape")
}

/// SwiGLU hidden: silu(x Wg) * (x Wu).
pub fn swiglu(x: &Tensor, w_gate: &Tensor, w_up: &Tensor) -> Result<Tensor, TensorError> {
    let g = silu(&matmul(x, w_gate)?);
    let u = matmul(x, w_up)?;
    mul(&g, &u)
}

/// RMSNorm over the last axis of a `[rows, d]` tensor with scale `gamma[d]`.
pub fn rms_norm(x: &Tensor, gamma: &Tensor, eps: f32) -> Result<Tensor, TensorError> {
    let (rows, d) = rank2(x)?;
    if gamma.shape() != [d] {
        return Err(TensorError::ShapeMismatch { expected: vec![d], actual: gamma.shape().to_vec() });
    }
    if !(eps.is_finite() && eps > 0.0) {
        return Err(TensorError::InvalidArgument("eps must be finite and > 0"));
    }
    let mut out = Tensor::zeros(&[rows, d])?;
    for r in 0..rows {
        let row = &x.data()[r * d..(r + 1) * d];
        let ms = row.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for i in 0..d {
            out.data_mut()[r * d + i] = row[i] * inv * gamma.data()[i];
        }
    }
    Ok(out)
}

/// Adjacent-pair RoPE on `[T, d_h]` with absolute positions `pos_offset + t`.
pub fn rope(x: &Tensor, theta: f32, pos_offset: usize) -> Result<Tensor, TensorError> {
    let (t, dh) = rank2(x)?;
    if dh % 2 != 0 {
        return Err(TensorError::InvalidArgument("head dim must be even"));
    }
    if !(theta.is_finite() && theta > 1.0) {
        return Err(TensorError::InvalidArgument("theta must be finite and > 1"));
    }
    let mut out = x.clone();
    for pos in 0..t {
        let m = (pos_offset + pos) as f32;
        for k in 0..dh / 2 {
            let freq = (theta as f64).powf(-2.0 * k as f64 / dh as f64) as f32;
            let (s, c) = (m * freq).sin_cos();
            let (a, b) = (x.data()[pos * dh + 2 * k], x.data()[pos * dh + 2 * k + 1]);
            out.data_mut()[pos * dh + 2 * k] = a * c - b * s;
            out.data_mut()[pos * dh + 2 * k + 1] = a * s + b * c;
        }
    }
    Ok(out)
}

/// Numerically stable softmax over a slice, in place. Entries with `valid=false` get exactly 0.
pub fn masked_softmax_row(scores: &mut [f32], valid: &[bool]) -> bool {
    let max = scores
        .iter()
        .zip(valid)
        .filter(|(_, &v)| v)
        .map(|(&s, _)| s)
        .fold(f32::NEG_INFINITY, f32::max);
    if max == f32::NEG_INFINITY {
        return false;
    }
    let mut sum = 0.0;
    for (s, &v) in scores.iter_mut().zip(valid) {
        *s = if v { (*s - max).exp() } else { 0.0 };
        sum += *s;
    }
    for s in scores.iter_mut() {
        *s /= sum;
    }
    true
}
