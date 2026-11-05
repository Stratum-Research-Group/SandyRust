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

/// `[m,k] x [n,k]^T -> [m,n]`.
/// Computes `a * b^T` directly without allocating or transposing `b`.
pub fn matmul_trans_b(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    let (m, k) = rank2(a)?;
    let (n, k2) = rank2(b)?;
    if k != k2 {
        return Err(TensorError::ShapeMismatch { expected: vec![n, k], actual: vec![n, k2] });
    }
    let mut out = Tensor::zeros(&[m, n])?;
    let (ad, bd) = (a.data(), b.data());
    let od = out.data_mut();
    for i in 0..m {
        let a_row = &ad[i * k..(i + 1) * k];
        for j in 0..n {
            let b_row = &bd[j * k..(j + 1) * k];
            let mut s = 0.0f32;
            for p in 0..k {
                s += a_row[p] * b_row[p];
            }
            od[i * n + j] = s;
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
        for (i, &val) in row.iter().enumerate().take(d) {
            out.data_mut()[r * d + i] = val * inv * gamma.data()[i];
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
        let m = (pos_offset + pos) as f64;
        for k in 0..dh / 2 {
            let freq = (theta as f64).powf(-2.0 * k as f64 / dh as f64);
            let (s, c) = (m * freq).sin_cos();
            let (s, c) = (s as f32, c as f32);
            let (a, b) = (x.data()[pos * dh + 2 * k], x.data()[pos * dh + 2 * k + 1]);
            out.data_mut()[pos * dh + 2 * k] = a * c - b * s;
            out.data_mut()[pos * dh + 2 * k + 1] = a * s + b * c;
        }
    }
    Ok(out)
}

/// Numerically stable softmax over a slice, in place. Entries with `valid=false` get exactly 0.
pub fn masked_softmax_row(scores: &mut [f32], valid: &[bool]) -> bool {
    if scores.len() != valid.len() {
        return false;
    }
    let mut max = f32::NEG_INFINITY;
    for (&s, &v) in scores.iter().zip(valid) {
        if v {
            if !s.is_finite() {
                return false;
            }
            if s > max {
                max = s;
            }
        }
    }
    if max == f32::NEG_INFINITY {
        return false;
    }
    let mut sum = 0.0f32;
    for (s, &v) in scores.iter_mut().zip(valid) {
        *s = if v { (*s - max).exp() } else { 0.0 };
        sum += *s;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return false;
    }
    for s in scores.iter_mut() {
        *s /= sum;
    }
    true
}

/// Numerically stable logsumexp over a slice.
pub fn logsumexp(x: &[f32]) -> f32 {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return max;
    }
    let s: f32 = x.iter().map(|&v| (v - max).exp()).sum();
    max + s.ln()
}

/// Sum tensor along axis 0 or 1.
pub fn sum_axis(t: &Tensor, axis: usize) -> Result<Tensor, TensorError> {
    let (rows, cols) = rank2(t)?;
    match axis {
        0 => {
            let mut out = vec![0.0f32; cols];
            for r in 0..rows {
                for (c, slot) in out.iter_mut().enumerate().take(cols) {
                    *slot += t.data()[r * cols + c];
                }
            }
            Tensor::from_vec(&[cols], out)
        }
        1 => {
            let mut out = vec![0.0f32; rows];
            for (r, slot) in out.iter_mut().enumerate().take(rows) {
                for c in 0..cols {
                    *slot += t.data()[r * cols + c];
                }
            }
            Tensor::from_vec(&[rows], out)
        }
        _ => Err(TensorError::InvalidArgument("axis out of bounds")),
    }
}

/// Batched matmul `[B, M, K] x [B, K, N] -> [B, M, N]`.
pub fn batched_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    if a.shape().len() != 3 || b.shape().len() != 3 {
        return Err(TensorError::Rank { expected: 3, actual: a.shape().len() });
    }
    let (b_a, m, k_a) = (a.shape()[0], a.shape()[1], a.shape()[2]);
    let (b_b, k_b, n) = (b.shape()[0], b.shape()[1], b.shape()[2]);
    if b_a != b_b || k_a != k_b {
        return Err(TensorError::ShapeMismatch {
            expected: vec![b_a, k_a, n],
            actual: vec![b_b, k_b, n],
        });
    }
    let mut out = Tensor::zeros(&[b_a, m, n])?;
    for bi in 0..b_a {
        let a_slice = &a.data()[bi * m * k_a..(bi + 1) * m * k_a];
        let b_slice = &b.data()[bi * k_a * n..(bi + 1) * k_a * n];
        let out_slice = &mut out.data_mut()[bi * m * n..(bi + 1) * m * n];
        for i in 0..m {
            for p in 0..k_a {
                let av = a_slice[i * k_a + p];
                for j in 0..n {
                    out_slice[i * n + j] += av * b_slice[p * n + j];
                }
            }
        }
    }
    Ok(out)
}
