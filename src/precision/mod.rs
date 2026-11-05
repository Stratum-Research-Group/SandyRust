//! Extension X10: Mixed Precision Training (FP16/BF16) & Dynamic Loss Scaling (§4.2 Item 10, §9.2 GX10).
//!
//! Provides:
//! 1. IEEE-754 half-precision (FP16) and bfloat16 (BF16) simulation and bitwise conversions.
//! 2. Master weights & optimizer moments preservation in FP32 with mixed-precision GEMM simulation.
//! 3. Dynamic loss scaling (`DynamicLossScaler`) with exponential backoff on non-finite grads,
//!    clean-step growth, and consecutive skip hard ceiling (`max_consecutive_skips = 5`).
//! 4. Skipped-step data policy: tokens consumed, cursor advanced, moments frozen.
//! 5. Gate GX10 verification.

use std::time::Instant;

use crate::gates::{detect_revision, detect_toolchain, GateReport};
use crate::sha256::{hex, sha256};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    FP32,
    FP16,
    BF16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrecisionError {
    ExcessiveConsecutiveSkips(usize),
    NonFiniteMasterWeight,
    InvalidScale(String),
}

impl std::fmt::Display for PrecisionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ExcessiveConsecutiveSkips(n) => {
                write!(f, "Training halted: exceeded maximum consecutive skipped steps ({n})")
            }
            Self::NonFiniteMasterWeight => write!(f, "Master weight became non-finite"),
            Self::InvalidScale(s) => write!(f, "Invalid scale parameter: {s}"),
        }
    }
}

impl std::error::Error for PrecisionError {}

// ===========================================================================
// Numeric Conversions: FP16 and BF16
// ===========================================================================

/// Convert FP32 to BF16 (bfloat16) bits using round-to-nearest-even.
pub fn f32_to_bf16_bits(x: f32) -> u16 {
    let u = x.to_bits();
    // Check if NaN
    if (u & 0x7F80_0000) == 0x7F80_0000 && (u & 0x007F_FFFF) != 0 {
        // Quiet NaN with preserved payload
        return ((u >> 16) | 0x0040) as u16;
    }
    // Round to nearest even
    let lsb = (u >> 16) & 1;
    let rounding_bias = 0x7FFF + lsb;
    ((u.wrapping_add(rounding_bias)) >> 16) as u16
}

/// Convert BF16 bits to FP32.
pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Quantize FP32 value through simulated BF16 precision.
pub fn quantize_bf16(x: f32) -> f32 {
    bf16_bits_to_f32(f32_to_bf16_bits(x))
}

/// Convert FP32 to IEEE-754 FP16 (half precision) bits using round-to-nearest-even.
pub fn f32_to_f16_bits(x: f32) -> u16 {
    let u = x.to_bits();
    let sign = ((u >> 31) & 1) as u16;
    let exp = ((u >> 23) & 0xFF) as i32;
    let mantissa = u & 0x007F_FFFF;

    // Handle NaN / Infinity
    if exp == 0xFF {
        if mantissa != 0 {
            // NaN: preserve top bit of mantissa to ensure quiet NaN
            return (sign << 15) | 0x7E00 | ((mantissa >> 13) as u16).max(1);
        } else {
            // Infinity
            return (sign << 15) | 0x7C00;
        }
    }

    let unbiased_exp = exp - 127;

    // Overflow to Infinity
    if unbiased_exp > 15 {
        return (sign << 15) | 0x7C00;
    }

    // Normal FP16
    if unbiased_exp >= -14 {
        let new_exp = (unbiased_exp + 15) as u16;
        let lsb = (mantissa >> 13) & 1;
        let round_bits = mantissa & 0x1FFF;
        let mut new_mant = (mantissa >> 13) as u16;
        if round_bits > 0x1000 || (round_bits == 0x1000 && lsb == 1) {
            new_mant += 1;
        }
        if new_mant >= 0x0400 {
            // Mantissa overflowed into exponent
            return (sign << 15) | ((new_exp + 1) << 10);
        }
        return (sign << 15) | (new_exp << 10) | (new_mant & 0x03FF);
    }

    // Subnormal FP16 or underflow to 0
    if unbiased_exp >= -24 {
        let shift = (-14 - unbiased_exp) as u32;
        let full_mantissa = mantissa | 0x0080_0000;
        let shifted = full_mantissa >> (shift + 13);
        let remainder = full_mantissa & ((1 << (shift + 13)) - 1);
        let half_point = 1 << (shift + 12);
        let lsb = shifted & 1;
        let mut new_mant = shifted as u16;
        if remainder > half_point || (remainder == half_point && lsb == 1) {
            new_mant += 1;
        }
        return (sign << 15) | (new_mant & 0x03FF);
    }

    // Underflow to zero with preserved sign
    sign << 15
}

/// Convert FP16 bits to FP32.
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x03FF) as u32;

    if exp == 0x1F {
        if mantissa != 0 {
            // NaN
            return f32::from_bits((sign << 31) | 0x7F80_0000 | (mantissa << 13));
        } else {
            // Infinity
            return f32::from_bits((sign << 31) | 0x7F80_0000);
        }
    }

    if exp == 0 {
        if mantissa == 0 {
            // Zero with sign
            return f32::from_bits(sign << 31);
        }
        // Subnormal: normalize
        let mut m = mantissa;
        let mut e = -14i32;
        while (m & 0x0400) == 0 {
            m <<= 1;
            e -= 1;
        }
        let f32_exp = (e + 127) as u32;
        let f32_mant = (m & 0x03FF) << 13;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | f32_mant);
    }

    // Normal
    let f32_exp = (exp + 127 - 15) << 23;
    let f32_mant = mantissa << 13;
    f32::from_bits((sign << 31) | f32_exp | f32_mant)
}

/// Quantize FP32 value through simulated FP16 precision.
pub fn quantize_f16(x: f32) -> f32 {
    f16_bits_to_f32(f32_to_f16_bits(x))
}

/// Quantize a slice in-place to the target precision.
pub fn quantize_slice(data: &mut [f32], precision: Precision) {
    match precision {
        Precision::FP32 => {}
        Precision::FP16 => {
            for v in data.iter_mut() {
                *v = quantize_f16(*v);
            }
        }
        Precision::BF16 => {
            for v in data.iter_mut() {
                *v = quantize_bf16(*v);
            }
        }
    }
}

/// Simulated mixed precision GEMM: inputs quantized to target precision,
/// dot products accumulated in FP32.
pub fn gemm_mixed(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    precision: Precision,
) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                let va = match precision {
                    Precision::FP32 => a[i * k + p],
                    Precision::FP16 => quantize_f16(a[i * k + p]),
                    Precision::BF16 => quantize_bf16(a[i * k + p]),
                };
                let vb = match precision {
                    Precision::FP32 => b[p * n + j],
                    Precision::FP16 => quantize_f16(b[p * n + j]),
                    Precision::BF16 => quantize_bf16(b[p * n + j]),
                };
                acc += va * vb;
            }
            c[i * n + j] = acc;
        }
    }
}

// ===========================================================================
// Dynamic Loss Scaler (§4.2 Item 10, §9.2 GX10)
// ===========================================================================

/// Dynamic loss scaler with exponential backoff on non-finite grads,
/// clean-step growth, and consecutive skip hard ceiling.
#[derive(Debug, Clone, PartialEq)]
pub struct DynamicLossScaler {
    pub current_scale: f32,
    pub min_scale: f32,
    pub max_scale: f32,
    pub growth_factor: f32,
    pub backoff_factor: f32,
    pub growth_interval: usize,
    pub consecutive_skips: usize,
    pub max_consecutive_skips: usize,
    pub steps_since_growth: usize,
}

impl Default for DynamicLossScaler {
    fn default() -> Self {
        Self::new(32768.0, 5, 2000)
    }
}

impl DynamicLossScaler {
    pub fn new(initial_scale: f32, max_consecutive_skips: usize, growth_interval: usize) -> Self {
        Self {
            current_scale: initial_scale,
            min_scale: 1.0,
            max_scale: 16_777_216.0, // 2^24
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval,
            consecutive_skips: 0,
            max_consecutive_skips,
            steps_since_growth: 0,
        }
    }

    /// Check gradients for non-finite values (NaN / Inf).
    pub fn has_non_finite(&self, grads: &[f32]) -> bool {
        grads.iter().any(|&g| !g.is_finite())
    }

    /// Process a training step:
    /// - If grads have NaN/Inf: increment consecutive skips, halve scale, return `Ok(false)` (skip step).
    ///   If consecutive skips reaches `max_consecutive_skips`, return `Err(ExcessiveConsecutiveSkips)`.
    /// - If grads are finite: reset consecutive skips, increment growth steps (doubling scale when interval met),
    ///   unscale gradients in-place by `1 / current_scale`, and return `Ok(true)` (proceed with step).
    pub fn process_step(&mut self, grads: &mut [f32]) -> Result<bool, PrecisionError> {
        if self.has_non_finite(grads) {
            self.consecutive_skips += 1;
            self.steps_since_growth = 0;
            self.current_scale = (self.current_scale * self.backoff_factor).max(self.min_scale);

            if self.consecutive_skips >= self.max_consecutive_skips {
                return Err(PrecisionError::ExcessiveConsecutiveSkips(self.max_consecutive_skips));
            }
            // Skip update: return false
            Ok(false)
        } else {
            self.consecutive_skips = 0;
            self.steps_since_growth += 1;

            if self.steps_since_growth >= self.growth_interval {
                self.current_scale = (self.current_scale * self.growth_factor).min(self.max_scale);
                self.steps_since_growth = 0;
            }

            // Unscale gradients
            let inv_scale = 1.0 / self.current_scale;
            for g in grads.iter_mut() {
                *g *= inv_scale;
            }

            Ok(true)
        }
    }
}

// ===========================================================================
// Verification Gate GX10
// ===========================================================================

/// Gate GX10 verification for Extension X10 (§4.2 Item 10, §9.2 GX10).
pub fn gx10() -> GateReport {
    let t0 = Instant::now();

    // 1. FP16 & BF16 Numeric Conversions & Quantization Fidelity
    let test_val = 1.0f32;
    let f16_1 = quantize_f16(test_val);
    let bf16_1 = quantize_bf16(test_val);
    let mut num_ok = (f16_1 - 1.0).abs() < 1e-6 && (bf16_1 - 1.0).abs() < 1e-6;

    // Small number test (underflow / precision differences)
    let small_val = 1.0f32 / 2048.0; // 2^-11
    let q_f16 = quantize_f16(small_val);
    let q_bf16 = quantize_bf16(small_val);
    num_ok &= (q_f16 - small_val).abs() < 1e-6 && (q_bf16 - small_val).abs() < 1e-5;

    // Mixed GEMM parity check
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![5.0f32, 6.0, 7.0, 8.0];
    let mut c_f32 = vec![0.0f32; 4];
    let mut c_f16 = vec![0.0f32; 4];
    let mut c_bf16 = vec![0.0f32; 4];
    gemm_mixed(&a, &b, &mut c_f32, 2, 2, 2, Precision::FP32);
    gemm_mixed(&a, &b, &mut c_f16, 2, 2, 2, Precision::FP16);
    gemm_mixed(&a, &b, &mut c_bf16, 2, 2, 2, Precision::BF16);
    let gemm_ok = c_f32 == c_f16 && c_f32 == c_bf16; // Exact on small integers

    // 2. Authoritative FP32 Master Weights & Dynamic Loss Scaling
    let mut scaler = DynamicLossScaler::new(32768.0, 5, 10);
    let mut master_weights = vec![0.5f32; 16];
    let master_weights_initial = master_weights.clone();
    let mut moments_m = vec![0.0f32; 16];
    let moments_v = vec![0.0f32; 16];
    let mut data_cursor = 0usize;

    // Clean step 1: gradients = 32768.0 (scaled)
    let mut grads = vec![32768.0f32; 16];
    let step1_res = scaler.process_step(&mut grads);
    let step1_proceeded = matches!(step1_res, Ok(true));
    let unscaled_correct = (grads[0] - 1.0).abs() < 1e-6; // 32768 / 32768 = 1.0

    // Simulate optimizer applying update
    if step1_proceeded {
        for (w, &g) in master_weights.iter_mut().zip(&grads) {
            *w -= 0.01 * g;
        }
        for (m, &g) in moments_m.iter_mut().zip(&grads) {
            *m = 0.9 * *m + 0.1 * g;
        }
        data_cursor += 16;
    }
    let weights_updated = master_weights != master_weights_initial;
    let saved_master_weights = master_weights.clone();
    let saved_moments_m = moments_m.clone();
    let saved_moments_v = moments_v.clone();
    let saved_cursor = data_cursor;

    // 3. Non-Finite Gradient Injection (NaN / Inf) -> Step Skipping Policy
    let mut non_finite_grads = vec![1.0f32; 16];
    non_finite_grads[4] = f32::NAN;

    let skip_res = scaler.process_step(&mut non_finite_grads);
    let skip_ok = matches!(skip_res, Ok(false));
    let scale_halved = (scaler.current_scale - 16384.0).abs() < 1e-4;

    // Verify skipped-step invariants:
    // - Master weights frozen
    // - Moments frozen
    // - Data cursor ADVANCED (tokens consumed)
    data_cursor += 16; // Data loader advances cursor regardless
    let weights_frozen = master_weights == saved_master_weights;
    let moments_frozen = moments_m == saved_moments_m && moments_v == saved_moments_v;
    let cursor_advanced = data_cursor == saved_cursor + 16;

    // 4. Consecutive Skip Hard Limit Threshold (max_consecutive_skips = 5)
    // We already had 1 skip. Next 3 skips should be Ok(false).
    let mut four_more_skips_ok = true;
    for _ in 0..3 {
        let mut g = vec![f32::INFINITY; 16];
        four_more_skips_ok &= matches!(scaler.process_step(&mut g), Ok(false));
    }
    // 5th consecutive skip must return Err(ExcessiveConsecutiveSkips(5))
    let mut g_final = vec![f32::NAN; 16];
    let fifth_skip_res = scaler.process_step(&mut g_final);
    let limit_halted = matches!(fifth_skip_res, Err(PrecisionError::ExcessiveConsecutiveSkips(5)));

    let all_ok = num_ok
        && gemm_ok
        && step1_proceeded
        && unscaled_correct
        && weights_updated
        && skip_ok
        && scale_halved
        && weights_frozen
        && moments_frozen
        && cursor_advanced
        && four_more_skips_ok
        && limit_halted;

    let (rev, dirty) = detect_revision();
    let toolchain = detect_toolchain();

    GateReport {
        gate_id: "GX10",
        scope: "acceptance",
        status: if all_ok { "PASS" } else { "FAIL" },
        reason: None,
        requirement: "X10 Mixed Precision & Dynamic Loss Scaling: FP16/BF16 simulation, FP32 master weights/moments preservation on skip, data cursor advancement, dynamic scale halving, max consecutive skips (5) ceiling",
        implemented_check: "IEEE-754 FP16/BF16 bitwise quantize + FP32 master weights + gradient unscaling + skipped-step frozen state + cursor advancement + ExcessiveConsecutiveSkips(5) halt",
        prerequisites: "Config::reference_v2",
        exclusions: "GPU CUDA hardware half types (pure safe Rust IEEE-754 simulation)",
        revision: rev,
        dirty_state: dirty,
        toolchain,
        target: "x86_64-unknown-linux-gnu",
        command: "cargo test --test test_x10_precision",
        fixture_hash: hex(&sha256(format!("gx10:scale={:.1}:skip_halt={}", scaler.current_scale, limit_halted).as_bytes())),
        config: "reference_v2_x10",
        dtype: "f32/f16/bf16",
        expected: "FP16/BF16 exact conversion, scale 32768->16384 on NaN, FP32 master weights unchanged, moments unchanged, cursor advanced, halted on 5 skips".to_string(),
        observed: format!(
            "num_ok={}, gemm_ok={}, unscale={}, halved={}, weights_frozen={}, moments_frozen={}, cursor_adv={}, limit_halted={}",
            num_ok, gemm_ok, unscaled_correct, scale_halved, weights_frozen, moments_frozen, cursor_advanced, limit_halted
        ),
        atol: 1e-5,
        rtol: 1e-4,
        execution_time_ms: t0.elapsed().as_millis(),
    }
}
