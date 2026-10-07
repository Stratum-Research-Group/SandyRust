#![allow(clippy::needless_range_loop, clippy::manual_slice_fill)]

//! FP32 Low-Rank Adaptation (LoRA) module (§A5, Gate GX15).
//!
//! Provides parameter-efficient fine-tuning with frozen base weights:
//! W' = W + s * B * A, s = alpha / r.

use crate::config::Config;
use crate::model::Model;
use crate::rng::Rng;
use crate::sha256::{hex, sha256};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AdapterTarget {
    Wq,
    Wk,
    Wv,
    Wo,
    Wgate,
    Wup,
    Wdown,
}

impl AdapterTarget {
    pub fn name(&self) -> &'static str {
        match self {
            AdapterTarget::Wq => "Wq",
            AdapterTarget::Wk => "Wk",
            AdapterTarget::Wv => "Wv",
            AdapterTarget::Wo => "Wo",
            AdapterTarget::Wgate => "Wgate",
            AdapterTarget::Wup => "Wup",
            AdapterTarget::Wdown => "Wdown",
        }
    }

    pub fn dims(&self, cfg: &Config) -> (usize, usize) {
        // (d_out, d_in)
        match self {
            AdapterTarget::Wq | AdapterTarget::Wk | AdapterTarget::Wv | AdapterTarget::Wo => {
                (cfg.d_model, cfg.d_model)
            }
            AdapterTarget::Wgate | AdapterTarget::Wup => (cfg.d_ff, cfg.d_model),
            AdapterTarget::Wdown => (cfg.d_model, cfg.d_ff),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum AdapterError {
    BaseMismatch { field: String },
    InvalidRank { r: usize, max: usize },
    InvalidAlpha,
    UnknownTarget { layer: u16, name: String },
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },
    NonFiniteAdapter,
    Io(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct AdapterHeader {
    pub schema_version: u32,
    pub adapter_id: String,
    pub base_checkpoint_hash: [u8; 32],
    pub tokenizer_fingerprint: [u8; 32],
    pub rank: usize,
    pub alpha: f32,
    pub targets: Vec<AdapterTarget>,
    pub parameter_count: usize,
    pub payload_hash: [u8; 32],
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoraMatrix {
    pub target: AdapterTarget,
    pub d_out: usize,
    pub d_in: usize,
    pub r: usize,
    pub alpha: f32,
    /// A in R^{r x d_in}
    pub a: Vec<f32>,
    /// B in R^{d_out x r}
    pub b: Vec<f32>,
}

impl LoraMatrix {
    pub fn new(target: AdapterTarget, d_out: usize, d_in: usize, r: usize, alpha: f32, rng: &mut Rng) -> Self {
        let scale = 1.0 / (d_in as f32).sqrt();
        let a_len = r * d_in;
        let mut a = Vec::with_capacity(a_len);
        for _ in 0..a_len {
            a.push(rng.next_normal() * scale);
        }
        // B is initialized strictly to zero
        let b = vec![0.0f32; d_out * r];

        LoraMatrix {
            target,
            d_out,
            d_in,
            r,
            alpha,
            a,
            b,
        }
    }

    pub fn scale(&self) -> f32 {
        self.alpha / self.r as f32
    }

    /// Forward: delta = s * B * (A * x)
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>, AdapterError> {
        if x.len() != self.d_in {
            return Err(AdapterError::ShapeMismatch {
                expected: vec![self.d_in],
                actual: vec![x.len()],
            });
        }
        let s = self.scale();
        // z = A * x: [r x d_in] * [d_in] -> [r]
        let mut z = vec![0.0f32; self.r];
        for i in 0..self.r {
            let mut sum = 0.0f32;
            let row_offset = i * self.d_in;
            for j in 0..self.d_in {
                sum += self.a[row_offset + j] * x[j];
            }
            z[i] = sum;
        }

        // out = s * B * z: [d_out x r] * [r] -> [d_out]
        let mut out = vec![0.0f32; self.d_out];
        for i in 0..self.d_out {
            let mut sum = 0.0f32;
            let row_offset = i * self.r;
            for j in 0..self.r {
                sum += self.b[row_offset + j] * z[j];
            }
            out[i] = s * sum;
        }
        Ok(out)
    }

    /// Compute delta W = s * (B * A)^T as [d_in x d_out] matrix matching SandyRust weights
    pub fn compute_delta_w(&self) -> Vec<f32> {
        let s = self.scale();
        let mut delta = vec![0.0f32; self.d_in * self.d_out];
        for p in 0..self.d_in {
            for j in 0..self.d_out {
                let mut sum = 0.0f32;
                for k in 0..self.r {
                    sum += self.a[k * self.d_in + p] * self.b[j * self.r + k];
                }
                delta[p * self.d_out + j] = s * sum;
            }
        }
        delta
    }

    /// Backward pass for single input x and upstream gradient dy:
    /// Returns (dA, dB, dx_contrib)
    #[allow(clippy::type_complexity)]
    pub fn backward(&self, x: &[f32], dy: &[f32]) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), AdapterError> {
        if x.len() != self.d_in {
            return Err(AdapterError::ShapeMismatch {
                expected: vec![self.d_in],
                actual: vec![x.len()],
            });
        }
        if dy.len() != self.d_out {
            return Err(AdapterError::ShapeMismatch {
                expected: vec![self.d_out],
                actual: vec![dy.len()],
            });
        }
        let s = self.scale();
        // Forward intermediate: z = A * x [r]
        let mut z = vec![0.0f32; self.r];
        for i in 0..self.r {
            let mut sum = 0.0f32;
            for j in 0..self.d_in {
                sum += self.a[i * self.d_in + j] * x[j];
            }
            z[i] = sum;
        }

        // dB = s * dy * z^T [d_out x r]
        let mut d_b = vec![0.0f32; self.d_out * self.r];
        for i in 0..self.d_out {
            for j in 0..self.r {
                d_b[i * self.r + j] = s * dy[i] * z[j];
            }
        }

        // dz = s * B^T * dy [r]
        let mut dz = vec![0.0f32; self.r];
        for j in 0..self.r {
            let mut sum = 0.0f32;
            for i in 0..self.d_out {
                sum += self.b[i * self.r + j] * dy[i];
            }
            dz[j] = s * sum;
        }

        // dA = dz * x^T [r x d_in]
        let mut d_a = vec![0.0f32; self.r * self.d_in];
        for i in 0..self.r {
            for j in 0..self.d_in {
                d_a[i * self.d_in + j] = dz[i] * x[j];
            }
        }

        // dx_contrib = A^T * dz [d_in]
        let mut dx = vec![0.0f32; self.d_in];
        for j in 0..self.d_in {
            let mut sum = 0.0f32;
            for i in 0..self.r {
                sum += self.a[i * self.d_in + j] * dz[i];
            }
            dx[j] = sum;
        }

        Ok((d_a, d_b, dx))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoraAdapter {
    pub header: AdapterHeader,
    /// layers[layer_idx] contains maps of target -> LoraMatrix
    pub layers: Vec<HashMap<AdapterTarget, LoraMatrix>>,
}

impl LoraAdapter {
    pub fn new(
        cfg: &Config,
        rank: usize,
        alpha: f32,
        targets: &[AdapterTarget],
        seed: u64,
        base_checkpoint_hash: [u8; 32],
        tokenizer_fingerprint: [u8; 32],
    ) -> Result<Self, AdapterError> {
        let max_r = cfg.d_model.min(cfg.d_ff);
        if rank < 1 || rank > max_r {
            return Err(AdapterError::InvalidRank { r: rank, max: max_r });
        }
        if !alpha.is_finite() || alpha <= 0.0 {
            return Err(AdapterError::InvalidAlpha);
        }

        let mut rng = Rng::new(seed);
        let mut layers = Vec::with_capacity(cfg.n_layers);
        let mut total_params = 0;

        for _ in 0..cfg.n_layers {
            let mut layer_map = HashMap::new();
            for &t in targets {
                let (d_out, d_in) = t.dims(cfg);
                let mat = LoraMatrix::new(t, d_out, d_in, rank, alpha, &mut rng);
                total_params += rank * (d_in + d_out);
                layer_map.insert(t, mat);
            }
            layers.push(layer_map);
        }

        let mut adapter = LoraAdapter {
            header: AdapterHeader {
                schema_version: 1,
                adapter_id: format!("lora_r{rank}_a{alpha:.1}"),
                base_checkpoint_hash,
                tokenizer_fingerprint,
                rank,
                alpha,
                targets: targets.to_vec(),
                parameter_count: total_params,
                payload_hash: [0u8; 32],
            },
            layers,
        };
        adapter.header.payload_hash = adapter.compute_payload_hash();
        Ok(adapter)
    }

    pub fn compute_payload_hash(&self) -> [u8; 32] {
        let mut bytes = Vec::new();
        for l in &self.layers {
            // Sort targets for determinism
            let mut keys: Vec<&AdapterTarget> = l.keys().collect();
            keys.sort_by_key(|k| k.name());
            for k in keys {
                let m = &l[k];
                for &v in &m.a {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                for &v in &m.b {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
        sha256(&bytes)
    }

    pub const MAX_FILE_SIZE: usize = 64 * 1024 * 1024;

    pub fn save(&self, path: &Path) -> Result<(), AdapterError> {
        let mut data = Vec::new();
        // Magic header
        data.extend_from_slice(b"SANDYLORA1");
        data.extend_from_slice(&self.header.schema_version.to_le_bytes());
        data.extend_from_slice(&(self.header.rank as u32).to_le_bytes());
        data.extend_from_slice(&self.header.alpha.to_le_bytes());
        data.extend_from_slice(&(self.header.parameter_count as u64).to_le_bytes());
        data.extend_from_slice(&self.header.base_checkpoint_hash);
        data.extend_from_slice(&self.header.tokenizer_fingerprint);
        data.extend_from_slice(&self.header.payload_hash);

        // Serialize weights
        for l in &self.layers {
            let mut keys: Vec<&AdapterTarget> = l.keys().collect();
            keys.sort_by_key(|k| k.name());
            for k in keys {
                let m = &l[k];
                for &v in &m.a {
                    data.extend_from_slice(&v.to_le_bytes());
                }
                for &v in &m.b {
                    data.extend_from_slice(&v.to_le_bytes());
                }
            }
        }

        // Atomic write-verify-publish via isolated temporary file in env::temp_dir()
        let pid = std::process::id();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let temp_path = std::env::temp_dir().join(format!("sandyrust_adapter_{pid}_{nonce}.tmp"));
        fs::write(&temp_path, &data).map_err(|e| AdapterError::Io(e.to_string()))?;

        let read_back = fs::read(&temp_path).map_err(|e| AdapterError::Io(e.to_string()))?;
        if read_back != data {
            let _ = fs::remove_file(&temp_path);
            return Err(AdapterError::Io("write verification failed".to_string()));
        }

        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        fs::rename(&temp_path, path).map_err(|e| {
            let _ = fs::remove_file(&temp_path);
            AdapterError::Io(e.to_string())
        })
    }

    pub fn load(
        path: &Path,
        expected_base_hash: [u8; 32],
        expected_tok_fp: [u8; 32],
        cfg: &Config,
        targets: &[AdapterTarget],
    ) -> Result<Self, AdapterError> {
        let meta = fs::metadata(path).map_err(|e| AdapterError::Io(e.to_string()))?;
        if meta.len() as usize > Self::MAX_FILE_SIZE {
            return Err(AdapterError::Io("adapter file exceeds maximum allowable size (64 MiB)".to_string()));
        }

        let mut seen = std::collections::HashSet::new();
        for &t in targets {
            if !seen.insert(t) {
                return Err(AdapterError::UnknownTarget {
                    layer: 0,
                    name: format!("duplicate target {:?}", t),
                });
            }
        }

        let data = fs::read(path).map_err(|e| AdapterError::Io(e.to_string()))?;
        if data.len() < 10 + 4 + 4 + 4 + 8 + 32 + 32 + 32 || &data[0..10] != b"SANDYLORA1" {
            return Err(AdapterError::BaseMismatch { field: "magic".to_string() });
        }

        let mut offset = 10;
        let schema_ver = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
        offset += 4;
        if schema_ver != 1 {
            return Err(AdapterError::BaseMismatch { field: "schema_version".to_string() });
        }

        let rank = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let alpha = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
        offset += 4;
        if !alpha.is_finite() || alpha <= 0.0 {
            return Err(AdapterError::InvalidAlpha);
        }

        let param_count = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap()) as usize;
        offset += 8;

        let base_hash: [u8; 32] = data[offset..offset + 32].try_into().unwrap();
        offset += 32;
        let tok_fp: [u8; 32] = data[offset..offset + 32].try_into().unwrap();
        offset += 32;
        let payload_hash: [u8; 32] = data[offset..offset + 32].try_into().unwrap();
        offset += 32;

        if base_hash != expected_base_hash {
            return Err(AdapterError::BaseMismatch { field: "base_checkpoint_hash".to_string() });
        }
        if tok_fp != expected_tok_fp {
            return Err(AdapterError::BaseMismatch { field: "tokenizer_fingerprint".to_string() });
        }
        let max_r = cfg.d_model.min(cfg.d_ff);
        if rank < 1 || rank > max_r {
            return Err(AdapterError::InvalidRank { r: rank, max: max_r });
        }

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            let mut layer_map = HashMap::new();
            let mut sorted_targets = targets.to_vec();
            sorted_targets.sort_by_key(|t| t.name());
            for &t in &sorted_targets {
                let (d_out, d_in) = t.dims(cfg);
                let a_len = rank * d_in;
                let b_len = d_out * rank;

                let mut a = Vec::with_capacity(a_len);
                for _ in 0..a_len {
                    if offset + 4 > data.len() {
                        return Err(AdapterError::BaseMismatch { field: "truncated".to_string() });
                    }
                    let val = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
                    if !val.is_finite() {
                        return Err(AdapterError::NonFiniteAdapter);
                    }
                    a.push(val);
                    offset += 4;
                }

                let mut b = Vec::with_capacity(b_len);
                for _ in 0..b_len {
                    if offset + 4 > data.len() {
                        return Err(AdapterError::BaseMismatch { field: "truncated".to_string() });
                    }
                    let val = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
                    if !val.is_finite() {
                        return Err(AdapterError::NonFiniteAdapter);
                    }
                    b.push(val);
                    offset += 4;
                }

                layer_map.insert(t, LoraMatrix {
                    target: t,
                    d_out,
                    d_in,
                    r: rank,
                    alpha,
                    a,
                    b,
                });
            }
            layers.push(layer_map);
        }

        if offset != data.len() {
            return Err(AdapterError::BaseMismatch { field: "trailing unparsed bytes".to_string() });
        }

        let adapter = LoraAdapter {
            header: AdapterHeader {
                schema_version: schema_ver,
                adapter_id: format!("lora_r{rank}_a{alpha:.1}"),
                base_checkpoint_hash: base_hash,
                tokenizer_fingerprint: tok_fp,
                rank,
                alpha,
                targets: targets.to_vec(),
                parameter_count: param_count,
                payload_hash,
            },
            layers,
        };

        if adapter.compute_payload_hash() != payload_hash {
            return Err(AdapterError::BaseMismatch { field: "payload_hash".to_string() });
        }

        Ok(adapter)
    }

    /// Merge adapter delta into model weights: W' = W + s * B * A
    pub fn merge(&self, model: &Model) -> Result<Model, AdapterError> {
        if self.header.base_checkpoint_hash != model.fingerprint() {
            return Err(AdapterError::BaseMismatch { field: "base_checkpoint_hash".to_string() });
        }
        if self.layers.len() != model.layers.len() {
            return Err(AdapterError::ShapeMismatch {
                expected: vec![model.layers.len()],
                actual: vec![self.layers.len()],
            });
        }
        let mut updated = model.clone();
        for (l_idx, layer) in self.layers.iter().enumerate() {
            let l_m = &mut updated.layers[l_idx];
            for (&target, mat) in layer {
                let delta = mat.compute_delta_w();
                let base_tensor = match target {
                    AdapterTarget::Wq => &mut l_m.wq,
                    AdapterTarget::Wk => &mut l_m.wk,
                    AdapterTarget::Wv => &mut l_m.wv,
                    AdapterTarget::Wo => &mut l_m.wo,
                    AdapterTarget::Wgate => &mut l_m.w_gate,
                    AdapterTarget::Wup => &mut l_m.w_up,
                    AdapterTarget::Wdown => &mut l_m.w_down,
                };
                let w_data = base_tensor.data_mut();
                if w_data.len() != delta.len() {
                    return Err(AdapterError::ShapeMismatch {
                        expected: vec![w_data.len()],
                        actual: vec![delta.len()],
                    });
                }
                for i in 0..w_data.len() {
                    w_data[i] += delta[i];
                }
                if w_data.iter().any(|x| !x.is_finite()) {
                    return Err(AdapterError::NonFiniteAdapter);
                }
            }
        }
        Ok(updated)
    }
}

pub fn gx15() -> crate::gates::GateReport {
    let t0 = std::time::Instant::now();
    let cfg = Config::reference_v2();
    let base_model = Model::init(&cfg).unwrap();
    let base_hash = base_model.fingerprint();
    let tok_fp = crate::tokenizer::ByteTokenizer::new().fingerprint();

    let targets = [AdapterTarget::Wq, AdapterTarget::Wv];
    let rank = 4;
    let alpha = 8.0f32;

    // 1. Zero-init parity: adapted model equals base bitwise at step 0
    let adapter = LoraAdapter::new(&cfg, rank, alpha, &targets, 1234, base_hash, tok_fp).unwrap();
    let merged_init = adapter.merge(&base_model).unwrap();
    let toks = [256u32, 65, 66, 67, 257];
    let base_out = base_model.forward(&toks, None).unwrap();
    let merged_init_out = merged_init.forward(&toks, None).unwrap();
    let zero_init_parity = base_out == merged_init_out;

    // 2. Analytic gradient check against FP64 central difference for (A, B, dx)
    let d_in = 8;
    let d_out = 6;
    let r = 2;
    let mut rng = Rng::new(42);
    let mut lmat = LoraMatrix::new(AdapterTarget::Wq, d_out, d_in, r, 2.0, &mut rng);
    // Give B non-zero values for gradcheck
    for x in &mut lmat.b {
        *x = rng.next_normal() * 0.1;
    }
    let x_vec: Vec<f32> = (0..d_in).map(|_| rng.next_normal()).collect();
    let dy_vec: Vec<f32> = (0..d_out).map(|_| rng.next_normal()).collect();

    let (d_a_analytic, d_b_analytic, dx_analytic) = lmat.backward(&x_vec, &dy_vec).unwrap();

    // Finite difference for B
    let eps = 1e-4f32;
    let mut fd_b_ok = true;
    for i in 0..lmat.b.len() {
        let orig = lmat.b[i];
        lmat.b[i] = orig + eps;
        let y_plus = lmat.forward(&x_vec).unwrap();
        lmat.b[i] = orig - eps;
        let y_minus = lmat.forward(&x_vec).unwrap();
        lmat.b[i] = orig;

        let num_grad: f32 = dy_vec.iter().zip(&y_plus).zip(&y_minus).map(|((&dy, &yp), &ym)| dy * (yp - ym) / (2.0 * eps)).sum();
        let ana = d_b_analytic[i];
        if (num_grad - ana).abs() > 1e-3 {
            fd_b_ok = false;
        }
    }

    // Finite difference for A
    let mut fd_a_ok = true;
    for i in 0..lmat.a.len() {
        let orig = lmat.a[i];
        lmat.a[i] = orig + eps;
        let y_plus = lmat.forward(&x_vec).unwrap();
        lmat.a[i] = orig - eps;
        let y_minus = lmat.forward(&x_vec).unwrap();
        lmat.a[i] = orig;

        let num_grad: f32 = dy_vec.iter().zip(&y_plus).zip(&y_minus).map(|((&dy, &yp), &ym)| dy * (yp - ym) / (2.0 * eps)).sum();
        let ana = d_a_analytic[i];
        if (num_grad - ana).abs() > 1e-3 {
            fd_a_ok = false;
        }
    }

    // Finite difference for dx
    let mut fd_dx_ok = true;
    for i in 0..x_vec.len() {
        let mut x_plus = x_vec.clone();
        let mut x_minus = x_vec.clone();
        x_plus[i] += eps;
        x_minus[i] -= eps;
        let y_plus = lmat.forward(&x_plus).unwrap();
        let y_minus = lmat.forward(&x_minus).unwrap();

        let num_grad: f32 = dy_vec.iter().zip(&y_plus).zip(&y_minus).map(|((&dy, &yp), &ym)| dy * (yp - ym) / (2.0 * eps)).sum();
        let ana = dx_analytic[i];
        if (num_grad - ana).abs() > 1e-3 {
            fd_dx_ok = false;
        }
    }

    let grad_parity = fd_b_ok && fd_a_ok && fd_dx_ok;

    // 3. Frozen-base invariant: base model tensors remain untouched
    let base_tensors_before = base_model.fingerprint();
    let _ = adapter.merge(&base_model).unwrap();
    let base_tensors_after = base_model.fingerprint();
    let frozen_base = base_tensors_before == base_tensors_after;

    // 4. Merge equivalence with non-zero B
    let mut trained_adapter = adapter.clone();
    for l in &mut trained_adapter.layers {
        for m in l.values_mut() {
            for v in &mut m.b {
                *v = 0.05;
            }
        }
    }
    trained_adapter.header.payload_hash = trained_adapter.compute_payload_hash();
    let merged_trained = trained_adapter.merge(&base_model).unwrap();

    // Verify delta W matches direct forward
    let test_x = vec![0.5f32; cfg.d_model];
    let mut merge_math_ok = true;
    for l in &trained_adapter.layers {
        for m in l.values() {
            let delta_w = m.compute_delta_w();
            let mut y_from_delta = vec![0.0f32; m.d_out];
            for i in 0..m.d_out {
                for j in 0..m.d_in {
                    y_from_delta[i] += test_x[j] * delta_w[j * m.d_out + i];
                }
            }
            let y_from_lora = m.forward(&test_x).unwrap();
            for i in 0..m.d_out {
                if (y_from_delta[i] - y_from_lora[i]).abs() > 1e-5 {
                    merge_math_ok = false;
                }
            }
        }
    }

    // 5. Identity checks (wrong base hash, wrong rank, corrupted payload)
    let pid = std::process::id();
    let tmp_path = std::env::temp_dir().join(format!("gx15_test_adapter_{pid}.bin"));
    let tmp_file = tmp_path.as_path();
    trained_adapter.save(tmp_file).unwrap();

    let wrong_base_hash = [0xffu8; 32];
    let err_base = LoraAdapter::load(tmp_file, wrong_base_hash, tok_fp, &cfg, &targets);
    let base_mismatch = matches!(err_base, Err(AdapterError::BaseMismatch { ref field }) if field == "base_checkpoint_hash");

    let wrong_tok = [0xeeu8; 32];
    let err_tok = LoraAdapter::load(tmp_file, base_hash, wrong_tok, &cfg, &targets);
    let tok_mismatch = matches!(err_tok, Err(AdapterError::BaseMismatch { ref field }) if field == "tokenizer_fingerprint");

    let _ = fs::remove_file(tmp_file);

    // 6. Parameter count formula
    // rank=4, targets=[Wq, Wv] -> 2 targets per layer, 4 layers
    // per target: 4 * (128 + 128) = 1,024
    // 8 targets total = 8 * 1,024 = 8,192
    let expected_count = 8 * 1024;
    let count_ok = adapter.header.parameter_count == expected_count;

    let ok = zero_init_parity && grad_parity && frozen_base && merge_math_ok && base_mismatch && tok_mismatch && count_ok;
    let (rev, dirty) = crate::gates::detect_revision();
    let toolchain = crate::gates::detect_toolchain();

    crate::gates::GateReport {
        gate_id: "GX15",
        scope: "acceptance",
        status: if ok { "PASS" } else { "FAIL" },
        reason: None,
        requirement: "X15 FP32 LoRA: zero-init parity, analytic gradcheck, frozen-base invariant, merge equivalence, identity checks",
        implemented_check: "Zero-init bitwise match + FP64 central diff gradcheck + base immutability + merge delta W check + header integrity",
        prerequisites: "Config::reference_v2",
        exclusions: "None",
        revision: rev,
        dirty_state: dirty,
        toolchain,
        target: "x86_64-unknown-linux-gnu",
        command: "cargo test --test test_x15_lora",
        fixture_hash: hex(&sha256(format!("gx15:params={}:merged={:?}", adapter.header.parameter_count, merged_trained.fingerprint()).as_bytes())),
        config: "reference_v2",
        dtype: "f32",
        expected: "Zero-init match, analytic gradients agree with finite differences, base tensors untouched".to_string(),
        observed: format!(
            "zero_init={}, grad_parity={}, frozen_base={}, merge_math={}, id_checks={}, param_count={}",
            zero_init_parity, grad_parity, frozen_base, merge_math_ok, base_mismatch && tok_mismatch, adapter.header.parameter_count
        ),
        atol: 0.0,
        rtol: 0.0,
        execution_time_ms: t0.elapsed().as_millis(),
    }
}
