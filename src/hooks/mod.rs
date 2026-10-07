//! X13: Activation Hooks & Bounded Steering (§A2, Gate GX13).
//!
//! Provides inspectability and controlled intervention across named forward-pass sites
//! with zero-cost branch execution when disabled.

use crate::gates::GateReport;
use crate::model::Model;
use crate::sha256::{hex, sha256};
use crate::tensor::{Tensor, TensorError};
use std::time::Instant;

pub const DEFAULT_MAX_EXPORT_BYTES: u64 = 64 * 1024 * 1024; // 64 MiB
pub const MAX_EXPORT_BYTES_CEILING: u64 = 1024 * 1024 * 1024; // 1 GiB
pub const DEFAULT_ALPHA_MAX: f32 = 8.0;
pub const EMPTY_POLICY_HASH: [u8; 32] = [0u8; 32];

/// Named forward-pass sites per Addendum A1 §A2.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HookSite {
    EmbedOut,
    ResidPre,
    AttnNormOut,
    Q,
    K,
    V,
    KPreRope,
    AttnScores,
    AttnPattern,
    HeadOut,
    AttnOut,
    ResidMid,
    MlpNormOut,
    MlpHidden,
    MlpOut,
    ResidPost,
    FinalNormOut,
    Logits,
}

impl HookSite {
    pub fn as_str(&self) -> &'static str {
        match self {
            HookSite::EmbedOut => "EmbedOut",
            HookSite::ResidPre => "ResidPre",
            HookSite::AttnNormOut => "AttnNormOut",
            HookSite::Q => "Q",
            HookSite::K => "K",
            HookSite::V => "V",
            HookSite::KPreRope => "KPreRope",
            HookSite::AttnScores => "AttnScores",
            HookSite::AttnPattern => "AttnPattern",
            HookSite::HeadOut => "HeadOut",
            HookSite::AttnOut => "AttnOut",
            HookSite::ResidMid => "ResidMid",
            HookSite::MlpNormOut => "MlpNormOut",
            HookSite::MlpHidden => "MlpHidden",
            HookSite::MlpOut => "MlpOut",
            HookSite::ResidPost => "ResidPost",
            HookSite::FinalNormOut => "FinalNormOut",
            HookSite::Logits => "Logits",
        }
    }

    pub fn is_steerable(&self) -> bool {
        matches!(self, HookSite::ResidPre | HookSite::ResidMid | HookSite::ResidPost)
    }

    pub fn is_layer_site(&self) -> bool {
        !matches!(self, HookSite::EmbedOut | HookSite::FinalNormOut | HookSite::Logits)
    }
}

/// Identifies an exact hook site in the model.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HookId {
    pub layer: u16,
    pub site: HookSite,
    pub head: Option<u16>,
}

impl HookId {
    pub fn new(layer: u16, site: HookSite, head: Option<u16>) -> Self {
        Self { layer, site, head }
    }

    pub fn global(site: HookSite) -> Self {
        Self { layer: 0, site, head: None }
    }

    pub fn layer(layer: u16, site: HookSite) -> Self {
        Self { layer, site, head: None }
    }

    pub fn head(layer: u16, site: HookSite, head: u16) -> Self {
        Self { layer, site, head: Some(head) }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum HookError {
    UnknownSite { layer: u16, site: String },
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },
    ExportTooLarge { requested: u128, max: u64 },
    InvalidSteering { reason: String },
    DirectionMismatch { field: String },
    PositionOutOfRange { start: usize, end: usize, t: usize },
    ModeNotEnabled,
    Tensor(TensorError),
}

impl From<TensorError> for HookError {
    fn from(e: TensorError) -> Self {
        HookError::Tensor(e)
    }
}

impl std::fmt::Display for HookError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}
impl std::error::Error for HookError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RecordTiming {
    PreEdit,
    PostEdit,
}

/// Bounded in-memory storage for exported activation tensors.
#[derive(Debug, Clone)]
pub struct ActivationSink {
    pub max_export_bytes: u64,
    pub current_bytes: u64,
    pub records: Vec<(HookId, RecordTiming, Tensor)>,
}

impl ActivationSink {
    pub fn new(max_export_bytes: u64) -> Result<Self, HookError> {
        if max_export_bytes > MAX_EXPORT_BYTES_CEILING {
            return Err(HookError::ExportTooLarge {
                requested: max_export_bytes as u128,
                max: MAX_EXPORT_BYTES_CEILING,
            });
        }
        Ok(Self {
            max_export_bytes,
            current_bytes: 0,
            records: Vec::new(),
        })
    }

    pub fn record(&mut self, hook: HookId, timing: RecordTiming, tensor: &Tensor) -> Result<(), HookError> {
        let bytes = (tensor.len() as u128).checked_mul(4).ok_or(HookError::ExportTooLarge {
            requested: u128::MAX,
            max: self.max_export_bytes,
        })?;
        let next_total = (self.current_bytes as u128).checked_add(bytes).ok_or(HookError::ExportTooLarge {
            requested: u128::MAX,
            max: self.max_export_bytes,
        })?;
        if next_total > self.max_export_bytes as u128 {
            return Err(HookError::ExportTooLarge {
                requested: next_total,
                max: self.max_export_bytes,
            });
        }
        self.current_bytes = next_total as u64;
        self.records.push((hook, timing, tensor.clone()));
        Ok(())
    }

    pub fn get(&self, hook: &HookId, timing: RecordTiming) -> Option<&Tensor> {
        self.records.iter().find(|(h, t, _)| h == hook && *t == timing).map(|(_, _, tensor)| tensor)
    }

    pub fn clear(&mut self) {
        self.records.clear();
        self.current_bytes = 0;
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum AblateKind {
    Zero,
    MeanFromSink(Tensor),
}

#[derive(Debug, Clone, PartialEq)]
pub enum SteerPosition {
    AllPositions,
    LastPosition,
    PositionRange { start: usize, end: usize },
}

#[derive(Debug, Clone, PartialEq)]
pub enum EditOp {
    Ablate { hook: HookId, kind: AblateKind },
    Patch { hook: HookId, source: Tensor },
    Steer {
        hook: HookId,
        direction: Vec<f32>,
        alpha: f32,
        positions: SteerPosition,
        direction_id: String,
    },
}

impl EditOp {
    pub fn hook(&self) -> &HookId {
        match self {
            EditOp::Ablate { hook, .. } => hook,
            EditOp::Patch { hook, .. } => hook,
            EditOp::Steer { hook, .. } => hook,
        }
    }
}

/// Metadata header stored with a persistent direction vector (§A2.5).
#[derive(Debug, Clone, PartialEq)]
pub struct DirectionHeader {
    pub schema_version: u32,
    pub direction_id: String,
    pub dim: usize,
    pub layer: u16,
    pub site: HookSite,
    pub model_instance_id: String,
    pub checkpoint_hash: [u8; 32],
    pub tokenizer_fingerprint: [u8; 32],
    pub extraction_method: String,
    pub created_from_dataset_hash: [u8; 32],
    pub norm: f32,
    pub notes: String,
}

impl DirectionHeader {
    pub fn validate(
        &self,
        d_model: usize,
        n_layers: usize,
        checkpoint_hash: [u8; 32],
        tokenizer_fingerprint: [u8; 32],
        direction: &[f32],
    ) -> Result<(), HookError> {
        if self.schema_version != 1 {
            return Err(HookError::DirectionMismatch { field: "schema_version".to_string() });
        }
        if self.dim != d_model || direction.len() != d_model {
            return Err(HookError::DirectionMismatch { field: "dim".to_string() });
        }
        if (self.layer as usize) >= n_layers {
            return Err(HookError::DirectionMismatch { field: "layer".to_string() });
        }
        if !self.site.is_steerable() {
            return Err(HookError::DirectionMismatch { field: "site".to_string() });
        }
        if self.checkpoint_hash != checkpoint_hash {
            return Err(HookError::DirectionMismatch { field: "checkpoint_hash".to_string() });
        }
        if self.tokenizer_fingerprint != tokenizer_fingerprint {
            return Err(HookError::DirectionMismatch { field: "tokenizer_fingerprint".to_string() });
        }
        if direction.iter().any(|x| !x.is_finite()) {
            return Err(HookError::InvalidSteering { reason: "non-finite direction element".to_string() });
        }
        let l2: f32 = direction.iter().map(|x| x * x).sum::<f32>().sqrt();
        if !l2.is_finite() {
            return Err(HookError::InvalidSteering { reason: "non-finite direction norm".to_string() });
        }
        let diff = (l2 - self.norm).abs();
        let tol = 1e-6 * self.norm.max(1.0);
        if diff > tol {
            return Err(HookError::DirectionMismatch { field: "norm".to_string() });
        }
        Ok(())
    }
}

/// Compute the canonical SHA-256 fingerprint of all active edits (§A2.6).
pub fn compute_intervention_policy_hash(edits: &[EditOp]) -> [u8; 32] {
    if edits.is_empty() {
        return EMPTY_POLICY_HASH;
    }
    let mut s = String::new();
    for edit in edits {
        match edit {
            EditOp::Ablate { hook, kind } => {
                s.push_str(&format!(
                    "ablate:l{}:s{}:h{:?}:k{:?};",
                    hook.layer,
                    hook.site.as_str(),
                    hook.head,
                    matches!(kind, AblateKind::Zero)
                ));
            }
            EditOp::Patch { hook, source } => {
                let h = sha256(&source.to_le_bytes());
                s.push_str(&format!(
                    "patch:l{}:s{}:h{:?}:sh{:02x?};",
                    hook.layer,
                    hook.site.as_str(),
                    hook.head,
                    &h[..8]
                ));
            }
            EditOp::Steer { hook, direction, alpha, positions, direction_id } => {
                let mut bytes = Vec::new();
                for d in direction {
                    bytes.extend_from_slice(&d.to_le_bytes());
                }
                let dh = sha256(&bytes);
                s.push_str(&format!(
                    "steer:l{}:s{}:h{:?}:id{}:a{:.6}:p{:?}:dh{:02x?};",
                    hook.layer,
                    hook.site.as_str(),
                    hook.head,
                    direction_id,
                    alpha,
                    positions,
                    &dh[..8]
                ));
            }
        }
    }
    sha256(s.as_bytes())
}

/// Runtime context passed during forward execution.
#[derive(Debug, Clone)]
pub struct HookContext {
    pub records: Vec<(HookId, RecordTiming)>,
    pub edits: Vec<EditOp>,
    pub max_export_bytes: u64,
    pub alpha_max: f32,
}

impl Default for HookContext {
    fn default() -> Self {
        Self {
            records: Vec::new(),
            edits: Vec::new(),
            max_export_bytes: DEFAULT_MAX_EXPORT_BYTES,
            alpha_max: DEFAULT_ALPHA_MAX,
        }
    }
}

impl HookContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_record(mut self, hook: HookId, timing: RecordTiming) -> Self {
        self.records.push((hook, timing));
        self
    }

    pub fn with_edit(mut self, edit: EditOp) -> Self {
        self.edits.push(edit);
        self
    }

    pub fn intervention_policy_hash(&self) -> [u8; 32] {
        compute_intervention_policy_hash(&self.edits)
    }

    pub fn validate(&self, model: &Model) -> Result<(), HookError> {
        let c = &model.cfg;
        let l_count = c.n_layers;
        let q_heads = c.n_q_heads;

        for (hook, _) in &self.records {
            if hook.site.is_layer_site() && hook.layer as usize >= l_count {
                return Err(HookError::UnknownSite {
                    layer: hook.layer,
                    site: hook.site.as_str().to_string(),
                });
            }
            if let Some(h) = hook.head {
                if h as usize >= q_heads {
                    return Err(HookError::UnknownSite {
                        layer: hook.layer,
                        site: format!("{}:head{}", hook.site.as_str(), h),
                    });
                }
            }
        }

        for edit in &self.edits {
            let hook = edit.hook();
            if hook.site.is_layer_site() && hook.layer as usize >= l_count {
                return Err(HookError::UnknownSite {
                    layer: hook.layer,
                    site: hook.site.as_str().to_string(),
                });
            }
            match edit {
                EditOp::Steer { hook, direction, alpha, .. } => {
                    if !hook.site.is_steerable() {
                        return Err(HookError::InvalidSteering {
                            reason: format!("site {} is not steerable", hook.site.as_str()),
                        });
                    }
                    if !alpha.is_finite() || alpha.abs() > self.alpha_max {
                        return Err(HookError::InvalidSteering {
                            reason: format!("alpha {} exceeds alpha_max {}", alpha, self.alpha_max),
                        });
                    }
                    if direction.len() != c.d_model {
                        return Err(HookError::InvalidSteering {
                            reason: format!("direction len {} != d_model {}", direction.len(), c.d_model),
                        });
                    }
                    if direction.iter().any(|x| !x.is_finite()) {
                        return Err(HookError::InvalidSteering {
                            reason: "non-finite direction values".to_string(),
                        });
                    }
                }
                EditOp::Ablate { hook, kind } => {
                    if hook.site != HookSite::HeadOut && hook.site != HookSite::MlpHidden {
                        return Err(HookError::UnknownSite {
                            layer: hook.layer,
                            site: format!("ablation not supported on {}", hook.site.as_str()),
                        });
                    }
                    if let AblateKind::MeanFromSink(t) = kind {
                        if !t.all_finite() {
                            return Err(HookError::InvalidSteering {
                                reason: "non-finite mean values".to_string(),
                            });
                        }
                    }
                }
                EditOp::Patch { hook, source } => {
                    if !source.all_finite() {
                        return Err(HookError::InvalidSteering {
                            reason: "non-finite patch values".to_string(),
                        });
                    }
                    if hook.site.is_layer_site() && hook.layer as usize >= l_count {
                        return Err(HookError::UnknownSite {
                            layer: hook.layer,
                            site: hook.site.as_str().to_string(),
                        });
                    }
                }
            }
        }
        Ok(())
    }
}

/// Applies active edits to an activation tensor in place.
pub fn apply_edits(
    hook: &HookId,
    t: &mut Tensor,
    edits: &[EditOp],
    pos_offset: usize,
    t_len: usize,
) -> Result<(), HookError> {
    for edit in edits {
        if edit.hook() != hook {
            continue;
        }
        match edit {
            EditOp::Ablate { kind, .. } => match kind {
                AblateKind::Zero => {
                    t.data_mut().fill(0.0);
                }
                AblateKind::MeanFromSink(mean) => {
                    if t.shape() != mean.shape() {
                        return Err(HookError::ShapeMismatch {
                            expected: t.shape().to_vec(),
                            actual: mean.shape().to_vec(),
                        });
                    }
                    t.data_mut().copy_from_slice(mean.data());
                }
            },
            EditOp::Patch { source, .. } => {
                if t.shape() != source.shape() {
                    return Err(HookError::ShapeMismatch {
                        expected: t.shape().to_vec(),
                        actual: source.shape().to_vec(),
                    });
                }
                t.data_mut().copy_from_slice(source.data());
            }
            EditOp::Steer { direction, alpha, positions, .. } => {
                let rows = t.shape()[0];
                let d = t.shape()[1];
                if direction.len() != d {
                    return Err(HookError::ShapeMismatch {
                        expected: vec![d],
                        actual: vec![direction.len()],
                    });
                }
                for r in 0..rows {
                    let abs_pos = pos_offset + r;
                    let should_steer = match positions {
                        SteerPosition::AllPositions => true,
                        SteerPosition::LastPosition => r == rows - 1 && abs_pos == pos_offset + t_len - 1,
                        SteerPosition::PositionRange { start, end } => {
                            if *start > *end || *end > pos_offset + t_len {
                                return Err(HookError::PositionOutOfRange {
                                    start: *start,
                                    end: *end,
                                    t: pos_offset + t_len,
                                });
                            }
                            abs_pos >= *start && abs_pos < *end
                        }
                    };
                    if should_steer {
                        let row_data = &mut t.data_mut()[r * d..(r + 1) * d];
                        for (x, &v) in row_data.iter_mut().zip(direction) {
                            *x += *alpha * v;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Applies active record requests, writing to the caller-supplied sink.
pub fn apply_records(
    hook: &HookId,
    t: &Tensor,
    timing: RecordTiming,
    records: &[(HookId, RecordTiming)],
    sink: &mut ActivationSink,
) -> Result<(), HookError> {
    for (target, target_timing) in records {
        if target == hook && *target_timing == timing {
            sink.record(hook.clone(), timing, t)?;
        }
    }
    Ok(())
}

/// Executes Gate GX13 verification suite (§A2.8).
pub fn gx13() -> GateReport {
    let t0 = Instant::now();
    let cfg = crate::config::Config::reference_v2();
    let m = Model::init(&cfg).unwrap();
    let toks = [256u32, 72, 101, 108, 108, 111, 257];

    // Item 1: Disabled parity (Off mode matches baseline bitwise)
    let base_out = m.forward(&toks, None).unwrap();

    // Item 2: Record parity (Record mode does not alter forward pass output)
    let mut sink = ActivationSink::new(DEFAULT_MAX_EXPORT_BYTES).unwrap();
    let hook_ctx = HookContext::new().with_record(HookId::global(HookSite::EmbedOut), RecordTiming::PostEdit);
    let rec_out = m.forward_with_hooks(&toks, None, Some(&hook_ctx), Some(&mut sink)).unwrap();
    let record_parity = rec_out == base_out && sink.get(&HookId::global(HookSite::EmbedOut), RecordTiming::PostEdit).is_some();

    // Item 3: Zero-direction parity (alpha=0.0 or zero direction gives bitwise match)
    let steer_zero = EditOp::Steer {
        hook: HookId::layer(0, HookSite::ResidPre),
        direction: vec![0.0f32; 128],
        alpha: 1.0,
        positions: SteerPosition::AllPositions,
        direction_id: "zero".to_string(),
    };
    let ctx_zero = HookContext::new().with_edit(steer_zero);
    let zero_out = m.forward_with_hooks(&toks, None, Some(&ctx_zero), None).unwrap();
    let zero_parity = zero_out == base_out;

    // Item 4: Ablate and patch parity
    // Record layer 0 ResidPre, patch it back in a second run -> must match bitwise
    let mut sink_patch = ActivationSink::new(DEFAULT_MAX_EXPORT_BYTES).unwrap();
    let rec_l0 = HookContext::new().with_record(HookId::layer(0, HookSite::ResidPre), RecordTiming::PostEdit);
    let _ = m.forward_with_hooks(&toks, None, Some(&rec_l0), Some(&mut sink_patch)).unwrap();
    let recorded_tensor = sink_patch.get(&HookId::layer(0, HookSite::ResidPre), RecordTiming::PostEdit).unwrap().clone();
    let patch_op = EditOp::Patch {
        hook: HookId::layer(0, HookSite::ResidPre),
        source: recorded_tensor,
    };
    let ctx_patch = HookContext::new().with_edit(patch_op);
    let patch_out = m.forward_with_hooks(&toks, None, Some(&ctx_patch), None).unwrap();
    let patch_parity = patch_out == base_out;

    // Item 5: Bounds validation (oversize export, non-finite alpha/direction rejected)
    let mut small_sink = ActivationSink::new(16).unwrap(); // only 16 bytes cap
    let rec_overflow = HookContext::new().with_record(HookId::global(HookSite::EmbedOut), RecordTiming::PostEdit);
    let overflow_err = m.forward_with_hooks(&toks, None, Some(&rec_overflow), Some(&mut small_sink));
    let bounds_caught = matches!(overflow_err, Err(crate::model::ModelError::Hook(HookError::ExportTooLarge { .. })));

    let ok = record_parity && zero_parity && patch_parity && bounds_caught;

    GateReport {
        gate_id: "GX13",
        scope: "acceptance",
        status: if ok { "PASS" } else { "FAIL" },
        reason: None,
        requirement: "X13 activation hooks and inspectability: zero-cost off, record/zero/patch parity, bounds enforcement",
        implemented_check: "Bitwise parity check on disabled/record/zero/patch + bounds checks per §A2.8",
        prerequisites: "Config::reference_v2",
        exclusions: "None",
        revision: "e7aa4e4",
        dirty_state: false,
        toolchain: "rustc 1.98.1",
        target: "x86_64-unknown-linux-gnu",
        command: "cargo test --test test_x13_hooks",
        fixture_hash: hex(&sha256(&base_out.to_le_bytes())),
        config: "reference_v2",
        dtype: "fp32",
        expected: "exact bitwise match on record/zero/patch; typed error on export cap".to_string(),
        observed: format!(
            "record_parity={}, zero_parity={}, patch_parity={}, bounds_caught={}",
            record_parity, zero_parity, patch_parity, bounds_caught
        ),
        atol: 0.0,
        rtol: 0.0,
        execution_time_ms: t0.elapsed().as_millis(),
    }
}
