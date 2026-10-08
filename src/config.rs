//! Hyperparameters, validation (specv3 §2.2, §3.6) and parameter inventory (§2.3).

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum ConfigError {
    NonFinite(&'static str),
    OutOfRange(&'static str),
    Divisibility(&'static str),
    SpecialTokens(&'static str),
    ResourceCap(&'static str),
    UnsupportedSchema(u32),
    ExtensionNotEnabled(&'static str),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for ConfigError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeScalingType {
    Linear,
    Yarn,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeScalingConfig {
    pub scaling_type: RopeScalingType,
    pub train_context: usize,
    pub target_context: usize,
    pub yarn_alpha: f32,
    pub yarn_beta: f32,
}

impl RopeScalingConfig {
    pub fn linear(train_context: usize, target_context: usize) -> Self {
        Self {
            scaling_type: RopeScalingType::Linear,
            train_context,
            target_context,
            yarn_alpha: 1.0,
            yarn_beta: 32.0,
        }
    }

    pub fn yarn(train_context: usize, target_context: usize) -> Self {
        Self {
            scaling_type: RopeScalingType::Yarn,
            train_context,
            target_context,
            yarn_alpha: 1.0,
            yarn_beta: 32.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub schema_version: u32,
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_q_heads: usize,
    pub n_kv_heads: usize,
    pub d_ff: usize,
    pub max_seq_len: usize,
    pub batch_size: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
    pub bos_id: u32,
    pub eos_id: u32,
    pub pad_id: u32,
    pub tied_embeddings: bool,
    pub seed: u64,
    pub ext_x1_gqa: bool,
    pub ext_x2_untied_embeddings: bool,
    pub ext_x3_qk_norm: bool,
    pub ext_x4_rope: bool,
    pub rope_scaling: Option<RopeScalingConfig>,
    pub ext_x5_gemm: bool,
    pub ext_x6_arena: bool,
    pub max_arena_bytes: usize,
    pub ext_x7_chunked_loss: bool,
    pub chunk_size: usize,
    pub ext_x8_flash_attn: bool,
    pub flash_br: usize,
    pub flash_bc: usize,
    pub ext_x13_hooks: bool,
    pub ext_x13_edit: bool,
    pub ext_x12e_entropy_sampling: bool,
    pub ext_x15_lora: bool,
    pub ext_x17_import: bool,
}

pub const MAX_SEQ_CAP: usize = 8192;
pub const MAX_BATCH_CAP: usize = 256;
pub const MAX_LAYERS_CAP: usize = 32;
pub const MAX_D_MODEL_CAP: usize = 2048;
pub const MAX_VOCAB_CAP: usize = 65536;
pub const MAX_PARAM_BYTES_CAP: usize = 128 * 1024 * 1024; // 128 MiB

pub const MAX_IMPORT_VOCAB_CAP: usize = 262_144;
pub const MAX_IMPORT_PARAM_BYTES_CAP: usize = 1024 * 1024 * 1024; // 1 GiB
pub const MAX_IMPORT_SINGLE_TENSOR_BYTES_CAP: usize = 512 * 1024 * 1024; // 512 MiB

impl Config {
    /// `reference_v2` baseline: MHA, tied, V=259, D=128, L=4, P=886,272.
    pub fn reference_v2() -> Self {
        Config {
            schema_version: 1,
            vocab_size: 259,
            d_model: 128,
            n_layers: 4,
            n_q_heads: 4,
            n_kv_heads: 4,
            d_ff: 384,
            max_seq_len: 64,
            batch_size: 2,
            rope_theta: 10000.0,
            rms_eps: 1e-5,
            bos_id: 256,
            eos_id: 257,
            pad_id: 258,
            tied_embeddings: true,
            seed: 0x5EED_5A9D,
            ext_x1_gqa: false,
            ext_x2_untied_embeddings: false,
            ext_x3_qk_norm: false,
            ext_x4_rope: false,
            rope_scaling: None,
            ext_x5_gemm: false,
            ext_x6_arena: false,
            max_arena_bytes: 512 * 1024 * 1024,
            ext_x7_chunked_loss: false,
            chunk_size: 16,
            ext_x8_flash_attn: false,
            flash_br: 16,
            flash_bc: 16,
            ext_x13_hooks: false,
            ext_x13_edit: false,
            ext_x12e_entropy_sampling: false,
            ext_x15_lora: false,
            ext_x17_import: false,
        }
    }

    /// GQA scaling profile (§4.2 Item 1): H_q = 4, H_kv = 2, tied, P = 820,736.
    pub fn gqa_v2() -> Self {
        let mut c = Self::reference_v2();
        c.ext_x1_gqa = true;
        c.n_kv_heads = 2;
        c
    }

    /// MQA scaling profile (§4.2 Item 1): H_q = 4, H_kv = 1, tied, P = 787,968.
    pub fn mqa_v2() -> Self {
        let mut c = Self::reference_v2();
        c.ext_x1_gqa = true;
        c.n_kv_heads = 1;
        c
    }

    /// Untied embeddings scaling profile (§4.2 Item 2): MHA, untied, P = 919,424.
    pub fn untied_v2() -> Self {
        let mut c = Self::reference_v2();
        c.tied_embeddings = false;
        c.ext_x2_untied_embeddings = true;
        c
    }

    /// Untied GQA scaling profile (§4.2 Item 2): H_kv = 2, untied, P = 853,888.
    pub fn untied_gqa_v2() -> Self {
        let mut c = Self::gqa_v2();
        c.tied_embeddings = false;
        c.ext_x2_untied_embeddings = true;
        c
    }

    /// Untied MQA scaling profile (§4.2 Item 2): H_kv = 1, untied, P = 821,120.
    pub fn untied_mqa_v2() -> Self {
        let mut c = Self::mqa_v2();
        c.tied_embeddings = false;
        c.ext_x2_untied_embeddings = true;
        c
    }

    /// QK-Norm scaling profile (§4.2 Item 3): MHA, tied, P = 887,296 (+1,024 params).
    pub fn qk_norm_v2() -> Self {
        let mut c = Self::reference_v2();
        c.ext_x3_qk_norm = true;
        c
    }

    /// YaRN RoPE context extension profile (§4.2 Item 4): MHA, tied, extended target context.
    pub fn rope_yarn_v2(target_context: usize) -> Self {
        let mut c = Self::reference_v2();
        c.ext_x4_rope = true;
        c.max_seq_len = target_context;
        c.rope_scaling = Some(RopeScalingConfig::yarn(64, target_context));
        c
    }

    /// Linear RoPE context extension profile (§4.2 Item 4): MHA, tied, extended target context.
    pub fn rope_linear_v2(target_context: usize) -> Self {
        let mut c = Self::reference_v2();
        c.ext_x4_rope = true;
        c.max_seq_len = target_context;
        c.rope_scaling = Some(RopeScalingConfig::linear(64, target_context));
        c
    }

    /// Optimized CPU GEMM profile (§4.2 Item 5): MHA, tied.
    pub fn gemm_opt_v2() -> Self {
        let mut c = Self::reference_v2();
        c.ext_x5_gemm = true;
        c
    }

    /// Typed buffer reuse & in-place allocator profile (§4.2 Item 6): MHA, tied, arena allocator.
    pub fn arena_v2(max_arena_bytes: usize) -> Self {
        let mut c = Self::reference_v2();
        c.ext_x6_arena = true;
        c.max_arena_bytes = max_arena_bytes;
        c
    }

    /// Memory-efficient chunked cross-entropy profile (§4.2 Item 7): MHA, tied, chunked loss.
    pub fn chunked_loss_v2(chunk_size: usize) -> Self {
        let mut c = Self::reference_v2();
        c.ext_x7_chunked_loss = true;
        c.chunk_size = chunk_size;
        c
    }

    /// FlashAttention-2 custom tiled attention profile (§4.2 Item 8): MHA, tied, Br=16, Bc=16.
    pub fn flash_attn_v2() -> Self {
        let mut c = Self::reference_v2();
        c.ext_x8_flash_attn = true;
        c.flash_br = 16;
        c.flash_bc = 16;
        c
    }

    pub fn head_dim(&self) -> usize {
        self.d_model.checked_div(self.n_q_heads).unwrap_or(0)
    }

    /// Canonical cryptographic fingerprint of forward-affecting hyperparameters.
    pub fn fingerprint(&self) -> [u8; 32] {
        let s = if !self.ext_x1_gqa && !self.ext_x2_untied_embeddings && !self.ext_x3_qk_norm && !self.ext_x4_rope && !self.ext_x5_gemm && !self.ext_x6_arena && !self.ext_x7_chunked_loss && !self.ext_x8_flash_attn && !self.ext_x13_hooks && !self.ext_x13_edit && !self.ext_x12e_entropy_sampling && !self.ext_x15_lora && !self.ext_x17_import {
            format!(
                "config:v{}:vs{}:d{}:l{}:q{}:kv{}:ff{}:seq{}:b{}:th_bits{:08x}:eps_bits{:08x}:bos{}:eos{}:pad{}:tied{}:seed{}",
                self.schema_version, self.vocab_size, self.d_model, self.n_layers,
                self.n_q_heads, self.n_kv_heads, self.d_ff, self.max_seq_len,
                self.batch_size, self.rope_theta.to_bits(), self.rms_eps.to_bits(),
                self.bos_id, self.eos_id, self.pad_id, self.tied_embeddings, self.seed
            )
        } else {
            let rs_str = match &self.rope_scaling {
                None => "none".to_string(),
                Some(rs) => format!("{:?}:{}:{}:{:.2}:{:.2}", rs.scaling_type, rs.train_context, rs.target_context, rs.yarn_alpha, rs.yarn_beta),
            };
            format!(
                "config:v{}:vs{}:d{}:l{}:q{}:kv{}:ff{}:seq{}:b{}:th_bits{:08x}:eps_bits{:08x}:bos{}:eos{}:pad{}:tied{}:seed{}:x1g{}:x2u{}:x3q{}:x4r{}:{}:x5g{}:x6a{}:{}:x7c{}:{}:x8f{}:{}:{}:x13h{}:x13e{}:x12e{}:x15l{}:x17i{}",
                self.schema_version, self.vocab_size, self.d_model, self.n_layers,
                self.n_q_heads, self.n_kv_heads, self.d_ff, self.max_seq_len,
                self.batch_size, self.rope_theta.to_bits(), self.rms_eps.to_bits(),
                self.bos_id, self.eos_id, self.pad_id, self.tied_embeddings, self.seed,
                self.ext_x1_gqa, self.ext_x2_untied_embeddings, self.ext_x3_qk_norm, self.ext_x4_rope, rs_str, self.ext_x5_gemm, self.ext_x6_arena, self.max_arena_bytes, self.ext_x7_chunked_loss, self.chunk_size, self.ext_x8_flash_attn, self.flash_br, self.flash_bc, self.ext_x13_hooks, self.ext_x13_edit, self.ext_x12e_entropy_sampling, self.ext_x15_lora, self.ext_x17_import
            )
        };
        crate::sha256::sha256(s.as_bytes())
    }

    /// Validate every precondition of §3.6 before any allocation.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.schema_version != 1 {
            return Err(ConfigError::UnsupportedSchema(self.schema_version));
        }
        if !(self.rms_eps.is_finite() && self.rms_eps > 0.0) {
            return Err(ConfigError::NonFinite("rms_eps"));
        }
        if !(self.rope_theta.is_finite() && self.rope_theta > 1.0) {
            return Err(ConfigError::NonFinite("rope_theta"));
        }
        for (v, name) in [
            (self.d_model, "d_model"),
            (self.n_q_heads, "n_q_heads"),
            (self.n_kv_heads, "n_kv_heads"),
            (self.n_layers, "n_layers"),
            (self.d_ff, "d_ff"),
            (self.batch_size, "batch_size"),
        ] {
            if v == 0 {
                return Err(ConfigError::OutOfRange(name));
            }
        }
        if self.d_model % self.n_q_heads != 0 {
            return Err(ConfigError::Divisibility("d_model % n_q_heads"));
        }
        if self.n_q_heads % self.n_kv_heads != 0 {
            return Err(ConfigError::Divisibility("n_q_heads % n_kv_heads"));
        }
        if self.head_dim() == 0 || self.head_dim() % 2 != 0 {
            return Err(ConfigError::Divisibility("head_dim must be even"));
        }
        let max_vocab = if self.ext_x17_import {
            MAX_IMPORT_VOCAB_CAP
        } else {
            MAX_VOCAB_CAP
        };
        if self.vocab_size > max_vocab {
            return Err(ConfigError::ResourceCap("vocab_size"));
        }
        if !self.ext_x17_import && self.vocab_size < 259 {
            return Err(ConfigError::OutOfRange("vocab_size < 259"));
        }
        if self.ext_x17_import && self.vocab_size == 0 {
            return Err(ConfigError::OutOfRange("vocab_size == 0"));
        }
        if self.max_seq_len < 1 {
            return Err(ConfigError::OutOfRange("max_seq_len"));
        }
        let v = self.vocab_size as u64;
        let ids = [self.bos_id, self.eos_id, self.pad_id];
        if self.ext_x17_import {
            if ids.iter().any(|&i| u64::from(i) >= v) {
                return Err(ConfigError::SpecialTokens("special id must be < vocab_size"));
            }
        } else {
            if ids.iter().any(|&i| i < 256 || u64::from(i) >= v) {
                return Err(ConfigError::SpecialTokens("special id must be in 256..vocab_size"));
            }
            if ids[0] == ids[1] || ids[1] == ids[2] || ids[0] == ids[2] {
                return Err(ConfigError::SpecialTokens("special ids must be pairwise distinct"));
            }
        }
        if self.max_seq_len > MAX_SEQ_CAP {
            return Err(ConfigError::ResourceCap("max_seq_len"));
        }
        if self.batch_size > MAX_BATCH_CAP {
            return Err(ConfigError::ResourceCap("batch_size"));
        }
        if self.n_layers > MAX_LAYERS_CAP {
            return Err(ConfigError::ResourceCap("n_layers"));
        }
        if self.d_model > MAX_D_MODEL_CAP {
            return Err(ConfigError::ResourceCap("d_model"));
        }
        let p_count = self.param_count().ok_or(ConfigError::ResourceCap("param_count overflow"))?;
        let param_bytes = p_count.checked_mul(4).ok_or(ConfigError::ResourceCap("param_bytes overflow"))?;
        let max_param_bytes = if self.ext_x17_import {
            MAX_IMPORT_PARAM_BYTES_CAP
        } else {
            MAX_PARAM_BYTES_CAP
        };
        if param_bytes > max_param_bytes {
            return Err(ConfigError::ResourceCap("total param bytes exceeds MAX_PARAM_BYTES_CAP"));
        }

        // Strict baseline extension rejection: baseline must be tied MHA
        if !self.ext_x17_import {
            if !self.ext_x2_untied_embeddings && !self.tied_embeddings {
                return Err(ConfigError::ExtensionNotEnabled("tied_embeddings=false is extension X2, disabled in baseline"));
            }
            if !self.ext_x1_gqa && self.n_kv_heads != self.n_q_heads {
                return Err(ConfigError::ExtensionNotEnabled("n_kv_heads != n_q_heads is extension X1 (GQA/MQA), disabled in baseline"));
            }
        }

        if self.rope_scaling.is_some() && !self.ext_x4_rope {
            return Err(ConfigError::ExtensionNotEnabled("rope_scaling requires ext_x4_rope (extension X4)"));
        }
        if let Some(ref rs) = self.rope_scaling {
            if rs.train_context == 0 || rs.target_context == 0 {
                return Err(ConfigError::OutOfRange("rope_scaling contexts must be > 0"));
            }
            if rs.target_context < rs.train_context {
                return Err(ConfigError::OutOfRange("target_context must be >= train_context"));
            }
            if !rs.yarn_alpha.is_finite() || rs.yarn_alpha < 0.0 || !rs.yarn_beta.is_finite() || rs.yarn_beta <= rs.yarn_alpha {
                return Err(ConfigError::OutOfRange("yarn_alpha and yarn_beta invalid"));
            }
        }

        if self.ext_x6_arena && self.max_arena_bytes == 0 {
            return Err(ConfigError::OutOfRange("max_arena_bytes must be > 0"));
        }

        if self.ext_x7_chunked_loss && self.chunk_size == 0 {
            return Err(ConfigError::OutOfRange("chunk_size must be > 0"));
        }

        if self.ext_x8_flash_attn && (self.flash_br == 0 || self.flash_bc == 0) {
            return Err(ConfigError::OutOfRange("flash_br and flash_bc must be > 0"));
        }

        Ok(())
    }

    /// Exact parameter count per §2.3 (checked arithmetic, no biases).
    pub fn param_count(&self) -> Option<usize> {
        let d = self.d_model;
        let dh = self.head_dim();
        let kv = self.n_kv_heads.checked_mul(dh)?;
        let q = self.n_q_heads.checked_mul(dh)?;
        let mut per_layer = d // rms1
            .checked_add(d.checked_mul(q)?)? // Wq
            .checked_add(d.checked_mul(kv)?)? // Wk
            .checked_add(d.checked_mul(kv)?)? // Wv
            .checked_add(q.checked_mul(d)?)? // Wo
            .checked_add(d)? // rms2
            .checked_add(d.checked_mul(self.d_ff)?.checked_mul(3)?)?; // gate+up+down
        if self.ext_x3_qk_norm {
            per_layer = per_layer.checked_add(q)?.checked_add(kv)?;
        }
        let embed = self.vocab_size.checked_mul(d)?;
        let mut total = embed
            .checked_add(per_layer.checked_mul(self.n_layers)?)?
            .checked_add(d)?; // final rms
        if !self.tied_embeddings {
            total = total.checked_add(embed)?;
        }
        Some(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_param_count_g6() {
        let c = Config::reference_v2();
        c.validate().unwrap();
        assert_eq!(c.param_count(), Some(886_272));
    }

    #[test]
    fn qk_norm_counts() {
        let c_mha = Config::qk_norm_v2();
        assert_eq!(c_mha.param_count(), Some(887_296));
        assert!(c_mha.validate().is_ok());

        let mut c_gqa = Config::gqa_v2();
        c_gqa.ext_x3_qk_norm = true;
        assert_eq!(c_gqa.param_count(), Some(821_504));
        assert!(c_gqa.validate().is_ok());

        let mut c_mqa = Config::mqa_v2();
        c_mqa.ext_x3_qk_norm = true;
        assert_eq!(c_mqa.param_count(), Some(788_608));
        assert!(c_mqa.validate().is_ok());
    }

    #[test]
    fn untied_gqa_mqa_counts() {
        let mut c = Config::reference_v2();
        c.tied_embeddings = false;
        assert_eq!(c.param_count(), Some(919_424));
        assert!(matches!(c.validate(), Err(ConfigError::ExtensionNotEnabled(_))));
        c.ext_x2_untied_embeddings = true;
        assert!(c.validate().is_ok());

        let u_mha = Config::untied_v2();
        assert_eq!(u_mha.param_count(), Some(919_424));
        assert!(u_mha.validate().is_ok());

        let u_gqa = Config::untied_gqa_v2();
        assert_eq!(u_gqa.param_count(), Some(853_888));
        assert!(u_gqa.validate().is_ok());

        let u_mqa = Config::untied_mqa_v2();
        assert_eq!(u_mqa.param_count(), Some(821_120));
        assert!(u_mqa.validate().is_ok());

        c.n_kv_heads = 2;
        c.ext_x2_untied_embeddings = false;
        assert_eq!(c.param_count(), Some(853_888));
        c.tied_embeddings = true;
        assert_eq!(c.param_count(), Some(820_736));
        assert!(matches!(c.validate(), Err(ConfigError::ExtensionNotEnabled(_))));
        c.n_kv_heads = 1;
        assert_eq!(c.param_count(), Some(787_968));
        c.tied_embeddings = false;
        assert_eq!(c.param_count(), Some(821_120));
    }

    #[test]
    fn rejects_bad_configs() {
        let mut c = Config::reference_v2();
        c.rms_eps = f32::INFINITY;
        assert!(c.validate().is_err());
        let mut c = Config::reference_v2();
        c.rms_eps = f32::NAN;
        assert!(c.validate().is_err());
        let mut c = Config::reference_v2();
        c.n_q_heads = 3;
        assert!(c.validate().is_err());
        let mut c = Config::reference_v2();
        c.pad_id = c.eos_id;
        assert!(c.validate().is_err());
        let mut c = Config::reference_v2();
        c.bos_id = 5; // collides with byte token 5
        assert!(c.validate().is_err());
        let mut c = Config::reference_v2();
        c.bos_id = c.pad_id;
        assert!(c.validate().is_err());
        let mut c = Config::reference_v2();
        c.max_seq_len = MAX_SEQ_CAP + 1;
        assert!(c.validate().is_err());
        let mut c = Config::reference_v2();
        c.n_layers = MAX_LAYERS_CAP + 1;
        assert!(matches!(c.validate(), Err(ConfigError::ResourceCap(_))));
        let mut c = Config::reference_v2();
        c.schema_version = 2;
        assert!(c.validate().is_err());
    }
}
