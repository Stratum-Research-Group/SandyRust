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
    pub ext_x13_hooks: bool,
    pub ext_x13_edit: bool,
    pub ext_x12e_entropy_sampling: bool,
}

pub const MAX_SEQ_CAP: usize = 8192;
pub const MAX_BATCH_CAP: usize = 256;
pub const MAX_LAYERS_CAP: usize = 32;
pub const MAX_D_MODEL_CAP: usize = 2048;
pub const MAX_VOCAB_CAP: usize = 65536;
pub const MAX_PARAM_BYTES_CAP: usize = 128 * 1024 * 1024; // 128 MiB

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
            ext_x13_hooks: false,
            ext_x13_edit: false,
            ext_x12e_entropy_sampling: false,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_q_heads
    }

    /// Canonical cryptographic fingerprint of forward-affecting hyperparameters.
    pub fn fingerprint(&self) -> [u8; 32] {
        let s = if !self.ext_x13_hooks && !self.ext_x13_edit && !self.ext_x12e_entropy_sampling {
            format!(
                "config:v{}:vs{}:d{}:l{}:q{}:kv{}:ff{}:seq{}:b{}:th{:.6}:eps{:.8}:bos{}:eos{}:pad{}:tied{}:seed{}",
                self.schema_version, self.vocab_size, self.d_model, self.n_layers,
                self.n_q_heads, self.n_kv_heads, self.d_ff, self.max_seq_len,
                self.batch_size, self.rope_theta, self.rms_eps,
                self.bos_id, self.eos_id, self.pad_id, self.tied_embeddings, self.seed
            )
        } else {
            format!(
                "config:v{}:vs{}:d{}:l{}:q{}:kv{}:ff{}:seq{}:b{}:th{:.6}:eps{:.8}:bos{}:eos{}:pad{}:tied{}:seed{}:x13h{}:x13e{}:x12e{}",
                self.schema_version, self.vocab_size, self.d_model, self.n_layers,
                self.n_q_heads, self.n_kv_heads, self.d_ff, self.max_seq_len,
                self.batch_size, self.rope_theta, self.rms_eps,
                self.bos_id, self.eos_id, self.pad_id, self.tied_embeddings, self.seed,
                self.ext_x13_hooks, self.ext_x13_edit, self.ext_x12e_entropy_sampling
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
        if self.vocab_size < 259 {
            return Err(ConfigError::OutOfRange("vocab_size < 259"));
        }
        if self.max_seq_len < 1 {
            return Err(ConfigError::OutOfRange("max_seq_len"));
        }
        let v = self.vocab_size as u64;
        let ids = [self.bos_id, self.eos_id, self.pad_id];
        if ids.iter().any(|&i| i < 256 || u64::from(i) >= v) {
            return Err(ConfigError::SpecialTokens("special id must be in 256..vocab_size"));
        }
        if ids[0] == ids[1] || ids[1] == ids[2] || ids[0] == ids[2] {
            return Err(ConfigError::SpecialTokens("special ids must be pairwise distinct"));
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
        if self.vocab_size > MAX_VOCAB_CAP {
            return Err(ConfigError::ResourceCap("vocab_size"));
        }
        let p_count = self.param_count().ok_or(ConfigError::ResourceCap("param_count overflow"))?;
        let param_bytes = p_count.checked_mul(4).ok_or(ConfigError::ResourceCap("param_bytes overflow"))?;
        if param_bytes > MAX_PARAM_BYTES_CAP {
            return Err(ConfigError::ResourceCap("total param bytes exceeds MAX_PARAM_BYTES_CAP"));
        }

        // Strict baseline extension rejection: baseline must be tied MHA
        if !self.tied_embeddings {
            return Err(ConfigError::ExtensionNotEnabled("tied_embeddings=false is extension X2, disabled in baseline"));
        }
        if self.n_kv_heads != self.n_q_heads {
            return Err(ConfigError::ExtensionNotEnabled("n_kv_heads != n_q_heads is extension X1 (GQA/MQA), disabled in baseline"));
        }

        Ok(())
    }

    /// Exact parameter count per §2.3 (checked arithmetic, no biases).
    pub fn param_count(&self) -> Option<usize> {
        let d = self.d_model;
        let dh = self.head_dim();
        let kv = self.n_kv_heads.checked_mul(dh)?;
        let q = self.n_q_heads.checked_mul(dh)?;
        let per_layer = d // rms1
            .checked_add(d.checked_mul(q)?)? // Wq
            .checked_add(d.checked_mul(kv)?)? // Wk
            .checked_add(d.checked_mul(kv)?)? // Wv
            .checked_add(q.checked_mul(d)?)? // Wo
            .checked_add(d)? // rms2
            .checked_add(d.checked_mul(self.d_ff)?.checked_mul(3)?)?; // gate+up+down
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
    fn untied_gqa_mqa_counts() {
        let mut c = Config::reference_v2();
        c.tied_embeddings = false;
        assert_eq!(c.param_count(), Some(919_424));
        assert!(matches!(c.validate(), Err(ConfigError::ExtensionNotEnabled(_))));
        c.n_kv_heads = 2;
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
