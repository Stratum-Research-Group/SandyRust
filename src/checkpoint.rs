//! Safe checkpoint export/import (§8.2): validated header, SHA-256 payload checksums,
//! non-finite scan, byte cap, and write-verify-publish saving.
//!
//! Layout: `RCHK | u32 version | arch(7 x u32) | max_seq_len u32 | rope_theta f32 | rms_eps f32 |
//! config fp[32] | tokenizer fp[32] | u32 count | entries(...) | payload`.
//! Training layout: `RTRN | u32 version | config fp[32] | corpus hash[32] | ...`.

use crate::config::Config;
use crate::model::{inventory, Model};
use crate::sha256::sha256;
use crate::tensor::Tensor;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

pub const MAGIC: &[u8; 4] = b"RCHK";
pub const TRAINING_MAGIC: &[u8; 4] = b"RTRN";
pub const VERSION: u32 = 1;
pub const DEFAULT_MAX_BYTES: u64 = 64 * 1024 * 1024;

static TMP_COUNTER: AtomicU64 = AtomicU64::new(1);

fn unique_tmp_path(path: &Path) -> std::path::PathBuf {
    let pid = std::process::id();
    let ctr = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(format!(".tmp.{pid}.{ctr}"));
    std::path::PathBuf::from(tmp)
}

#[derive(Debug, Clone, PartialEq)]
pub enum CheckpointError {
    Io(String),
    BadMagic,
    UnsupportedVersion(u32),
    ArchMismatch,
    ConfigMismatch(String),
    TokenizerMismatch,
    CorpusMismatch,
    InventoryMismatch(String),
    Truncated,
    OutOfBounds,
    ChecksumMismatch(String),
    NonFinite(String),
    TooLarge { size: u64, max: u64 },
    Model(String),
    TrainingState(String),
}

pub fn tokenizer_fingerprint(cfg: &Config) -> [u8; 32] {
    sha256(format!("bytelevel-v1:{}:{},{},{}", cfg.vocab_size, cfg.bos_id, cfg.eos_id, cfg.pad_id).as_bytes())
}

fn arch(cfg: &Config) -> [u32; 7] {
    [
        cfg.vocab_size as u32,
        cfg.d_model as u32,
        cfg.n_layers as u32,
        cfg.n_q_heads as u32,
        cfg.n_kv_heads as u32,
        cfg.d_ff as u32,
        cfg.tied_embeddings as u32,
    ]
}

pub fn serialize(model: &Model) -> Vec<u8> {
    let tensors = model.tensors();
    let mut payload = Vec::new();
    let mut entries = Vec::new();
    for (name, t) in &tensors {
        let off = payload.len() as u64;
        for x in t.data() {
            payload.extend_from_slice(&x.to_le_bytes());
        }
        let len = payload.len() as u64 - off;
        let digest = sha256(&payload[off as usize..]);
        let mut e = Vec::new();
        e.extend_from_slice(&(name.len() as u32).to_le_bytes());
        e.extend_from_slice(name.as_bytes());
        e.extend_from_slice(&(t.shape().len() as u32).to_le_bytes());
        for &d in t.shape() {
            e.extend_from_slice(&(d as u64).to_le_bytes());
        }
        e.push(0); // dtype f32
        e.extend_from_slice(&off.to_le_bytes());
        e.extend_from_slice(&len.to_le_bytes());
        e.extend_from_slice(&digest);
        entries.push(e);
    }
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    for a in arch(&model.cfg) {
        out.extend_from_slice(&a.to_le_bytes());
    }
    out.extend_from_slice(&(model.cfg.max_seq_len as u32).to_le_bytes());
    out.extend_from_slice(&model.cfg.rope_theta.to_bits().to_le_bytes());
    out.extend_from_slice(&model.cfg.rms_eps.to_bits().to_le_bytes());
    out.extend_from_slice(&model.cfg.fingerprint());
    out.extend_from_slice(&tokenizer_fingerprint(&model.cfg));
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for e in entries {
        out.extend_from_slice(&e);
    }
    out.extend_from_slice(&payload);
    out
}

struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], CheckpointError> {
        let end = self.pos.checked_add(n).ok_or(CheckpointError::OutOfBounds)?;
        if end > self.b.len() {
            return Err(CheckpointError::Truncated);
        }
        let s = &self.b[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, CheckpointError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, CheckpointError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
}

/// Validate and decode a checkpoint for `cfg`. Never panics on malformed input.
pub fn parse(bytes: &[u8], cfg: &Config, max_bytes: u64) -> Result<Model, CheckpointError> {
    if bytes.len() as u64 > max_bytes {
        return Err(CheckpointError::TooLarge { size: bytes.len() as u64, max: max_bytes });
    }
    let mut r = Reader { b: bytes, pos: 0 };
    if r.take(4)? != MAGIC {
        return Err(CheckpointError::BadMagic);
    }
    let v = r.u32()?;
    if v != VERSION {
        return Err(CheckpointError::UnsupportedVersion(v));
    }
    for a in arch(cfg) {
        if r.u32()? != a {
            return Err(CheckpointError::ArchMismatch);
        }
    }
    let max_seq = r.u32()? as usize;
    if max_seq != cfg.max_seq_len {
        return Err(CheckpointError::ConfigMismatch(format!("max_seq_len mismatch: ckpt {max_seq} != cfg {}", cfg.max_seq_len)));
    }
    let rope_theta = f32::from_bits(r.u32()?);
    if rope_theta != cfg.rope_theta {
        return Err(CheckpointError::ConfigMismatch(format!("rope_theta mismatch: ckpt {rope_theta} != cfg {}", cfg.rope_theta)));
    }
    let rms_eps = f32::from_bits(r.u32()?);
    if rms_eps != cfg.rms_eps {
        return Err(CheckpointError::ConfigMismatch(format!("rms_eps mismatch: ckpt {rms_eps} != cfg {}", cfg.rms_eps)));
    }
    let cfg_fp: [u8; 32] = r.take(32)?.try_into().unwrap();
    if cfg_fp != cfg.fingerprint() {
        return Err(CheckpointError::ConfigMismatch("config fingerprint mismatch".into()));
    }
    if r.take(32)? != tokenizer_fingerprint(cfg) {
        return Err(CheckpointError::TokenizerMismatch);
    }
    let inv = inventory(cfg);
    if r.u32()? as usize != inv.len() {
        return Err(CheckpointError::InventoryMismatch("tensor count".into()));
    }
    struct Entry {
        off: u64,
        len: u64,
        digest: [u8; 32],
    }
    let mut entries = Vec::new();
    for (name, shape) in &inv {
        let nl = r.u32()? as usize;
        if nl > 256 {
            return Err(CheckpointError::OutOfBounds);
        }
        if r.take(nl)? != name.as_bytes() {
            return Err(CheckpointError::InventoryMismatch(format!("name != {name}")));
        }
        let rank = r.u32()? as usize;
        if rank != shape.len() {
            return Err(CheckpointError::InventoryMismatch(format!("rank of {name}")));
        }
        for &d in shape {
            if r.u64()? != d as u64 {
                return Err(CheckpointError::InventoryMismatch(format!("shape of {name}")));
            }
        }
        if r.take(1)?[0] != 0 {
            return Err(CheckpointError::InventoryMismatch(format!("dtype of {name}")));
        }
        let (off, len) = (r.u64()?, r.u64()?);
        let digest: [u8; 32] = r.take(32)?.try_into().unwrap();
        let expect = shape
            .iter()
            .try_fold(4usize, |a, &d| a.checked_mul(d))
            .ok_or(CheckpointError::OutOfBounds)? as u64;
        if len != expect {
            return Err(CheckpointError::InventoryMismatch(format!("byte length of {name}")));
        }
        entries.push(Entry { off, len, digest });
    }
    let payload = &bytes[r.pos..];
    let mut tensors = Vec::new();
    let mut expected_off = 0u64;
    for ((name, shape), e) in inv.iter().zip(&entries) {
        if e.off != expected_off {
            return Err(CheckpointError::InventoryMismatch(format!("offset of {name}")));
        }
        let end = e.off.checked_add(e.len).ok_or(CheckpointError::OutOfBounds)?;
        if end > payload.len() as u64 {
            return Err(CheckpointError::Truncated);
        }
        let raw = &payload[e.off as usize..end as usize];
        if sha256(raw) != e.digest {
            return Err(CheckpointError::ChecksumMismatch(name.clone()));
        }
        let data: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        if data.iter().any(|x| !x.is_finite()) {
            return Err(CheckpointError::NonFinite(name.clone()));
        }
        tensors.push(Tensor::from_vec(shape, data).map_err(|e| CheckpointError::Model(format!("{e:?}")))?);
        expected_off = end;
    }
    if expected_off != payload.len() as u64 {
        return Err(CheckpointError::InventoryMismatch("trailing payload bytes".into()));
    }
    Model::from_tensors(cfg, tensors).map_err(|e| CheckpointError::Model(format!("{e:?}")))
}

pub fn load(path: &Path, cfg: &Config, max_bytes: u64) -> Result<Model, CheckpointError> {
    let mut file = File::open(path).map_err(|e| CheckpointError::Io(e.to_string()))?;
    let mut bytes = Vec::new();
    (&mut file)
        .take(max_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|e| CheckpointError::Io(e.to_string()))?;
    if bytes.len() as u64 > max_bytes {
        return Err(CheckpointError::TooLarge { size: bytes.len() as u64, max: max_bytes });
    }
    parse(&bytes, cfg, max_bytes)
}

/// Write-verify-publish: unique tmp file -> fsync -> read back + full validation -> atomic rename.
pub fn save(model: &Model, path: &Path) -> Result<(), CheckpointError> {
    let io = |e: std::io::Error| CheckpointError::Io(e.to_string());
    let tmp = unique_tmp_path(path);
    let bytes = serialize(model);
    let result = (|| {
        let mut f = File::create(&tmp).map_err(io)?;
        f.write_all(&bytes).map_err(io)?;
        f.flush().map_err(io)?;
        f.sync_all().map_err(io)?;
        drop(f);
        load(&tmp, &model.cfg, DEFAULT_MAX_BYTES)?; // read-back verification
        fs::rename(&tmp, path).map_err(io)?;
        if let Some(parent) = path.parent() {
            if let Ok(dir) = File::open(parent) {
                let _ = dir.sync_all();
            }
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

/// Complete training resumption state (§8.2, Item 10).
#[derive(Debug, Clone, PartialEq)]
pub struct TrainingState {
    pub model: Model,
    pub optimizer: crate::optimizer::AdamW,
    pub loader_state: crate::data::LoaderState,
    pub rng_state: ([u32; 16], [u32; 16], usize),
    pub schedule_step: usize,
}

pub fn serialize_training(state: &TrainingState) -> Vec<u8> {
    let model_bytes = serialize(&state.model);
    let mut body = Vec::new();

    body.extend_from_slice(&state.model.cfg.fingerprint());
    body.extend_from_slice(&state.loader_state.corpus_hash);
    body.extend_from_slice(&(state.schedule_step as u64).to_le_bytes());
    body.extend_from_slice(&state.optimizer.step.to_le_bytes());
    body.extend_from_slice(&(state.optimizer.consecutive_skips as u64).to_le_bytes());

    body.extend_from_slice(&state.loader_state.position.to_le_bytes());
    body.extend_from_slice(&state.loader_state.epoch.to_le_bytes());
    body.extend_from_slice(&state.loader_state.shuffle_seed.to_le_bytes());

    for s in &state.rng_state.0 {
        body.extend_from_slice(&s.to_le_bytes());
    }
    for s in &state.rng_state.1 {
        body.extend_from_slice(&s.to_le_bytes());
    }
    body.extend_from_slice(&(state.rng_state.2 as u64).to_le_bytes());

    body.extend_from_slice(&(model_bytes.len() as u64).to_le_bytes());
    body.extend_from_slice(&model_bytes);

    body.extend_from_slice(&(state.optimizer.m.len() as u32).to_le_bytes());
    for m_vec in &state.optimizer.m {
        body.extend_from_slice(&(m_vec.len() as u64).to_le_bytes());
        for &val in m_vec {
            body.extend_from_slice(&val.to_le_bytes());
        }
    }

    body.extend_from_slice(&(state.optimizer.v.len() as u32).to_le_bytes());
    for v_vec in &state.optimizer.v {
        body.extend_from_slice(&(v_vec.len() as u64).to_le_bytes());
        for &val in v_vec {
            body.extend_from_slice(&val.to_le_bytes());
        }
    }

    let checksum = sha256(&body);
    let mut out = Vec::new();
    out.extend_from_slice(TRAINING_MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&body);
    out.extend_from_slice(&checksum);
    out
}

pub fn parse_training(
    bytes: &[u8],
    cfg: &Config,
    corpus_hash: &[u8; 32],
    max_bytes: u64,
) -> Result<TrainingState, CheckpointError> {
    if bytes.len() as u64 > max_bytes {
        return Err(CheckpointError::TooLarge { size: bytes.len() as u64, max: max_bytes });
    }
    if bytes.len() < 4 + 4 + 32 {
        return Err(CheckpointError::Truncated);
    }
    let body_end = bytes.len().checked_sub(32).ok_or(CheckpointError::Truncated)?;
    let expected_checksum = &bytes[body_end..];
    let computed_checksum = sha256(&bytes[8..body_end]);
    if computed_checksum.as_slice() != expected_checksum {
        return Err(CheckpointError::ChecksumMismatch("training payload".into()));
    }

    let mut r = Reader { b: &bytes[..body_end], pos: 0 };
    if r.take(4)? != TRAINING_MAGIC {
        return Err(CheckpointError::BadMagic);
    }
    let v = r.u32()?;
    if v != VERSION {
        return Err(CheckpointError::UnsupportedVersion(v));
    }

    let stored_cfg_fp: [u8; 32] = r.take(32)?.try_into().unwrap();
    if stored_cfg_fp != cfg.fingerprint() {
        return Err(CheckpointError::ConfigMismatch("config fingerprint mismatch in training checkpoint".into()));
    }

    let stored_corpus_hash: [u8; 32] = r.take(32)?.try_into().unwrap();
    if stored_corpus_hash != *corpus_hash {
        return Err(CheckpointError::CorpusMismatch);
    }

    let schedule_step = r.u64()? as usize;
    let opt_step = r.u64()?;
    let consecutive_skips = r.u64()? as usize;

    let loader_position = r.u64()?;
    let loader_epoch = r.u64()?;
    let loader_seed = r.u64()?;

    let mut rng_s0 = [0u32; 16];
    for s in &mut rng_s0 {
        *s = r.u32()?;
    }
    let mut rng_s1 = [0u32; 16];
    for s in &mut rng_s1 {
        *s = r.u32()?;
    }
    let rng_idx = r.u64()? as usize;

    let model_bytes_len = r.u64()? as usize;
    let model_bytes = r.take(model_bytes_len)?;
    let model = parse(model_bytes, cfg, max_bytes)?;

    let num_m = r.u32()? as usize;
    let mut m = Vec::with_capacity(num_m);
    for _ in 0..num_m {
        let len = r.u64()? as usize;
        let mut vec = Vec::with_capacity(len);
        let raw = r.take(len.checked_mul(4).ok_or(CheckpointError::OutOfBounds)?)?;
        for chunk in raw.chunks_exact(4) {
            let val = f32::from_le_bytes(chunk.try_into().unwrap());
            if !val.is_finite() {
                return Err(CheckpointError::NonFinite("optimizer moment m".into()));
            }
            vec.push(val);
        }
        m.push(vec);
    }

    let num_v = r.u32()? as usize;
    let mut v_vecs = Vec::with_capacity(num_v);
    for _ in 0..num_v {
        let len = r.u64()? as usize;
        let mut vec = Vec::with_capacity(len);
        let raw = r.take(len.checked_mul(4).ok_or(CheckpointError::OutOfBounds)?)?;
        for chunk in raw.chunks_exact(4) {
            let val = f32::from_le_bytes(chunk.try_into().unwrap());
            if !val.is_finite() {
                return Err(CheckpointError::NonFinite("optimizer moment v".into()));
            }
            vec.push(val);
        }
        v_vecs.push(vec);
    }

    let optimizer = crate::optimizer::AdamW {
        m,
        v: v_vecs,
        step: opt_step,
        consecutive_skips,
    };

    let loader_state = crate::data::LoaderState {
        position: loader_position,
        epoch: loader_epoch,
        corpus_hash: *corpus_hash,
        shuffle_seed: loader_seed,
    };

    Ok(TrainingState {
        model,
        optimizer,
        loader_state,
        rng_state: (rng_s0, rng_s1, rng_idx),
        schedule_step,
    })
}

pub fn load_training(
    path: &Path,
    cfg: &Config,
    corpus_hash: &[u8; 32],
    max_bytes: u64,
) -> Result<TrainingState, CheckpointError> {
    let mut file = File::open(path).map_err(|e| CheckpointError::Io(e.to_string()))?;
    let mut bytes = Vec::new();
    (&mut file)
        .take(max_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|e| CheckpointError::Io(e.to_string()))?;
    if bytes.len() as u64 > max_bytes {
        return Err(CheckpointError::TooLarge { size: bytes.len() as u64, max: max_bytes });
    }
    parse_training(&bytes, cfg, corpus_hash, max_bytes)
}

pub fn save_training(state: &TrainingState, path: &Path) -> Result<(), CheckpointError> {
    let io = |e: std::io::Error| CheckpointError::Io(e.to_string());
    let tmp = unique_tmp_path(path);
    let bytes = serialize_training(state);
    let result = (|| {
        let mut f = File::create(&tmp).map_err(io)?;
        f.write_all(&bytes).map_err(io)?;
        f.flush().map_err(io)?;
        f.sync_all().map_err(io)?;
        drop(f);
        load_training(&tmp, &state.model.cfg, &state.loader_state.corpus_hash, DEFAULT_MAX_BYTES)?;
        fs::rename(&tmp, path).map_err(io)?;
        if let Some(parent) = path.parent() {
            if let Ok(dir) = File::open(parent) {
                let _ = dir.sync_all();
            }
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> Model {
        Model::init(&Config::reference_v2()).unwrap()
    }

    #[test]
    fn roundtrip_is_bitwise_identical_g10() {
        let m = model();
        let b = serialize(&m);
        let m2 = parse(&b, &m.cfg, DEFAULT_MAX_BYTES).unwrap();
        let toks = [256u32, 82, 117, 115, 116];
        assert_eq!(m.forward(&toks, None).unwrap(), m2.forward(&toks, None).unwrap());
    }

    #[test]
    fn corruption_is_rejected() {
        let m = model();
        let good = serialize(&m);
        let cfg = &m.cfg;
        let mut bad = good.clone();
        bad[0] ^= 1;
        assert_eq!(parse(&bad, cfg, DEFAULT_MAX_BYTES).unwrap_err(), CheckpointError::BadMagic);
        let mut bad = good.clone();
        let last = bad.len() - 1;
        bad[last] ^= 0xFF;
        assert!(matches!(parse(&bad, cfg, DEFAULT_MAX_BYTES), Err(CheckpointError::ChecksumMismatch(_))));
        assert!(parse(&good[..good.len() / 2], cfg, DEFAULT_MAX_BYTES).is_err());
        assert!(parse(&good[..10], cfg, DEFAULT_MAX_BYTES).is_err());
        assert!(matches!(parse(&good, cfg, 100), Err(CheckpointError::TooLarge { .. })));
        let mut other = cfg.clone();
        other.d_ff = 256;
        assert_eq!(parse(&good, &other, DEFAULT_MAX_BYTES).unwrap_err(), CheckpointError::ArchMismatch);
    }

    #[test]
    fn config_changes_rejected() {
        let m = model();
        let b = serialize(&m);
        let mut c_theta = m.cfg.clone();
        c_theta.rope_theta += 1.0;
        assert!(matches!(parse(&b, &c_theta, DEFAULT_MAX_BYTES), Err(CheckpointError::ConfigMismatch(_))));

        let mut c_eps = m.cfg.clone();
        c_eps.rms_eps *= 2.0;
        assert!(matches!(parse(&b, &c_eps, DEFAULT_MAX_BYTES), Err(CheckpointError::ConfigMismatch(_))));
    }

    #[test]
    fn training_state_roundtrip() {
        let m = model();
        let p = crate::autodiff::Params {
            shapes: vec![vec![2]],
            data: vec![vec![1.0, 2.0]],
        };
        let opt = crate::optimizer::AdamW::new(&p);
        let corpus_hash = [42u8; 32];
        let loader_state = crate::data::LoaderState {
            position: 10,
            epoch: 1,
            corpus_hash,
            shuffle_seed: 1234,
        };
        let rng = crate::rng::Rng::new(999);
        let rng_state = rng.dump_state();
        let state = TrainingState {
            model: m,
            optimizer: opt,
            loader_state,
            rng_state,
            schedule_step: 5,
        };

        let bytes = serialize_training(&state);
        let loaded = parse_training(&bytes, &state.model.cfg, &corpus_hash, DEFAULT_MAX_BYTES).unwrap();
        assert_eq!(state.schedule_step, loaded.schedule_step);
        assert_eq!(state.optimizer.step, loaded.optimizer.step);
        assert_eq!(state.loader_state.position, loaded.loader_state.position);
        assert_eq!(state.rng_state, loaded.rng_state);

        // Mismatched corpus rejected
        let bad_corpus = [99u8; 32];
        assert_eq!(
            parse_training(&bytes, &state.model.cfg, &bad_corpus, DEFAULT_MAX_BYTES).unwrap_err(),
            CheckpointError::CorpusMismatch
        );
    }

    #[test]
    fn non_finite_weights_rejected() {
        let mut m = model();
        m.embed.data_mut()[3] = f32::NAN;
        let b = serialize(&m);
        assert!(matches!(parse(&b, &m.cfg, DEFAULT_MAX_BYTES), Err(CheckpointError::NonFinite(_))));
    }

    #[test]
    fn save_and_load_file() {
        let m = model();
        let p = std::env::temp_dir().join("sandy_rust_test.ckpt");
        save(&m, &p).unwrap();
        assert!(!p.with_extension("ckpt.tmp").exists());
        let m2 = load(&p, &m.cfg, DEFAULT_MAX_BYTES).unwrap();
        assert_eq!(m.embed, m2.embed);
        let _ = fs::remove_file(&p);
    }
}
