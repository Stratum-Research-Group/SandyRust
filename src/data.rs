//! Deterministic window loader over in-memory documents and safe binary `.rtok` reader (§5.1–5.3).
//!
//! Each document becomes `BOS + bytes + EOS`, cut into `T+1`-token windows at stride `T`
//! (targets continuous, windows never cross documents), right-padded with PAD.
//! Window order is a seeded Fisher-Yates shuffle per epoch, so a saved
//! `{corpus_hash, epoch, position, shuffle_seed}` resumes to the identical next window.

use crate::autodiff::TrainError;
use crate::rng::Rng;
use crate::sha256::sha256;
use crate::tokenizer::{BOS, EOS, PAD};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::Path;

#[derive(Debug)]
pub enum TokenReaderError {
    Io(std::io::Error),
    InvalidMagic,
    UnsupportedVersion(u16),
    InvalidTokenWidth(u16),
    NonZeroReservedHeader,
    ExceedsMaxBytes { max_bytes: u64, actual_bytes: u64 },
    LengthMismatch { expected_bytes: u64, actual_bytes: u64 },
    UnsupportedExtentOnTarget,
    InvalidVocabularyBound { bound: usize },
    TokenOutOfVocab { index: usize, token_id: u32, vocab_size: usize },
    ReaderPoisoned,
    SeekFailed(std::io::Error),
    ArithmeticOverflow,
}

impl std::fmt::Display for TokenReaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for TokenReaderError {}

/// Safe default buffered `.rtok` shard reader (specv3 §5.3).
pub struct BufferedTokenShard {
    reader: BufReader<File>,
    token_count: usize,
    token_width: usize,
    vocab_size: usize,
    current_index: usize,
    is_poisoned: bool,
}

impl BufferedTokenShard {
    pub const HEADER_SIZE: u64 = 64;

    pub fn open(path: &Path, model_vocab: usize, max_bytes: u64) -> Result<Self, TokenReaderError> {
        let file = File::open(path).map_err(TokenReaderError::Io)?;
        let file_len = file.metadata().map_err(TokenReaderError::Io)?.len();

        if file_len > max_bytes {
            return Err(TokenReaderError::ExceedsMaxBytes {
                max_bytes,
                actual_bytes: file_len,
            });
        }

        let mut reader = BufReader::new(file);
        let mut header = [0u8; 64];
        reader.read_exact(&mut header).map_err(TokenReaderError::Io)?;

        if &header[0..4] != b"RTOK" {
            return Err(TokenReaderError::InvalidMagic);
        }
        let version = u16::from_le_bytes([header[4], header[5]]);
        if version != 1 {
            return Err(TokenReaderError::UnsupportedVersion(version));
        }
        let width = u16::from_le_bytes([header[6], header[7]]);
        if width != 2 && width != 4 {
            return Err(TokenReaderError::InvalidTokenWidth(width));
        }
        let count_u64 = u64::from_le_bytes(header[8..16].try_into().unwrap());
        let vocab_u64 = u64::from_le_bytes(header[16..24].try_into().unwrap());

        if header[24..64].iter().any(|&b| b != 0) {
            return Err(TokenReaderError::NonZeroReservedHeader);
        }

        let count = usize::try_from(count_u64).map_err(|_| TokenReaderError::UnsupportedExtentOnTarget)?;
        let vocab = usize::try_from(vocab_u64).map_err(|_| TokenReaderError::UnsupportedExtentOnTarget)?;

        if vocab == 0 || vocab > model_vocab {
            return Err(TokenReaderError::InvalidVocabularyBound { bound: vocab });
        }

        let payload_bytes = count_u64
            .checked_mul(width as u64)
            .ok_or(TokenReaderError::ArithmeticOverflow)?;
        let expected_total = Self::HEADER_SIZE
            .checked_add(payload_bytes)
            .ok_or(TokenReaderError::ArithmeticOverflow)?;

        if file_len != expected_total {
            return Err(TokenReaderError::LengthMismatch {
                expected_bytes: expected_total,
                actual_bytes: file_len,
            });
        }

        Ok(Self {
            reader,
            token_count: count,
            token_width: width as usize,
            vocab_size: vocab,
            current_index: 0,
            is_poisoned: false,
        })
    }

    pub fn token_count(&self) -> usize {
        self.token_count
    }

    pub fn current_index(&self) -> usize {
        self.current_index
    }

    pub fn next_token(&mut self) -> Result<Option<u32>, TokenReaderError> {
        if self.is_poisoned {
            return Err(TokenReaderError::ReaderPoisoned);
        }
        if self.current_index >= self.token_count {
            return Ok(None);
        }

        let token_id = if self.token_width == 2 {
            let mut buf = [0u8; 2];
            if let Err(e) = self.reader.read_exact(&mut buf) {
                self.is_poisoned = true;
                self.current_index = usize::MAX;
                return Err(TokenReaderError::Io(e));
            }
            u16::from_le_bytes(buf) as u32
        } else {
            let mut buf = [0u8; 4];
            if let Err(e) = self.reader.read_exact(&mut buf) {
                self.is_poisoned = true;
                self.current_index = usize::MAX;
                return Err(TokenReaderError::Io(e));
            }
            u32::from_le_bytes(buf)
        };

        if token_id as usize >= self.vocab_size {
            let idx = self.current_index;
            self.is_poisoned = true;
            self.current_index = usize::MAX;
            return Err(TokenReaderError::TokenOutOfVocab {
                index: idx,
                token_id,
                vocab_size: self.vocab_size,
            });
        }

        self.current_index += 1;
        Ok(Some(token_id))
    }

    pub fn seek_to_token(&mut self, token_index: usize) -> Result<(), TokenReaderError> {
        if token_index > self.token_count {
            return Err(TokenReaderError::ArithmeticOverflow);
        }
        let byte_offset = (token_index as u64)
            .checked_mul(self.token_width as u64)
            .and_then(|offset| Self::HEADER_SIZE.checked_add(offset))
            .ok_or(TokenReaderError::ArithmeticOverflow)?;

        self.reader
            .seek(SeekFrom::Start(byte_offset))
            .map_err(TokenReaderError::SeekFailed)?;
        self.current_index = token_index;
        self.is_poisoned = false;
        Ok(())
    }
}

pub fn write_rtok_file(path: &Path, tokens: &[u32], vocab_size: usize, width: u16) -> Result<(), TokenReaderError> {
    if width != 2 && width != 4 {
        return Err(TokenReaderError::InvalidTokenWidth(width));
    }
    let mut f = File::create(path).map_err(TokenReaderError::Io)?;
    let mut header = [0u8; 64];
    header[0..4].copy_from_slice(b"RTOK");
    header[4..6].copy_from_slice(&1u16.to_le_bytes());
    header[6..8].copy_from_slice(&width.to_le_bytes());
    header[8..16].copy_from_slice(&(tokens.len() as u64).to_le_bytes());
    header[16..24].copy_from_slice(&(vocab_size as u64).to_le_bytes());
    f.write_all(&header).map_err(TokenReaderError::Io)?;

    let mut payload = Vec::new();
    for &tok in tokens {
        if width == 2 {
            payload.extend_from_slice(&(tok as u16).to_le_bytes());
        } else {
            payload.extend_from_slice(&tok.to_le_bytes());
        }
    }
    f.write_all(&payload).map_err(TokenReaderError::Io)?;
    f.flush().map_err(TokenReaderError::Io)?;
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoaderState {
    pub corpus_hash: [u8; 32],
    pub epoch: u64,
    pub position: u64,
    pub shuffle_seed: u64,
}

#[derive(Debug, Clone)]
pub struct WindowLoader {
    t: usize,
    windows: Vec<Vec<u32>>,
    order: Vec<usize>,
    state: LoaderState,
    corpus_hash: [u8; 32],
}

fn compute_corpus_hash(docs: &[Vec<u32>], t: usize, bos: u32, eos: u32, pad: u32) -> [u8; 32] {
    let mut b = Vec::new();
    b.extend_from_slice(&(t as u64).to_le_bytes());
    b.extend_from_slice(&bos.to_le_bytes());
    b.extend_from_slice(&eos.to_le_bytes());
    b.extend_from_slice(&pad.to_le_bytes());
    for doc in docs {
        b.extend_from_slice(&(doc.len() as u64).to_le_bytes());
        for &tok in doc {
            b.extend_from_slice(&tok.to_le_bytes());
        }
    }
    sha256(&b)
}

fn shuffled(len: usize, seed: u64, epoch: u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..len).collect();
    let mut rng = Rng::new(seed ^ epoch.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    for i in (1..len).rev() {
        let j = (rng.next_u32() as usize) % (i + 1);
        order.swap(i, j);
    }
    order
}

impl WindowLoader {
    /// `docs` hold raw byte token ids (`< 256`); boundary tokens BOS/EOS are added here.
    pub fn new(docs: &[Vec<u32>], t: usize, seed: u64) -> Result<Self, TrainError> {
        if t == 0 {
            return Err(TrainError::InvalidArgument("window length must be >= 1"));
        }
        for doc in docs {
            for &tok in doc {
                if tok >= 256 {
                    return Err(TrainError::InvalidArgument("document tokens must be raw byte values < 256"));
                }
            }
        }
        let chash = compute_corpus_hash(docs, t, BOS, EOS, PAD);
        let mut windows = Vec::new();
        for doc in docs {
            let mut tok = Vec::with_capacity(doc.len().checked_add(2).ok_or(TrainError::Data("doc length overflow"))?);
            tok.push(BOS);
            tok.extend_from_slice(doc);
            tok.push(EOS);
            let n = tok.len();
            let mut s = 0;
            while s + 1 < n {
                let end = (s + t + 1).min(n);
                let mut w: Vec<u32> = tok[s..end].to_vec();
                w.resize(t + 1, PAD);
                windows.push(w);
                s += t;
            }
        }
        if windows.is_empty() {
            return Err(TrainError::Data("no windows"));
        }
        let order = shuffled(windows.len(), seed, 0);
        Ok(WindowLoader {
            t,
            windows,
            order,
            state: LoaderState {
                corpus_hash: chash,
                epoch: 0,
                position: 0,
                shuffle_seed: seed,
            },
            corpus_hash: chash,
        })
    }

    pub fn t(&self) -> usize {
        self.t
    }

    pub fn len(&self) -> usize {
        self.windows.len()
    }
    pub fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }
    pub fn windows(&self) -> &[Vec<u32>] {
        &self.windows
    }

    pub fn state(&self) -> LoaderState {
        self.state
    }
    pub fn corpus_hash(&self) -> [u8; 32] {
        self.corpus_hash
    }

    pub fn restore(&mut self, s: LoaderState) -> Result<(), TrainError> {
        if s.corpus_hash != self.corpus_hash {
            return Err(TrainError::Data("corpus hash mismatch upon restore"));
        }
        if s.position > self.windows.len() as u64 {
            return Err(TrainError::Data("loader position beyond epoch"));
        }
        self.order = shuffled(self.windows.len(), s.shuffle_seed, s.epoch);
        self.state = s;
        Ok(())
    }

    /// Next window (cursor advances even if the caller later skips the step).
    pub fn next_window(&mut self) -> Vec<u32> {
        if self.state.position as usize >= self.windows.len() {
            self.state.epoch += 1;
            self.state.position = 0;
            self.order = shuffled(self.windows.len(), self.state.shuffle_seed, self.state.epoch);
        }
        let w = self.windows[self.order[self.state.position as usize]].clone();
        self.state.position += 1;
        w
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn docs() -> Vec<Vec<u32>> {
        vec![(0..20).collect(), (30..34).collect(), vec![]]
    }

    #[test]
    fn windows_respect_boundaries_and_padding() {
        let l = WindowLoader::new(&docs(), 8, 1).unwrap();
        // doc0: 22 tokens -> starts 0,8,16 (3); doc1: 6 tokens -> 1; doc2: [BOS,EOS] -> 1
        assert_eq!(l.len(), 5);
        assert!(l.windows.iter().all(|w| w.len() == 9));
        assert!(l.windows.iter().any(|w| w[0] == BOS && w[1] == 30));
        assert!(l.windows.iter().all(|w| w.iter().filter(|&&t| t == BOS).count() <= 1));
    }

    #[test]
    fn restore_resumes_identical_stream_across_epochs() {
        let mut a = WindowLoader::new(&docs(), 8, 42).unwrap();
        for _ in 0..7 {
            a.next_window();
        }
        let st = a.state();
        let tail_a: Vec<_> = (0..9).map(|_| a.next_window()).collect();
        let mut b = WindowLoader::new(&docs(), 8, 42).unwrap();
        b.restore(st).unwrap();
        let tail_b: Vec<_> = (0..9).map(|_| b.next_window()).collect();
        assert_eq!(tail_a, tail_b);
        assert!(a.state().epoch >= 2);
        assert!(b.restore(LoaderState { corpus_hash: a.corpus_hash(), epoch: 0, position: 999, shuffle_seed: 42 }).is_err());
        // Different corpus rejected
        assert!(b.restore(LoaderState { corpus_hash: [0u8; 32], epoch: 0, position: 0, shuffle_seed: 42 }).is_err());

        // N3 regression: restoring state from t=8 into loader with t=16 over identical docs must be rejected
        let mut loader_t16 = WindowLoader::new(&docs(), 16, 42).unwrap();
        assert!(loader_t16.restore(st).is_err(), "Restoring state from different window length must fail");
    }

    #[test]
    fn rtok_shard_roundtrip_and_poisoning() {
        let p = std::env::temp_dir().join("test_shard.rtok");
        let toks = vec![10u32, 20, 30, 40, 50];
        write_rtok_file(&p, &toks, 259, 2).unwrap();
        let mut shard = BufferedTokenShard::open(&p, 259, 1024 * 1024).unwrap();
        assert_eq!(shard.token_count(), 5);
        assert_eq!(shard.next_token().unwrap(), Some(10));
        assert_eq!(shard.next_token().unwrap(), Some(20));
        shard.seek_to_token(4).unwrap();
        assert_eq!(shard.next_token().unwrap(), Some(50));
        assert_eq!(shard.next_token().unwrap(), None);
        let _ = std::fs::remove_file(&p);
    }
}
