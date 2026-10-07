//! Byte-level tokenizer (V=259): ids 0..=255 are bytes, 256=BOS, 257=EOS, 258=PAD (§5.1).

pub const BOS: u32 = 256;
pub const EOS: u32 = 257;
pub const PAD: u32 = 258;
pub const VOCAB: usize = 259;

#[derive(Debug, Clone, PartialEq)]
pub enum TokenizerError {
    /// A token id outside the vocabulary was supplied to `decode`.
    InvalidId(u32),
    UnsupportedConfig,
}

#[derive(Debug, Clone, Default)]
pub struct ByteTokenizer;

impl ByteTokenizer {
    pub fn new() -> Self {
        ByteTokenizer
    }

    pub fn validate_config(cfg: &crate::config::Config) -> Result<(), TokenizerError> {
        if cfg.vocab_size != 259 || cfg.bos_id != 256 || cfg.eos_id != 257 || cfg.pad_id != 258 {
            return Err(TokenizerError::UnsupportedConfig);
        }
        Ok(())
    }

    /// Raw encoding never emits special tokens.
    pub fn encode_bytes(&self, bytes: &[u8]) -> Vec<u32> {
        bytes.iter().map(|&b| u32::from(b)).collect()
    }

    pub fn encode(&self, s: &str) -> Vec<u32> {
        self.encode_bytes(s.as_bytes())
    }

    /// Single-document training window: BOS + bytes + EOS (§5.1 boundary policy).
    pub fn encode_document(&self, s: &str) -> Vec<u32> {
        let mut v = Vec::with_capacity(s.len() + 2);
        v.push(BOS);
        v.extend(s.bytes().map(u32::from));
        v.push(EOS);
        v
    }

    /// Decode to raw bytes; special tokens are skipped, ids >= V are rejected.
    pub fn decode_bytes(&self, ids: &[u32]) -> Result<Vec<u8>, TokenizerError> {
        let mut out = Vec::with_capacity(ids.len());
        for &id in ids {
            match id {
                0..=255 => out.push(id as u8),
                BOS | EOS | PAD => {}
                other => return Err(TokenizerError::InvalidId(other)),
            }
        }
        Ok(out)
    }

    /// Lossy UTF-8 decode (invalid sequences become U+FFFD).
    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        Ok(String::from_utf8_lossy(&self.decode_bytes(ids)?).into_owned())
    }
}

/// Incremental UTF-8 byte stream buffer (§5.1, Item 17).
#[derive(Debug, Clone, Default)]
pub struct Utf8ByteStreamBuffer {
    pending: Vec<u8>,
}

impl Utf8ByteStreamBuffer {
    pub fn new() -> Self {
        Self { pending: Vec::new() }
    }

    pub fn push_byte(&mut self, b: u8) -> String {
        self.push_bytes(&[b])
    }

    pub fn push_bytes(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        let mut consumed = 0;

        while consumed < self.pending.len() {
            let slice = &self.pending[consumed..];
            match std::str::from_utf8(slice) {
                Ok(s) => {
                    out.push_str(s);
                    consumed = self.pending.len();
                    break;
                }
                Err(e) => {
                    let valid_len = e.valid_up_to();
                    if valid_len > 0 {
                        let valid_str = std::str::from_utf8(&slice[..valid_len]).unwrap();
                        out.push_str(valid_str);
                        consumed += valid_len;
                    }
                    if let Some(error_len) = e.error_len() {
                        out.push('\u{FFFD}');
                        consumed += error_len;
                    } else {
                        break;
                    }
                }
            }
        }
        self.pending.drain(..consumed);
        out
    }

    pub fn flush(&mut self) -> String {
        let mut out = String::new();
        while !self.pending.is_empty() {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    out.push_str(s);
                    self.pending.clear();
                    break;
                }
                Err(e) => {
                    let valid_len = e.valid_up_to();
                    if valid_len > 0 {
                        out.push_str(std::str::from_utf8(&self.pending[..valid_len]).unwrap());
                        self.pending.drain(..valid_len);
                        continue;
                    }
                    out.push('\u{FFFD}');
                    let step = e.error_len().unwrap_or(1);
                    self.pending.drain(..step);
                }
            }
        }
        out
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_256_bytes_roundtrip_g11() {
        let t = ByteTokenizer::new();
        let all: Vec<u8> = (0..=255u8).collect();
        let ids = t.encode_bytes(&all);
        assert_eq!(ids.len(), 256);
        assert_eq!(t.decode_bytes(&ids).unwrap(), all);
    }

    #[test]
    fn utf8_roundtrip_and_specials() {
        let t = ByteTokenizer::new();
        let s = "héllo → 世界 🦀";
        assert_eq!(t.decode(&t.encode(s)).unwrap(), s);
        assert_eq!(t.decode(&t.encode_document(s)).unwrap(), s);
        assert!(t.encode(s).iter().all(|&i| i < 256));
        assert_eq!(t.decode_bytes(&[259]), Err(TokenizerError::InvalidId(259)));
    }

    #[test]
    fn invalid_utf8_renders_replacement() {
        let t = ByteTokenizer::new();
        assert_eq!(t.decode(&[0xFF]).unwrap(), "\u{FFFD}");
    }

    #[test]
    fn utf8_byte_stream_buffer_incremental() {
        let mut buf = Utf8ByteStreamBuffer::new();
        let crab = "🦀".as_bytes(); // 4 bytes: [240, 159, 166, 128]
        assert_eq!(buf.push_byte(crab[0]), "");
        assert_eq!(buf.push_byte(crab[1]), "");
        assert_eq!(buf.push_byte(crab[2]), "");
        assert_eq!(buf.push_byte(crab[3]), "🦀");
        assert!(buf.is_empty());

        // Incomplete flush produces replacement char
        assert_eq!(buf.push_byte(crab[0]), "");
        assert_eq!(buf.flush(), "\u{FFFD}");
    }
}
