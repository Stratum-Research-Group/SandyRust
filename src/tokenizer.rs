//! Byte-level tokenizer (V=259): ids 0..=255 are bytes, 256=BOS, 257=EOS, 258=PAD (§5.1).

pub const BOS: u32 = 256;
pub const EOS: u32 = 257;
pub const PAD: u32 = 258;
pub const VOCAB: usize = 259;

#[derive(Debug, Clone, PartialEq)]
pub enum TokenizerError {
    /// A token id outside the vocabulary was supplied to `decode`.
    InvalidId(u32),
}

#[derive(Debug, Clone, Default)]
pub struct ByteTokenizer;

impl ByteTokenizer {
    pub fn new() -> Self {
        ByteTokenizer
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
}
