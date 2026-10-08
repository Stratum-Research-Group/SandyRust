//! HuggingFace Byte-Level BPE Tokenizer for Extension X17 (specv3 §3.6, §13.2, Phase I4).
//!
//! Provides a pure-Rust, standard-library-only BPE tokenizer capable of parsing
//! HuggingFace `tokenizer.json` files, implementing GPT-2 and LLaMA-3 style
//! byte-to-unicode bijective mapping, hand-written regex-free pre-tokenization,
//! ranked pair merges, special token handling, and round-trip decode.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use super::ImportError;
use crate::sha256::{hex, sha256};

pub const MAX_BPE_VOCAB_CAP: usize = 262_144;
pub const MAX_BPE_MERGES_CAP: usize = 1_000_000;
pub const MAX_TOKEN_STRING_LEN: usize = 256;

/// Supported pre-tokenizer kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreTokenizerKind {
    Gpt2,
    Llama3,
    ByteLevelNoRegex,
}

/// Compute the bijective GPT-2 / HuggingFace Byte-to-Unicode mapping table.
///
/// In ByteLevel BPE, all 256 raw bytes are mapped to unique printable Unicode characters:
/// - 188 direct bytes: 33..=126 ('!'..='~'), 161..=172 ('¡'..='¬'), 174..=255 ('®'..='ÿ')
/// - 68 shifted bytes: control characters, space, DEL, and non-printables are mapped
///   sequentially to U+0100..=U+0143 (256..=323).
///
/// In particular, space (32) maps to U+0120 ('Ġ').
pub fn bytes_to_unicode() -> ([char; 256], HashMap<char, u8>) {
    let mut bs = Vec::with_capacity(256);
    let mut cs = Vec::with_capacity(256);

    for b in 33u8..=126 {
        bs.push(b);
        cs.push(b as u32);
    }
    for b in 161u8..=172 {
        bs.push(b);
        cs.push(b as u32);
    }
    for b in 174u8..=255 {
        bs.push(b);
        cs.push(b as u32);
    }

    let mut n = 0u32;
    for b in 0u8..=255 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }

    let mut byte_to_char = ['\0'; 256];
    let mut char_to_byte = HashMap::with_capacity(256);

    for (b, c) in bs.into_iter().zip(cs) {
        let ch = char::from_u32(c).expect("valid unicode scalar");
        byte_to_char[b as usize] = ch;
        char_to_byte.insert(ch, b);
    }

    (byte_to_char, char_to_byte)
}

/// Hand-written GPT-2 pre-tokenizer scanner without the regex crate.
///
/// Splits text into word chunks matching:
/// `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
pub fn split_gpt2(s: &str) -> Vec<&str> {
    let mut chunks = Vec::new();
    let mut pos = 0;

    while pos < s.len() {
        let rem = &s[pos..];

        // 1. Contractions: 's|'t|'re|'ve|'m|'ll|'d
        let contraction_len = match_gpt2_contraction(rem);
        if contraction_len > 0 {
            chunks.push(&s[pos..pos + contraction_len]);
            pos += contraction_len;
            continue;
        }

        // 2. Letters: ?\p{L}+
        if let Some(len) = match_opt_space_then(rem, |c| c.is_alphabetic()) {
            chunks.push(&s[pos..pos + len]);
            pos += len;
            continue;
        }

        // 3. Numbers: ?\p{N}+
        if let Some(len) = match_opt_space_then(rem, |c| c.is_numeric()) {
            chunks.push(&s[pos..pos + len]);
            pos += len;
            continue;
        }

        // 4. Punctuation/Symbols: ?[^\s\p{L}\p{N}]+
        if let Some(len) = match_opt_space_then(rem, |c| !c.is_whitespace() && !c.is_alphabetic() && !c.is_numeric()) {
            chunks.push(&s[pos..pos + len]);
            pos += len;
            continue;
        }

        // 5. Whitespace: \s+(?!\S)|\s+
        let mut ws_len = 0;
        for c in rem.chars() {
            if c.is_whitespace() {
                ws_len += c.len_utf8();
            } else {
                break;
            }
        }
        if ws_len > 0 {
            chunks.push(&s[pos..pos + ws_len]);
            pos += ws_len;
            continue;
        }

        // Fallback: advance one unicode scalar
        let next_char_len = rem.chars().next().map_or(1, |c| c.len_utf8());
        chunks.push(&s[pos..pos + next_char_len]);
        pos += next_char_len;
    }

    chunks
}

fn match_gpt2_contraction(rem: &str) -> usize {
    const CONTRACTIONS: &[&str] = &["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];
    for &pat in CONTRACTIONS {
        if rem.starts_with(pat) {
            return pat.len();
        }
    }
    0
}

fn match_opt_space_then<F>(rem: &str, pred: F) -> Option<usize>
where
    F: Fn(char) -> bool,
{
    if rem.is_empty() {
        return None;
    }

    if let Some(after_space) = rem.strip_prefix(' ') {
        let first_char = after_space.chars().next()?;
        if pred(first_char) {
            let mut matched_len = 1;
            for c in after_space.chars() {
                if pred(c) {
                    matched_len += c.len_utf8();
                } else {
                    break;
                }
            }
            Some(matched_len)
        } else {
            None
        }
    } else {
        let first_char = rem.chars().next()?;
        if pred(first_char) {
            let mut matched_len = 0;
            for c in rem.chars() {
                if pred(c) {
                    matched_len += c.len_utf8();
                } else {
                    break;
                }
            }
            Some(matched_len)
        } else {
            None
        }
    }
}

/// Hand-written LLaMA-3 pre-tokenizer scanner without the regex crate.
///
/// Implements the LLaMA-3 splitting rule:
/// - Case-insensitive contractions
/// - Letter sequences with optional single non-newline-non-letter-non-digit prefix and optional trailing contraction
/// - Numbers chunked into 1 to 3 digits (\p{N}{1,3})
/// - Punctuation/symbols with optional space and trailing newlines
/// - Whitespace / newline sequences
pub fn split_llama3(s: &str) -> Vec<&str> {
    let mut chunks = Vec::new();
    let mut pos = 0;

    while pos < s.len() {
        let rem = &s[pos..];

        // 1. Contractions: (?i:'s|'t|'re|'ve|'m|'ll|'d)
        let cont_len = match_case_insensitive_contraction(rem);
        if cont_len > 0 {
            chunks.push(&s[pos..pos + cont_len]);
            pos += cont_len;
            continue;
        }

        // 2. Letters with optional prefix and optional trailing contraction
        if let Some(len) = match_llama3_letters(rem) {
            chunks.push(&s[pos..pos + len]);
            pos += len;
            continue;
        }

        // 3. Numbers: \p{N}{1,3}
        if let Some(len) = match_llama3_digits(rem) {
            chunks.push(&s[pos..pos + len]);
            pos += len;
            continue;
        }

        // 4. Punctuation with optional space and optional newlines: ?[^\s\p{L}\p{N}]+[\r\n]*
        if let Some(len) = match_llama3_punct(rem) {
            chunks.push(&s[pos..pos + len]);
            pos += len;
            continue;
        }

        // 5. Newline runs: \s*[\r\n]+
        if let Some(len) = match_newline_run(rem) {
            chunks.push(&s[pos..pos + len]);
            pos += len;
            continue;
        }

        // 6. Whitespace: \s+(?!\S)|\s+
        let mut ws_len = 0;
        for c in rem.chars() {
            if c.is_whitespace() {
                ws_len += c.len_utf8();
            } else {
                break;
            }
        }
        if ws_len > 0 {
            chunks.push(&s[pos..pos + ws_len]);
            pos += ws_len;
            continue;
        }

        let next_char_len = rem.chars().next().map_or(1, |c| c.len_utf8());
        chunks.push(&s[pos..pos + next_char_len]);
        pos += next_char_len;
    }

    chunks
}

fn match_case_insensitive_contraction(rem: &str) -> usize {
    if !rem.starts_with('\'') && !rem.starts_with('’') {
        return 0;
    }
    let quote_len = if rem.starts_with('\'') { 1 } else { '’'.len_utf8() };
    let after_quote = &rem[quote_len..];

    let mut rest_chars = after_quote.chars();
    let c1 = match rest_chars.next() {
        Some(c) => c.to_ascii_lowercase(),
        None => return 0,
    };

    match c1 {
        's' | 't' | 'm' | 'd' => quote_len + 1,
        'r' if rest_chars.next().map(|c| c.to_ascii_lowercase()) == Some('e') => quote_len + 2,
        'v' if rest_chars.next().map(|c| c.to_ascii_lowercase()) == Some('e') => quote_len + 2,
        'l' if rest_chars.next().map(|c| c.to_ascii_lowercase()) == Some('l') => quote_len + 2,
        _ => 0,
    }
}

fn match_llama3_letters(rem: &str) -> Option<usize> {
    let mut total_len = 0;
    let mut chars = rem.chars();

    let first = chars.next()?;
    let (has_prefix, letter_start_char) = if first != '\r' && first != '\n' && !first.is_alphabetic() && !first.is_numeric() {
        let second = chars.next()?;
        if second.is_alphabetic() {
            total_len += first.len_utf8();
            (true, second)
        } else {
            return None;
        }
    } else if first.is_alphabetic() {
        (false, first)
    } else {
        return None;
    };

    total_len += letter_start_char.len_utf8();

    for c in chars {
        if c.is_alphabetic() {
            total_len += c.len_utf8();
        } else {
            break;
        }
    }

    // Optional trailing contraction
    let tail = &rem[total_len..];
    let cont_len = match_case_insensitive_contraction(tail);
    if cont_len > 0 {
        total_len += cont_len;
    }

    if has_prefix && total_len == first.len_utf8() {
        None
    } else {
        Some(total_len)
    }
}

fn match_llama3_digits(rem: &str) -> Option<usize> {
    let mut len = 0;
    let mut count = 0;
    for c in rem.chars() {
        if c.is_numeric() && count < 3 {
            len += c.len_utf8();
            count += 1;
        } else {
            break;
        }
    }
    if count > 0 {
        Some(len)
    } else {
        None
    }
}

fn match_llama3_punct(rem: &str) -> Option<usize> {
    let has_space = rem.starts_with(' ');
    let target = if has_space { &rem[1..] } else { rem };

    let mut punct_len = 0;
    let mut iter = target.chars();
    let first = iter.next()?;
    if first.is_whitespace() || first.is_alphabetic() || first.is_numeric() {
        return None;
    }
    punct_len += first.len_utf8();

    for c in iter {
        if !c.is_whitespace() && !c.is_alphabetic() && !c.is_numeric() {
            punct_len += c.len_utf8();
        } else {
            break;
        }
    }

    let mut total_len = if has_space { 1 + punct_len } else { punct_len };

    // Optional trailing [\r\n]*
    for c in rem[total_len..].chars() {
        if c == '\r' || c == '\n' {
            total_len += c.len_utf8();
        } else {
            break;
        }
    }

    Some(total_len)
}

fn match_newline_run(rem: &str) -> Option<usize> {
    let mut has_nl = false;
    let mut len = 0;
    for c in rem.chars() {
        if c == '\r' || c == '\n' {
            has_nl = true;
            len += c.len_utf8();
        } else if c.is_whitespace() && !has_nl {
            len += c.len_utf8();
        } else {
            break;
        }
    }
    if has_nl {
        Some(len)
    } else {
        None
    }
}

/// HuggingFace Byte-Level BPE Tokenizer.
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    vocab: HashMap<String, u32>,
    inv_vocab: Vec<Option<String>>,
    merges: HashMap<String, usize>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    added_tokens: HashMap<String, u32>,
    inv_added_tokens: HashMap<u32, String>,
    special_tokens: Vec<(String, u32)>,
    pre_tokenizer_kind: PreTokenizerKind,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
    pad_id: Option<u32>,
}

impl BpeTokenizer {
    /// Load and construct a BPE tokenizer from raw `tokenizer.json` UTF-8 bytes.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, ImportError> {
        let (byte_to_char, char_to_byte) = bytes_to_unicode();

        let mut parser = TokenizerJsonReader::new(bytes);
        let parsed = parser.parse()?;

        // Check model type
        if parsed.model_type != "BPE" {
            return Err(ImportError::UnsupportedTokenizer(format!(
                "model.type must be 'BPE', got '{}'",
                parsed.model_type
            )));
        }

        // Check normalizer
        if parsed.has_non_null_normalizer {
            return Err(ImportError::UnsupportedTokenizer(
                "non-null normalizer is not supported".to_string(),
            ));
        }

        // Check caps
        if parsed.vocab.len() > MAX_BPE_VOCAB_CAP {
            return Err(ImportError::TooLarge {
                what: "vocab entries",
                size: parsed.vocab.len(),
                cap: MAX_BPE_VOCAB_CAP,
            });
        }
        if parsed.merges.len() > MAX_BPE_MERGES_CAP {
            return Err(ImportError::TooLarge {
                what: "merges",
                size: parsed.merges.len(),
                cap: MAX_BPE_MERGES_CAP,
            });
        }

        let mut max_id = 0u32;
        for &(id, _) in &parsed.inv_vocab {
            if id > max_id {
                max_id = id;
            }
        }
        for &id in parsed.added_tokens.values() {
            if id > max_id {
                max_id = id;
            }
        }

        let mut inv_vocab_vec = vec![None; (max_id + 1) as usize];
        for (id, token_str) in parsed.inv_vocab {
            inv_vocab_vec[id as usize] = Some(token_str);
        }

        let mut special_tokens = Vec::new();
        let mut inv_added_tokens = HashMap::with_capacity(parsed.added_tokens.len());
        for (content, &id) in &parsed.added_tokens {
            inv_added_tokens.insert(id, content.clone());
            if parsed.special_ids.contains(&id) {
                special_tokens.push((content.clone(), id));
            }
        }
        // Sort special tokens by length descending so longer tokens match first
        special_tokens.sort_by_key(|a| std::cmp::Reverse(a.0.len()));

        let mut bos_id = None;
        let mut eos_id = None;
        let mut pad_id = None;

        for (content, id) in &special_tokens {
            if content == "<|begin_of_text|>" || content == "<s>" || content == "<bos>" {
                bos_id = Some(*id);
            } else if content == "<|end_of_text|>" || content == "<|endoftext|>" || content == "</s>" || content == "<eos>" {
                eos_id = Some(*id);
            } else if content == "<pad>" || content == "<|pad|>" {
                pad_id = Some(*id);
            }
        }

        Ok(BpeTokenizer {
            vocab: parsed.vocab,
            inv_vocab: inv_vocab_vec,
            merges: parsed.merges,
            byte_to_char,
            char_to_byte,
            added_tokens: parsed.added_tokens,
            inv_added_tokens,
            special_tokens,
            pre_tokenizer_kind: parsed.pre_tokenizer_kind,
            bos_id,
            eos_id,
            pad_id,
        })
    }

    /// Load from a filesystem path.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, ImportError> {
        let bytes = fs::read(path.as_ref()).map_err(|e| ImportError::Io(e.to_string()))?;
        Self::from_json_bytes(&bytes)
    }

    /// Compute cryptographic fingerprint for the tokenizer files (§B4.3).
    ///
    /// Formatted as: `sha256("import-tok-v1:" + hex(sha256(tokenizer.json)) + hex(sha256(tokenizer_config.json)))`.
    pub fn fingerprint(tokenizer_json_bytes: &[u8], tokenizer_config_bytes: Option<&[u8]>) -> [u8; 32] {
        let tok_sha = sha256(tokenizer_json_bytes);
        let tok_cfg_sha = sha256(tokenizer_config_bytes.unwrap_or(b""));
        let combined = format!("import-tok-v1:{}{}", hex(&tok_sha), hex(&tok_cfg_sha));
        sha256(combined.as_bytes())
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len() + self.added_tokens.len()
    }

    pub fn bos_id(&self) -> Option<u32> {
        self.bos_id
    }

    pub fn eos_id(&self) -> Option<u32> {
        self.eos_id
    }

    pub fn pad_id(&self) -> Option<u32> {
        self.pad_id
    }

    /// Encode input text into token IDs.
    ///
    /// If `allowed_special` is true, special added tokens (e.g. `<|endoftext|>`)
    /// are parsed directly into their special token IDs.
    pub fn encode(&self, text: &str, allowed_special: bool) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        let mut out_ids = Vec::new();

        if allowed_special && !self.special_tokens.is_empty() {
            let mut pos = 0;
            while pos < text.len() {
                let rem = &text[pos..];

                // Check if any special token starts at pos
                let mut matched_special = None;
                for (spec_str, id) in &self.special_tokens {
                    if rem.starts_with(spec_str) {
                        matched_special = Some((spec_str.len(), *id));
                        break;
                    }
                }

                if let Some((spec_len, id)) = matched_special {
                    out_ids.push(id);
                    pos += spec_len;
                } else {
                    // Find nearest special token occurrence
                    let mut next_spec_pos = text.len();
                    for (spec_str, _) in &self.special_tokens {
                        if let Some(idx) = rem.find(spec_str) {
                            if idx > 0 && pos + idx < next_spec_pos {
                                next_spec_pos = pos + idx;
                            }
                        }
                    }

                    let normal_slice = &text[pos..next_spec_pos];
                    self.encode_normal_text(normal_slice, &mut out_ids);
                    pos = next_spec_pos;
                }
            }
        } else {
            self.encode_normal_text(text, &mut out_ids);
        }

        out_ids
    }

    fn encode_normal_text(&self, text: &str, out: &mut Vec<u32>) {
        let chunks = match self.pre_tokenizer_kind {
            PreTokenizerKind::Gpt2 => split_gpt2(text),
            PreTokenizerKind::Llama3 => split_llama3(text),
            PreTokenizerKind::ByteLevelNoRegex => {
                if text.is_empty() {
                    Vec::new()
                } else {
                    vec![text]
                }
            }
        };

        let mut key_buf = String::with_capacity(64);

        for chunk in chunks {
            // Check direct added token match first
            if let Some(&id) = self.added_tokens.get(chunk) {
                out.push(id);
                continue;
            }

            // Convert bytes of chunk to unicode symbols
            let mut symbols: Vec<String> = Vec::with_capacity(chunk.len());
            for &b in chunk.as_bytes() {
                symbols.push(self.byte_to_char[b as usize].to_string());
            }

            // BPE merge loop
            while symbols.len() >= 2 {
                let mut min_rank = usize::MAX;
                let mut best_pair = None;

                for i in 0..symbols.len() - 1 {
                    key_buf.clear();
                    key_buf.push_str(&symbols[i]);
                    key_buf.push(' ');
                    key_buf.push_str(&symbols[i + 1]);

                    if let Some(&rank) = self.merges.get(&key_buf) {
                        if rank < min_rank {
                            min_rank = rank;
                            best_pair = Some((symbols[i].clone(), symbols[i + 1].clone()));
                        }
                    }
                }

                let (first, second) = match best_pair {
                    Some(p) => p,
                    None => break,
                };

                let mut next_symbols = Vec::with_capacity(symbols.len());
                let mut i = 0;
                while i < symbols.len() {
                    if i + 1 < symbols.len() && symbols[i] == first && symbols[i + 1] == second {
                        let mut merged = String::with_capacity(symbols[i].len() + symbols[i + 1].len());
                        merged.push_str(&symbols[i]);
                        merged.push_str(&symbols[i + 1]);
                        next_symbols.push(merged);
                        i += 2;
                    } else {
                        next_symbols.push(symbols[i].clone());
                        i += 1;
                    }
                }
                symbols = next_symbols;
            }

            // Map merged symbols to token IDs
            for sym in symbols {
                if let Some(&id) = self.vocab.get(&sym) {
                    out.push(id);
                } else if let Some(&id) = self.added_tokens.get(&sym) {
                    out.push(id);
                }
            }
        }
    }

    /// Decode token IDs back to a UTF-8 String, returning an error on invalid UTF-8 or unknown IDs.
    pub fn decode(&self, ids: &[u32]) -> Result<String, ImportError> {
        let bytes = self.decode_to_bytes(ids)?;
        String::from_utf8(bytes).map_err(|e| {
            ImportError::UnsupportedTokenizer(format!("decoded bytes are not valid UTF-8: {e}"))
        })
    }

    /// Decode token IDs back to a UTF-8 String with lossy replacement for any invalid byte sequences.
    pub fn decode_lossy(&self, ids: &[u32]) -> String {
        match self.decode_to_bytes(ids) {
            Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
            Err(_) => String::new(),
        }
    }

    fn decode_to_bytes(&self, ids: &[u32]) -> Result<Vec<u8>, ImportError> {
        let mut raw_bytes = Vec::new();

        for &id in ids {
            if let Some(content) = self.inv_added_tokens.get(&id) {
                raw_bytes.extend_from_slice(content.as_bytes());
                continue;
            }

            if let Some(Some(token_str)) = self.inv_vocab.get(id as usize) {
                for ch in token_str.chars() {
                    if let Some(&b) = self.char_to_byte.get(&ch) {
                        raw_bytes.push(b);
                    } else {
                        let mut buf = [0u8; 4];
                        raw_bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    }
                }
            } else {
                return Err(ImportError::UnsupportedTokenizer(format!(
                    "unknown token id {id}"
                )));
            }
        }

        Ok(raw_bytes)
    }
}

// ---------------------------------------------------------------------------
// Dedicated Fast Tokenizer JSON Parser
// ---------------------------------------------------------------------------

struct ParsedTokenizerData {
    model_type: String,
    has_non_null_normalizer: bool,
    vocab: HashMap<String, u32>,
    inv_vocab: Vec<(u32, String)>,
    merges: HashMap<String, usize>,
    added_tokens: HashMap<String, u32>,
    special_ids: HashSet<u32>,
    pre_tokenizer_kind: PreTokenizerKind,
}

struct TokenizerJsonReader<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> TokenizerJsonReader<'a> {
    fn new(src: &'a [u8]) -> Self {
        TokenizerJsonReader { src, pos: 0 }
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.src.len() {
            match self.src[self.pos] {
                b' ' | b'\t' | b'\n' | b'\r' => self.pos += 1,
                _ => break,
            }
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_whitespace();
        if self.pos < self.src.len() {
            Some(self.src[self.pos])
        } else {
            None
        }
    }

    fn next_byte(&mut self) -> Option<u8> {
        self.skip_whitespace();
        if self.pos < self.src.len() {
            let b = self.src[self.pos];
            self.pos += 1;
            Some(b)
        } else {
            None
        }
    }

    fn expect_byte(&mut self, expected: u8) -> Result<(), ImportError> {
        match self.next_byte() {
            Some(b) if b == expected => Ok(()),
            Some(b) => Err(ImportError::InvalidJson(format!(
                "expected '{}', found '{}'",
                expected as char, b as char
            ))),
            None => Err(ImportError::InvalidJson(format!(
                "expected '{}', found EOF",
                expected as char
            ))),
        }
    }

    fn parse_string(&mut self) -> Result<String, ImportError> {
        self.skip_whitespace();
        if self.pos >= self.src.len() || self.src[self.pos] != b'"' {
            return Err(ImportError::InvalidJson("expected '\"'".to_string()));
        }
        self.pos += 1;

        let mut out = String::new();
        while self.pos < self.src.len() {
            let b = self.src[self.pos];
            self.pos += 1;
            if b == b'"' {
                return Ok(out);
            }
            if b < 0x20 {
                return Err(ImportError::InvalidJson("unescaped control char".to_string()));
            }
            if b == b'\\' {
                if self.pos >= self.src.len() {
                    return Err(ImportError::InvalidJson("unterminated string escape".to_string()));
                }
                let esc = self.src[self.pos];
                self.pos += 1;
                match esc {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\x08'),
                    b'f' => out.push('\x0C'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        if self.pos + 4 > self.src.len() {
                            return Err(ImportError::InvalidJson("truncated unicode escape".to_string()));
                        }
                        let hex_bytes = &self.src[self.pos..self.pos + 4];
                        self.pos += 4;
                        let hex_str = std::str::from_utf8(hex_bytes)
                            .map_err(|_| ImportError::InvalidJson("invalid unicode escape".to_string()))?;
                        let code_point = u16::from_str_radix(hex_str, 16)
                            .map_err(|_| ImportError::InvalidJson("invalid unicode hex".to_string()))?;

                        if (0xD800..=0xDBFF).contains(&code_point) {
                            if self.pos + 6 <= self.src.len() && &self.src[self.pos..self.pos + 2] == b"\\u" {
                                self.pos += 2;
                                let low_hex = &self.src[self.pos..self.pos + 4];
                                self.pos += 4;
                                let low_str = std::str::from_utf8(low_hex)
                                    .map_err(|_| ImportError::InvalidJson("invalid low surrogate escape".to_string()))?;
                                let low_cp = u16::from_str_radix(low_str, 16)
                                    .map_err(|_| ImportError::InvalidJson("invalid low surrogate hex".to_string()))?;
                                if (0xDC00..=0xDFFF).contains(&low_cp) {
                                    let scalar = 0x10000 + (((code_point - 0xD800) as u32) << 10) + ((low_cp - 0xDC00) as u32);
                                    let ch = char::from_u32(scalar)
                                        .ok_or_else(|| ImportError::InvalidJson("invalid surrogate pair scalar".to_string()))?;
                                    out.push(ch);
                                } else {
                                    return Err(ImportError::InvalidJson("invalid low surrogate".to_string()));
                                }
                            } else {
                                return Err(ImportError::InvalidJson("unpaired high surrogate".to_string()));
                            }
                        } else {
                            let ch = char::from_u32(code_point as u32)
                                .ok_or_else(|| ImportError::InvalidJson("invalid unicode code point".to_string()))?;
                            out.push(ch);
                        }
                    }
                    other => {
                        return Err(ImportError::InvalidJson(format!("invalid escape '\\{}'", other as char)));
                    }
                }
            } else if b >= 0x80 {
                self.pos -= 1;
                let s = std::str::from_utf8(&self.src[self.pos..])
                    .map_err(|_| ImportError::InvalidJson("invalid UTF-8 in JSON string".to_string()))?;
                let ch = s.chars().next().unwrap();
                self.pos += ch.len_utf8();
                out.push(ch);
            } else {
                out.push(b as char);
            }
        }
        Err(ImportError::InvalidJson("unterminated string".to_string()))
    }

    fn parse_u64(&mut self) -> Result<u64, ImportError> {
        self.skip_whitespace();
        let start = self.pos;
        while self.pos < self.src.len() && self.src[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        if start == self.pos {
            return Err(ImportError::InvalidJson("expected integer".to_string()));
        }
        let s = std::str::from_utf8(&self.src[start..self.pos])
            .map_err(|_| ImportError::InvalidJson("invalid integer bytes".to_string()))?;
        s.parse::<u64>().map_err(|e| ImportError::InvalidJson(e.to_string()))
    }

    fn skip_value(&mut self) -> Result<(), ImportError> {
        self.skip_whitespace();
        match self.peek() {
            Some(b'{') => {
                self.pos += 1;
                let mut depth = 1usize;
                while depth > 0 && self.pos < self.src.len() {
                    match self.src[self.pos] {
                        b'{' => depth += 1,
                        b'}' => depth -= 1,
                        b'"' => {
                            self.parse_string()?;
                            continue;
                        }
                        _ => {}
                    }
                    self.pos += 1;
                }
                Ok(())
            }
            Some(b'[') => {
                self.pos += 1;
                let mut depth = 1usize;
                while depth > 0 && self.pos < self.src.len() {
                    match self.src[self.pos] {
                        b'[' => depth += 1,
                        b']' => depth -= 1,
                        b'"' => {
                            self.parse_string()?;
                            continue;
                        }
                        _ => {}
                    }
                    self.pos += 1;
                }
                Ok(())
            }
            Some(b'"') => {
                self.parse_string()?;
                Ok(())
            }
            Some(_) => {
                while self.pos < self.src.len() {
                    match self.src[self.pos] {
                        b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r' => break,
                        _ => self.pos += 1,
                    }
                }
                Ok(())
            }
            None => Err(ImportError::InvalidJson("unexpected EOF in skip_value".to_string())),
        }
    }

    fn parse(&mut self) -> Result<ParsedTokenizerData, ImportError> {
        self.expect_byte(b'{')?;

        let mut model_type = String::new();
        let mut has_non_null_normalizer = false;
        let mut vocab = HashMap::new();
        let mut inv_vocab = Vec::new();
        let mut merges = HashMap::new();
        let mut added_tokens = HashMap::new();
        let mut special_ids = HashSet::new();
        let mut pre_tokenizer_kind = PreTokenizerKind::Gpt2;

        loop {
            self.skip_whitespace();
            if self.peek() == Some(b'}') {
                self.pos += 1;
                break;
            }

            let key = self.parse_string()?;
            self.expect_byte(b':')?;

            match key.as_str() {
                "normalizer" => {
                    self.skip_whitespace();
                    if self.peek() == Some(b'n') {
                        // "null"
                        self.skip_value()?;
                    } else {
                        has_non_null_normalizer = true;
                        self.skip_value()?;
                    }
                }
                "pre_tokenizer" => {
                    pre_tokenizer_kind = self.parse_pre_tokenizer()?;
                }
                "model" => {
                    self.expect_byte(b'{')?;
                    loop {
                        self.skip_whitespace();
                        if self.peek() == Some(b'}') {
                            self.pos += 1;
                            break;
                        }
                        let mkey = self.parse_string()?;
                        self.expect_byte(b':')?;
                        match mkey.as_str() {
                            "type" => {
                                model_type = self.parse_string()?;
                            }
                            "vocab" => {
                                self.parse_vocab(&mut vocab, &mut inv_vocab)?;
                            }
                            "merges" => {
                                self.parse_merges(&mut merges)?;
                            }
                            _ => {
                                self.skip_value()?;
                            }
                        }

                        self.skip_whitespace();
                        match self.peek() {
                            Some(b',') => {
                                self.pos += 1;
                            }
                            Some(b'}') => {
                                self.pos += 1;
                                break;
                            }
                            _ => return Err(ImportError::InvalidJson("expected ',' or '}' in model".to_string())),
                        }
                    }
                }
                "added_tokens" => {
                    self.parse_added_tokens(&mut added_tokens, &mut special_ids)?;
                }
                _ => {
                    self.skip_value()?;
                }
            }

            self.skip_whitespace();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(ImportError::InvalidJson("expected ',' or '}' in root object".to_string())),
            }
        }

        Ok(ParsedTokenizerData {
            model_type,
            has_non_null_normalizer,
            vocab,
            inv_vocab,
            merges,
            added_tokens,
            special_ids,
            pre_tokenizer_kind,
        })
    }

    fn parse_vocab(
        &mut self,
        vocab: &mut HashMap<String, u32>,
        inv_vocab: &mut Vec<(u32, String)>,
    ) -> Result<(), ImportError> {
        self.expect_byte(b'{')?;
        loop {
            self.skip_whitespace();
            if self.peek() == Some(b'}') {
                self.pos += 1;
                break;
            }

            let token_str = self.parse_string()?;
            if token_str.len() > MAX_TOKEN_STRING_LEN {
                return Err(ImportError::TooLarge {
                    what: "token byte length",
                    size: token_str.len(),
                    cap: MAX_TOKEN_STRING_LEN,
                });
            }

            self.expect_byte(b':')?;
            let id = self.parse_u64()? as u32;

            inv_vocab.push((id, token_str.clone()));
            vocab.insert(token_str, id);

            if vocab.len() > MAX_BPE_VOCAB_CAP {
                return Err(ImportError::TooLarge {
                    what: "vocab entries",
                    size: vocab.len(),
                    cap: MAX_BPE_VOCAB_CAP,
                });
            }

            self.skip_whitespace();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(ImportError::InvalidJson("expected ',' or '}' in vocab".to_string())),
            }
        }
        Ok(())
    }

    fn parse_merges(&mut self, merges: &mut HashMap<String, usize>) -> Result<(), ImportError> {
        self.expect_byte(b'[')?;
        let mut rank = 0usize;
        loop {
            self.skip_whitespace();
            if self.peek() == Some(b']') {
                self.pos += 1;
                break;
            }

            let merge_str = self.parse_string()?;
            merges.insert(merge_str, rank);
            rank += 1;

            if rank > MAX_BPE_MERGES_CAP {
                return Err(ImportError::TooLarge {
                    what: "merges",
                    size: rank,
                    cap: MAX_BPE_MERGES_CAP,
                });
            }

            self.skip_whitespace();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b']') => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(ImportError::InvalidJson("expected ',' or ']' in merges".to_string())),
            }
        }
        Ok(())
    }

    fn parse_added_tokens(
        &mut self,
        added: &mut HashMap<String, u32>,
        special: &mut HashSet<u32>,
    ) -> Result<(), ImportError> {
        self.expect_byte(b'[')?;
        loop {
            self.skip_whitespace();
            if self.peek() == Some(b']') {
                self.pos += 1;
                break;
            }

            self.expect_byte(b'{')?;
            let mut id = None;
            let mut content = None;
            let mut is_special = false;

            loop {
                self.skip_whitespace();
                if self.peek() == Some(b'}') {
                    self.pos += 1;
                    break;
                }

                let key = self.parse_string()?;
                self.expect_byte(b':')?;
                match key.as_str() {
                    "id" => {
                        id = Some(self.parse_u64()? as u32);
                    }
                    "content" => {
                        content = Some(self.parse_string()?);
                    }
                    "special" => {
                        self.skip_whitespace();
                        if self.src[self.pos..].starts_with(b"true") {
                            is_special = true;
                            self.pos += 4;
                        } else if self.src[self.pos..].starts_with(b"false") {
                            is_special = false;
                            self.pos += 5;
                        } else {
                            self.skip_value()?;
                        }
                    }
                    _ => {
                        self.skip_value()?;
                    }
                }

                self.skip_whitespace();
                match self.peek() {
                    Some(b',') => {
                        self.pos += 1;
                    }
                    Some(b'}') => {
                        self.pos += 1;
                        break;
                    }
                    _ => return Err(ImportError::InvalidJson("expected ',' or '}' in added_token".to_string())),
                }
            }

            if let (Some(id), Some(content)) = (id, content) {
                if is_special {
                    special.insert(id);
                }
                added.insert(content, id);
            }

            self.skip_whitespace();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b']') => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(ImportError::InvalidJson("expected ',' or ']' in added_tokens".to_string())),
            }
        }
        Ok(())
    }

    fn parse_pre_tokenizer(&mut self) -> Result<PreTokenizerKind, ImportError> {
        self.skip_whitespace();
        if self.peek() == Some(b'n') {
            self.skip_value()?;
            return Ok(PreTokenizerKind::Gpt2);
        }

        self.expect_byte(b'{')?;
        let mut ptype = String::new();
        let mut use_regex = true;
        let mut regex_pattern = None;

        loop {
            self.skip_whitespace();
            if self.peek() == Some(b'}') {
                self.pos += 1;
                break;
            }

            let key = self.parse_string()?;
            self.expect_byte(b':')?;

            match key.as_str() {
                "type" => {
                    ptype = self.parse_string()?;
                }
                "use_regex" => {
                    self.skip_whitespace();
                    if self.src[self.pos..].starts_with(b"true") {
                        use_regex = true;
                        self.pos += 4;
                    } else if self.src[self.pos..].starts_with(b"false") {
                        use_regex = false;
                        self.pos += 5;
                    } else {
                        self.skip_value()?;
                    }
                }
                "pre_tokenizers" => {
                    // Sequence of pre-tokenizers: scan for Split with Regex
                    self.expect_byte(b'[')?;
                    while self.peek() != Some(b']') {
                        self.skip_whitespace();
                        if self.peek() == Some(b'{') {
                            self.pos += 1;
                            while self.peek() != Some(b'}') {
                                let subkey = self.parse_string()?;
                                self.expect_byte(b':')?;
                                if subkey == "pattern" {
                                    self.expect_byte(b'{')?;
                                    while self.peek() != Some(b'}') {
                                        let pat_key = self.parse_string()?;
                                        self.expect_byte(b':')?;
                                        if pat_key == "Regex" {
                                            regex_pattern = Some(self.parse_string()?);
                                        } else {
                                            self.skip_value()?;
                                        }
                                        if self.peek() == Some(b',') {
                                            self.pos += 1;
                                        }
                                    }
                                    self.expect_byte(b'}')?;
                                } else {
                                    self.skip_value()?;
                                }
                                if self.peek() == Some(b',') {
                                    self.pos += 1;
                                }
                            }
                            self.expect_byte(b'}')?;
                        }
                        if self.peek() == Some(b',') {
                            self.pos += 1;
                        }
                    }
                    self.expect_byte(b']')?;
                }
                _ => {
                    self.skip_value()?;
                }
            }

            self.skip_whitespace();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(ImportError::InvalidJson("expected ',' or '}' in pre_tokenizer".to_string())),
            }
        }

        if let Some(ref pat) = regex_pattern {
            if pat.contains("(?i:'s|'t") || pat.contains("\\p{N}{1,3}") {
                Ok(PreTokenizerKind::Llama3)
            } else if pat.contains("'s|'t|'re") || pat.contains("?\\p{L}+") {
                Ok(PreTokenizerKind::Gpt2)
            } else {
                Err(ImportError::UnsupportedTokenizer(format!(
                    "unsupported regex pattern: {pat}"
                )))
            }
        } else if ptype == "ByteLevel" {
            if use_regex {
                Ok(PreTokenizerKind::Gpt2)
            } else {
                Ok(PreTokenizerKind::ByteLevelNoRegex)
            }
        } else if ptype == "Sequence" {
            Ok(PreTokenizerKind::Gpt2)
        } else {
            Err(ImportError::UnsupportedTokenizer(format!(
                "unsupported pre_tokenizer type: '{ptype}'"
            )))
        }
    }
}
