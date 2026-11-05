use sandy_rust::bpe::{StreamError, TokenShardStream};
use sandy_rust::gates;
use sandy_rust::import::bpe::BpeTokenizer;

#[test]
fn test_gx9_gate_pass() {
    let rep = gates::run(&["gx9"]).expect("run gx9");
    assert_eq!(rep.len(), 1);
    assert_eq!(rep[0].status, "PASS", "Gate GX9 failed: {:?}", rep[0]);
}

#[test]
fn test_bpe_roundtrip_and_merges() {
    let bpe_json = sandy_rust::import::make_synthetic_bpe_tokenizer_json();
    let tokenizer = BpeTokenizer::from_json_bytes(bpe_json.as_bytes()).expect("BpeTokenizer");
    assert!(tokenizer.vocab_size() > 259, "Vocab should be expanded (V > 259)");

    let text = "hello world";
    let ids = tokenizer.encode(text, false);
    assert_eq!(ids, vec![261, 266], "Should merge into subwords");

    let decoded = tokenizer.decode(&ids).expect("decode");
    assert_eq!(decoded, text);

    // Special token isolation
    let spec_ids = tokenizer.encode("<|im_start|>hello world<|endoftext|>", true);
    assert_eq!(spec_ids, vec![1, 261, 266, 0]);
    let spec_decoded = tokenizer.decode(&spec_ids).expect("decode with specials");
    assert_eq!(spec_decoded, "<|im_start|>hello world<|endoftext|>");
}

#[test]
fn test_token_shard_stream_bounds_and_rewind() {
    let temp_dir = std::env::temp_dir();
    let shard_path = temp_dir.join("test_x9_shard_file.bin");

    let tokens: Vec<u32> = (0..128).map(|i| 1000 + i).collect();
    let mut bytes = Vec::with_capacity(tokens.len() * 4);
    for &tok in &tokens {
        bytes.extend_from_slice(&tok.to_le_bytes());
    }
    std::fs::write(&shard_path, &bytes).expect("write temp shard");

    // Exceed capacity
    let err_stream = TokenShardStream::open(&shard_path, 64);
    assert!(matches!(err_stream, Err(StreamError::CapacityExceeded { .. })));

    // Open with valid capacity
    let mut stream = TokenShardStream::open(&shard_path, 1024).expect("open stream");
    assert_eq!(stream.total_tokens, 128);

    let mut buf = vec![0u32; 64];
    let n = stream.read_tokens(&mut buf).expect("read");
    assert_eq!(n, 64);
    assert_eq!(&buf[0..64], &tokens[0..64]);

    // Rewind
    stream.rewind_to(0).expect("rewind");
    assert_eq!(stream.cursor_token_idx, 0);

    let mut buf2 = vec![0u32; 64];
    let n2 = stream.read_tokens(&mut buf2).expect("reread");
    assert_eq!(n2, 64);
    assert_eq!(buf, buf2);

    // Seek past EOF
    assert!(matches!(stream.rewind_to(200), Err(StreamError::SeekError(_))));

    let _ = std::fs::remove_file(shard_path);
}
