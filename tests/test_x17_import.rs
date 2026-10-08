//! Integration tests for Extension X17: External Pretrained Weight Import (Phase I1).

use sandy_rust::config::{Config, ConfigError, MAX_IMPORT_VOCAB_CAP};
use sandy_rust::gates;
use sandy_rust::import::{
    build_safetensors_bytes, decode_tensor_f32, f16_to_f32, bf16_to_f32, f32_to_f32,
    parse_safetensors_header, Dtype,
};

#[test]
fn test_gx17_0_baseline_untouched() {
    let rep = sandy_rust::import::gx17_0();
    assert_eq!(rep.status, "PASS");
    assert_eq!(rep.gate_id, "GX17-0");
    assert_eq!(rep.scope, "integrity");

    // Also assert directly on Config
    let cfg = Config::reference_v2();
    assert!(!cfg.ext_x17_import);
    assert_eq!(cfg.param_count(), Some(886_272));
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_gx17_1_negative_suite() {
    let rep = sandy_rust::import::gx17_1();
    assert_eq!(rep.status, "PASS");
    assert_eq!(rep.gate_id, "GX17-1");
    assert_eq!(rep.scope, "regression");
}

#[test]
fn test_gx17_2_fuzz_deterministic() {
    let rep = sandy_rust::import::gx17_2();
    assert_eq!(rep.status, "PASS");
    assert_eq!(rep.gate_id, "GX17-2");
    assert_eq!(rep.scope, "integrity");
}

#[test]
fn test_gx17_3_dtype_conversion_parity() {
    let rep = sandy_rust::import::gx17_3();
    assert_eq!(rep.status, "PASS");
    assert_eq!(rep.gate_id, "GX17-3");
    assert_eq!(rep.scope, "acceptance");
}

#[test]
fn test_gx17_composite_gate() {
    let rep = sandy_rust::import::gx17();
    assert_eq!(rep.status, "PASS");
    assert_eq!(rep.gate_id, "GX17");
}

#[test]
fn test_gx17_4_synthetic_roundtrip() {
    let rep = sandy_rust::import::gx17_4();
    assert_eq!(rep.status, "PASS");
    assert_eq!(rep.gate_id, "GX17-4");
    assert_eq!(rep.scope, "acceptance");
}

#[test]
fn test_gx17_6_identity_and_binding() {
    let rep = sandy_rust::import::gx17_6();
    assert_eq!(rep.status, "PASS");
    assert_eq!(rep.gate_id, "GX17-6");
    assert_eq!(rep.scope, "integrity");
}

#[test]
fn test_gx17_runner_dispatch() {
    let reps = gates::run(&[
        "gx17", "gx17_0", "gx17_1", "gx17_2", "gx17_3", "gx17_4", "gx17_6", "gx17_8",
    ])
    .unwrap();
    assert_eq!(reps.len(), 8);
    for r in reps {
        assert_eq!(r.status, "PASS", "Gate {} failed", r.gate_id);
    }
}

#[test]
fn test_safetensors_valid_header_parsing_and_metadata() {
    let header_json = r#"{"__metadata__":{"source":"smollm2","license":"apache-2.0"},"tensor_a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"tensor_b":{"dtype":"F16","shape":[4],"data_offsets":[16,24]}}"#;
    let data = vec![0u8; 24];
    let file_bytes = build_safetensors_bytes(header_json, &data, 8);

    let parsed = parse_safetensors_header(&file_bytes).unwrap();
    assert_eq!(parsed.tensors.len(), 2);

    let meta = parsed.metadata.expect("metadata should be present");
    assert_eq!(meta.len(), 2);
    assert_eq!(meta[0], ("source".to_string(), "smollm2".to_string()));
    assert_eq!(meta[1], ("license".to_string(), "apache-2.0".to_string()));

    let t_a = &parsed.tensors[0];
    assert_eq!(t_a.name, "tensor_a");
    assert_eq!(t_a.dtype, Dtype::F32);
    assert_eq!(t_a.shape, vec![2, 2]);
    assert_eq!(t_a.data_offsets, (0, 16));

    let t_b = &parsed.tensors[1];
    assert_eq!(t_b.name, "tensor_b");
    assert_eq!(t_b.dtype, Dtype::F16);
    assert_eq!(t_b.shape, vec![4]);
    assert_eq!(t_b.data_offsets, (16, 24));
}

#[test]
fn test_safetensors_bf16_and_float_converters() {
    // Test known FP32, F16, and BF16 conversions
    assert_eq!(f32_to_f32([0x00, 0x00, 0x80, 0x3F]), 1.0f32);
    assert_eq!(f16_to_f32(0x3C00), 1.0f32);
    assert_eq!(bf16_to_f32(0x3F80), 1.0f32);

    // Decode slice of BF16
    let raw_bf16 = [0x80, 0x3F, 0x00, 0x40]; // 1.0, 2.0 in BF16
    let decoded = decode_tensor_f32(Dtype::BF16, &raw_bf16, "bf16_test").unwrap();
    assert_eq!(decoded, vec![1.0f32, 2.0f32]);
}

#[test]
fn test_config_import_profile_rules() {
    let mut import_cfg = Config::reference_v2();
    import_cfg.ext_x17_import = true;

    // GQA allowed in import profile
    import_cfg.n_kv_heads = 2;
    assert!(import_cfg.validate().is_ok(), "GQA should be permitted under ext_x17_import");

    // Untied embeddings allowed in import profile
    import_cfg.tied_embeddings = false;
    assert!(import_cfg.validate().is_ok(), "Untied embeddings should be permitted under ext_x17_import");

    // Special tokens with donor values (e.g. 0, 0, 0) allowed in import profile
    import_cfg.bos_id = 0;
    import_cfg.eos_id = 0;
    import_cfg.pad_id = 0;
    assert!(import_cfg.validate().is_ok(), "Donor special tokens should be permitted under ext_x17_import");

    // Vocab size up to 262,144 allowed in import profile
    import_cfg.vocab_size = 49152;
    assert!(import_cfg.validate().is_ok(), "Vocab 49152 should be permitted under ext_x17_import");

    // Exceeding import caps should still be rejected
    import_cfg.vocab_size = MAX_IMPORT_VOCAB_CAP + 1;
    assert!(matches!(import_cfg.validate(), Err(ConfigError::ResourceCap("vocab_size"))));

    // Restore vocab and test baseline immutability:
    // With ext_x17_import=false, GQA must still be rejected
    let mut base_cfg = Config::reference_v2();
    base_cfg.n_kv_heads = 2;
    assert!(matches!(base_cfg.validate(), Err(ConfigError::ExtensionNotEnabled(_))));
}

#[test]
fn test_parse_donor_config_smollm2_json() {
    let json_str = r#"{
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 576,
        "intermediate_size": 1536,
        "num_hidden_layers": 30,
        "num_attention_heads": 9,
        "num_key_value_heads": 3,
        "vocab_size": 49152,
        "max_position_embeddings": 8192,
        "rms_norm_eps": 1e-5,
        "rope_theta": 100000.0,
        "bos_token_id": 0,
        "eos_token_id": 0,
        "pad_token_id": 0,
        "tie_word_embeddings": true
    }"#;

    let cfg = sandy_rust::import::parse_donor_config(json_str.as_bytes(), Some(2048)).unwrap();
    assert_eq!(cfg.d_model, 576);
    assert_eq!(cfg.d_ff, 1536);
    assert_eq!(cfg.n_layers, 30);
    assert_eq!(cfg.n_q_heads, 9);
    assert_eq!(cfg.n_kv_heads, 3);
    assert_eq!(cfg.vocab_size, 49152);
    assert_eq!(cfg.max_seq_len, 2048);
    assert_eq!(cfg.rope_theta, 100000.0);
    assert_eq!(cfg.rms_eps, 1e-5);
    assert!(cfg.tied_embeddings);
    assert!(cfg.ext_x17_import);
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_import_manifest_format_and_hash() {
    let manifest = sandy_rust::import::ImportManifest {
        source_repo: "HuggingFaceTB/SmolLM2-135M".to_string(),
        source_commit: "277ec9b".to_string(),
        source_license_spdx: "Apache-2.0".to_string(),
        sha256_model_safetensors: [0x11u8; 32],
        sha256_config_json: [0x22u8; 32],
        sha256_tokenizer_json: [0x33u8; 32],
        sha256_tokenizer_config_json: [0x44u8; 32],
        importer_version: "sandy_rust-0.1.0-x17".to_string(),
        mapping_table_version: "llama_dense_v1".to_string(),
        sandyrust_commit: "unknown".to_string(),
        dtype_source: "bf16".to_string(),
        conversion: "lossless_widen".to_string(),
    };

    let text = manifest.canonical_text();
    assert!(text.contains("source_repo=HuggingFaceTB/SmolLM2-135M\n"));
    assert!(text.contains("conversion=lossless_widen\n"));
    // Canonical text must end with LF
    assert!(text.ends_with('\n'));
    // Fingerprint should be deterministic
    let h1 = manifest.fingerprint();
    let h2 = manifest.fingerprint();
    assert_eq!(h1, h2);
}

#[test]
fn test_rimp_serialize_deserialize_roundtrip() {
    use sandy_rust::model::Model;

    let cfg = Config {
        schema_version: 1,
        vocab_size: 16,
        d_model: 8,
        n_layers: 1,
        n_q_heads: 2,
        n_kv_heads: 1,
        d_ff: 16,
        max_seq_len: 16,
        batch_size: 2,
        rope_theta: 10000.0,
        rms_eps: 1e-5,
        bos_id: 0,
        eos_id: 1,
        pad_id: 2,
        tied_embeddings: false, // test untied lm_head as well
        seed: 42,
        ext_x13_hooks: false,
        ext_x13_edit: false,
        ext_x12e_entropy_sampling: false,
        ext_x15_lora: false,
        ext_x17_import: true,
    };

    let model = Model::init(&cfg).unwrap();
    assert!(model.lm_head.is_some());

    let manifest_hash = [0xAAu8; 32];
    let tok_fp = [0xBBu8; 32];

    let bytes = sandy_rust::import::serialize_rimp(&model, manifest_hash, tok_fp).unwrap();
    assert_eq!(&bytes[..4], b"RIMP");

    let loaded = sandy_rust::import::deserialize_rimp(&bytes, Some(manifest_hash), 16 * 1024 * 1024).unwrap();
    assert_eq!(loaded.cfg, cfg);
    assert_eq!(loaded.fingerprint(), model.fingerprint());

    // Forward output parity between original and loaded
    let tokens = [0u32, 5, 10];
    let out_orig = model.forward(&tokens, None).unwrap();
    let out_loaded = loaded.forward(&tokens, None).unwrap();
    assert_eq!(out_orig.data(), out_loaded.data());
}

#[test]
fn test_permute_rope_rows_and_transpose_2d() {
    let dh = 4;
    let n_heads = 1;
    let d_in = 2;
    // Input matrix [4, 2]:
    // row 0: [1, 2]  (HF k=0, first half)
    // row 1: [3, 4]  (HF k=1, first half)
    // row 2: [5, 6]  (HF k=0, second half)
    // row 3: [7, 8]  (HF k=1, second half)
    let raw = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let perm = sandy_rust::import::permute_rope_rows(&raw, n_heads, dh, d_in);
    // Sandy adjacent pairs:
    // row 0 (2*0 = 0) gets old row 0: [1, 2]
    // row 1 (2*0+1 = 1) gets old row 2: [5, 6]
    // row 2 (2*1 = 2) gets old row 1: [3, 4]
    // row 3 (2*1+1 = 3) gets old row 3: [7, 8]
    assert_eq!(perm, vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);

    // Transpose [4, 2] to [2, 4]:
    let trans = sandy_rust::import::transpose_2d(&perm, 4, 2);
    // Row 0 of trans: [1.0, 5.0, 3.0, 7.0]
    // Row 1 of trans: [2.0, 6.0, 4.0, 8.0]
    assert_eq!(trans, vec![1.0, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0, 8.0]);
}

#[test]
fn test_synthetic_llama_import_parity() {
    let (config_json, safetensors_bytes) = sandy_rust::import::build_synthetic_llama_safetensors();
    let model = sandy_rust::import::remap_safetensors_from_bytes(
        config_json.as_bytes(),
        &safetensors_bytes,
        Some(16),
    ).unwrap();

    assert_eq!(model.cfg.d_model, 8);
    assert_eq!(model.cfg.n_layers, 2);
    assert_eq!(model.cfg.n_q_heads, 2);
    assert_eq!(model.cfg.n_kv_heads, 1);
    assert!(model.cfg.tied_embeddings);

    // Forward pass
    let prompt = [0u32, 1, 2];
    let out = model.forward(&prompt, None).unwrap();
    assert_eq!(out.shape(), &[3, 16]);
    assert!(out.data().iter().all(|x| x.is_finite()));
}

#[test]
fn test_negative_mapping_rejections() {
    let (config_json, mut safetensors_bytes) = sandy_rust::import::build_synthetic_llama_safetensors();

    // 1. Bias rejection
    let bad_bias_header = r#"{"model.layers.0.self_attn.q_proj.bias":{"dtype":"F32","shape":[8],"data_offsets":[0,32]}}"#;
    let bad_bias_file = sandy_rust::import::build_safetensors_bytes(bad_bias_header, &[0u8; 32], 8);
    let res = sandy_rust::import::remap_safetensors_from_bytes(config_json.as_bytes(), &bad_bias_file, Some(16));
    assert!(matches!(res, Err(sandy_rust::import::ImportError::UnsupportedArchitecture(_))));

    // 2. Q-norm / K-norm rejection
    let bad_qnorm_header = r#"{"model.layers.0.self_attn.q_norm.weight":{"dtype":"F32","shape":[8],"data_offsets":[0,32]}}"#;
    let bad_qnorm_file = sandy_rust::import::build_safetensors_bytes(bad_qnorm_header, &[0u8; 32], 8);
    let res = sandy_rust::import::remap_safetensors_from_bytes(config_json.as_bytes(), &bad_qnorm_file, Some(16));
    assert!(matches!(res, Err(sandy_rust::import::ImportError::UnsupportedArchitecture(_))));

    // 3. Unknown tensor rejection
    let unknown_header = r#"{"some_unknown_tensor":{"dtype":"F32","shape":[8],"data_offsets":[0,32]}}"#;
    let unknown_file = sandy_rust::import::build_safetensors_bytes(unknown_header, &[0u8; 32], 8);
    let res = sandy_rust::import::remap_safetensors_from_bytes(config_json.as_bytes(), &unknown_file, Some(16));
    assert!(matches!(res, Err(sandy_rust::import::ImportError::UnknownTensor(_))));

    // 4. Missing required tensor rejection
    let empty_header = r#"{}"#;
    let empty_file = sandy_rust::import::build_safetensors_bytes(empty_header, &[], 8);
    let res = sandy_rust::import::remap_safetensors_from_bytes(config_json.as_bytes(), &empty_file, Some(16));
    assert!(matches!(res, Err(sandy_rust::import::ImportError::MissingTensor(_))));

    // 5. Non-finite weight rejection
    // Inject NaN into safetensors payload (after 8 bytes len + N header bytes)
    let header_len = u64::from_le_bytes(safetensors_bytes[..8].try_into().unwrap()) as usize;
    let nan_bytes = f32::NAN.to_le_bytes();
    safetensors_bytes[8 + header_len..8 + header_len + 4].copy_from_slice(&nan_bytes);
    let res = sandy_rust::import::remap_safetensors_from_bytes(config_json.as_bytes(), &safetensors_bytes, Some(16));
    assert!(matches!(res, Err(sandy_rust::import::ImportError::NonFinite { .. })));
}

#[test]
fn test_gx17_8_gate() {
    let rep = sandy_rust::import::gx17_8();
    assert_eq!(rep.status, "PASS");
    assert_eq!(rep.gate_id, "GX17-8");
    assert_eq!(rep.scope, "acceptance");
}

#[test]
fn test_bpe_byte_to_unicode_bijection() {
    let (byte_to_char, char_to_byte) = sandy_rust::import::bpe::bytes_to_unicode();

    // Verify all 256 bytes are bijective
    for b in 0u8..=255 {
        let ch = byte_to_char[b as usize];
        assert_eq!(char_to_byte.get(&ch), Some(&b));
    }

    // Verify space is 'Ġ'
    assert_eq!(byte_to_char[32], 'Ġ');
    assert_eq!(char_to_byte.get(&'Ġ'), Some(&32));

    // Verify exactly 256 unique characters
    let set: std::collections::HashSet<char> = byte_to_char.iter().copied().collect();
    assert_eq!(set.len(), 256);
}

#[test]
fn test_bpe_pre_tokenizer_scanners() {
    // 1. GPT-2 scanner tests
    let chunks_gpt2 = sandy_rust::import::bpe::split_gpt2("Hello, world! It's 123 dogs.");
    assert_eq!(chunks_gpt2, vec!["Hello", ",", " world", "!", " It", "'s", " 123", " dogs", "."]);

    let ws_gpt2 = sandy_rust::import::bpe::split_gpt2("  hello   world  ");
    assert_eq!(ws_gpt2, vec!["  ", "hello", "   ", "world", "  "]);

    // 2. LLaMA-3 scanner tests
    let chunks_llama = sandy_rust::import::bpe::split_llama3("Hello, world! It's 123456 dogs.");
    // Digits chunked into up to 3 digits (\p{N}{1,3}): "123", "456"
    assert!(chunks_llama.contains(&"123"));
    assert!(chunks_llama.contains(&"456"));
}

#[test]
fn test_bpe_synthetic_tokenizer_encode_decode_roundtrip() {
    let json_str = sandy_rust::import::make_synthetic_bpe_tokenizer_json();
    let tokenizer = sandy_rust::import::BpeTokenizer::from_json_bytes(json_str.as_bytes()).unwrap();

    // 1. Golden ASCII
    let ascii = "hello world";
    let ids = tokenizer.encode(ascii, false);
    assert_eq!(ids, vec![261, 266]); // "hello", " world"
    let decoded = tokenizer.decode(&ids).unwrap();
    assert_eq!(decoded, ascii);

    // 2. Special tokens
    let special_text = "<|im_start|>hello<|endoftext|>";
    let ids_spec = tokenizer.encode(special_text, true);
    assert_eq!(ids_spec, vec![1, 261, 0]);
    let dec_spec = tokenizer.decode(&ids_spec).unwrap();
    assert_eq!(dec_spec, special_text);

    // 3. Multilingual and emojis roundtrip
    let multi = "café München déjà vu Привет 你好 🦀🚀 🌟";
    let ids_multi = tokenizer.encode(multi, false);
    let dec_multi = tokenizer.decode(&ids_multi).unwrap();
    assert_eq!(dec_multi, multi);

    // 4. 10 KB payload roundtrip
    let payload = "SandyRust BPE pure-Rust std-only test. ".repeat(300);
    assert!(payload.len() > 10 * 1024);
    let ids_payload = tokenizer.encode(&payload, false);
    let dec_payload = tokenizer.decode(&ids_payload).unwrap();
    assert_eq!(dec_payload, payload);
}

#[test]
fn test_bpe_caps_and_rejections() {
    // 1. Unsupported model type
    let bad_model = br#"{"model":{"type":"Unigram"}}"#;
    let res = sandy_rust::import::BpeTokenizer::from_json_bytes(bad_model);
    assert!(matches!(res, Err(sandy_rust::import::ImportError::UnsupportedTokenizer(_))));

    // 2. Non-null normalizer
    let bad_norm = br#"{"model":{"type":"BPE","vocab":{},"merges":[]},"normalizer":{"type":"NFC"}}"#;
    let res = sandy_rust::import::BpeTokenizer::from_json_bytes(bad_norm);
    assert!(matches!(res, Err(sandy_rust::import::ImportError::UnsupportedTokenizer(_))));

    // 3. Invalid JSON
    let bad_json = b"{\"model\":";
    let res = sandy_rust::import::BpeTokenizer::from_json_bytes(bad_json);
    assert!(matches!(res, Err(sandy_rust::import::ImportError::InvalidJson(_))));

    // 4. Token length cap (> 256 bytes)
    let long_key = "x".repeat(300);
    let bad_len_json = format!(r#"{{"model":{{"type":"BPE","vocab":{{"{}":1}},"merges":[]}}}}"#, long_key);
    let res = sandy_rust::import::BpeTokenizer::from_json_bytes(bad_len_json.as_bytes());
    assert!(matches!(res, Err(sandy_rust::import::ImportError::TooLarge { .. })));
}
