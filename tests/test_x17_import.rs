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
fn test_gx17_runner_dispatch() {
    let reps = gates::run(&["gx17", "gx17_0", "gx17_1", "gx17_2", "gx17_3"]).unwrap();
    assert_eq!(reps.len(), 5);
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
