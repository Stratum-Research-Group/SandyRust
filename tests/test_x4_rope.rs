//! Integration test suite for Extension X4: Advanced RoPE Extensions & Layouts & Gate GX4 (§4.2 Item 4, §9.2 GX4).

use sandy_rust::attention::rope_ext::{
    adjacent_to_split, compute_frequencies, gx4, l2_norm, rope_split_with_freqs, rope_with_freqs,
    split_to_adjacent, verify_layout_bijection, yarn_temperature_scale,
};
use sandy_rust::config::{Config, ConfigError, RopeScalingConfig, RopeScalingType};
use sandy_rust::gates;
use sandy_rust::tensor::Tensor;

#[test]
fn test_x4_config_validation_and_rejections() {
    let mut c_base = Config::reference_v2();
    assert!(c_base.validate().is_ok());

    // Without extension flag, setting rope_scaling is rejected
    c_base.rope_scaling = Some(RopeScalingConfig::yarn(64, 256));
    assert!(matches!(
        c_base.validate(),
        Err(ConfigError::ExtensionNotEnabled(_))
    ));

    // With ext_x4_rope, validates
    c_base.ext_x4_rope = true;
    c_base.max_seq_len = 256;
    assert!(c_base.validate().is_ok());

    // Invalid parameters rejected
    let mut c_invalid = c_base.clone();
    c_invalid.rope_scaling = Some(RopeScalingConfig {
        scaling_type: RopeScalingType::Yarn,
        train_context: 256,
        target_context: 64, // target < train invalid
        yarn_alpha: 1.0,
        yarn_beta: 32.0,
    });
    assert!(matches!(
        c_invalid.validate(),
        Err(ConfigError::OutOfRange(_))
    ));

    let mut c_invalid_beta = c_base.clone();
    c_invalid_beta.rope_scaling = Some(RopeScalingConfig {
        scaling_type: RopeScalingType::Yarn,
        train_context: 64,
        target_context: 256,
        yarn_alpha: 32.0,
        yarn_beta: 1.0, // beta <= alpha invalid
    });
    assert!(matches!(
        c_invalid_beta.validate(),
        Err(ConfigError::OutOfRange(_))
    ));

    // Preset constructors validate cleanly
    assert!(Config::rope_yarn_v2(256).validate().is_ok());
    assert!(Config::rope_linear_v2(256).validate().is_ok());
}

#[test]
fn test_x4_layout_conversions_and_inverses() {
    for &dh in &[16, 32, 64] {
        // 1. Single head row
        let v: Vec<f32> = (0..dh).map(|i| (i as f32 * 0.31 + 0.1).sin()).collect();
        assert!(verify_layout_bijection(&v, dh));

        let split = adjacent_to_split(&v, dh);
        let back = split_to_adjacent(&split, dh);
        assert_eq!(v, back);

        // Verify explicit half-split mapping
        let half = dh / 2;
        for k in 0..half {
            assert_eq!(split[k], v[2 * k]);
            assert_eq!(split[k + half], v[2 * k + 1]);
        }

        // 2. Multi-row matrix
        let mat: Vec<f32> = (0..dh * 12).map(|i| (i as f32 * 0.19).cos()).collect();
        assert!(verify_layout_bijection(&mat, dh));

        let mat_split = adjacent_to_split(&mat, dh);
        let mat_back = split_to_adjacent(&mat_split, dh);
        assert_eq!(mat, mat_back);
    }
}

#[test]
fn test_x4_split_vs_adjacent_rope_rotation_equivalence() {
    let dh = 32;
    let seq_len = 8;
    let base_theta = 10000.0f64;

    let yarn_cfg = RopeScalingConfig::yarn(64, 256);
    let freqs = compute_frequencies(dh, base_theta, Some(&yarn_cfg));

    let raw_data: Vec<f32> = (0..seq_len * dh)
        .map(|i| ((i as f32 * 0.41).sin()) * 2.0)
        .collect();

    let adj_tensor = Tensor::from_vec(&[seq_len, dh], raw_data.clone()).unwrap();
    let split_data = adjacent_to_split(&raw_data, dh);
    let split_tensor = Tensor::from_vec(&[seq_len, dh], split_data).unwrap();

    // Rotate at multiple position offsets
    for pos in [0, 5, 63, 128, 255] {
        let rot_adj = rope_with_freqs(&adj_tensor, &freqs, pos).unwrap();
        let rot_split = rope_split_with_freqs(&split_tensor, &freqs, pos).unwrap();
        let rot_split_to_adj = split_to_adjacent(rot_split.data(), dh);

        for (i, (&a, &b)) in rot_adj.data().iter().zip(rot_split_to_adj.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(
                diff <= 1e-6,
                "pos {pos} idx {i} mismatch: adj {a} vs split->adj {b}, diff {diff}"
            );
        }
    }
}

#[test]
fn test_x4_yarn_frequency_bands() {
    let dh = 32;
    let base_theta = 10000.0f64;
    let train_ctx = 64;
    let target_ctx = 256;
    let s = 4.0f64;

    let yarn_cfg = RopeScalingConfig::yarn(train_ctx, target_ctx);
    let freqs_base = compute_frequencies(dh, base_theta, None);
    let freqs_yarn = compute_frequencies(dh, base_theta, Some(&yarn_cfg));
    let freqs_lin = compute_frequencies(
        dh,
        base_theta,
        Some(&RopeScalingConfig::linear(train_ctx, target_ctx)),
    );

    assert_eq!(freqs_base.len(), dh / 2);
    assert_eq!(freqs_yarn.len(), dh / 2);
    assert_eq!(freqs_lin.len(), dh / 2);

    let alpha_bound = 1.0 * train_ctx as f64; // 64
    let beta_bound = 32.0 * train_ctx as f64; // 2048

    for k in 0..dh / 2 {
        let th_base = freqs_base[k];
        let th_yarn = freqs_yarn[k];
        let th_lin = freqs_lin[k];

        // Linear scaling is strictly theta / s everywhere
        assert!((th_lin - th_base / s).abs() < 1e-12);

        let wavelength = 2.0 * std::f64::consts::PI / th_base;
        if wavelength < alpha_bound {
            // High frequency: strictly unscaled
            assert_eq!(th_yarn, th_base, "k={k} wavelength={wavelength} < {alpha_bound}");
        } else if wavelength > beta_bound {
            // Low frequency: linearly scaled
            assert!(
                (th_yarn - th_base / s).abs() < 1e-12,
                "k={k} wavelength={wavelength} > {beta_bound}"
            );
        } else {
            // Intermediate: smoothly interpolated
            let gamma = (wavelength / train_ctx as f64 - 1.0) / (32.0 - 1.0);
            let expected = (1.0 - gamma) * th_base + gamma * (th_base / s);
            assert!(
                (th_yarn - expected).abs() < 1e-12,
                "k={k} intermediate ramp mismatch: {th_yarn} vs {expected}"
            );
        }
    }
}

#[test]
fn test_x4_yarn_attention_temperature_scaling() {
    let yarn_cfg = RopeScalingConfig::yarn(64, 256);
    let s = 256.0f64 / 64.0f64; // 4.0
    let t = 0.1 * s.ln() + 1.0;
    let expected_scale = (1.0 / t).sqrt() as f32;

    let scale = yarn_temperature_scale(Some(&yarn_cfg));
    assert!((scale - expected_scale).abs() < 1e-6);

    // Linear and None have 1.0 scale
    let lin_cfg = RopeScalingConfig::linear(64, 256);
    assert_eq!(yarn_temperature_scale(Some(&lin_cfg)), 1.0);
    assert_eq!(yarn_temperature_scale(None), 1.0);
}

#[test]
fn test_x4_vector_norm_preservation_isometry() {
    let dh = 32;
    let test_v: Vec<f32> = (0..dh).map(|i| (i as f32 * 0.29 + 0.5).sin()).collect();
    let norm_orig = l2_norm(&test_v);
    assert!(norm_orig > 0.0);

    let v_tensor = Tensor::from_vec(&[1, dh], test_v).unwrap();
    let yarn_cfg = RopeScalingConfig::yarn(64, 256);
    let freqs_yarn = compute_frequencies(dh, 10000.0, Some(&yarn_cfg));
    let freqs_lin = compute_frequencies(
        dh,
        10000.0,
        Some(&RopeScalingConfig::linear(64, 256)),
    );
    let freqs_base = compute_frequencies(dh, 10000.0, None);

    for pos in 0..=256 {
        let r_y = rope_with_freqs(&v_tensor, &freqs_yarn, pos).unwrap();
        let r_l = rope_with_freqs(&v_tensor, &freqs_lin, pos).unwrap();
        let r_b = rope_with_freqs(&v_tensor, &freqs_base, pos).unwrap();

        let norm_y = l2_norm(r_y.data());
        let norm_l = l2_norm(r_l.data());
        let norm_b = l2_norm(r_b.data());

        assert!((norm_y - norm_orig).abs() <= 1e-5, "YaRN norm drift at pos {pos}");
        assert!((norm_l - norm_orig).abs() <= 1e-5, "Linear norm drift at pos {pos}");
        assert!((norm_b - norm_orig).abs() <= 1e-5, "Base norm drift at pos {pos}");
    }
}

#[test]
fn test_x4_gate_gx4_execution() {
    let rep = gx4();
    assert_eq!(rep.gate_id, "GX4");
    assert_eq!(rep.status, "PASS", "Gate GX4 must pass: {:?}", rep.reason);

    let dispatched = gates::run(&["gx4"]).unwrap();
    assert_eq!(dispatched.len(), 1);
    assert_eq!(dispatched[0].status, "PASS");
}
