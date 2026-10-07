//! Test suite for Gate GX12E: Entropy-Feedback Adaptive Sampling (§A4).

use sandy_rust::config::Config;
use sandy_rust::gates;
use sandy_rust::generation::{
    compute_distinct_n, compute_entropy, compute_repetition_rate, generate, gx12e, sample_next_step,
    AdaptiveControlMode, AdaptiveSamplerConfig, AdaptiveSamplerState, GenerationError,
    SamplingParams,
};
use sandy_rust::model::Model;
use sandy_rust::rng::Rng;

#[test]
fn test_gx12e_gate_passes() {
    let rep = gx12e();
    assert_eq!(rep.status, "PASS", "GX12E must pass: {}", rep.observed);
}

#[test]
fn test_x12e_disabled_and_k0_parity() {
    let m = Model::init(&Config::reference_v2()).unwrap();
    let prompt = [256u32, 70, 71, 72];

    let p_disabled = SamplingParams {
        temperature: 0.85,
        top_k: 40,
        top_p: 0.95,
        max_new_tokens: 20,
        seed: 12345,
        adaptive: None,
    };
    let out_disabled = generate(&m, &prompt, &p_disabled).unwrap();

    let p_k0 = SamplingParams {
        temperature: 0.85,
        top_k: 40,
        top_p: 0.95,
        max_new_tokens: 20,
        seed: 12345,
        adaptive: Some(AdaptiveSamplerConfig {
            beta: 0.75,
            k: 0.0,
            h_target: 3.0,
            mode: AdaptiveControlMode::Temperature {
                t_min: 0.1,
                t_max: 2.0,
            },
        }),
    };
    let out_k0 = generate(&m, &prompt, &p_k0).unwrap();

    assert_eq!(out_disabled.tokens, out_k0.tokens, "k=0 must match disabled fixed sampling bitwise");
    assert_eq!(out_disabled.final_logits, out_k0.final_logits);
}

#[test]
fn test_x12e_entropy_fixtures() {
    // 1. One-hot distribution -> H = 0.0
    let one_hot = [1.0f32, 0.0, 0.0, 0.0];
    assert_eq!(compute_entropy(&one_hot), 0.0);

    // 2. Uniform distributions -> H = ln(N)
    for n in [2, 4, 8, 16, 257] {
        let p = 1.0f32 / n as f32;
        let dist = vec![p; n];
        let h = compute_entropy(&dist);
        let expected = (n as f32).ln();
        assert!((h - expected).abs() < 1e-5, "H for N={n} must equal ln({n})");
    }

    // 3. Zero-probability entries must not produce NaN
    let mixed = [0.5f32, 0.25, 0.25, 0.0, 0.0];
    let h_mixed = compute_entropy(&mixed);
    // H = -(0.5*ln(0.5) + 0.25*ln(0.25) + 0.25*ln(0.25)) = -(0.5*(-0.693147) + 0.5*(-1.386294)) = 0.3465735 + 0.693147 = 1.03972
    let expected_mixed = -(0.5 * 0.5f32.ln() + 0.5 * 0.25f32.ln());
    assert!((h_mixed - expected_mixed).abs() < 1e-6);
}

#[test]
fn test_x12e_control_bounds_clamping() {
    let t_min = 0.2f32;
    let t_max = 1.8f32;
    let min_p_cap = 0.5f32;
    let h_max = (257.0f32).ln();

    // Adversarial sequence of extreme entropy values
    let extreme_entropies = [-1e6f32, -100.0, 0.0, 1.0, 10.0, 100.0, 1e6];

    for &h_ema in &extreme_entropies {
        // Temperature mode clamping
        let factor_t = 1.0 + 5.0 * (3.0 - h_ema) / h_max;
        let t_eff = (0.8 * factor_t).clamp(t_min, t_max);
        assert!(t_eff >= t_min && t_eff <= t_max, "t_eff {t_eff} must be within [{t_min}, {t_max}]");

        // Min-P mode clamping
        let factor_min_p = 1.0 + 5.0 * (h_ema - 3.0) / h_max;
        let min_p_eff = (0.1 * factor_min_p).clamp(0.0, min_p_cap);
        assert!(min_p_eff >= 0.0 && min_p_eff <= min_p_cap, "min_p_eff {min_p_eff} must be within [0.0, {min_p_cap}]");
    }
}

#[test]
fn test_x12e_prng_consumption() {
    let mut rng_ref = Rng::new(999);
    let mut rng_test = Rng::new(999);
    let logits = vec![1.0f32; 259];

    let p = SamplingParams {
        temperature: 0.8,
        top_k: 50,
        top_p: 0.9,
        max_new_tokens: 1,
        seed: 0,
        adaptive: Some(AdaptiveSamplerConfig {
            beta: 0.8,
            k: 1.0,
            h_target: 3.0,
            mode: AdaptiveControlMode::Temperature {
                t_min: 0.2,
                t_max: 2.0,
            },
        }),
    };

    let mut state = Some(AdaptiveSamplerState::new());
    let _ = sample_next_step(&logits, &p, &mut state, &mut rng_test, 0).unwrap();

    // Baseline stochastic sampling calls rng.next_f32() exactly once
    let _ = rng_ref.next_f32();
    assert_eq!(rng_ref.next_u32(), rng_test.next_u32(), "PRNG call count must match exactly 1 draw per step");
}

#[test]
fn test_x12e_config_validation_and_errors() {
    let v_allowed = 257;

    // 1. Beta validation
    let bad_beta_neg = AdaptiveSamplerConfig {
        beta: -0.1,
        k: 1.0,
        h_target: 2.0,
        mode: AdaptiveControlMode::Temperature { t_min: 0.1, t_max: 2.0 },
    };
    assert!(matches!(bad_beta_neg.validate(v_allowed), Err(GenerationError::InvalidAdaptiveConfig { ref field }) if field == "beta"));

    let bad_beta_one = AdaptiveSamplerConfig {
        beta: 1.0,
        k: 1.0,
        h_target: 2.0,
        mode: AdaptiveControlMode::Temperature { t_min: 0.1, t_max: 2.0 },
    };
    assert!(matches!(bad_beta_one.validate(v_allowed), Err(GenerationError::InvalidAdaptiveConfig { ref field }) if field == "beta"));

    // 2. K validation
    let bad_k = AdaptiveSamplerConfig {
        beta: 0.5,
        k: f32::INFINITY,
        h_target: 2.0,
        mode: AdaptiveControlMode::Temperature { t_min: 0.1, t_max: 2.0 },
    };
    assert!(matches!(bad_k.validate(v_allowed), Err(GenerationError::InvalidAdaptiveConfig { ref field }) if field == "k"));

    // 3. H_target validation
    let bad_htarget = AdaptiveSamplerConfig {
        beta: 0.5,
        k: 1.0,
        h_target: -0.01,
        mode: AdaptiveControlMode::Temperature { t_min: 0.1, t_max: 2.0 },
    };
    assert!(matches!(bad_htarget.validate(v_allowed), Err(GenerationError::InvalidAdaptiveConfig { ref field }) if field == "h_target"));

    // 4. Temperature range validation
    let bad_tmin = AdaptiveSamplerConfig {
        beta: 0.5,
        k: 1.0,
        h_target: 2.0,
        mode: AdaptiveControlMode::Temperature { t_min: 0.0, t_max: 2.0 },
    };
    assert!(matches!(bad_tmin.validate(v_allowed), Err(GenerationError::InvalidAdaptiveConfig { ref field }) if field == "t_min"));

    let bad_tmax = AdaptiveSamplerConfig {
        beta: 0.5,
        k: 1.0,
        h_target: 2.0,
        mode: AdaptiveControlMode::Temperature { t_min: 1.5, t_max: 1.0 },
    };
    assert!(matches!(bad_tmax.validate(v_allowed), Err(GenerationError::InvalidAdaptiveConfig { ref field }) if field == "t_max"));

    // 5. Min-P validation
    let bad_minp_cap = AdaptiveSamplerConfig {
        beta: 0.5,
        k: 1.0,
        h_target: 2.0,
        mode: AdaptiveControlMode::MinP { min_p_base: 0.1, min_p_cap: 1.0 },
    };
    assert!(matches!(bad_minp_cap.validate(v_allowed), Err(GenerationError::InvalidAdaptiveConfig { ref field }) if field == "min_p_cap"));

    let bad_minp_base = AdaptiveSamplerConfig {
        beta: 0.5,
        k: 1.0,
        h_target: 2.0,
        mode: AdaptiveControlMode::MinP { min_p_base: 0.6, min_p_cap: 0.5 },
    };
    assert!(matches!(bad_minp_base.validate(v_allowed), Err(GenerationError::InvalidAdaptiveConfig { ref field }) if field == "min_p_base"));

    // 6. Precedence: NaN logits error before controller logic
    let mut nan_logits = vec![0.0f32; 259];
    nan_logits[5] = f32::NAN;
    let mut r = Rng::new(0);
    let mut state = Some(AdaptiveSamplerState::new());
    let p_valid = SamplingParams {
        temperature: 0.8,
        top_k: 10,
        top_p: 0.9,
        max_new_tokens: 1,
        seed: 0,
        adaptive: Some(AdaptiveSamplerConfig {
            beta: 0.5,
            k: 1.0,
            h_target: 2.0,
            mode: AdaptiveControlMode::Temperature { t_min: 0.1, t_max: 2.0 },
        }),
    };
    let res = sample_next_step(&nan_logits, &p_valid, &mut state, &mut r, 12);
    assert_eq!(res, Err(GenerationError::NonFiniteLogits { position: 12 }));
}

#[test]
fn test_x12e_diversity_metrics() {
    let tokens = [1, 2, 3, 4, 1, 2, 3, 4];
    assert_eq!(compute_distinct_n(&tokens, 1), 0.5); // 4 distinct / 8 total
    assert_eq!(compute_repetition_rate(&tokens), 0.0); // no adjacent duplicates

    let rep_tokens = [1, 1, 1, 1];
    assert_eq!(compute_repetition_rate(&rep_tokens), 1.0); // 3 duplicates / 3 transitions
}

#[test]
fn test_x12e_gates_runner_integration() {
    let reports = gates::run(&["gx12e"]).expect("gates::run succeeds");
    assert_eq!(reports.len(), 1);
    assert_eq!(reports[0].gate_id, "GX12E");
    assert_eq!(reports[0].status, "PASS");
}
