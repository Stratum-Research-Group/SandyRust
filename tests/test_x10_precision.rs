use sandy_rust::gates;
use sandy_rust::precision::{
    quantize_bf16, quantize_f16, DynamicLossScaler, PrecisionError,
};

#[test]
fn test_gx10_gate_pass() {
    let rep = gates::run(&["gx10"]).expect("run gx10");
    assert_eq!(rep.len(), 1);
    assert_eq!(rep[0].status, "PASS", "Gate GX10 failed: {:?}", rep[0]);
}

#[test]
fn test_fp16_bf16_conversions_and_quantization() {
    // Exact powers of 2
    for p in -10..=10 {
        let val = 2.0f32.powi(p);
        let q16 = quantize_f16(val);
        let qbf = quantize_bf16(val);
        assert_eq!(q16, val, "FP16 power of 2: {val}");
        assert_eq!(qbf, val, "BF16 power of 2: {val}");
    }

    // Zero with sign
    let z_pos = 0.0f32;
    let z_neg = -0.0f32;
    assert_eq!(quantize_f16(z_pos).to_bits(), 0);
    assert_eq!(quantize_f16(z_neg).to_bits(), 0x8000_0000);
    assert_eq!(quantize_bf16(z_pos).to_bits(), 0);
    assert_eq!(quantize_bf16(z_neg).to_bits(), 0x8000_0000);

    // Inf and NaN
    let inf_pos = f32::INFINITY;
    let inf_neg = f32::NEG_INFINITY;
    assert!(quantize_f16(inf_pos).is_infinite() && quantize_f16(inf_pos) > 0.0);
    assert!(quantize_f16(inf_neg).is_infinite() && quantize_f16(inf_neg) < 0.0);
    assert!(quantize_bf16(inf_pos).is_infinite() && quantize_bf16(inf_pos) > 0.0);
    assert!(quantize_bf16(inf_neg).is_infinite() && quantize_bf16(inf_neg) < 0.0);

    let nan_val = f32::NAN;
    assert!(quantize_f16(nan_val).is_nan());
    assert!(quantize_bf16(nan_val).is_nan());
}

#[test]
fn test_dynamic_loss_scaler_flow_and_exhaustion() {
    let mut scaler = DynamicLossScaler::new(32768.0, 5, 2000);
    assert_eq!(scaler.current_scale, 32768.0);

    // Clean step unscales grads
    let mut grads = vec![32768.0f32; 4];
    let ok = scaler.process_step(&mut grads).unwrap();
    assert!(ok);
    assert_eq!(grads, vec![1.0f32; 4]);
    assert_eq!(scaler.consecutive_skips, 0);

    // First NaN skip: scale halved to 16384.0
    let mut nan_grads = vec![1.0f32, f32::NAN, 3.0f32, 4.0f32];
    let skip1 = scaler.process_step(&mut nan_grads).unwrap();
    assert!(!skip1);
    assert_eq!(scaler.current_scale, 16384.0);
    assert_eq!(scaler.consecutive_skips, 1);

    // Second skip: scale halved to 8192.0
    let mut inf_grads = vec![f32::INFINITY, 2.0f32, 3.0f32, 4.0f32];
    let skip2 = scaler.process_step(&mut inf_grads).unwrap();
    assert!(!skip2);
    assert_eq!(scaler.current_scale, 8192.0);
    assert_eq!(scaler.consecutive_skips, 2);

    // Clean step resets consecutive skips and leaves scale
    let mut clean_grads = vec![8192.0f32; 4];
    let ok2 = scaler.process_step(&mut clean_grads).unwrap();
    assert!(ok2);
    assert_eq!(clean_grads, vec![1.0f32; 4]);
    assert_eq!(scaler.consecutive_skips, 0);
    assert_eq!(scaler.current_scale, 8192.0);

    // Trigger 5 consecutive skips to halt
    for i in 1..=4 {
        let mut bad = vec![f32::NAN; 4];
        let s = scaler.process_step(&mut bad).unwrap();
        assert!(!s);
        assert_eq!(scaler.consecutive_skips, i);
    }
    // 5th consecutive skip must return Err(ExcessiveConsecutiveSkips(5))
    let mut bad_final = vec![f32::NAN; 4];
    let res_final = scaler.process_step(&mut bad_final);
    assert_eq!(
        res_final,
        Err(PrecisionError::ExcessiveConsecutiveSkips(5))
    );
}
