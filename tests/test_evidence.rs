//! Verification and evidence contract test suite (03-sandyrust-math-plan EV1..EV7).

use sandy_rust::gates;
use sandy_rust::config::Config;
use sandy_rust::model::Model;
use sandy_rust::checkpoint::{serialize, parse, DEFAULT_MAX_BYTES, CheckpointError};
use std::fs;

#[test]
fn test_ev1_unknown_and_empty_gate_selections_rejected() {
    // Empty selection returns Err
    assert!(gates::run(&[]).is_err());

    // Unknown gate returns Err
    let err = gates::run(&["g1", "nonexistent_gate"]).unwrap_err();
    assert!(err.contains("unknown gate id"));

    // Valid list returns exact matching count
    let reps = gates::run(&["g1", "g2", "g6"]).unwrap();
    assert_eq!(reps.len(), 3);
}

#[test]
fn test_ev2_config_fingerprint_and_reproducibility() {
    let c1 = Config::reference_v2();
    let c2 = Config::reference_v2();
    assert_eq!(c1.fingerprint(), c2.fingerprint());

    let m1 = Model::init(&c1).unwrap();
    let m2 = Model::init(&c2).unwrap();
    assert_eq!(m1.fingerprint(), m2.fingerprint());
}

#[test]
fn test_ev3_not_run_and_non_finite_classification() {
    // G13 must be classified as NOT_RUN
    let g13_rep = gates::g13();
    assert_eq!(g13_rep.status, "NOT_RUN");
    assert_eq!(g13_rep.scope, "acceptance");
    assert!(g13_rep.reason.is_some());

    // Non-finite weights must be rejected during parse
    let mut m = Model::init(&Config::reference_v2()).unwrap();
    m.embed.data_mut()[0] = f32::NAN;
    let bytes = serialize(&m);
    assert!(matches!(parse(&bytes, &m.cfg, DEFAULT_MAX_BYTES), Err(CheckpointError::NonFinite(_))));
}

#[test]
fn test_ev4_tampering_invalidates_integrity() {
    let m = Model::init(&Config::reference_v2()).unwrap();
    let good = serialize(&m);
    let mut tampered = good.clone();
    let len = tampered.len();
    tampered[len - 1] ^= 0xAA;

    assert!(matches!(
        parse(&tampered, &m.cfg, DEFAULT_MAX_BYTES),
        Err(CheckpointError::ChecksumMismatch(_))
    ));
}

#[test]
fn test_ev5_scope_separation_smoke_vs_acceptance() {
    let all_gates = ["g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9", "g10", "g11", "g12", "g13", "g14"];
    let reps = gates::run(&all_gates).unwrap();

    let g9 = reps.iter().find(|r| r.gate_id == "G9").unwrap();
    assert_eq!(g9.scope, "smoke", "G9 must be tagged as smoke check, not full acceptance");

    let g1 = reps.iter().find(|r| r.gate_id == "G1").unwrap();
    assert_eq!(g1.scope, "integrity");

    let g6 = reps.iter().find(|r| r.gate_id == "G6").unwrap();
    assert_eq!(g6.scope, "integrity");

    let g14 = reps.iter().find(|r| r.gate_id == "G14").unwrap();
    assert_eq!(g14.scope, "regression");

    let g2 = reps.iter().find(|r| r.gate_id == "G2").unwrap();
    assert_eq!(g2.scope, "acceptance");
}

#[test]
fn test_ev6_and_ev7_clean_replay_in_fresh_directory() {
    let temp_dir = std::env::temp_dir().join("sandy_rust_ev_replay_test");
    let _ = fs::remove_dir_all(&temp_dir);
    fs::create_dir_all(&temp_dir).unwrap();

    let all_gates = ["g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9", "g10", "g11", "g12", "g13", "g14"];
    let reps = gates::run(&all_gates).unwrap();

    for r in &reps {
        let path = temp_dir.join(format!("{}.json", r.gate_id));
        let json = r.to_json();
        fs::write(&path, &json).unwrap();

        // Readback verification (EV7)
        let readback = fs::read_to_string(&path).unwrap();
        assert_eq!(json, readback);
        assert!(readback.contains("\"schema_version\": \"3.1\""));
        assert!(readback.contains(&format!("\"gate_id\": \"{}\"", r.gate_id)));
    }

    let _ = fs::remove_dir_all(&temp_dir);
}
