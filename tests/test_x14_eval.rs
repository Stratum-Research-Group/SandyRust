//! Test suite for Gate GX14: Paired Perturbation & Semantic Flip-Rate Evaluation (§A3).

use sandy_rust::eval::{
    assign_splits, evaluate_pairs, evaluate_trivial_baselines, family_clustered_bootstrap,
    generate_copy_marker, generate_digit_comparison, generate_kv_lookup, generate_symbol_count,
    gx14, EvalError, Family, Pair, PredictionResult, Relation,
};
use sandy_rust::gates;

#[test]
fn test_gx14_gate_passes() {
    let rep = gx14();
    assert_eq!(rep.status, "PASS", "GX14 must pass: {}", rep.observed);
}

#[test]
fn test_x14_generator_determinism_and_invariants() {
    // Determinism under identical seed
    let kv1 = generate_kv_lookup(20, 101);
    let kv2 = generate_kv_lookup(20, 101);
    assert_eq!(kv1, kv2);

    let copy1 = generate_copy_marker(20, 102);
    let copy2 = generate_copy_marker(20, 102);
    assert_eq!(copy1, copy2);

    let cmp1 = generate_digit_comparison(20, 103);
    let cmp2 = generate_digit_comparison(20, 103);
    assert_eq!(cmp1, cmp2);

    let cnt1 = generate_symbol_count(20, 104);
    let cnt2 = generate_symbol_count(20, 104);
    assert_eq!(cnt1, cnt2);

    // Differentiation under different seed
    let kv3 = generate_kv_lookup(20, 999);
    assert_ne!(kv1, kv3);

    // Invariant check: every family must have at least 1 MustFlip and at least 1 MustNotFlip
    for fam in kv1.iter().chain(&copy1).chain(&cmp1).chain(&cnt1) {
        assert!(fam.pairs.iter().any(|p| p.relation == Relation::MustFlip));
        assert!(fam.pairs.iter().any(|p| p.relation == Relation::MustNotFlip));
        // All pairs in the family must share the exact same base prompt
        let base0 = &fam.pairs[0].base_prompt;
        for p in &fam.pairs {
            assert_eq!(&p.base_prompt, base0);
        }
    }
}

#[test]
fn test_x14_split_leakage_detection() {
    let mut families = Vec::new();
    families.extend(generate_kv_lookup(30, 1));
    families.extend(generate_copy_marker(30, 2));
    families.extend(generate_digit_comparison(30, 3));
    families.extend(generate_symbol_count(30, 4));

    // Splitting valid families succeeds with 0% prompt leakage
    let (train, dev, test) = assign_splits(&families, 42).expect("split assignment succeeds");
    assert!(!train.is_empty());
    assert!(!dev.is_empty());
    assert!(!test.is_empty());

    // Artificial leakage test: create two families with the exact same base prompt
    // and force them into different buckets.
    let leak_prompt = "[leak_test] duplicate base prompt for split test";
    let f1 = Family {
        family_id: "fam_leak_train_bucket".to_string(), // will hash into train
        task_name: "leak_test".to_string(),
        pairs: vec![
            Pair {
                base_prompt: leak_prompt.to_string(),
                variant_prompt: "var1".to_string(),
                relation: Relation::MustFlip,
                oracle_base_answer: "1".to_string(),
                oracle_variant_answer: "2".to_string(),
            },
            Pair {
                base_prompt: leak_prompt.to_string(),
                variant_prompt: "var2".to_string(),
                relation: Relation::MustNotFlip,
                oracle_base_answer: "1".to_string(),
                oracle_variant_answer: "1".to_string(),
            },
        ],
    };

    let f2 = Family {
        family_id: "fam_leak_test_bucket".to_string(),
        task_name: "leak_test".to_string(),
        pairs: vec![
            Pair {
                base_prompt: leak_prompt.to_string(),
                variant_prompt: "var3".to_string(),
                relation: Relation::MustFlip,
                oracle_base_answer: "1".to_string(),
                oracle_variant_answer: "3".to_string(),
            },
            Pair {
                base_prompt: leak_prompt.to_string(),
                variant_prompt: "var4".to_string(),
                relation: Relation::MustNotFlip,
                oracle_base_answer: "1".to_string(),
                oracle_variant_answer: "1".to_string(),
            },
        ],
    };

    // Find a seed where f1 and f2 end up in different splits
    let mut found_leak = false;
    for seed in 0..1000 {
        let test_set = [f1.clone(), f2.clone()];
        if let Err(EvalError::SplitLeak { prompt }) = assign_splits(&test_set, seed) {
            assert_eq!(prompt, leak_prompt);
            found_leak = true;
            break;
        }
    }
    assert!(found_leak, "Split leakage must be detected when identical base prompt exists across splits");
}

#[test]
fn test_x14_hand_worked_metrics() {
    // 4 MustFlip pairs: 3 flipped, 1 not flipped -> flip_rate_mf = 3/4 = 0.75
    // 4 MustNotFlip pairs: 1 flipped, 3 not flipped -> flip_rate_mnf = 1/4 = 0.25
    // Selectivity = 0.75 - 0.25 = 0.50
    // Base correct: 6 of 8 = 0.75
    // Variant correct: 5 of 8 = 0.625

    let make_mf = |id: usize, flipped: bool, b_correct: bool, v_correct: bool| {
        let p = Pair {
            base_prompt: format!("base_mf_{id}"),
            variant_prompt: format!("var_mf_{id}"),
            relation: Relation::MustFlip,
            oracle_base_answer: "A".to_string(),
            oracle_variant_answer: "B".to_string(),
        };
        let b_ans = if b_correct { "A" } else { "X" };
        let v_ans = if flipped {
            if v_correct { "B" } else { "Y" }
        } else {
            b_ans
        };
        (p, PredictionResult {
            model_base_answer: b_ans.to_string(),
            model_variant_answer: v_ans.to_string(),
        })
    };

    let make_mnf = |id: usize, flipped: bool, b_correct: bool, v_correct: bool| {
        let p = Pair {
            base_prompt: format!("base_mnf_{id}"),
            variant_prompt: format!("var_mnf_{id}"),
            relation: Relation::MustNotFlip,
            oracle_base_answer: "A".to_string(),
            oracle_variant_answer: "A".to_string(),
        };
        let b_ans = if b_correct { "A" } else { "X" };
        let v_ans = if flipped { "Z" } else { b_ans };
        let final_v_ans = if v_correct { "A" } else { v_ans };
        (p, PredictionResult {
            model_base_answer: b_ans.to_string(),
            model_variant_answer: final_v_ans.to_string(),
        })
    };

    let pairs_with_preds = [
        // MF: 3 flipped, 1 same
        make_mf(1, true, true, true),
        make_mf(2, true, true, true),
        make_mf(3, true, true, false),
        make_mf(4, false, true, false),
        // MNF: 1 flipped, 3 same
        make_mnf(5, true, true, false),
        make_mnf(6, false, true, true),
        make_mnf(7, false, false, false),
        make_mnf(8, false, false, false),
    ];

    let refs: Vec<(&Pair, PredictionResult)> = pairs_with_preds.iter().map(|(p, pred)| {
        (p, PredictionResult {
            model_base_answer: pred.model_base_answer.clone(),
            model_variant_answer: pred.model_variant_answer.clone(),
        })
    }).collect();

    let (flip_mf, flip_mnf, selectivity, base_acc, var_acc) = evaluate_pairs(&refs).unwrap();

    assert!((flip_mf - 0.75).abs() < 1e-9);
    assert!((flip_mnf - 0.25).abs() < 1e-9);
    assert!((selectivity - 0.50).abs() < 1e-9);
    assert!((base_acc - 0.75).abs() < 1e-9); // 6 of 8 base correct
    assert!((var_acc - 0.375).abs() < 1e-9);  // 3 of 8 var correct
}

#[test]
fn test_x14_bootstrap_reproducibility() {
    let families = generate_kv_lookup(20, 777);
    let pairs: Vec<&Pair> = families.iter().flat_map(|f| &f.pairs).collect();
    let family_refs: Vec<&Family> = families.iter().collect();

    let preds: Vec<(&Pair, PredictionResult)> = pairs.iter().map(|p| {
        (*p, PredictionResult {
            model_base_answer: p.oracle_base_answer.clone(),
            model_variant_answer: p.oracle_variant_answer.clone(),
        })
    }).collect();

    let bs1 = family_clustered_bootstrap(&family_refs, &preds, 500, 42).unwrap();
    let bs2 = family_clustered_bootstrap(&family_refs, &preds, 500, 42).unwrap();

    assert_eq!(bs1, bs2, "Bootstrap with identical seed must produce bitwise identical intervals");
    assert!(bs1.lower <= bs1.mean);
    assert!(bs1.mean <= bs1.upper);
    assert!((bs1.mean - 1.0).abs() < 1e-6);
}

#[test]
fn test_x14_trivial_baselines_selectivity_zero() {
    let families = generate_copy_marker(10, 888);
    let pairs: Vec<&Pair> = families.iter().flat_map(|f| &f.pairs).collect();

    let baselines = evaluate_trivial_baselines(&pairs);
    assert_eq!(baselines.always_flip_selectivity, 0.0);
    assert_eq!(baselines.always_same_selectivity, 0.0);
    assert_eq!(baselines.copy_base_selectivity, 0.0);
    assert_eq!(baselines.random_selectivity, 0.0);
}

#[test]
fn test_x14_gates_runner_integration() {
    let reports = gates::run(&["gx14"]).expect("gates::run succeeds");
    assert_eq!(reports.len(), 1);
    assert_eq!(reports[0].gate_id, "GX14");
    assert_eq!(reports[0].status, "PASS");
}
