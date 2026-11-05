use sandy_rust::autodiff::{loss_and_grad, Params};
use sandy_rust::checkpoint::{self, DEFAULT_MAX_BYTES};
use sandy_rust::config::Config;
use sandy_rust::data::WindowLoader;
use sandy_rust::model::Model;
use sandy_rust::optimizer::{AdamW, Schedule};
use std::fs;
use std::path::PathBuf;

fn create_deterministic_docs() -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let train_docs = vec![
        b"Document one with structured token bytes for training.".iter().map(|&b| b as u32).collect(),
        b"Second document containing diverse byte sequences.".iter().map(|&b| b as u32).collect(),
        b"Third document providing additional coverage for windows.".iter().map(|&b| b as u32).collect(),
        b"Fourth document completing the minimal training split.".iter().map(|&b| b as u32).collect(),
    ];
    let val_docs = vec![
        b"Held-out validation document alpha for evaluation.".iter().map(|&b| b as u32).collect(),
        b"Held-out validation document beta testing generalisation.".iter().map(|&b| b as u32).collect(),
    ];
    (train_docs, val_docs)
}

#[test]
fn test_n2_training_state_resume_identical_to_uninterrupted() {
    let cfg = Config::reference_v2();
    let (train_docs, _) = create_deterministic_docs();
    let total_steps = 6usize;
    let split_step = 3usize;
    let lr = 0.01f32;

    // --- Uninterrupted Run (1..=total_steps) ---
    let model_uninterrupted = Model::init(&cfg).expect("model init");
    let mut params_uninterrupted = Params::<f32>::from_model(&model_uninterrupted);
    let mut opt_uninterrupted = AdamW::new(&params_uninterrupted);
    let mut loader_uninterrupted = WindowLoader::new(&train_docs, cfg.max_seq_len, 42).expect("loader init");
    let sched = Schedule {
        eta_max: lr,
        eta_min: lr * 0.1,
        warmup: 2,
        max_steps: total_steps as u64,
    };

    for step in 1..=total_steps {
        let win = loader_uninterrupted.next_window();
        let mut grads = params_uninterrupted.zeros_like();
        let n_denom = win.len().saturating_sub(1) as f32;
        let _ = loss_and_grad(&cfg, &params_uninterrupted, &win, n_denom, &mut grads).expect("loss_and_grad");
        let eta = sched.eta(step as u64);
        let ok = opt_uninterrupted.step_with_clipping(&mut params_uninterrupted, &mut grads, eta, 1.0).expect("step");
        assert!(ok, "step {step} must succeed");
    }

    // --- Interrupted Run: Part 1 (1..=split_step) ---
    let model_part1 = Model::init(&cfg).expect("model init");
    let mut params_part1 = Params::<f32>::from_model(&model_part1);
    let mut opt_part1 = AdamW::new(&params_part1);
    let mut loader_part1 = WindowLoader::new(&train_docs, cfg.max_seq_len, 42).expect("loader init");

    for step in 1..=split_step {
        let win = loader_part1.next_window();
        let mut grads = params_part1.zeros_like();
        let n_denom = win.len().saturating_sub(1) as f32;
        let _ = loss_and_grad(&cfg, &params_part1, &win, n_denom, &mut grads).expect("loss_and_grad");
        let eta = sched.eta(step as u64);
        let ok = opt_part1.step_with_clipping(&mut params_part1, &mut grads, eta, 1.0).expect("step");
        assert!(ok, "step {step} must succeed");
    }

    // Save state at step K
    let tmp_path = PathBuf::from(format!("./target/test_resume_step_{split_step}.tmp"));
    let state_k = checkpoint::TrainingState {
        model: Model::from_params(&cfg, &params_part1).expect("model from params"),
        optimizer: opt_part1.clone(),
        loader_state: loader_part1.state(),
        rng_state: ([1u32; 16], [2u32; 16], 3),
        schedule_step: split_step,
        schedule: sched.clone(),
    };
    checkpoint::save_training(&state_k, &tmp_path).expect("save_training");

    // --- Interrupted Run: Part 2 (Resume from step K, run (K+1)..=total_steps) ---
    let loaded_state = checkpoint::load_training(
        &tmp_path,
        &cfg,
        &loader_part1.corpus_hash(),
        DEFAULT_MAX_BYTES,
    ).expect("load_training");

    assert_eq!(loaded_state.schedule_step, split_step);
    assert_eq!(loaded_state.optimizer.step, split_step as u64);

    let model_resumed = loaded_state.model;
    let mut params_resumed = Params::<f32>::from_model(&model_resumed);
    let mut opt_resumed = loaded_state.optimizer;
    let mut loader_resumed = WindowLoader::new(&train_docs, cfg.max_seq_len, 42).expect("loader init");
    loader_resumed.restore(loaded_state.loader_state).expect("restore loader");

    for step in (split_step + 1)..=total_steps {
        let win = loader_resumed.next_window();
        let mut grads = params_resumed.zeros_like();
        let n_denom = win.len().saturating_sub(1) as f32;
        let _ = loss_and_grad(&cfg, &params_resumed, &win, n_denom, &mut grads).expect("loss_and_grad");
        let eta = sched.eta(step as u64);
        let ok = opt_resumed.step_with_clipping(&mut params_resumed, &mut grads, eta, 1.0).expect("step");
        assert!(ok, "step {step} must succeed");
    }

    // Clean up temporary checkpoint
    let _ = fs::remove_file(&tmp_path);

    // --- Bitwise Parity Assertions ---
    // 1. Model parameter data
    assert_eq!(
        params_uninterrupted.data,
        params_resumed.data,
        "resumed model parameters must match uninterrupted run bitwise"
    );

    // 2. Optimizer state (m, v, step)
    assert_eq!(opt_uninterrupted.step, opt_resumed.step, "optimizer step must match");
    assert_eq!(opt_uninterrupted.consecutive_skips, opt_resumed.consecutive_skips);
    for (idx, (m_u, m_r)) in opt_uninterrupted.m.iter().zip(&opt_resumed.m).enumerate() {
        assert_eq!(m_u, m_r, "optimizer moment m[{idx}] must match bitwise");
    }
    for (idx, (v_u, v_r)) in opt_uninterrupted.v.iter().zip(&opt_resumed.v).enumerate() {
        assert_eq!(v_u, v_r, "optimizer moment v[{idx}] must match bitwise");
    }

    // 3. Next window from loaders must be identical
    let next_win_uninterrupted = loader_uninterrupted.next_window();
    let next_win_resumed = loader_resumed.next_window();
    assert_eq!(
        next_win_uninterrupted,
        next_win_resumed,
        "loader resumed stream must match uninterrupted stream"
    );
}

#[test]
fn test_n1_eval_leaves_train_loader_cursor_and_rng_unchanged() {
    let cfg = Config::reference_v2();
    let (train_docs, val_docs) = create_deterministic_docs();

    let mut train_loader = WindowLoader::new(&train_docs, cfg.max_seq_len, 42).expect("train loader");
    let mut val_loader = WindowLoader::new(&val_docs, cfg.max_seq_len, 1337).expect("val loader");

    let initial_train_state = train_loader.state();

    // Advance train loader 3 times
    let _ = train_loader.next_window();
    let _ = train_loader.next_window();
    let _ = train_loader.next_window();
    let train_state_before_eval = train_loader.state();
    assert_ne!(initial_train_state.position, train_state_before_eval.position);

    // Perform multiple evaluation steps with val_loader
    for _ in 0..5 {
        let val_win = val_loader.next_window();
        assert_eq!(val_win.len(), cfg.max_seq_len + 1);
    }

    // Verify train_loader state was completely unaffected by evaluation
    assert_eq!(train_loader.state(), train_state_before_eval);
}

#[test]
fn test_n5_adamw_skip_budget_and_loader_consumption() {
    let cfg = Config::reference_v2();
    let (train_docs, _) = create_deterministic_docs();

    let model = Model::init(&cfg).expect("model init");
    let mut params = Params::<f32>::from_model(&model);
    let mut opt = AdamW::new(&params);
    let mut loader = WindowLoader::new(&train_docs, cfg.max_seq_len, 42).expect("loader init");

    let initial_weights = params.data.clone();
    let mut consecutive_skips = 0usize;

    for step in 1..=5 {
        let _win = loader.next_window();
        let total_consumed = loader.state().epoch * (loader.len() as u64) + loader.state().position;
        assert_eq!(total_consumed, step as u64);

        // Inject non-finite gradients (NaN)
        let mut bad_grads = params.zeros_like();
        bad_grads.data[0][0] = f32::NAN;

        let stepped = opt.step_with_clipping(&mut params, &mut bad_grads, 0.01, 1.0).expect("step_with_clipping");
        assert!(!stepped, "step must be skipped for NaN gradients");

        consecutive_skips = opt.consecutive_skips;
        assert_eq!(consecutive_skips, step);
        // Weights must remain untouched
        assert_eq!(params.data, initial_weights);
    }

    // Verify skip budget threshold
    assert!(consecutive_skips >= 5, "consecutive skips must reach 5");
}
