use sandy_rust::checkpoint::{self, DEFAULT_MAX_BYTES};
use sandy_rust::config::Config;
use sandy_rust::data::WindowLoader;
use sandy_rust::gates;
use sandy_rust::generation::{generate, SamplingParams};
use sandy_rust::model::Model;
use sandy_rust::optimizer::{AdamW, Schedule};
use sandy_rust::tokenizer::{ByteTokenizer, BOS};
use std::{env, fs, path::Path, process::ExitCode, str::FromStr};

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(String::as_str)
}

fn parse_flag<T: FromStr>(args: &[String], name: &str, default: T) -> Result<T, String> {
    match flag(args, name) {
        Some(val) => val.parse::<T>().map_err(|_| format!("invalid value for flag {name}: '{val}'")),
        None => Ok(default),
    }
}

type CorpusSplits = (Vec<Vec<u32>>, Vec<Vec<u32>>);

fn load_or_generate_corpus(
    dataset_path: Option<&str>,
    vocab_size: usize,
) -> Result<CorpusSplits, String> {
    if let Some(path_str) = dataset_path {
        let p = Path::new(path_str);
        if path_str.ends_with(".rtok") {
            let mut shard = sandy_rust::data::BufferedTokenShard::open(p, vocab_size, DEFAULT_MAX_BYTES)
                .map_err(|e| format!("failed to open .rtok shard: {e:?}"))?;
            let mut all_tokens = Vec::new();
            while let Some(tok) = shard.next_token().map_err(|e| format!("shard read error: {e:?}"))? {
                if tok < 256 {
                    all_tokens.push(tok);
                }
            }
            if all_tokens.is_empty() {
                return Err("dataset shard contained no byte tokens < 256".into());
            }
            let mut docs = Vec::new();
            for chunk in all_tokens.chunks(64) {
                docs.push(chunk.to_vec());
            }
            let split_idx = (docs.len() * 4 / 5).max(1);
            let train_docs = docs[..split_idx].to_vec();
            let val_docs = if split_idx < docs.len() {
                docs[split_idx..].to_vec()
            } else {
                train_docs.clone()
            };
            Ok((train_docs, val_docs))
        } else {
            let content = fs::read(p).map_err(|e| format!("failed to read dataset file: {e}"))?;
            let mut docs = Vec::new();
            for line in content.split(|&b| b == b'\n') {
                if !line.is_empty() {
                    docs.push(line.iter().map(|&b| b as u32).collect());
                }
            }
            if docs.is_empty() {
                return Err("dataset file was empty".into());
            }
            let split_idx = (docs.len() * 4 / 5).max(1);
            let train_docs = docs[..split_idx].to_vec();
            let val_docs = if split_idx < docs.len() {
                docs[split_idx..].to_vec()
            } else {
                train_docs.clone()
            };
            Ok((train_docs, val_docs))
        }
    } else {
        // Deterministic multi-document synthetic corpus
        let train_docs = vec![
            (b"Hello world from SandyRust transformer!".to_vec()).into_iter().map(|b| b as u32).collect(),
            (b"Pure Rust autoregressive language model baseline.".to_vec()).into_iter().map(|b| b as u32).collect(),
            (b"The quick brown fox jumps over the lazy dog.".to_vec()).into_iter().map(|b| b as u32).collect(),
            (b"Crouzeix ratio constant verification in FP64.".to_vec()).into_iter().map(|b| b as u32).collect(),
            (b"Zero third-party crate dependencies in R1 std.".to_vec()).into_iter().map(|b| b as u32).collect(),
            (b"Rotary positional embeddings preserve isometric norms.".to_vec()).into_iter().map(|b| b as u32).collect(),
            (b"Attention masking isolates padded future query tokens.".to_vec()).into_iter().map(|b| b as u32).collect(),
            (b"Deterministic state resumption reproduces all moments.".to_vec()).into_iter().map(|b| b as u32).collect(),
        ];
        let val_docs = vec![
            (b"Held-out validation set document number one.".to_vec()).into_iter().map(|b| b as u32).collect(),
            (b"Testing generalisation on unseen evaluation tokens.".to_vec()).into_iter().map(|b| b as u32).collect(),
            (b"Bounded cross-entropy loss reduction over epochs.".to_vec()).into_iter().map(|b| b as u32).collect(),
            (b"Evaluating forward loss on separate loader cursor.".to_vec()).into_iter().map(|b| b as u32).collect(),
        ];
        Ok((train_docs, val_docs))
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let cfg = Config::reference_v2();
    match args.first().map(String::as_str) {
        Some("verify") => {
            let default_gates = "g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14,gx12e,gx13,gx14,gx15";
            let list = match flag(&args, "--gates") {
                Some("all") => default_gates,
                Some("baseline") => "g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14",
                Some(custom) => custom,
                None => default_gates,
            };
            let ids: Vec<&str> = list.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
            let reports = match gates::run(&ids) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("verify error: {e}");
                    return ExitCode::from(2);
                }
            };
            let art_dir = flag(&args, "--artifacts").unwrap_or("artifacts");
            if let Err(e) = fs::create_dir_all(art_dir) {
                eprintln!("failed to create evidence directory '{art_dir}': {e}");
                return ExitCode::FAILURE;
            }

            let mut failed = false;
            let mut acceptance_pass = 0;
            let mut smoke_pass = 0;
            let mut integrity_pass = 0;
            let mut regression_pass = 0;
            let mut not_run_count = 0;

            for r in &reports {
                if let Some(reason) = &r.reason {
                    println!("{:5} [{:<10}] {:<7} ({})", r.gate_id, r.scope, r.status, reason);
                } else {
                    println!("{:5} [{:<10}] {:<7}", r.gate_id, r.scope, r.status);
                }
                match (r.status, r.scope) {
                    ("NOT_RUN", _) => not_run_count += 1,
                    ("PASS", "acceptance") => acceptance_pass += 1,
                    ("PASS", "smoke") => smoke_pass += 1,
                    ("PASS", "integrity") => integrity_pass += 1,
                    ("PASS", "regression") => regression_pass += 1,
                    _ => failed = true,
                }

                // Atomic write and readback verification (EV1, EV7)
                let final_path = format!("{art_dir}/{}.json", r.gate_id);
                let tmp_path = format!("{art_dir}/.{}.tmp", r.gate_id);
                let json_data = r.to_json();
                if let Err(e) = fs::write(&tmp_path, &json_data) {
                    eprintln!("failed to write artifact '{tmp_path}': {e}");
                    return ExitCode::FAILURE;
                }
                let readback = match fs::read_to_string(&tmp_path) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("failed to readback artifact '{tmp_path}': {e}");
                        let _ = fs::remove_file(&tmp_path);
                        return ExitCode::FAILURE;
                    }
                };
                if readback != json_data {
                    eprintln!("readback artifact content mismatch for '{}'", r.gate_id);
                    let _ = fs::remove_file(&tmp_path);
                    return ExitCode::FAILURE;
                }
                if let Err(e) = fs::rename(&tmp_path, &final_path) {
                    eprintln!("failed to publish artifact '{final_path}': {e}");
                    let _ = fs::remove_file(&tmp_path);
                    return ExitCode::FAILURE;
                }
            }

            println!(
                "\nVerification Summary: {} executed (acceptance: {}, integrity: {}, regression: {}, smoke: {}, not_run: {})",
                reports.len(), acceptance_pass, integrity_pass, regression_pass, smoke_pass, not_run_count
            );
            if failed { ExitCode::FAILURE } else { ExitCode::SUCCESS }
        }
        Some("checkpoint") => {
            if let Some(path) = flag(&args, "--init") {
                let m = match Model::init(&cfg) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("model init error: {e:?}");
                        return ExitCode::FAILURE;
                    }
                };
                match checkpoint::save(&m, Path::new(path)) {
                    Ok(()) => println!("saved {} params to {path}", m.param_count()),
                    Err(e) => { eprintln!("save failed: {e:?}"); return ExitCode::FAILURE; }
                }
            } else if let Some(path) = flag(&args, "--inspect") {
                match checkpoint::load(Path::new(path), &cfg, DEFAULT_MAX_BYTES) {
                    Ok(m) => println!("valid checkpoint: {} params, {} layers", m.param_count(), m.layers.len()),
                    Err(e) => { eprintln!("rejected: {e:?}"); return ExitCode::FAILURE; }
                }
            } else {
                eprintln!("usage: checkpoint --init <path> | --inspect <path>");
                return ExitCode::from(2);
            }
            ExitCode::SUCCESS
        }
        Some("generate") => {
            let model = match flag(&args, "--checkpoint") {
                Some(p) => match checkpoint::load(Path::new(p), &cfg, DEFAULT_MAX_BYTES) {
                    Ok(m) => m,
                    Err(e) => { eprintln!("rejected: {e:?}"); return ExitCode::FAILURE; }
                },
                None => {
                    eprintln!("note: random-initialised weights (untrained); output is a smoke test only");
                    match Model::init(&cfg) {
                        Ok(m) => m,
                        Err(e) => {
                            eprintln!("model init error: {e:?}");
                            return ExitCode::FAILURE;
                        }
                    }
                }
            };
            let tok = ByteTokenizer::new();
            let mut prompt = vec![BOS];
            prompt.extend(tok.encode(flag(&args, "--prompt").unwrap_or("Rust")));

            let temp: f32 = match parse_flag(&args, "--temperature", 0.0f32) {
                Ok(v) => v,
                Err(e) => { eprintln!("{e}"); return ExitCode::from(2); }
            };
            let top_k: usize = match parse_flag(&args, "--top-k", usize::MAX) {
                Ok(v) => v,
                Err(e) => { eprintln!("{e}"); return ExitCode::from(2); }
            };
            let top_p: f32 = match parse_flag(&args, "--top-p", 1.0f32) {
                Ok(v) => v,
                Err(e) => { eprintln!("{e}"); return ExitCode::from(2); }
            };
            let max_new_tokens: usize = match parse_flag(&args, "--max-tokens", 16usize) {
                Ok(v) => v,
                Err(e) => { eprintln!("{e}"); return ExitCode::from(2); }
            };
            let seed: u64 = match parse_flag(&args, "--seed", 0u64) {
                Ok(v) => v,
                Err(e) => { eprintln!("{e}"); return ExitCode::from(2); }
            };

            let params = SamplingParams {
                temperature: temp,
                top_k,
                top_p,
                max_new_tokens,
                seed,
                adaptive: None,
            };
            match generate(&model, &prompt, &params) {
                Ok(out) => {
                    println!("{:?}", tok.decode(&out.tokens).unwrap_or_default());
                    ExitCode::SUCCESS
                }
                Err(e) => { eprintln!("generation error: {e:?}"); ExitCode::FAILURE }
            }
        }
        Some("gradcheck") => {
            let m = match Model::init(&cfg) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("model init error: {e:?}");
                    return ExitCode::FAILURE;
                }
            };
            let p_f64 = sandy_rust::autodiff::Params::<f64>::from_model(&m);
            let window = [256u32, 65, 66, 65, 257];
            println!("Running analytic vs finite-difference gradcheck on reference_v2 (f64)...");
            match sandy_rust::autodiff::gradcheck(&cfg, &p_f64, &window, &[1e-5, 1e-6, 1e-7], 1e-5, 1e-3, 1) {
                Ok(rep) => {
                    println!(
                        "Checked {} coordinates. Failures: {}. Max abs err: {:e}, max rel err: {:e}",
                        rep.checked, rep.failures, rep.max_abs_err, rep.max_rel_err
                    );
                    if rep.failures == 0 {
                        println!("PASS: gradcheck verified");
                        ExitCode::SUCCESS
                    } else {
                        eprintln!("FAIL: gradcheck failures detected");
                        ExitCode::FAILURE
                    }
                }
                Err(e) => {
                    eprintln!("gradcheck error: {e:?}");
                    ExitCode::FAILURE
                }
            }
        }
        Some("train") => {
            let steps: usize = match parse_flag(&args, "--steps", 20usize) {
                Ok(v) => v,
                Err(e) => { eprintln!("{e}"); return ExitCode::from(2); }
            };
            let lr: f32 = match parse_flag(&args, "--lr", 0.01f32) {
                Ok(v) => v,
                Err(e) => { eprintln!("{e}"); return ExitCode::from(2); }
            };
            let val_every: usize = match parse_flag(&args, "--val-every", (steps / 5).max(1)) {
                Ok(v) => v,
                Err(e) => { eprintln!("{e}"); return ExitCode::from(2); }
            };

            let (train_docs, val_docs) = match load_or_generate_corpus(flag(&args, "--dataset"), cfg.vocab_size) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("dataset error: {e}");
                    return ExitCode::FAILURE;
                }
            };

            let mut train_loader = match WindowLoader::new(&train_docs, cfg.max_seq_len, 42) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("train loader error: {e:?}");
                    return ExitCode::FAILURE;
                }
            };
            let mut val_loader = match WindowLoader::new(&val_docs, cfg.max_seq_len, 1337) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("val loader error: {e:?}");
                    return ExitCode::FAILURE;
                }
            };

            let mut m = match Model::init(&cfg) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("model init error: {e:?}");
                    return ExitCode::FAILURE;
                }
            };
            let mut p = sandy_rust::autodiff::Params::<f32>::from_model(&m);
            let mut opt = AdamW::new(&p);
            let mut start_step = 1usize;

            if let Some(resume_path) = flag(&args, "--resume") {
                let state = match checkpoint::load_training(Path::new(resume_path), &cfg, &train_loader.corpus_hash(), DEFAULT_MAX_BYTES) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("resume error: {e:?}");
                        return ExitCode::FAILURE;
                    }
                };
                m = state.model;
                p = sandy_rust::autodiff::Params::<f32>::from_model(&m);
                opt = state.optimizer;
                if let Err(e) = train_loader.restore(state.loader_state) {
                    eprintln!("loader restore error: {e:?}");
                    return ExitCode::FAILURE;
                }
                start_step = state.schedule_step + 1;
                println!("Resumed training state from '{resume_path}' at step {start_step}");
            }

            println!("Starting bounded train: steps={steps}, lr={lr}, start_step={start_step}...");
            let sched = Schedule { eta_max: lr, eta_min: lr * 0.1, warmup: (steps / 5).max(1) as u64, max_steps: steps as u64 };

            for step in start_step..=steps {
                let train_window = train_loader.next_window();
                let mut grads = p.zeros_like();
                let n_denom = train_window.len().saturating_sub(1) as f32;
                let o = match sandy_rust::autodiff::loss_and_grad(&cfg, &p, &train_window, n_denom, &mut grads) {
                    Ok(o) => o,
                    Err(e) => { eprintln!("training step {step} error: {e:?}"); return ExitCode::FAILURE; }
                };
                let train_loss = o.loss_sum / o.n_valid as f32;
                let eta = sched.eta(step as u64);
                match opt.step_with_clipping(&mut p, &mut grads, eta, 1.0) {
                    Ok(true) => {},
                    Ok(false) => {
                        eprintln!("warning: step {step} skipped due to non-finite gradients (consecutive skips: {})", opt.consecutive_skips);
                        if opt.consecutive_skips >= 5 {
                            eprintln!("aborted training: exceeded 5 consecutive gradient skips");
                            return ExitCode::FAILURE;
                        }
                    }
                    Err(e) => {
                        eprintln!("optimizer step {step} error: {e:?}");
                        return ExitCode::FAILURE;
                    }
                }
                if step % val_every == 0 || step == steps {
                    let val_window = val_loader.next_window();
                    let val_out = match sandy_rust::autodiff::forward_loss(&cfg, &p, &val_window) {
                        Ok(l) => l,
                        Err(e) => { eprintln!("eval error: {e:?}"); return ExitCode::FAILURE; }
                    };
                    let val_loss = val_out.loss_sum / val_out.n_valid as f32;
                    println!("step {step:3}/{steps} | train_loss: {train_loss:.4} | val_loss: {val_loss:.4} | lr: {eta:.6}");
                }
            }

            if let Some(state_path) = flag(&args, "--save-state") {
                let updated_model = match Model::from_params(&cfg, &p) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("model from params error: {e:?}");
                        return ExitCode::FAILURE;
                    }
                };
                let state = sandy_rust::checkpoint::TrainingState {
                    model: updated_model,
                    optimizer: opt.clone(),
                    loader_state: train_loader.state(),
                    rng_state: ([0u32; 16], [0u32; 16], 0),
                    schedule_step: steps,
                };
                if let Err(e) = checkpoint::save_training(&state, Path::new(state_path)) {
                    eprintln!("save-state error: {e:?}");
                    return ExitCode::FAILURE;
                }
                println!("saved training state to {state_path}");
            }

            if let Some(ckpt_path) = flag(&args, "--checkpoint") {
                let updated_model = match Model::from_params(&cfg, &p) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("model from params error: {e:?}");
                        return ExitCode::FAILURE;
                    }
                };
                if let Err(e) = checkpoint::save(&updated_model, Path::new(ckpt_path)) {
                    eprintln!("checkpoint save error: {e:?}");
                    return ExitCode::FAILURE;
                }
                println!("saved final checkpoint to {ckpt_path}");
            }
            ExitCode::SUCCESS
        }
        Some("inspect") => {
            let prompt = flag(&args, "--prompt").unwrap_or("Hello");
            let mode = flag(&args, "--mode").unwrap_or("record");
            let tok = ByteTokenizer::new();
            let mut prompt_tokens = vec![BOS];
            prompt_tokens.extend(tok.encode_bytes(prompt.as_bytes()));
            let m = match Model::init(&cfg) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("model init error: {e:?}");
                    return ExitCode::FAILURE;
                }
            };
            match mode {
                "record" => {
                    let mut sink = match sandy_rust::hooks::ActivationSink::new(sandy_rust::hooks::DEFAULT_MAX_EXPORT_BYTES) {
                        Ok(s) => s,
                        Err(e) => {
                            eprintln!("activation sink error: {e:?}");
                            return ExitCode::FAILURE;
                        }
                    };
                    let ctx = sandy_rust::hooks::HookContext::new().with_record(
                        sandy_rust::hooks::HookId::global(sandy_rust::hooks::HookSite::EmbedOut),
                        sandy_rust::hooks::RecordTiming::PostEdit,
                    );
                    let out = match m.forward_with_hooks(&prompt_tokens, None, Some(&ctx), Some(&mut sink)) {
                        Ok(o) => o,
                        Err(e) => {
                            eprintln!("forward error: {e:?}");
                            return ExitCode::FAILURE;
                        }
                    };
                    println!("Recorded {} activation tensors; logits shape: {:?}", sink.records.len(), out.shape());
                }
                "steer" => {
                    let alpha = match parse_flag(&args, "--alpha", 1.0f32) {
                        Ok(v) => v,
                        Err(e) => {
                            eprintln!("{e}");
                            return ExitCode::from(2);
                        }
                    };
                    let steer_op = sandy_rust::hooks::EditOp::Steer {
                        hook: sandy_rust::hooks::HookId::layer(0, sandy_rust::hooks::HookSite::ResidPre),
                        direction: vec![0.1f32; 128],
                        alpha,
                        positions: sandy_rust::hooks::SteerPosition::AllPositions,
                        direction_id: "cli_steer".to_string(),
                    };
                    let ctx = sandy_rust::hooks::HookContext::new().with_edit(steer_op);
                    let out = match m.forward_with_hooks(&prompt_tokens, None, Some(&ctx), None) {
                        Ok(o) => o,
                        Err(e) => {
                            eprintln!("forward error: {e:?}");
                            return ExitCode::FAILURE;
                        }
                    };
                    println!("Steered forward complete with alpha={alpha}; logits shape: {:?}", out.shape());
                }
                other => {
                    eprintln!("unknown inspect mode: '{other}' (expected 'record' or 'steer')");
                    return ExitCode::from(2);
                }
            }
            ExitCode::SUCCESS
        }
        Some("eval") => {
            let sub = args.get(1).map(String::as_str);
            match sub {
                Some("flip") => {
                    let task = flag(&args, "--task").unwrap_or("all");
                    let split = flag(&args, "--split").unwrap_or("test");
                    println!("Running Paired Perturbation & Flip-Rate Evaluation (task: {task}, split: {split})...");

                    let mut families = Vec::new();
                    if task == "all" || task == "kv_lookup" {
                        families.extend(sandy_rust::eval::generate_kv_lookup(100, 42));
                    }
                    if task == "all" || task == "copy_marker" {
                        families.extend(sandy_rust::eval::generate_copy_marker(100, 43));
                    }
                    if task == "all" || task == "digit_comparison" {
                        families.extend(sandy_rust::eval::generate_digit_comparison(100, 44));
                    }
                    if task == "all" || task == "symbol_count" {
                        families.extend(sandy_rust::eval::generate_symbol_count(100, 45));
                    }

                    let (train_f, dev_f, test_f) = match sandy_rust::eval::assign_splits(&families, 1337) {
                        Ok(s) => s,
                        Err(e) => {
                            eprintln!("split assignment error: {e}");
                            return ExitCode::FAILURE;
                        }
                    };

                    let target_families = match split {
                        "train" => train_f,
                        "dev" => dev_f,
                        "test" => test_f,
                        _ => {
                            eprintln!("invalid split '{split}'; expected train, dev, or test");
                            return ExitCode::from(2);
                        }
                    };

                    let pairs: Vec<&sandy_rust::eval::Pair> = target_families.iter().flat_map(|f| &f.pairs).collect();
                    println!("Evaluated {} families, {} pairs on split '{}'", target_families.len(), pairs.len(), split);

                    let oracle_preds: Vec<(&sandy_rust::eval::Pair, sandy_rust::eval::PredictionResult)> = pairs.iter().map(|p| {
                        (*p, sandy_rust::eval::PredictionResult {
                            model_base_answer: p.oracle_base_answer.clone(),
                            model_variant_answer: p.oracle_variant_answer.clone(),
                        })
                    }).collect();

                    let (flip_mf, flip_mnf, selectivity, base_acc, var_acc) = match sandy_rust::eval::evaluate_pairs(&oracle_preds) {
                        Ok(m) => m,
                        Err(e) => {
                            eprintln!("metrics computation error: {e}");
                            return ExitCode::FAILURE;
                        }
                    };

                    let ci = match sandy_rust::eval::family_clustered_bootstrap(&target_families, &oracle_preds, 2000, 20261007) {
                        Ok(c) => c,
                        Err(e) => {
                            eprintln!("bootstrap error: {e}");
                            return ExitCode::FAILURE;
                        }
                    };

                    let baselines = sandy_rust::eval::evaluate_trivial_baselines(&pairs);

                    println!("--- Metrics (Oracle Reference) ---");
                    println!("Base Accuracy:       {:.2}%", base_acc * 100.0);
                    println!("Variant Accuracy:    {:.2}%", var_acc * 100.0);
                    println!("Must-Flip Flip Rate: {:.2}%", flip_mf * 100.0);
                    println!("Must-Not-Flip Rate:  {:.2}%", flip_mnf * 100.0);
                    println!("Selectivity:         {:.4}", selectivity);
                    println!("Selectivity 95% CI:  [{:.4}, {:.4}]", ci.lower, ci.upper);
                    println!("--- Trivial Baselines Selectivity ---");
                    println!("Always Flip:         {:.4}", baselines.always_flip_selectivity);
                    println!("Always Same:         {:.4}", baselines.always_same_selectivity);
                    println!("Random:              {:.4}", baselines.random_selectivity);

                    ExitCode::SUCCESS
                }
                _ => {
                    eprintln!("usage: sandy_rust eval flip [--task all|kv_lookup|copy_marker|digit_comparison|symbol_count] [--split train|dev|test]");
                    ExitCode::from(2)
                }
            }
        }
        _ => {
            eprintln!("usage: sandy_rust <command> [options]");
            eprintln!("commands:");
            eprintln!("  verify      [--gates <list>|all|baseline] [--artifacts <dir>]");
            eprintln!("  train       [--steps n] [--lr f] [--dataset path] [--save-state path] [--resume path] [--checkpoint path] [--val-every n]");
            eprintln!("  gradcheck");
            eprintln!("  checkpoint  --init <path> | --inspect <path>");
            eprintln!("  generate    [--prompt s] [--max-tokens n] [--temperature t] [--top-k k] [--top-p p] [--seed n] [--checkpoint path]");
            eprintln!("  inspect     [--prompt s] [--mode record|steer] [--alpha f]");
            eprintln!("  eval flip   [--task <name>] [--split <name>]");
            ExitCode::from(2)
        }
    }
}
