use sandy_rust::checkpoint::{self, DEFAULT_MAX_BYTES};
use sandy_rust::config::Config;
use sandy_rust::generation::{generate, SamplingParams};
use sandy_rust::model::Model;
use sandy_rust::optimizer::{AdamW, Schedule};
use sandy_rust::tokenizer::{ByteTokenizer, BOS};
use sandy_rust::gates;
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

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let cfg = Config::reference_v2();
    match args.first().map(String::as_str) {
        Some("verify") => {
            let list = flag(&args, "--gates").unwrap_or("g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14");
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
                    println!("{:4} [{:<10}] {:<7} ({})", r.gate_id, r.scope, r.status, reason);
                } else {
                    println!("{:4} [{:<10}] {:<7}", r.gate_id, r.scope, r.status);
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
                let m = Model::init(&cfg).expect("valid baseline config");
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
                    Model::init(&cfg).expect("valid baseline config")
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
            let m = Model::init(&cfg).expect("model init");
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
            println!("Starting bounded train smoke test: steps={steps}, lr={lr}...");
            let m = Model::init(&cfg).expect("model init");
            let mut p = sandy_rust::autodiff::Params::<f32>::from_model(&m);
            let mut opt = AdamW::new(&p);
            let sched = Schedule { eta_max: lr, eta_min: lr * 0.1, warmup: (steps / 5).max(1) as u64, max_steps: steps as u64 };
            let train_window = [256u32, 65, 66, 67, 257];
            let val_window = [256u32, 68, 69, 70, 257];

            for step in 1..=steps {
                let mut grads = p.zeros_like();
                let o = match sandy_rust::autodiff::loss_and_grad(&cfg, &p, &train_window, train_window.len().saturating_sub(1) as f32, &mut grads) {
                    Ok(o) => o,
                    Err(e) => { eprintln!("training step {step} error: {e:?}"); return ExitCode::FAILURE; }
                };
                let train_loss = o.loss_sum / o.n_valid as f32;
                let eta = sched.eta(step as u64);
                if let Err(e) = opt.step_with_clipping(&mut p, &mut grads, eta, 1.0) {
                    eprintln!("optimizer step {step} error: {e:?}");
                    return ExitCode::FAILURE;
                }
                if step % (steps / 5).max(1) == 0 || step == steps {
                    let val_out = match sandy_rust::autodiff::forward_loss(&cfg, &p, &val_window) {
                        Ok(l) => l,
                        Err(e) => { eprintln!("eval error: {e:?}"); return ExitCode::FAILURE; }
                    };
                    let val_loss = val_out.loss_sum / val_out.n_valid as f32;
                    println!("step {step:3}/{steps} | train_loss: {train_loss:.4} | val_loss: {val_loss:.4} | lr: {eta:.6}");
                }
            }
            if let Some(ckpt_path) = flag(&args, "--checkpoint") {
                let updated_model = Model::from_params(&cfg, &p).expect("model from params");
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
            let m = Model::init(&cfg).expect("model init");
            match mode {
                "record" => {
                    let mut sink = sandy_rust::hooks::ActivationSink::new(sandy_rust::hooks::DEFAULT_MAX_EXPORT_BYTES).unwrap();
                    let ctx = sandy_rust::hooks::HookContext::new().with_record(
                        sandy_rust::hooks::HookId::global(sandy_rust::hooks::HookSite::EmbedOut),
                        sandy_rust::hooks::RecordTiming::PostEdit,
                    );
                    let out = m.forward_with_hooks(&prompt_tokens, None, Some(&ctx), Some(&mut sink)).expect("forward");
                    println!("Recorded {} activation tensors; logits shape: {:?}", sink.records.len(), out.shape());
                }
                "steer" => {
                    let alpha = parse_flag(&args, "--alpha", 1.0f32).unwrap_or(1.0);
                    let steer_op = sandy_rust::hooks::EditOp::Steer {
                        hook: sandy_rust::hooks::HookId::layer(0, sandy_rust::hooks::HookSite::ResidPre),
                        direction: vec![0.1f32; 128],
                        alpha,
                        positions: sandy_rust::hooks::SteerPosition::AllPositions,
                        direction_id: "cli_steer".to_string(),
                    };
                    let ctx = sandy_rust::hooks::HookContext::new().with_edit(steer_op);
                    let out = m.forward_with_hooks(&prompt_tokens, None, Some(&ctx), None).expect("forward");
                    println!("Steered forward complete with alpha={alpha}; logits shape: {:?}", out.shape());
                }
                other => {
                    eprintln!("unknown inspect mode: '{other}' (expected 'record' or 'steer')");
                    return ExitCode::from(2);
                }
            }
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!("usage: sandy_rust <command> [options]");
            eprintln!("commands:");
            eprintln!("  verify      [--gates g1,g2,...] [--artifacts <dir>]");
            eprintln!("  train       [--steps n] [--lr f] [--checkpoint path]");
            eprintln!("  gradcheck");
            eprintln!("  checkpoint  --init <path> | --inspect <path>");
            eprintln!("  generate    [--prompt s] [--max-tokens n] [--temperature t] [--top-k k] [--top-p p] [--seed n] [--checkpoint path]");
            eprintln!("  inspect     [--prompt s] [--mode record|steer] [--alpha f]");
            ExitCode::from(2)
        }
    }
}
