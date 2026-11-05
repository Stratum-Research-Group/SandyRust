use sandy_rust::checkpoint::{self, DEFAULT_MAX_BYTES};
use sandy_rust::config::Config;
use sandy_rust::data::WindowLoader;
use sandy_rust::gates;
use sandy_rust::generation::{generate, SamplingParams};
use sandy_rust::model::Model;
use sandy_rust::optimizer::{AdamW, Schedule};
use sandy_rust::rng::Rng;
use sandy_rust::tokenizer::{ByteTokenizer, BOS};
use std::{env, fs, path::Path, process::ExitCode, str::FromStr};

fn flag<'a>(args: &'a [String], name: &str) -> Result<Option<&'a str>, String> {
    if let Some(pos) = args.iter().position(|a| a == name) {
        if let Some(val) = args.get(pos + 1) {
            if val.starts_with("--") {
                return Err(format!("missing value for flag '{name}'"));
            }
            Ok(Some(val.as_str()))
        } else {
            Err(format!("missing value for flag '{name}'"))
        }
    } else {
        Ok(None)
    }
}

fn parse_flag<T: FromStr>(args: &[String], name: &str, default: T) -> Result<T, String> {
    match flag(args, name)? {
        Some(val) => val.parse::<T>().map_err(|_| format!("invalid value for flag {name}: '{val}'")),
        None => Ok(default),
    }
}

const MAX_DATASET_BYTES: u64 = 10 * 1024 * 1024; // 10 MiB limit (Decision D5)

type CorpusSplits = (Vec<Vec<u32>>, Vec<Vec<u32>>);

fn load_or_generate_corpus(
    dataset_path: Option<&str>,
    vocab_size: usize,
) -> Result<CorpusSplits, String> {
    if let Some(path_str) = dataset_path {
        let p = Path::new(path_str);
        if path_str.ends_with(".rtok") {
            let mut shard = sandy_rust::data::BufferedTokenShard::open(p, vocab_size, MAX_DATASET_BYTES)
                .map_err(|e| format!("failed to open .rtok shard: {e:?}"))?;
            let mut docs = Vec::new();
            let mut current_doc = Vec::new();
            while let Some(tok) = shard.next_token().map_err(|e| format!("shard read error: {e:?}"))? {
                if tok < 256 {
                    current_doc.push(tok);
                } else if tok == 257 && !current_doc.is_empty() {
                    docs.push(current_doc);
                    current_doc = Vec::new();
                }
            }
            if !current_doc.is_empty() {
                docs.push(current_doc);
            }
            if docs.is_empty() {
                return Err("dataset shard contained no valid tokens".into());
            }
            if docs.len() < 2 {
                return Err("corpus must contain at least 2 documents for disjoint train/val splits".into());
            }
            let split_idx = (docs.len() * 4 / 5).clamp(1, docs.len() - 1);
            let train_docs = docs[..split_idx].to_vec();
            let val_docs = docs[split_idx..].to_vec();
            Ok((train_docs, val_docs))
        } else {
            let meta = fs::metadata(p).map_err(|e| format!("failed to stat dataset file: {e}"))?;
            if meta.len() > MAX_DATASET_BYTES {
                return Err(format!("dataset file exceeds 10 MiB budget ({} bytes)", meta.len()));
            }
            let content = fs::read(p).map_err(|e| format!("failed to read dataset file: {e}"))?;
            let mut docs = Vec::new();
            for line in content.split(|&b| b == b'\n') {
                if !line.is_empty() {
                    docs.push(line.iter().map(|&b| b as u32).collect());
                }
            }
            if docs.len() < 2 {
                return Err("corpus must contain at least 2 documents for disjoint train/val splits".into());
            }
            let split_idx = (docs.len() * 4 / 5).clamp(1, docs.len() - 1);
            let train_docs = docs[..split_idx].to_vec();
            let val_docs = docs[split_idx..].to_vec();
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
            let default_gates = "g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14,gx1,gx2,gx3,gx4,gx5,gx6,gx7,gx8,gx9,gx10,gx11,gx12e,gx13,gx14,gx15,gx16,gx17";
            let gates_flag = match flag(&args, "--gates") {
                Ok(v) => v,
                Err(e) => { eprintln!("{e}"); return ExitCode::from(2); }
            };
            let list = match gates_flag {
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
            let art_dir = match flag(&args, "--artifacts") {
                Ok(Some(d)) => d,
                Ok(None) => "artifacts",
                Err(e) => { eprintln!("{e}"); return ExitCode::from(2); }
            };
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
        Some("import") => {
            let model_dir = match flag(&args, "--model-dir") {
                Ok(Some(d)) => d,
                Ok(None) => {
                    eprintln!("missing required flag --model-dir");
                    return ExitCode::from(2);
                }
                Err(e) => {
                    eprintln!("{e}");
                    return ExitCode::from(2);
                }
            };
            let out_path_str = match flag(&args, "--out") {
                Ok(Some(o)) => o,
                Ok(None) => {
                    eprintln!("missing required flag --out");
                    return ExitCode::from(2);
                }
                Err(e) => {
                    eprintln!("{e}");
                    return ExitCode::from(2);
                }
            };
            let out_path = Path::new(out_path_str);
            if out_path.exists() {
                eprintln!("error: refusing to overwrite existing file '{}'", out_path.display());
                return ExitCode::FAILURE;
            }

            println!("Importing model from '{model_dir}'...");
            let (model, manifest) = match sandy_rust::import::import_model(Path::new(model_dir), None) {
                Ok((m, mf)) => (m, mf),
                Err(e) => {
                    eprintln!("import error: {e:?}");
                    return ExitCode::FAILURE;
                }
            };

            let manifest_text = manifest.canonical_text();
            let manifest_path = out_path.with_extension("manifest");
            if manifest_path.exists() {
                eprintln!("error: refusing to overwrite existing manifest file '{}'", manifest_path.display());
                return ExitCode::FAILURE;
            }

            if let Err(e) = sandy_rust::import::save_rimp(out_path, &model, manifest.fingerprint(), [0u8; 32]) {
                eprintln!("failed to save RIMP checkpoint: {e:?}");
                return ExitCode::FAILURE;
            }
            if let Err(e) = fs::write(&manifest_path, &manifest_text) {
                eprintln!("failed to write manifest: {e}");
                return ExitCode::FAILURE;
            }

            println!("Successfully imported model: {} params", model.param_count());
            println!("  RIMP checkpoint: {}", out_path.display());
            println!("  Manifest:        {}", manifest_path.display());
            println!("  SPDX License:    {}", manifest.source_license_spdx);
            ExitCode::SUCCESS
        }
        Some("import-inspect") => {
            let model_dir = match flag(&args, "--model-dir") {
                Ok(Some(d)) => d,
                Ok(None) => {
                    eprintln!("missing required flag --model-dir");
                    return ExitCode::from(2);
                }
                Err(e) => {
                    eprintln!("{e}");
                    return ExitCode::from(2);
                }
            };
            let dir_path = Path::new(model_dir);
            let sf_path = dir_path.join("model.safetensors");
            if !sf_path.exists() {
                eprintln!("error: 'model.safetensors' not found in '{}'", dir_path.display());
                return ExitCode::FAILURE;
            }

            let file_bytes = match fs::read(&sf_path) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("failed to read '{}': {e}", sf_path.display());
                    return ExitCode::FAILURE;
                }
            };

            let header = match sandy_rust::import::parse_safetensors_header(&file_bytes) {
                Ok(h) => h,
                Err(e) => {
                    eprintln!("failed to parse safetensors header: {e:?}");
                    return ExitCode::FAILURE;
                }
            };

            let cfg_path = dir_path.join("config.json");
            let donor_cfg = if cfg_path.exists() {
                match fs::read(&cfg_path) {
                    Ok(b) => sandy_rust::import::parse_donor_config(&b, None).ok(),
                    Err(_) => None,
                }
            } else {
                None
            };

            println!("Safetensors Header Inspection for '{}':", dir_path.display());
            if let Some(ref dc) = donor_cfg {
                println!("  Hidden Size (d_model): {}", dc.d_model);
                println!("  Layers:                {}", dc.n_layers);
                println!("  Heads:                 {} q / {} kv", dc.n_q_heads, dc.n_kv_heads);
                println!("  Intermediate (d_ff):   {}", dc.d_ff);
                println!("  Vocab Size:            {}", dc.vocab_size);
                println!("  Tied Embeddings:       {}", dc.tied_embeddings);
                println!("  RoPE Theta:            {}", dc.rope_theta);
            }
            if let Some(ref meta) = header.metadata {
                println!("  Metadata Entries:  {}", meta.len());
                for (k, v) in meta {
                    println!("    {k}: {v}");
                }
            }

            println!("\nTensors ({} total):", header.tensors.len());
            println!("{:<45} {:<6} {:<20} {:>12}", "Tensor Name", "Dtype", "Shape", "Bytes");
            println!("{:-<45} {:-<6} {:-<20} {:-<12}", "", "", "", "");
            let mut total_bytes = 0usize;
            for t in &header.tensors {
                let bytes = t.data_offsets.1 - t.data_offsets.0;
                total_bytes += bytes;
                let shape_str = format!("{:?}", t.shape);
                println!("{:<45} {:<6?} {:<20} {:>12}", t.name, t.dtype, shape_str, bytes);
            }
            println!("{:-<45} {:-<6} {:-<20} {:-<12}", "", "", "", "");
            println!("{:<45} {:<6} {:<20} {:>12}", "Total Payload", "", "", total_bytes);

            ExitCode::SUCCESS
        }
        Some("import-parity") => {
            let import_path = match flag(&args, "--import") {
                Ok(Some(p)) => p,
                Ok(None) => {
                    eprintln!("missing required flag --import <file.rimp>");
                    return ExitCode::from(2);
                }
                Err(e) => { eprintln!("{e}"); return ExitCode::from(2); }
            };
            let fixture_path = match flag(&args, "--fixture") {
                Ok(Some(p)) => p,
                Ok(None) => {
                    eprintln!("missing required flag --fixture <file.json>");
                    return ExitCode::from(2);
                }
                Err(e) => { eprintln!("{e}"); return ExitCode::from(2); }
            };
            let atol: f32 = match parse_flag(&args, "--atol", 1e-3f32) {
                Ok(v) => v,
                Err(e) => { eprintln!("{e}"); return ExitCode::from(2); }
            };

            let model = match sandy_rust::import::load_rimp(Path::new(import_path), None, sandy_rust::import::MAX_RIMP_FILE_CAP) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("failed to load RIMP model: {e:?}");
                    return ExitCode::FAILURE;
                }
            };

            let fixture_bytes = match fs::read(fixture_path) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("failed to read fixture file: {e}");
                    return ExitCode::FAILURE;
                }
            };

            let report = match sandy_rust::import::run_parity_check(&model, &fixture_bytes, atol) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("parity check execution failed: {e:?}");
                    return ExitCode::FAILURE;
                }
            };

            println!("--- Extension X17 Parity Check Report ---");
            println!("Donor Repo:                  {}", report.donor_repo_id);
            println!("Donor Commit:                {}", report.donor_commit);
            println!("Test Cases Evaluated:        {}", report.num_cases_evaluated);
            println!("Max Absolute Logit Diff:     {:e}", report.max_abs_logit_diff);
            println!("Tolerance (atol):            {:e}", report.atol);
            println!("Top-1 Agreement:             {}/{}", report.top1_matches, report.total_top1_expected);
            println!("Greedy Continuation Matches: {}/{}", report.greedy_continuation_matches, report.total_greedy_expected);
            println!("Status:                      {}", if report.passed { "PASS" } else { "FAIL" });

            if let Some(ref reason) = report.failure_reason {
                eprintln!("Failure reason: {reason}");
            }

            if report.passed {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Some("checkpoint") => {
            let init_path = match flag(&args, "--init") {
                Ok(v) => v,
                Err(e) => { eprintln!("{e}"); return ExitCode::from(2); }
            };
            let inspect_path = match flag(&args, "--inspect") {
                Ok(v) => v,
                Err(e) => { eprintln!("{e}"); return ExitCode::from(2); }
            };
            if let Some(path) = init_path {
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
            } else if let Some(path) = inspect_path {
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
            let import_flag = match flag(&args, "--import") {
                Ok(v) => v,
                Err(e) => { eprintln!("{e}"); return ExitCode::from(2); }
            };
            let ckpt_flag = match flag(&args, "--checkpoint") {
                Ok(v) => v,
                Err(e) => { eprintln!("{e}"); return ExitCode::from(2); }
            };

            let (model, bpe_opt) = if let Some(rimp_path) = import_flag {
                let m = match sandy_rust::import::load_rimp(Path::new(rimp_path), None, sandy_rust::import::MAX_RIMP_FILE_CAP) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("import load rejected: {e:?}");
                        return ExitCode::FAILURE;
                    }
                };
                let tok_path_flag = match flag(&args, "--tokenizer") {
                    Ok(v) => v,
                    Err(e) => { eprintln!("{e}"); return ExitCode::from(2); }
                };
                let bpe = if let Some(tp) = tok_path_flag {
                    match sandy_rust::import::BpeTokenizer::from_file(Path::new(tp)) {
                        Ok(b) => Some(b),
                        Err(e) => {
                            eprintln!("warning: failed to load tokenizer from {tp}: {e:?}");
                            None
                        }
                    }
                } else {
                    let rimp_p = Path::new(rimp_path);
                    let candidate = rimp_p.parent().unwrap_or(Path::new(".")).join("tokenizer.json");
                    if candidate.exists() {
                        sandy_rust::import::BpeTokenizer::from_file(&candidate).ok()
                    } else {
                        None
                    }
                };
                (m, bpe)
            } else {
                let m = match ckpt_flag {
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
                (m, None)
            };

            let prompt_ids_flag = match flag(&args, "--prompt-ids") {
                Ok(v) => v,
                Err(e) => { eprintln!("{e}"); return ExitCode::from(2); }
            };
            let prompt = if let Some(ids_str) = prompt_ids_flag {
                let mut ids = Vec::new();
                for part in ids_str.split(',') {
                    let id = match part.trim().parse::<u32>() {
                        Ok(i) => i,
                        Err(_) => {
                            eprintln!("invalid token id in --prompt-ids: '{part}'");
                            return ExitCode::from(2);
                        }
                    };
                    ids.push(id);
                }
                ids
            } else {
                let prompt_text = match flag(&args, "--prompt") {
                    Ok(Some(s)) => s,
                    Ok(None) => "Rust",
                    Err(e) => { eprintln!("{e}"); return ExitCode::from(2); }
                };
                if let Some(ref bpe) = bpe_opt {
                    let mut p = Vec::new();
                    if let Some(bos) = bpe.bos_id() {
                        p.push(bos);
                    }
                    p.extend(bpe.encode(prompt_text, true));
                    p
                } else {
                    let tok = ByteTokenizer::new();
                    let mut p = vec![BOS];
                    p.extend(tok.encode(prompt_text));
                    p
                }
            };

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
                    if let Some(ref bpe) = bpe_opt {
                        println!("{}", bpe.decode_lossy(&out.tokens));
                    } else {
                        let tok = ByteTokenizer::new();
                        println!("{:?}", tok.decode(&out.tokens).unwrap_or_default());
                    }
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
            if steps == 0 {
                eprintln!("--steps must be > 0");
                return ExitCode::from(2);
            }
            let lr: f32 = match parse_flag(&args, "--lr", 0.01f32) {
                Ok(v) => v,
                Err(e) => { eprintln!("{e}"); return ExitCode::from(2); }
            };
            if !lr.is_finite() || lr <= 0.0 {
                eprintln!("--lr must be positive and finite");
                return ExitCode::from(2);
            }
            let stop_after: Option<usize> = match flag(&args, "--stop-after") {
                Ok(Some(v)) => match v.parse::<usize>() {
                    Ok(0) => { eprintln!("--stop-after must be > 0"); return ExitCode::from(2); }
                    Ok(k) => Some(k),
                    Err(e) => { eprintln!("invalid value for --stop-after: '{v}': {e}"); return ExitCode::from(2); }
                },
                Ok(None) => None,
                Err(e) => { eprintln!("{e}"); return ExitCode::from(2); }
            };
            let target_step = stop_after.unwrap_or(steps);

            let val_every: usize = match parse_flag(&args, "--val-every", (steps / 5).max(1)) {
                Ok(v) => v,
                Err(e) => { eprintln!("{e}"); return ExitCode::from(2); }
            };
            if val_every == 0 {
                eprintln!("--val-every must be > 0");
                return ExitCode::from(2);
            }

            let dataset_arg = match flag(&args, "--dataset") {
                Ok(v) => v,
                Err(e) => { eprintln!("{e}"); return ExitCode::from(2); }
            };
            let (train_docs, val_docs) = match load_or_generate_corpus(dataset_arg, cfg.vocab_size) {
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
            let val_loader = match WindowLoader::new(&val_docs, cfg.max_seq_len, 1337) {
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
            let mut train_rng = Rng::new(cfg.seed);
            let mut start_step = 1usize;
            let mut sched = Schedule { eta_max: lr, eta_min: lr * 0.1, warmup: (steps / 5).max(1) as u64, max_steps: steps as u64 };

            let resume_arg = match flag(&args, "--resume") {
                Ok(v) => v,
                Err(e) => { eprintln!("{e}"); return ExitCode::from(2); }
            };
            if let Some(resume_path) = resume_arg {
                let state = match checkpoint::load_training(Path::new(resume_path), &cfg, &train_loader.corpus_hash(), DEFAULT_MAX_BYTES) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("resume error: {e:?}");
                        return ExitCode::FAILURE;
                    }
                };
                if target_step <= state.schedule_step {
                    eprintln!("target step ({target_step}) must be greater than resumed schedule step ({})", state.schedule_step);
                    return ExitCode::from(2);
                }
                m = state.model;
                p = sandy_rust::autodiff::Params::<f32>::from_model(&m);
                opt = state.optimizer;
                sched = state.schedule;
                train_rng = Rng::from_state(state.rng_state.0, state.rng_state.1, state.rng_state.2);
                if let Err(e) = train_loader.restore(state.loader_state) {
                    eprintln!("loader restore error: {e:?}");
                    return ExitCode::FAILURE;
                }
                start_step = state.schedule_step + 1;
                println!("Resumed training state from '{resume_path}' at step {start_step}");
            }

            println!("Starting bounded train: steps={steps}, target_step={target_step}, lr={lr}, start_step={start_step}...");

            for step in start_step..=target_step {
                let train_window = train_loader.next_window();
                let mut grads = p.zeros_like();
                let n_valid_targets = train_window[..train_window.len().saturating_sub(1)]
                    .iter()
                    .zip(&train_window[1..])
                    .filter(|(&tok, &tgt)| tok != cfg.pad_id && tgt != cfg.pad_id)
                    .count();
                let n_denom = (n_valid_targets.max(1)) as f32;
                let o = match sandy_rust::autodiff::loss_and_grad(&cfg, &p, &train_window, n_denom, &mut grads) {
                    Ok(o) => o,
                    Err(e) => {
                        eprintln!("warning: step {step} loss error: {e:?}, skipping window");
                        opt.consecutive_skips += 1;
                        if opt.consecutive_skips >= 5 {
                            eprintln!("aborted training: exceeded 5 consecutive skips");
                            return ExitCode::FAILURE;
                        }
                        continue;
                    }
                };
                let train_loss = o.loss_sum / o.n_valid as f32;
                let eta = sched.eta(opt.step + 1);
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
                if step % val_every == 0 || step == target_step {
                    let mut val_loss_sum = 0.0f32;
                    let mut val_valid_sum = 0usize;
                    for vw in val_loader.windows() {
                        let vo = match sandy_rust::autodiff::forward_loss(&cfg, &p, vw) {
                            Ok(l) => l,
                            Err(e) => { eprintln!("eval error: {e:?}"); return ExitCode::FAILURE; }
                        };
                        val_loss_sum += vo.loss_sum;
                        val_valid_sum += vo.n_valid;
                    }
                    let val_loss = if val_valid_sum > 0 { val_loss_sum / val_valid_sum as f32 } else { 0.0 };
                    println!("step {step:3}/{steps} | train_loss: {train_loss:.4} | val_loss: {val_loss:.4} | lr: {eta:.6}");
                }
            }

            let save_state_arg = match flag(&args, "--save-state") {
                Ok(v) => v,
                Err(e) => { eprintln!("{e}"); return ExitCode::from(2); }
            };
            if let Some(state_path) = save_state_arg {
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
                    rng_state: train_rng.dump_state(),
                    schedule_step: opt.step as usize,
                    schedule: sched,
                };
                if let Err(e) = checkpoint::save_training(&state, Path::new(state_path)) {
                    eprintln!("save-state error: {e:?}");
                    return ExitCode::FAILURE;
                }
                println!("saved training state to {state_path}");
            }

            let ckpt_arg = match flag(&args, "--checkpoint") {
                Ok(v) => v,
                Err(e) => { eprintln!("{e}"); return ExitCode::from(2); }
            };
            if let Some(ckpt_path) = ckpt_arg {
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
            let prompt = match flag(&args, "--prompt") {
                Ok(Some(p)) => p,
                Ok(None) => "Hello",
                Err(e) => { eprintln!("{e}"); return ExitCode::from(2); }
            };
            let mode = match flag(&args, "--mode") {
                Ok(Some(m)) => m,
                Ok(None) => "record",
                Err(e) => { eprintln!("{e}"); return ExitCode::from(2); }
            };
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
                    let task = match flag(&args, "--task") {
                        Ok(Some(t)) => t,
                        Ok(None) => "all",
                        Err(e) => { eprintln!("{e}"); return ExitCode::from(2); }
                    };
                    let split = match flag(&args, "--split") {
                        Ok(Some(s)) => s,
                        Ok(None) => "test",
                        Err(e) => { eprintln!("{e}"); return ExitCode::from(2); }
                    };
                    let ckpt_opt = match flag(&args, "--checkpoint") {
                        Ok(c) => c,
                        Err(e) => { eprintln!("{e}"); return ExitCode::from(2); }
                    };
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

                    let preds: Vec<(&sandy_rust::eval::Pair, sandy_rust::eval::PredictionResult)> = if let Some(ref ckpt_path) = ckpt_opt {
                        let model = match sandy_rust::checkpoint::load(std::path::Path::new(ckpt_path), &cfg, DEFAULT_MAX_BYTES) {
                            Ok(m) => m,
                            Err(e) => {
                                eprintln!("failed to load checkpoint '{ckpt_path}': {e:?}");
                                return ExitCode::FAILURE;
                            }
                        };
                        let tok = sandy_rust::tokenizer::ByteTokenizer;
                        let sp = sandy_rust::generation::SamplingParams {
                            max_new_tokens: 8,
                            temperature: 0.0,
                            ..Default::default()
                        };
                        println!("Generating model predictions for {} pairs...", pairs.len());
                        pairs.iter().map(|p| {
                            let b_enc = tok.encode(&p.base_prompt);
                            let v_enc = tok.encode(&p.variant_prompt);
                            let b_gen = sandy_rust::generation::generate(&model, &b_enc, &sp).map(|o| tok.decode(&o.tokens).unwrap_or_default()).unwrap_or_default();
                            let v_gen = sandy_rust::generation::generate(&model, &v_enc, &sp).map(|o| tok.decode(&o.tokens).unwrap_or_default()).unwrap_or_default();
                            (*p, sandy_rust::eval::PredictionResult {
                                model_base_answer: b_gen.trim().to_string(),
                                model_variant_answer: v_gen.trim().to_string(),
                            })
                        }).collect()
                    } else {
                        pairs.iter().map(|p| {
                            (*p, sandy_rust::eval::PredictionResult {
                                model_base_answer: p.oracle_base_answer.clone(),
                                model_variant_answer: p.oracle_variant_answer.clone(),
                            })
                        }).collect()
                    };

                    let (flip_mf, flip_mnf, selectivity, base_acc, var_acc) = match sandy_rust::eval::evaluate_pairs(&preds) {
                        Ok(m) => m,
                        Err(e) => {
                            eprintln!("metrics computation error: {e}");
                            return ExitCode::FAILURE;
                        }
                    };

                    let ci = match sandy_rust::eval::family_clustered_bootstrap(&target_families, &preds, 2000, 20261007) {
                        Ok(c) => c,
                        Err(e) => {
                            eprintln!("bootstrap error: {e}");
                            return ExitCode::FAILURE;
                        }
                    };

                    let baselines = sandy_rust::eval::evaluate_trivial_baselines(&pairs);

                    if ckpt_opt.is_some() {
                        println!("--- Model Evaluation Metrics ---");
                    } else {
                        println!("--- Metrics (Oracle Reference - pass --checkpoint <path> for model evaluation) ---");
                    }
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
                    eprintln!("usage: sandy_rust eval flip [--task all|kv_lookup|copy_marker|digit_comparison|symbol_count] [--split train|dev|test] [--checkpoint <path>]");
                    ExitCode::from(2)
                }
            }
        }
        _ => {
            eprintln!("usage: sandy_rust <command> [options]");
            eprintln!("commands:");
            eprintln!("  verify      [--gates <list>|all|baseline] [--artifacts <dir>]");
            eprintln!("  import      --model-dir <dir> --out <file.rimp>");
            eprintln!("  import-inspect --model-dir <dir>");
            eprintln!("  import-parity --import <file.rimp> --fixture <file.json> [--atol f]");
            eprintln!("  train       [--steps n] [--lr f] [--dataset path] [--save-state path] [--resume path] [--checkpoint path] [--val-every n]");
            eprintln!("  gradcheck");
            eprintln!("  checkpoint  --init <path> | --inspect <path>");
            eprintln!("  generate    [--prompt s] [--prompt-ids list] [--import path] [--tokenizer path] [--max-tokens n] [--temperature t] [--top-k k] [--top-p p] [--seed n] [--checkpoint path]");
            eprintln!("  inspect     [--prompt s] [--mode record|steer] [--alpha f]");
            eprintln!("  eval flip   [--task <name>] [--split <name>] [--checkpoint <path>]");
            ExitCode::from(2)
        }
    }
}
