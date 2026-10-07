use sandy_rust::checkpoint::{self, DEFAULT_MAX_BYTES};
use sandy_rust::config::Config;
use sandy_rust::generation::{generate, SamplingParams};
use sandy_rust::model::Model;
use sandy_rust::tokenizer::{ByteTokenizer, BOS};
use sandy_rust::gates;
use std::{env, fs, path::Path, process::ExitCode};

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(String::as_str)
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let cfg = Config::reference_v2();
    match args.first().map(String::as_str) {
        Some("verify") => {
            let list = flag(&args, "--gates").unwrap_or("g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14");
            let ids: Vec<&str> = list.split(',').collect();
            let reports = gates::run(&ids);
            let _ = fs::create_dir_all("artifacts");
            let mut failed = false;
            for r in &reports {
                if let Some(reason) = &r.reason {
                    println!("{} {} ({})", r.gate_id, r.status, reason);
                } else {
                    println!("{} {}", r.gate_id, r.status);
                }
                failed |= r.status == "FAIL";
                let _ = fs::write(format!("artifacts/{}.json", r.gate_id), r.to_json());
            }
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
            let params = SamplingParams {
                temperature: flag(&args, "--temperature").and_then(|v| v.parse().ok()).unwrap_or(0.0),
                top_k: flag(&args, "--top-k").and_then(|v| v.parse().ok()).unwrap_or(usize::MAX),
                top_p: flag(&args, "--top-p").and_then(|v| v.parse().ok()).unwrap_or(1.0),
                max_new_tokens: flag(&args, "--max-tokens").and_then(|v| v.parse().ok()).unwrap_or(16),
                seed: flag(&args, "--seed").and_then(|v| v.parse().ok()).unwrap_or(0),
            };
            match generate(&model, &prompt, &params) {
                Ok(out) => {
                    println!("{:?}", tok.decode(&out.tokens).unwrap_or_default());
                    ExitCode::SUCCESS
                }
                Err(e) => { eprintln!("generation error: {e:?}"); ExitCode::FAILURE }
            }
        }
        _ => {
            eprintln!("usage: sandy_rust verify [--gates g1,g2,...] | checkpoint --init|--inspect <path> | generate [--prompt s] [--max-tokens n] [--temperature t] [--top-k k] [--top-p p] [--seed n] [--checkpoint path]");
            ExitCode::from(2)
        }
    }
}
