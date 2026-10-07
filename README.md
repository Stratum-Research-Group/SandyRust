# SandyRust

SandyRust is a standalone, pure-Rust autoregressive decoder-only transformer implementing the controlling `reference_v2` baseline specification.

## Architecture & Hyperparameters (`reference_v2`)

* **Model Hidden Dimension ($D$):** 128
* **Number of Layers ($L$):** 4
* **Query Attention Heads ($H_q$):** 4
* **Key/Value Attention Heads ($H_{kv}$):** 4 (Standard Multi-Head Attention)
* **Head Dimension ($d_h$):** 32
* **Feed-Forward Dimension ($d_{ff}$):** 384 ($3 \times D$)
* **Vocabulary Size ($V$):** 259 (256 raw byte tokens `0x00..=0xFF` + `BOS=256`, `EOS=257`, `PAD=258`)
* **Total Parameters:** **886,272** (tied token embeddings and unembedding projection)
* **Numerical Precision:** FP32 runtime (`Vec<f32>`) with scoped FP64 diagnostic precision path for autodiff verification (Gate G3)
* **Dependencies:** Zero external crates (100% pure Rust standard library under Decision D2-R1)

## Building & Offline Execution

SandyRust has zero third-party crate dependencies and requires no internet access or external tooling.

```bash
# Debug build and test (including primitive, evidence, and mathematical suites)
cargo test

# Release build
cargo build --release
```

## CLI Usage

### 1. Quality Acceptance Gates Verification
Run the verification suite across gates `G1`–`G14`:
```bash
cargo run --release -- verify
```
Reports and versioned machine-readable JSON artifacts (schema v3.1) are saved to `artifacts/G*.json`. Detailed coverage is documented in [`docs/evidence_manifest.md`](file:///home/akarta/Projects/SandyRusty/docs/evidence_manifest.md).

### 2. Autodiff Gradient Verification (Gradcheck)
Verify reverse-mode autodiff analytic gradients against central finite differences in FP64:
```bash
cargo run --release -- gradcheck
```

### 3. Bounded Training & Evaluation
Run bounded training steps with cosine schedule, gradient clipping, and held-out validation:
```bash
cargo run --release -- train --steps 20 --lr 0.01
```

### 4. Checkpoint Serialization & Inspection
Save an initialized baseline model checkpoint:
```bash
cargo run --release -- checkpoint --init checkpoint.bin
```

Inspect an existing checkpoint file (verifies SHA-256 integrity, magic bytes, tensor inventory):
```bash
cargo run --release -- checkpoint --inspect checkpoint.bin
```

### 5. Autoregressive Generation
Generate tokens using greedy or stochastic sampling:
```bash
cargo run --release -- generate --prompt "Hello world" --max-tokens 32 --temperature 0.8 --top-k 20 --top-p 0.9
```

## Quality Acceptance Gates

| Gate | Scope | Title | Description | Status |
| :--- | :--- | :--- | :--- | :--- |
| **G1** | `integrity` | **Build Boundary** | Zero C/CUDA/Python dependencies in R1; pure Rust std | **PASS** |
| **G2** | `acceptance` | **Forward Primitives** | RMSNorm, RoPE, Linear, SwiGLU match analytic fixtures | **PASS** |
| **G3** | `acceptance` | **Autodiff Gradients** | FP64 central finite-difference verification across epsilon sweep | **PASS** |
| **G4** | `acceptance` | **Causality & Masking** | Padded queries output zero; future positions masked | **PASS** |
| **G5** | `acceptance` | **RoPE & RMSNorm** | RoPE preserves vector norms; RMSNorm matches formulation | **PASS** |
| **G6** | `integrity` | **Parameter Inventory** | Exact parameter count calibration ($P = 886,272$); extensions rejected | **PASS** |
| **G7** | `acceptance` | **KV Cache Parity** | Incremental single-token cached decode matches full-sequence forward pass | **PASS** |
| **G8** | `acceptance` | **Optimizer Accumulation** | Microbatch gradient accumulation with AdamW update | **PASS** |
| **G9** | `smoke` | **Synthetic Overfit** | Bounded learning overfits sequence with $\ge 50\%$ loss reduction on 4-layer model | **PASS** |
| **G10** | `acceptance` | **Deterministic State Restore** | Bitwise checkpoint reload; corrupt/truncated payloads rejected; `TrainingState` resume | **PASS** |
| **G11** | `acceptance` | **Tokenizer Byte Roundtrip** | Bijective 256-byte roundtrip and UTF-8 string integrity; buffered `.rtok` shard | **PASS** |
| **G12** | `acceptance` | **Sampling Determinism** | Seeded sampling reproduces tokens; terminal logit parity; UTF-8 stream buffer | **PASS** |
| **G13** | `acceptance` | **GPU Spike** | Optional GPU spike deferred under CPU baseline | **NOT RUN** (D2-R1) |
| **G14** | `regression` | **Failure Mode Handling** | Oversized checkpoints, capacity overflows, and rank errors rejected safely | **PASS** |

## Unsupported Features & Non-Goals

Per Project Decisions ([`docs/decisions.md`](file:///home/akarta/Projects/SandyRusty/docs/decisions.md)):

| Feature | Policy | Rationale |
| :--- | :--- | :--- |
| External Pretrained Weights | Unsupported | Model is trained from scratch with deterministic seeded initialization (D6) |
| Third-Party Crates | Unsupported | Strict pure-Rust standard library implementation (D2-R1) |
| GPU Execution | Unsupported | Strict CPU pure Rust execution profile (D2-R1) |
| Extensions `X1`–`X16`, `X12-E` | Disabled | Modular extensions remain disabled in baseline; evaluated in Phase 6 |
