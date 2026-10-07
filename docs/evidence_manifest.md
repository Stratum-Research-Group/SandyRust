# SandyRust Gate Evidence Manifest & Verification Contract

**Specification Authority:** `specv3.md` §1, §9, §13  
**Schema Version:** 3.1  
**Target Profile:** `reference_v2` (V=259, D=128, L=4, H_q=H_kv=4, tied P=886,272)  
**Execution Baseline:** Strict CPU Pure Rust (`std` only, Decision D2-R1)

---

## 1. Evidence Contract Architecture

To prevent evidence inflation and distinguish superficial smoke tests from verified acceptance, every gate runner execution produces an atomic, versioned JSON record adhering to schema 3.1.

### 1.1 Scope Taxonomy

| Scope | Definition | Acceptance Policy |
| :--- | :--- | :--- |
| **`integrity`** | Structural repository checks (zero dependencies, parameter counting, extension gates). | Must PASS to ensure architectural invariants. |
| **`acceptance`** | Full numerical, algorithmic, and roundtrip correctness proofs across required inputs. | Required for milestone certification. |
| **`smoke`** | Bounded end-to-end convergence sanity checks (e.g., small synthetic overfit window). | Tagged as smoke; cannot substitute for full acceptance. |
| **`regression`** | Adversarial edge-case, resource cap, and typed error handling checks. | Must PASS to ensure crash safety and resilience. |

### 1.2 Status Taxonomy

- **`PASS`**: Verified implementation satisfied all declared predicates within specified tolerances.
- **`FAIL`**: Valid test execution detected violation of contract or tolerance breach.
- **`NOT_RUN`**: Explicitly excluded from execution under an approved decision record (e.g., G13 GPU spike under D2-R1).
- **`INCOMPLETE`**: Execution interrupted, timed out, or covered only partial domain.
- **`ERROR`**: Runner misconfiguration, invalid command-line invocation, or artifact I/O failure.

---

## 2. Requirement Coverage Manifest (Gates G1–G14)

| Gate ID | Scope | Status | Requirement | Implemented Check | Prerequisites | Exclusions |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **G1** | `integrity` | `PASS` | Zero external dependencies & build scripts (`specv3 §1`, `D2-R1`) | `Cargo.toml` dependency line count == 0 and SHA-256 hash validation | Rust `std` | Third-party crates, `build.rs`, native C/C++ |
| **G2** | `acceptance` | `PASS` | Core tensor contractions: matmul, batched_matmul, logsumexp (`§3.1`) | Exact analytic fixtures for 2x3*3x2, 2x2x2 batched matmul, and $\ln(2)$ logsumexp | Contiguous tensors | GPU GEMM kernels |
| **G3** | `acceptance` | `PASS` | Autodiff gradient correctness on tied parameters (`§3.6`) | Two-sided finite-difference gradcheck in FP64 across coordinate probes with repeated tokens | FP64 model & forward | Single-sided differences |
| **G4** | `acceptance` | `PASS` | Causal attention masking, rank validation, padded query isolation (`§3.3`) | Upper triangular attention weights == 0, padded query output == 0, rank-1 tensor rejection | Attention reference | Bidirectional attention |
| **G5** | `acceptance` | `PASS` | RMSNorm equation and RoPE isometry norm preservation (`§3.2`, `§3.4`) | RMSNorm closed formula match; RoPE vector L2 norm preservation with exact f64 angles | RMSNorm / RoPE ops | Learned positional embeddings |
| **G6** | `integrity` | `PASS` | Parameter inventory & extension rejection (`§2.1`, `§2.2`) | Exact tied count == 886,272; untied embeddings (X1) & GQA (X2) config rejection | Config engine | Extensions X1–X16 |
| **G7** | `acceptance` | `PASS` | KV cache incremental decode parity & model binding (`§4.1`, `§4.2`) | Stepwise cached logits match full forward row within componentwise 1e-4; stale model rejected | KVCache & Model | Unbound cache reuse |
| **G8** | `acceptance` | `PASS` | Microbatch gradient accumulation with AdamW clipping & skips (`§8.1`) | Unequal microbatches accumulated, clipped AdamW produces finite weights, NaN grad skipped | AdamW & autodiff | Unchecked NaN grad writes |
| **G9** | `smoke` | `PASS` | Overfit convergence on synthetic window with 4-layer model (`§8.1`, `G9`) | Full 4-layer `reference_v2` model reduces cross-entropy loss by $\ge 50\%$ in $\le 70$ steps | Full 4-layer autodiff | Multi-epoch corpus training |
| **G10** | `acceptance` | `PASS` | Deterministic checkpoint & TrainingState resumption (`§8.2`) | Bitwise forward logits parity; corruption/truncation rejected; moments/RNG/loader roundtrip | Checkpoint parser | Inference-only resume |
| **G11** | `acceptance` | `PASS` | Byte tokenizer roundtrip, next-token shift, buffered shard (`§5.1`, `§5.3`) | All 256 bytes & UTF-8 roundtrip; $y_t = x_{t+1}$; `.rtok` 64-byte header & token streaming | Tokenizer, `.rtok` shard | BPE / subword tokenizers |
| **G12** | `acceptance` | `PASS` | Deterministic sampling, terminal logit parity, UTF-8 streaming (`§8.3`) | Greedy ignores PRNG; seeded sampling repeats; terminal logits match full forward; UTF-8 buffer | Sampler, UTF-8 buffer | Lossy whole-buffer only decoding |
| **G13** | `acceptance` | `NOT_RUN` | GPU reference kernel acceleration spike (`§9 G13`) | None (CPU pure-Rust baseline selected under Decision D2-R1) | GPU / CUDA hardware | Decision D2-R1 boundary |
| **G14** | `regression` | `PASS` | Resource caps, bounded allocation, and typed error safety (`§2.2`, `§8.2`) | Oversize checkpoint rejected; context overflow rejected; rank error rejected; all-PAD rejected | Resource accounting | Unbounded buffer growth |
| **GX12E** | `acceptance` | `PASS` | Entropy-feedback adaptive sampling (`Addendum A1 §A4`) | $k=0.0$ parity, exact entropy fixtures, bounded temperature and Min-P control, error precedence | `Config::reference_v2` | Training-time entropy adjustments |
| **GX13** | `acceptance` | `PASS` | Activation hooks, inspectability & bounded steering (`Addendum A1 §A2`) | Bitwise match on disabled/record/zero/patch; export caps; cache intervention policy hash | `Config::reference_v2` | Training-time steering |
| **GX14** | `acceptance` | `PASS` | Paired perturbation & semantic flip-rate evaluation (`Addendum A1 §A3`) | Deterministic generators, 0% split prompt leakage, hand-worked metrics parity, bootstrap reproducibility, trivial baselines | `Config::reference_v2` | Model training dependencies |
| **GX15** | `acceptance` | `PASS` | FP32 Low-Rank Adaptation (LoRA) (`Addendum A1 §A5`) | Zero-init bitwise match, analytic gradcheck vs finite differences, base weight immutability, merge equivalence $W' = W + sBA$, header integrity & adapter compatibility | `Config::reference_v2` | Quantized LoRA, base weight training |

---

## 3. Runner Integrity & Verification Protocol (EV1–EV7)

The verification runner enforces atomic execution:
1. **EV1 (Set Validation):** All requested gate IDs validated upfront; empty or unrecognized IDs abort with exit code 2. Artifact write failures abort with exit code 1.
2. **EV2 (Configuration Binding):** Every artifact binds source commit, config identity (`reference_v2`), toolchain, and fixture hashes.
3. **EV3 (Defect Classification):** Non-finite values and invalid dimensions trigger explicit typed errors rather than silent acceptance. `NOT_RUN` gates remain clearly distinguished from failures.
4. **EV4 (Tamper Detection):** Cryptographic SHA-256 payload checksums invalidate tampered checkpoints or serialized states.
5. **EV5 (Scope Separation):** Smoke tests (`G9`) are segregated from full acceptance gates (`G2`–`G5`, `G7`, `G8`, `G10`–`G12`).
6. **EV6 (Replay Isolation):** Replaying verifications into isolated temporary directories matches declared tolerances bitwise.
7. **EV7 (Readback Verification):** Every artifact is written to a temporary staging file, read back and verified for byte parity before atomic rename.

---

## 4. Replay Commands

### Full Acceptance Gate Verification
```bash
cargo run --release -- verify
```

### Verification with Specific Gate Selection & Custom Artifact Directory
```bash
cargo run --release -- verify --gates g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14 --artifacts artifacts
```

### Analytic vs. Finite-Difference Gradcheck
```bash
cargo run --release -- gradcheck
```

### Bounded Reference Training Run
```bash
cargo run --release -- train --steps 20 --lr 0.01
```

### Activation Inspection & Steering (X13)
```bash
cargo run --release -- inspect --prompt "Hello" --mode record
cargo run --release -- inspect --prompt "Hello" --mode steer --alpha 1.5
```

### Paired Perturbation & Semantic Flip-Rate Evaluation (X14)
```bash
cargo run --release -- eval flip --task all --split test
```

### Complete Test Suite (Unit, Integration, Evidence, and Mathematical Fixtures)
```bash
cargo test
```
