# SandyRust Gate Evidence Manifest & Verification Contract

**Specification Authority:** `specv3.md` §1, §9, §13  
- Local Workspace Specification SHA-256: `93c9f564927d1dbf5cb47e204810fc44e1c0952ff38cc6558da91c97cbb32bb1`
- External Reference Specification SHA-256: `27f542a043c7b8e1f57d6eaef23df46497f5f9e578f2441a5fe2bbdf70f17e08`
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
| **G10** | `acceptance` | `PASS` | Deterministic checkpoint & TrainingState resumption (`§8.2`) | Bitwise forward logits parity; corruption/truncation rejected; moments/RNG/loader roundtrip tested via G10 and CLI `--save-state`/`--resume` | Checkpoint parser | Inference-only resume |
| **G11** | `acceptance` | `PASS` | Byte tokenizer roundtrip, next-token shift, buffered shard (`§5.1`, `§5.3`) | All 256 bytes & UTF-8 roundtrip; $y_t = x_{t+1}$; `.rtok` 64-byte header & token streaming | Tokenizer, `.rtok` shard | BPE / subword tokenizers |
| **G12** | `acceptance` | `PASS` | Deterministic sampling, terminal logit parity, UTF-8 streaming (`§8.3`) | Greedy ignores PRNG; seeded sampling repeats; terminal logits match full forward; UTF-8 buffer | Sampler, UTF-8 buffer | Lossy whole-buffer only decoding |
| **G13** | `acceptance` | `NOT_RUN` | GPU reference kernel acceleration spike (`§9 G13`) | None (CPU pure-Rust baseline selected under Decision D2-R1) | GPU / CUDA hardware | Decision D2-R1 boundary |
| **G14** | `regression` | `PASS` | Resource caps, bounded allocation, and typed error safety (`§2.2`, `§8.2`) | Oversize checkpoint rejected; context overflow rejected; rank error rejected; all-PAD rejected | Resource accounting | Unbounded buffer growth |
| **GX1** | `acceptance` | `PASS` | Grouped-Query Attention (GQA / MQA) scaling profiles & parity (`§4.2 Item 1`, `§9.2 GX1`) | Head ratio parity across $H_q / H_{kv} \in \{1, 2, 4\}$; verified parameter counts ($820,736$ for $H_{kv}=2$, $787,968$ for $H_{kv}=1$); cache allocation without duplicate expanded copies; one AdamW update parity; checkpoint reload rejection of mismatched $H_{kv}$ metadata | `Config::gqa_v2()`, `Config::mqa_v2()` | Untied embeddings (X2), QK-Norm (X3) |
| **GX2** | `integrity` | `PASS` | Untied Embeddings buffer separation, perturbation invariance & gradients (`§4.2 Item 2`, `§9.2 GX2`) | Exact parameter counts ($919,424$ on MHA, $853,888$ on GQA, $821,120$ on MQA); physical buffer separation ($id_E \neq id_{W_{\text{head}}}$); parameter perturbation independence; separate gradient accumulation and dedicated optimizer moments; checkpoint reload and cross-arch rejection | `Config::untied_v2()`, `Config::untied_gqa_v2()`, `Config::untied_mqa_v2()` | Tied baseline reference |
| **GX3** | `acceptance` | `PASS` | Attention score stability with per-head RMSNorm on Q and K (`§4.2 Item 3`, `§9.2 GX3`) | Parameter inventory counts ($+1,024$ on MHA $\to 887,296$, $+768$ on GQA $\to 821,504$, $+640$ on MQA $\to 788,608$); zero-input numerical stability; analytic VJP gradients vs finite differences ($err \le 10^{-5}$); attention score entropy stability under magnitude scaling ($\Delta H \le 10^{-3}$); checkpoint serialization & reload; cached incremental decode parity | `Config::qk_norm_v2()` | Baseline models without QK-Norm |
| **GX4** | `acceptance` | `PASS` | Long-context RoPE extrapolation check ($T=256$), linear/YaRN context scaling & invertible layouts (`§4.2 Item 4`, `§9.2 GX4`) | Invertible coordinate layout bijection & inverse checks ($f^{-1}(f(x)) == x$); adjacent vs split RoPE rotation equivalence ($\le 10^{-6}$); exact YaRN frequency banding (low, high, smooth ramp); attention temperature factor $\sqrt{1/t}$; vector norm preservation across $T=256$ ($\le 10^{-5}$); known-logit fixtures at context boundary ($pos=255$) | `Config::rope_yarn_v2(256)`, `Config::rope_linear_v2(256)` | Standard short-context baseline |
| **GX5** | `acceptance` | `PASS` | Optimized CPU GEMM Kernels with cache blocking and register unrolling (`§4.2 Item 5`, `§9.2 GX5`) | Tiled GEMM componentwise numerical equivalence to reference ($|x_i - \hat{x}_i| \le 10^{-5} + 10^{-4} \cdot |\hat{x}_i|$) across odd, strided, and batched inputs; gradient and update parity; thread-local accumulation policy; deterministic multithreading; wall-clock benchmark speedup | `Config::gemm_opt_v2()` | Non-CPU execution, GPU GEMM |
| **GX6** | `acceptance` | `PASS` | Typed Buffer Reuse & In-Place Scratch Arena Allocator (`§4.2 Item 6`, `§9.2 GX6`) | Zero allocation during steady-state forward and backward iterations; retained disjoint views with strict 64-byte alignment; graceful recovery on artificial OOM without corruption; multi-step gradient accumulation without memory creep | `Config::arena_v2(512 MiB)` | Active autograd tape references |
| **GX7** | `acceptance` | `PASS` | Memory-Efficient Chunked Cross-Entropy & Activation Recomputation (`§4.2 Item 7`, `§9.2 GX7`) | Exact loss equivalence ($\Delta \le 10^{-6}$) and gradient equivalence ($\Delta \le 10^{-5}$) to monolithic loss; zero retained chunk logits; activation recomputation immutability invariant; tail chunk, padding mask, and zero-valid token handling; custom VJP tape integration without duplicate backward passes | `Config::chunked_loss_v2(16)` | Non-tied and tied models |
| **GX8** | `acceptance` | `PASS` | FlashAttention-2 / Custom Tiled Attention (`§4.2 Item 8`, `§9.2 GX8`) | Block tiling ($B_r=16, B_c=16$) with online row-wise softmax recurrence; exact numerical parity across generalized causal offsets ($k \le q + \text{offset}$); unconditional padded query skip in forward ($O_q = \mathbf{0}$) and backward ($dQ_q = \mathbf{0}, dS = 0.0, \delta_q = 0.0$); tail block clamping; absolute sequence segment matching; analytic backward VJP grad parity; zero quadratic $T \times T$ tape memory ($O(B_r \cdot d_h + B_c \cdot d_h)$ bounded scratch) | `Config::flash_attn_v2()` | Baseline quadratic attention without tiling |
| **GX12E** | `acceptance` | `PASS` | Entropy-feedback adaptive sampling (`Addendum A1 §A4`) | $k=0.0$ parity, exact entropy fixtures, bounded temperature and Min-P control, error precedence | `Config::reference_v2` | Training-time entropy adjustments |
| **GX13** | `acceptance` | `PASS` | Activation hooks, inspectability & bounded steering (`Addendum A1 §A2`) | Bitwise match on disabled/record/zero/patch; export caps; cache intervention policy hash | `Config::reference_v2` | Training-time steering |
| **GX14** | `acceptance` | `PASS` | Paired perturbation & semantic flip-rate evaluation (`Addendum A1 §A3`) | Deterministic generators, 0% split prompt leakage, hand-worked metrics parity, bootstrap reproducibility, trivial baselines | `Config::reference_v2` | Model training dependencies |
| **GX15** | `acceptance` | `PASS` | FP32 Low-Rank Adaptation (LoRA) (`Addendum A1 §A5`) | Zero-init bitwise match, analytic gradcheck vs finite differences, base weight immutability, merge equivalence $W' = W + sBA$, header integrity & adapter compatibility | `Config::reference_v2` | Quantized LoRA, base weight training |
| **GX17** | `acceptance` | `PASS` | External Pretrained Weight Import Phases I1–I5 (`specv3 §3.6, §13.2`) | Safetensors reader, RIMP checkpoint format, float conversions, RoPE permutation, cryptographic bindings, BPE tokenizer, memory caps, greedy generation smoke | `import_profile` | Sharded models, quantization |
| **GX17-5** | `acceptance` | `NOT_RUN` | Real-model reference logit and continuation parity (`specv3 §B7.2, §B7.3, §B11`) | Offline reference logit fixture evaluation within atol <= 1e-3, 100% top-1 agreement, and greedy 8-token match | Donor SmolLM2-135M weights and offline reference logits fixture | Host missing external weights or fixture JSON |
| **GX17-7** | `acceptance` | `PASS` | Memory budget, bounded allocation, and streaming constraints (`specv3 §B5.3, §B11`) | Single-tensor 512 MiB cap; model 1 GiB cap; peak resident <= model + 2*tensor + 16 MiB formula verification | `import_profile` | Unbounded memory allocation |
| **GX17-8** | `acceptance` | `PASS` | Tokenizer golden vectors and roundtrip parity (`specv3 §B10.1, §B11`) | Byte-to-Unicode bijection, cap enforcement, ASCII, contraction, UTF-8, emoji, whitespace, special token golden vectors, 10 KB roundtrip | `import_profile` | Non-BPE tokenizers, non-null normalizers |
| **GX17-9** | `smoke` | `PASS` | Generation smoke test on imported model (`specv3 §B7.2, §B11`) | 32-token greedy generation, prefix-8 reference match, vocabulary bounds, and bitwise repeatability | `import_profile` | Multi-turn chat / instruction tuning |

---

## 3. Runner Integrity & Verification Protocol (EV1–EV7)

The verification runner enforces atomic execution:
1. **EV1 (Set Validation):** All requested gate IDs validated upfront; empty or unrecognized IDs abort with exit code 2. Artifact write failures abort with exit code 1.
2. **EV2 (Configuration Binding):** Every artifact dynamically binds source git commit revision (with working-tree dirty detection), config identity (`reference_v2`), active host toolchain (`rustc -V`), and fixture hashes at runtime.
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
