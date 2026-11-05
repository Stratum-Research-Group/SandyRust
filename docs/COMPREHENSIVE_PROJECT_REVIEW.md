# Comprehensive Project Review: `SandyRusty` (`sandy_rust` v0.1.0)

## 1. Baseline & Executive Verdict

| Field | Value |
| :--- | :--- |
| **Repository** | `/home/akarta/Projects/SandyRusty` (`origin/main`) |
| **Branch** | `main` |
| **Reviewed HEAD** | `1a0c1cf985975bed93719756b6d146c3cbea176e` |
| **Working Tree Status** | Clean tracked tree; 1 untracked file preserved untouched (`?? sandyrust_architectural_review_roadmap.md`) |
| **Languages & Edition** | Rust (`edition = "2021"`, `#![forbid(unsafe_code)]`, zero external crate dependencies) |
| **Toolchain** | `rustc 1.98.1 (48a229cea 2026-09-01)`, `cargo 1.98.1 (797e8a9bc 2026-08-05)` on `x86_64-unknown-linux-gnu` |
| **Review Scope** | Full-repository clean-sweep audit (`Cargo.toml`, `src/**`, `tests/**`, `docs/**`, `eval/**`, `artifacts/**`) |
| **Blunt Verdict** | **CONDITIONAL / NOT PRODUCTION-READY.** The baseline `reference_v2` transformer path ($P = 886,272$), cryptographic SHA-256 checkpointing (`RCHK` / `RIMP`), zero-dependency BPE/safetensors parsers, and 186 automated tests are mathematically clean and pass `cargo test` and `cargo clippy --all-targets` with zero warnings. **However, several extensions (X3, X4, X5, X6, X8, X14, X15, X17) are either only partially wired into the end-to-end model/autodiff/CLI paths or contain concrete correctness and performance defects.** Most notably: (1) training/autodiff on `ext_x3_qk_norm` models corrupts parameter indexing and panics out-of-bounds because `autodiff.rs` and `loss/mod.rs` hardcode 9 tensors per layer instead of 11; (2) imported `ext_x17_import` models with `tie_word_embeddings: false` load `lm_head.weight` into memory but silently ignore it during inference (`model.rs` and `cache.rs` require `ext_x2_untied_embeddings`, which `parse_donor_config` hardcodes to `false`); (3) `ext_x4_rope`, `ext_x5_gemm`, `ext_x6_arena`, and `ext_x8_flash_attn` pass isolated module gates (`GX4`–`GX8`) but are never called by `Model::forward`, `cache::forward_step`, or `autodiff::loss_and_grad`; and (4) `generation::generate` recomputes full uncached $O(T^2)$ forward passes on every token while `cache::forward_step` re-hashes the entire model via SHA-256 and transposes the full vocabulary matrix on every token step. |

---

## 2. Whole-Repository Coverage Inventory

Every module, test suite, specification document, evaluation protocol, and gate artifact in the repository was inspected at commit `1a0c1cf985975bed93719756b6d146c3cbea176e`.

| Area / Module | Path | Lines | Coverage Status | Purpose & Review Notes |
| :--- | :--- | ---: | :--- | :--- |
| **Crate Manifest** | [Cargo.toml](file:///home/akarta/Projects/SandyRusty/Cargo.toml#L1-L15) | 15 | Reviewed (100%) | Zero `[dependencies]` / `[dev-dependencies]`; release profile `opt-level = 3`, `lto = "thin"`. |
| **Library Root** | [src/lib.rs](file:///home/akarta/Projects/SandyRusty/src/lib.rs#L1-L26) | 26 | Reviewed (100%) | Enforces `#![forbid(unsafe_code)]`; exports all 20 top-level modules. |
| **CLI Entrypoint** | [src/main.rs](file:///home/akarta/Projects/SandyRusty/src/main.rs#L1-L1063) | 1,063 | Reviewed (100%) | Implements `verify`, `train`, `eval`, `generate`, `hooks`, `lora`, `import`, `inspect`, `parity`. Contains defects F-06, F-07, F-08, F-11. |
| **Configuration** | [src/config.rs](file:///home/akarta/Projects/SandyRusty/src/config.rs#L1-L497) | 497 | Reviewed (100%) | `Config`, `RopeScalingConfig`, parameter count invariants, SHA-256 config fingerprinting, baseline & X17 import profile validation. |
| **Model & Forward** | [src/model.rs](file:///home/akarta/Projects/SandyRusty/src/model.rs#L1-L598) | 598 | Reviewed (100%) | `LayerWeights`, `Model`, deterministic normal init, inventory, `forward_with_hooks`. Contains defects F-02, F-03, F-05. |
| **Autodiff & Gradcheck** | [src/autodiff.rs](file:///home/akarta/Projects/SandyRusty/src/autodiff.rs#L1-L824) | 824 | Reviewed (100%) | Generic `Scalar` (`f32`/`f64`) forward/backward tape, `loss_and_grad`, `forward_loss`, central-difference `gradcheck`. Contains defects F-01, F-03. |
| **KV-Cache Decode** | [src/cache.rs](file:///home/akarta/Projects/SandyRusty/src/cache.rs#L1-L654) | 654 | Reviewed (100%) | `KVCache`, session/fingerprint/policy validation, `forward_step_with_hooks`. Contains defects F-02, F-03, F-05. |
| **Tensor Core** | [src/tensor/mod.rs](file:///home/akarta/Projects/SandyRusty/src/tensor/mod.rs#L1-L163) | 163 | Reviewed (100%) | Row-major `Tensor`, checked shape products, `gather_rows`, `transpose2d`. |
| **Tensor Operations** | [src/tensor/ops.rs](file:///home/akarta/Projects/SandyRusty/src/tensor/ops.rs#L1-L261) | 261 | Reviewed (100%) | `matmul`, `batched_matmul`, `rms_norm` (FP64 accumulation), `rope` (FP64 trig), `silu`, `swiglu`, `masked_softmax_row`, `logsumexp`. |
| **Reference Attention** | [src/attention/mod.rs](file:///home/akarta/Projects/SandyRusty/src/attention/mod.rs#L1-L78) | 78 | Reviewed (100%) | Causal & pad-masked single-head attention with unconditional padded-query zero output (`G4`). |
| **GQA / MQA (X1)** | [src/attention/gqa.rs](file:///home/akarta/Projects/SandyRusty/src/attention/gqa.rs#L1-L343) | 343 | Reviewed (100%) | Grouped-query head mapping `kv_head_index`, `repeat_kv_heads`, `attention_gqa`, Gate `GX1`. |
| **QK-Norm (X3)** | [src/attention/qknorm.rs](file:///home/akarta/Projects/SandyRusty/src/attention/qknorm.rs#L1-L403) | 403 | Reviewed (100%) | Per-head RMSNorm before RoPE, FP64 VJP `rms_norm_vjp_f64`, Gate `GX3`. |
| **RoPE Scaling (X4)** | [src/attention/rope_ext.rs](file:///home/akarta/Projects/SandyRusty/src/attention/rope_ext.rs#L1-L347) | 347 | Reviewed (100%) | Linear & YaRN wavelength ramp, `yarn_temperature_scale`, adjacent/split layout bijection, Gate `GX4`. |
| **FlashAttention-2 (X8)** | [src/attention/flash.rs](file:///home/akarta/Projects/SandyRusty/src/attention/flash.rs#L1-L1094) | 1,094 | Reviewed (100%) | Online softmax tiled forward (`flash_attention`) and tiled backward VJP (`flash_attention_backward`), Gate `GX8`. |
| **CPU GEMM (X5)** | [src/gemm/mod.rs](file:///home/akarta/Projects/SandyRusty/src/gemm/mod.rs#L1-L450) | 450 | Reviewed (100%) | Cache-tiled (`MC=64, KC=64, NC=64`) & `4x4` register micro-kernel GEMM, `std::thread::scope` parallel M-partitioning, Gate `GX5`. |
| **Scratch Arena (X6)** | [src/arena/mod.rs](file:///home/akarta/Projects/SandyRusty/src/arena/mod.rs#L1-L527) | 527 | Reviewed (100%) | Safe 64-byte aligned bump arena (`ScratchArena`), non-overlapping slice splitting via `RefCell<&mut [u8]>`, Gate `GX6`. |
| **Chunked Loss (X7)** | [src/loss/mod.rs](file:///home/akarta/Projects/SandyRusty/src/loss/mod.rs#L1-L530) | 530 | Reviewed (100%) | Vocabulary-dimension chunked cross-entropy & gradient accumulation (`chunked_loss_and_grad`), Gate `GX7`. Contains defect F-01. |
| **Untied Embeddings (X2)** | [src/embedding/mod.rs](file:///home/akarta/Projects/SandyRusty/src/embedding/mod.rs#L1-L273) | 273 | Reviewed (100%) | Gate `GX2` verification of decoupled `embed` and `lm_head` buffers and gradients. |
| **Activation Hooks (X13)** | [src/hooks/mod.rs](file:///home/akarta/Projects/SandyRusty/src/hooks/mod.rs#L1-L654) | 654 | Reviewed (100%) | 17 hook sites, `RecordTiming`, `EditOp` (`Steer`, `Ablate`, `Patch`), `DirectionHeader`, `ActivationSink`, Gate `GX13`. |
| **Semantic Eval (X14)** | [src/eval/mod.rs](file:///home/akarta/Projects/SandyRusty/src/eval/mod.rs#L1-L624) | 624 | Reviewed (100%) | 4 synthetic paired-perturbation generators, split leakage detector, family-clustered bootstrap CI, Gate `GX14`. |
| **LoRA Adapters (X15)** | [src/adapter/mod.rs](file:///home/akarta/Projects/SandyRusty/src/adapter/mod.rs#L1-L654) | 654 | Reviewed (100%) | `LoraMatrix`, `LoraAdapter`, `RLRA` binary format, merge & VJP, Gate `GX15`. Contains defect F-09. |
| **Generation & X12E** | [src/generation.rs](file:///home/akarta/Projects/SandyRusty/src/generation.rs#L1-L725) | 725 | Reviewed (100%) | Greedy, temperature, top-k, top-p, X12E entropy-feedback adaptive sampling (`Temperature` & `MinP`), Gate `GX12E`. Contains defect F-04. |
| **Weight Import (X17)** | [src/import/mod.rs](file:///home/akarta/Projects/SandyRusty/src/import/mod.rs#L1-L3514) | 3,514 | Reviewed (100%) | Zero-dep `safetensors` & `config.json` parser, IEEE-754 FP16/BF16->FP32 conversion, HF RoPE row permutation, `RIMP` format, Gates `GX17-0`..`GX17-9`. Contains defects F-02, F-10, F-12. |
| **BPE Tokenizer (X17)** | [src/import/bpe.rs](file:///home/akarta/Projects/SandyRusty/src/import/bpe.rs#L1-L1345) | 1,345 | Reviewed (100%) | Pure-Rust HuggingFace `tokenizer.json` ByteLevel BPE parser, GPT-2 & Llama-3 pre-tokenizers, longest-match special token splitting, Gate `GX17-8`. |
| **Gates Runner** | [src/gates.rs](file:///home/akarta/Projects/SandyRusty/src/gates.rs#L1-L819) | 819 | Reviewed (100%) | `GateReport` v3.1 JSON schema, baseline gates `G1`–`G14`, dispatcher `gates::run`. |
| **Checkpointing** | [src/checkpoint.rs](file:///home/akarta/Projects/SandyRusty/src/checkpoint.rs#L1-L586) | 586 | Reviewed (100%) | `RCHK` model checkpoint and `RTRN` training state serialization with per-tensor and full-payload SHA-256 verification. Contains defect F-12. |
| **Optimizer** | [src/optimizer.rs](file:///home/akarta/Projects/SandyRusty/src/optimizer.rs#L1-L214) | 214 | Reviewed (100%) | Decoupled `AdamW` (weight decay only on 2D matrices), global L2 grad clipping, non-finite grad skip counter, cosine warmup `Schedule`. |
| **Data Loader** | [src/data.rs](file:///home/akarta/Projects/SandyRusty/src/data.rs#L1-L273) | 273 | Reviewed (100%) | `RTOK` v1 binary shard parser/writer, deterministic epoch-shuffled `WindowLoader` with exact cursor restoration. |
| **Byte Tokenizer** | [src/tokenizer.rs](file:///home/akarta/Projects/SandyRusty/src/tokenizer.rs#L1-L120) | 120 | Reviewed (100%) | Baseline 259-token byte-level tokenizer (`0..=255`, `BOS=256`, `EOS=257`, `PAD=258`) and `Utf8StreamDecoder`. |
| **PRNG** | [src/rng.rs](file:///home/akarta/Projects/SandyRusty/src/rng.rs#L1-L138) | 138 | Reviewed (100%) | Deterministic `ChaCha20` stream (`SANDY-CHACHA20-1`), Box-Muller normal pairs, Fisher-Yates shuffle. |
| **SHA-256** | [src/sha256.rs](file:///home/akarta/Projects/SandyRusty/src/sha256.rs#L1-L152) | 152 | Reviewed (100%) | Pure-Rust FIPS 180-4 SHA-256 implementation and hex formatting. |
| **Integration Tests** | `tests/*.rs` (17 files) | 3,823 | Reviewed (100%) | 146 integration tests across 17 test binaries + 40 unit tests in `src/lib.rs` (186 total). |
| **Docs & Artifacts** | `docs/*`, `eval/*`, `artifacts/*` | — | Reviewed (100%) | `README.md`, `docs/decisions.md`, `docs/dependency_audit.md`, `docs/evidence_manifest.md`, `eval/x14_protocol.json`, and 37 JSON gate reports in `artifacts/`. |

---

## 3. Prioritized Findings (Defects & Gaps)

Findings are prioritized by impact, likelihood, and reachability per review discipline:
- **P0**: Immediate severe exposure or data-loss risk (None found).
- **P1**: Major blocking failures, panics on valid configurations, or silent wrong mathematical results.
- **P2**: Ordinary functional defects, incomplete feature wiring, or broken CLI subcommands.
- **P3**: Low-impact cleanup, minor edge-case hardening, or documentation/artifact drift.

---

### [F-01] `autodiff` and `loss` Hardcode 9 Tensors per Layer, Breaking Training/Loss Evaluation when `ext_x3_qk_norm` is Enabled
- **Priority**: **P1**
- **Confidence**: **Confirmed**
- **Affected Modules**: [src/autodiff.rs](file:///home/akarta/Projects/SandyRusty/src/autodiff.rs#L289-L661), [src/loss/mod.rs](file:///home/akarta/Projects/SandyRusty/src/loss/mod.rs#L88-L246)
- **Reviewed Commit**: `1a0c1cf985975bed93719756b6d146c3cbea176e`
- **Exact Locations**:
  - [`autodiff::loss_and_grad`](file:///home/akarta/Projects/SandyRusty/src/autodiff.rs#L289-L379) and backward loop at [src/autodiff.rs](file:///home/akarta/Projects/SandyRusty/src/autodiff.rs#L422-L547)
  - [`autodiff::forward_loss`](file:///home/akarta/Projects/SandyRusty/src/autodiff.rs#L577-L661)
  - [`loss::chunked_loss_and_grad`](file:///home/akarta/Projects/SandyRusty/src/loss/mod.rs#L88-L246)
- **Expected Behavior**:
  When `cfg.ext_x3_qk_norm == true` (e.g., `Config::qk_norm_v2()`), `model::inventory` ([src/model.rs](file:///home/akarta/Projects/SandyRusty/src/model.rs#L72-L82)) and `Model::tensors` ([src/model.rs](file:///home/akarta/Projects/SandyRusty/src/model.rs#L196-L207)) emit **11 tensors per layer** (`rms1`, `wq`, `wk`, `wv`, `q_norm`, `k_norm`, `wo`, `rms2`, `w_gate`, `w_up`, `w_down`). `autodiff::loss_and_grad`, `autodiff::forward_loss`, and `loss::chunked_loss_and_grad` should index the per-layer parameter slices using the 11-tensor stride and apply per-head QK-Norm forward and backward (`rms_norm_vjp_f64`).
- **Observed Behavior**:
  `autodiff::loss_and_grad` ([src/autodiff.rs](file:///home/akarta/Projects/SandyRusty/src/autodiff.rs#L290), `#L373`, `#L422`, `#L431`), `autodiff::forward_loss` ([src/autodiff.rs](file:///home/akarta/Projects/SandyRusty/src/autodiff.rs#L578), `#L654`), and `loss::chunked_loss_and_grad` ([src/loss/mod.rs](file:///home/akarta/Projects/SandyRusty/src/loss/mod.rs#L89), `#L149`, `#L208`, `#L239`) unconditionally hardcode `let b = 1 + 9 * l;` and `let final_norm = &params.data[1 + 9 * c.n_layers];` (and `2 + 9 * c.n_layers` for untied `lm_head`).
- **Trigger Condition**:
  Calling `loss_and_grad`, `forward_loss`, `gradcheck`, or `chunked_loss_and_grad` with `Config::qk_norm_v2()` (or any config with `ext_x3_qk_norm: true`).
- **Impact**:
  1. In layer 0, `params.data[b + 4]` (which is `q_norm` of shape `[dh] = [32]`) is read as `wo` (expected shape `[d_model, d_model] = [128, 128]`), causing an immediate **index-out-of-bounds panic** inside `mm_nt(&attn_cat, wo, d, d)` when indexing `b[k * n..]`!
  2. Even if shapes happened to align, QK-Norm is completely absent from the training forward/backward tape. Gate `GX3` passed only because it tested `rms_norm_vjp_f64` in isolation rather than calling `autodiff::loss_and_grad` on `Config::qk_norm_v2()`.
- **Fix Direction**:
  Compute per-layer base offsets dynamically (`let stride = if c.ext_x3_qk_norm { 11 } else { 9 }; let b = 1 + stride * l;`), unpack `q_norm` and `k_norm` when `c.ext_x3_qk_norm` is true, apply per-head RMSNorm before RoPE in the forward pass of `loss_and_grad` / `forward_loss` / `chunked_loss_and_grad`, and propagate `d_q_norm` and `d_k_norm` via the RMSNorm VJP in the backward pass.
- **Regression Check**:
  Add a test in `tests/test_x3_qknorm.rs` that runs `autodiff::gradcheck` and `autodiff::loss_and_grad` on `Config::qk_norm_v2()` and verifies zero gradcheck failures and non-zero gradients on `q_norm` and `k_norm`.

---

### [F-02] Imported Untied Models (`ext_x17_import: true`, `tied_embeddings: false`) Silently Ignore `lm_head` During Inference
- **Priority**: **P1**
- **Confidence**: **Confirmed**
- **Affected Modules**: [src/import/mod.rs](file:///home/akarta/Projects/SandyRusty/src/import/mod.rs#L1051-L1091), [src/model.rs](file:///home/akarta/Projects/SandyRusty/src/model.rs#L530-L535), [src/cache.rs](file:///home/akarta/Projects/SandyRusty/src/cache.rs#L513-L518)
- **Reviewed Commit**: `1a0c1cf985975bed93719756b6d146c3cbea176e`
- **Exact Locations**:
  - [`parse_donor_config`](file:///home/akarta/Projects/SandyRusty/src/import/mod.rs#L1051-L1091) and [`deserialize_rimp`](file:///home/akarta/Projects/SandyRusty/src/import/mod.rs#L1372-L1392)
  - [`Model::forward_with_hooks`](file:///home/akarta/Projects/SandyRusty/src/model.rs#L530-L534)
  - [`cache::forward_step_with_hooks`](file:///home/akarta/Projects/SandyRusty/src/cache.rs#L513-L517)
- **Expected Behavior**:
  When an imported HuggingFace/RIMP model has `tie_word_embeddings: false` (`cfg.tied_embeddings == false`), `remap_safetensors_to_model` ([src/import/mod.rs](file:///home/akarta/Projects/SandyRusty/src/import/mod.rs#L1844-L1850)) and `deserialize_rimp` load `lm_head.weight` into `model.lm_head = Some(tensor)`. Both `Model::forward_with_hooks` and `cache::forward_step_with_hooks` must project final normalized hidden states through `model.lm_head` whenever `!c.tied_embeddings`.
- **Observed Behavior**:
  In `parse_donor_config` ([src/import/mod.rs](file:///home/akarta/Projects/SandyRusty/src/import/mod.rs#L1074)) and `deserialize_rimp` ([src/import/mod.rs](file:///home/akarta/Projects/SandyRusty/src/import/mod.rs#L1375)), `ext_x2_untied_embeddings` is hardcoded to `false` (to preserve the baseline `Config::fingerprint()` format when `ext_x17_import` is used). However, `Model::forward_with_hooks` ([src/model.rs](file:///home/akarta/Projects/SandyRusty/src/model.rs#L530)) and `cache::forward_step_with_hooks` ([src/cache.rs](file:///home/akarta/Projects/SandyRusty/src/cache.rs#L513)) check:
  ```rust
  let head_mat = if c.ext_x2_untied_embeddings && !c.tied_embeddings {
      self.lm_head.as_ref().unwrap_or(&self.embed)
  } else {
      &self.embed
  };
  ```
  Because `c.ext_x2_untied_embeddings` is `false`, `head_mat` evaluates to `&self.embed` even though `self.lm_head` is `Some(...)`!
- **Trigger Condition**:
  Importing or loading any model via X17 (`import_model`, `remap_safetensors_from_bytes`, `deserialize_rimp`) where `tie_word_embeddings` is `false` (for example, `test_rimp_serialize_deserialize_roundtrip` in [tests/test_x17_import.rs](file:///home/akarta/Projects/SandyRusty/tests/test_x17_import.rs#L281-L321)).
- **Impact**:
  Completely wrong output logits on every untied imported model (`Llama-3.2-3B`, untied `SmolLM2` variants, etc.): the output projection silently multiplies by `embed^T` instead of `lm_head^T`.
- **Fix Direction**:
  In [src/model.rs](file:///home/akarta/Projects/SandyRusty/src/model.rs#L530), [src/cache.rs](file:///home/akarta/Projects/SandyRusty/src/cache.rs#L513), [src/autodiff.rs](file:///home/akarta/Projects/SandyRusty/src/autodiff.rs#L373), and [src/loss/mod.rs](file:///home/akarta/Projects/SandyRusty/src/loss/mod.rs#L149), check `if (c.ext_x2_untied_embeddings || c.ext_x17_import) && !c.tied_embeddings` (or simply `if !c.tied_embeddings`, since `Config::validate()` already enforces that `!c.tied_embeddings` is only legal when `ext_x2_untied_embeddings || ext_x17_import`).
- **Regression Check**:
  In `tests/test_x17_import.rs`, mutate `model.lm_head` on an imported untied model (`ext_x17_import: true, tied_embeddings: false, ext_x2_untied_embeddings: false`) and assert that `model.forward` and `cache::forward_step` produce changed logits matching `final_norm * lm_head^T`.

---

### [F-03] Extensions X4 (`ext_x4_rope`), X5 (`ext_x5_gemm`), X6 (`ext_x6_arena`), and X8 (`ext_x8_flash_attn`) Are Disconnected from `Model::forward`, `cache::forward_step`, and `autodiff`
- **Priority**: **P1**
- **Confidence**: **Confirmed**
- **Affected Modules**: [src/model.rs](file:///home/akarta/Projects/SandyRusty/src/model.rs#L279-L545), [src/cache.rs](file:///home/akarta/Projects/SandyRusty/src/cache.rs#L168-L529), [src/autodiff.rs](file:///home/akarta/Projects/SandyRusty/src/autodiff.rs#L243-L661)
- **Reviewed Commit**: `1a0c1cf985975bed93719756b6d146c3cbea176e`
- **Exact Locations**:
  - RoPE call sites: [src/model.rs](file:///home/akarta/Projects/SandyRusty/src/model.rs#L323), [src/model.rs](file:///home/akarta/Projects/SandyRusty/src/model.rs#L355), [src/cache.rs](file:///home/akarta/Projects/SandyRusty/src/cache.rs#L293), [src/cache.rs](file:///home/akarta/Projects/SandyRusty/src/cache.rs#L355), [src/autodiff.rs](file:///home/akarta/Projects/SandyRusty/src/autodiff.rs#L314), [src/autodiff.rs](file:///home/akarta/Projects/SandyRusty/src/autodiff.rs#L321)
  - Matmul & attention call sites: [src/model.rs](file:///home/akarta/Projects/SandyRusty/src/model.rs#L305-L535), [src/cache.rs](file:///home/akarta/Projects/SandyRusty/src/cache.rs#L228-L518), [src/autodiff.rs](file:///home/akarta/Projects/SandyRusty/src/autodiff.rs#L295-L546)
- **Expected Behavior**:
  Enabling `cfg.ext_x4_rope` (`Config::rope_yarn_v2` / `Config::rope_linear_v2`), `cfg.ext_x5_gemm` (`Config::gemm_opt_v2`), `cfg.ext_x6_arena` (`Config::arena_v2`), or `cfg.ext_x8_flash_attn` (`Config::flash_attn_v2`) should cause `Model::forward_with_hooks`, `cache::forward_step_with_hooks`, and `autodiff::loss_and_grad` to dispatch to:
  - X4: `rope_ext::compute_frequencies(dh, c.rope_theta as f64, c.rope_scaling.as_ref())`, `rope_ext::rope_with_freqs`, and `yarn_temperature_scale(c.rope_scaling.as_ref())`
  - X5: `gemm::gemm` / `gemm::gemm_trans_b` / `gemm::gemm_trans_a`
  - X6: `arena::with_thread_scratch_arena(c.max_arena_bytes, ...)` for layer intermediates
  - X8: `flash::flash_attention` and `flash::flash_attention_backward` (when `HookSite::AttnPattern` is not being recorded/edited)
- **Observed Behavior**:
  All four extensions exist as standalone modules that pass their respective unit/gate tests (`GX4`, `GX5`, `GX6`, `GX8`), and their flags alter `Config::fingerprint()` ([src/config.rs](file:///home/akarta/Projects/SandyRusty/src/config.rs#L455-L486)), **but `Model::forward_with_hooks`, `cache::forward_step_with_hooks`, and `autodiff::loss_and_grad` never inspect `c.ext_x4_rope`, `c.ext_x5_gemm`, `c.ext_x6_arena`, or `c.ext_x8_flash_attn`**.
  - Specifically, for X4 (`Config::rope_yarn_v2(256)`), `Model::forward` still calls `tensor::ops::rope(..., c.rope_theta, 0)` with unscaled frequencies and unscaled softmax temperature!
  - For X7 (`ext_x7_chunked_loss`), `src/main.rs` (`cmd_train` at [src/main.rs](file:///home/akarta/Projects/SandyRusty/src/main.rs#L748) and `cmd_eval` at [src/main.rs](file:///home/akarta/Projects/SandyRusty/src/main.rs#L898)) also calls `autodiff::loss_and_grad` and `autodiff::forward_loss` directly instead of branching on `cfg.ext_x7_chunked_loss` to call `loss::chunked_loss_and_grad` / `loss::chunked_forward_loss`.
- **Trigger Condition**:
  Running `Model::forward`, `cache::forward_step`, `autodiff::loss_and_grad`, or CLI `train`/`generate` with any of `ext_x4_rope`, `ext_x5_gemm`, `ext_x6_arena`, `ext_x7_chunked_loss`, or `ext_x8_flash_attn` enabled.
- **Impact**:
  - **X4**: Silent mathematical bug—models configured with YaRN or Linear RoPE scaling run with unscaled baseline RoPE frequencies during `Model::forward`, `cache::forward_step`, and training.
  - **X5 / X6 / X7 / X8**: Zero runtime speedup or memory reduction in `Model::forward`, `cache::forward_step`, or CLI `train`, because the optimized kernels and chunked loss are never invoked outside their standalone test suites.
- **Fix Direction**:
  1. Wire `c.ext_x4_rope` into `Model::forward_with_hooks`, `cache::forward_step_with_hooks`, `autodiff`, and `loss` so RoPE frequencies and YaRN temperature scaling are applied whenever `c.ext_x4_rope` is enabled.
  2. Dispatch `matmul` through `gemm::gemm` when `c.ext_x5_gemm` is enabled.
  3. Dispatch attention through `flash::flash_attention` (and `flash_attention_backward` in `autodiff`) when `c.ext_x8_flash_attn` is enabled and no `HookSite::AttnPattern` hook is active.
  4. Dispatch `loss_and_grad` / `forward_loss` to `loss::chunked_loss_and_grad` / `loss::chunked_forward_loss` when `c.ext_x7_chunked_loss` is enabled.
- **Regression Check**:
  Add end-to-end integration tests verifying that:
  - `Model::init(&Config::rope_yarn_v2(256)).unwrap().forward(&tokens, None)` differs from unscaled RoPE and matches `rope_with_freqs` + `yarn_temperature_scale`.
  - `Model::forward` with `ext_x5_gemm: true` and `ext_x8_flash_attn: true` matches baseline `reference_v2` within `1e-5`.

---

### [F-04] `generation::generate` Uses Uncached $O(N \cdot T^2)$ Full-Sequence Re-evaluation Instead of `KVCache`
- **Priority**: **P1**
- **Confidence**: **Confirmed**
- **Affected Modules**: [src/generation.rs](file:///home/akarta/Projects/SandyRusty/src/generation.rs#L350-L412), [src/import/mod.rs](file:///home/akarta/Projects/SandyRusty/src/import/mod.rs#L3062-L3085)
- **Reviewed Commit**: `1a0c1cf985975bed93719756b6d146c3cbea176e`
- **Exact Locations**:
  - [`generation::generate`](file:///home/akarta/Projects/SandyRusty/src/generation.rs#L380-L405)
  - [`import::run_parity_check`](file:///home/akarta/Projects/SandyRusty/src/import/mod.rs#L3065-L3084)
  - [`import::generate_parity_fixture`](file:///home/akarta/Projects/SandyRusty/src/import/mod.rs#L3142-L3158)
- **Expected Behavior**:
  Autoregressive generation should prefill a `KVCache` on the prompt tokens and then execute incremental single-token decode steps via `cache::forward_step` ($O(T)$ per step), avoiding redundant recomputation of all prior token activations.
- **Observed Behavior**:
  In `generation::generate` ([src/generation.rs](file:///home/akarta/Projects/SandyRusty/src/generation.rs#L380-L405)):
  ```rust
  for step in 0..params.max_new_tokens {
      if seq.len() >= max_seq { break; }
      let logits_all = model.forward(&seq, None)?;
      ...
      seq.push(next);
  }
  let final_all = model.forward(&seq, None)?;
  ```
  Every single generated token runs a full uncached `model.forward(&seq, None)` over the entire prefix `0..seq.len()`, and after the loop finishes it runs one additional full `model.forward(&seq, None)` to populate `final_logits`.
- **Trigger Condition**:
  Any call to `generation::generate` or CLI `sandy_rust generate`.
- **Impact**:
  On an imported 135M parameter model (`SmolLM2-135M`, 30 layers, `d_model = 576`, `vocab_size = 49152`), generating 64 tokens executes 65 full multi-token forward passes and 65 full `[49152, 576]` matrix transposes + matmuls, turning seconds of inference into many minutes.
- **Fix Direction**:
  Use `KVCache::new(&model.cfg, model.fingerprint(), session_id)` inside `generate` (after fixing F-05 so `forward_step` does not re-hash the model on every step), prefill `0..prompt.len()`, and advance one token at a time via `cache::forward_step`.
- **Regression Check**:
  Existing `generation::tests::terminal_logits_parity_g12` ([src/generation.rs](file:///home/akarta/Projects/SandyRusty/src/generation.rs#L653-L667)) and `GX12` (`gates::g12`) verify that `generate` outputs match `cache::forward_step` within tolerance.

---

### [F-05] `cache::forward_step_with_hooks` Recomputes Full-Model SHA-256 and Transposes the Full Vocabulary Matrix on Every Token Step
- **Priority**: **P1**
- **Confidence**: **Confirmed**
- **Affected Modules**: [src/cache.rs](file:///home/akarta/Projects/SandyRusty/src/cache.rs#L176-L529), [src/model.rs](file:///home/akarta/Projects/SandyRusty/src/model.rs#L530-L536)
- **Reviewed Commit**: `1a0c1cf985975bed93719756b6d146c3cbea176e`
- **Exact Locations**:
  - Full-model SHA-256 on every decode step: [src/cache.rs](file:///home/akarta/Projects/SandyRusty/src/cache.rs#L183)
  - Per-layer KV prefix heap cloning on every decode step: [src/cache.rs](file:///home/akarta/Projects/SandyRusty/src/cache.rs#L332-L343)
  - Full `[vocab_size, d_model]` allocation & transpose on every decode step: [src/cache.rs](file:///home/akarta/Projects/SandyRusty/src/cache.rs#L518) and [src/model.rs](file:///home/akarta/Projects/SandyRusty/src/model.rs#L535)
- **Expected Behavior**:
  `forward_step_with_hooks` should execute in $O(P + L \cdot T \cdot d)$ floating-point operations with zero whole-model cryptographic hashing and zero weight-matrix transpositions during inference.
- **Observed Behavior**:
  1. Line 183 of [src/cache.rs](file:///home/akarta/Projects/SandyRusty/src/cache.rs#L183) executes `if cache.model_fingerprint != model.fingerprint()`. `model.fingerprint()` ([src/model.rs](file:///home/akarta/Projects/SandyRusty/src/model.rs#L254-L270)) serializes every weight float in the model into a newly allocated `Vec<u8>` and computes its SHA-256 hash on **every single token step**! For `SmolLM2-135M`, that allocates and SHA-256 hashes **540 MB of weights per generated token**.
  2. Lines 332–343 of [src/cache.rs](file:///home/akarta/Projects/SandyRusty/src/cache.rs#L332-L343) call `slice_rows(&cache.k[l_idx], 0, pos)` and `extend_from_slice` BEFORE calling `cache.append_step` at line 424, allocating a fresh `Vec<f32>` of size `(pos + 1) * kv_dim` per layer per token instead of writing row `pos` into `cache.k[l_idx]` first and slicing `&cache.k[l_idx].data()[0..(pos + 1) * kv_dim]`.
  3. Line 518 of [src/cache.rs](file:///home/akarta/Projects/SandyRusty/src/cache.rs#L518) (and line 535 of [src/model.rs](file:///home/akarta/Projects/SandyRusty/src/model.rs#L535)) calls `matmul(&x, &head_mat.transpose2d()?)`, allocating and transposing a `[vocab_size, d_model]` matrix (`49,152 * 576 * 4 = 113.2 MB` for `SmolLM2-135M`) on every token step instead of computing `x * head_mat^T` directly via row dot-products or `gemm::gemm_trans_b(&x, head_mat)`.
- **Trigger Condition**:
  Every call to `cache::forward_step` / `cache::forward_step_with_hooks` (and `Model::forward` for `head_mat.transpose2d()`).
- **Impact**:
  Severe memory churn and CPU bottleneck on any real imported model (`SmolLM2-135M`): >650 MB of heap allocation/copying and 540 MB of SHA-256 hashing per generated token.
- **Fix Direction**:
  1. Cache `fingerprint: [u8; 32]` as a precomputed field on `Model` (computed once in `Model::init` and `Model::from_tensors`, and invalidated/recomputed only if weights are mutated), making `model.fingerprint()` an $O(1)$ field read.
  2. Replace `matmul(&x, &head_mat.transpose2d()?)` with a transpose-free `matmul_nt` / `gemm_trans_b(&x, head_mat)`.
  3. Write `k_rot` and `v_cur` into `cache.k[l_idx]` / `cache.v[l_idx]` before attention (after validating all hook/step inputs) so attention can read the contiguous prefix slice directly from the cache without cloning.
- **Regression Check**:
  Run `cache::tests::test_kv_cache_rejects_stale_model_or_config`, `cache::tests::test_kv_cache_matches_full_forward_parity_g7`, and `gates::g7`.

---

### [F-06] CLI `eval flip` Evaluates Oracle Answers Against Oracle Answers Instead of Running the Model
- **Priority**: **P2**
- **Confidence**: **Confirmed**
- **Affected Modules**: [src/main.rs](file:///home/akarta/Projects/SandyRusty/src/main.rs#L912-L961)
- **Reviewed Commit**: `1a0c1cf985975bed93719756b6d146c3cbea176e`
- **Exact Location**: [`cmd_eval_flip`](file:///home/akarta/Projects/SandyRusty/src/main.rs#L939-L945)
- **Expected Behavior**:
  `sandy_rust eval flip --protocol eval/x14_protocol.json [--checkpoint <path>]` should load a `Model` (or initialize one), run greedy generation on each `p.base_prompt` and `p.variant_prompt` in the test split, and evaluate the model's actual predicted answers using `eval::evaluate_pairs`.
- **Observed Behavior**:
  Lines 939–945 of [src/main.rs](file:///home/akarta/Projects/SandyRusty/src/main.rs#L939-L945) construct `preds` by copying the ground-truth oracle strings directly into `PredictionResult`:
  ```rust
  let preds: Vec<(&eval::Pair, eval::PredictionResult)> = test_pairs
      .iter()
      .map(|p| {
          (*p, eval::PredictionResult {
              model_base_answer: p.oracle_base_answer.clone(),
              model_variant_answer: p.oracle_variant_answer.clone(),
          })
      })
      .collect();
  ```
  No `Model` is ever loaded or executed, and `eval flip` unconditionally reports `base_acc: 1.0000`, `var_acc: 1.0000`, `selectivity: 1.0000`.
- **Trigger Condition**:
  Running `sandy_rust eval flip --protocol eval/x14_protocol.json`.
- **Impact**:
  The CLI `eval flip` command is a mock/oracle self-check rather than a real model evaluation tool.
- **Fix Direction**:
  Accept `--checkpoint <path>` (or default `Model::init(&Config::reference_v2())`) in `cmd_eval_flip`, encode each `base_prompt` and `variant_prompt`, decode the greedy continuation up to the answer token(s), and pass the model's decoded predictions into `PredictionResult`.
- **Regression Check**:
  Add a CLI test verifying that an untrained random model achieves `< 1.0` accuracy on `eval flip`, while the oracle helper in `test_x14_eval.rs` continues to test `evaluate_pairs` at `1.0`.

---

### [F-07] CLI `train --save-state` Writes All-Zero `rng_state` and `--resume` Never Restores PRNG State
- **Priority**: **P2**
- **Confidence**: **Confirmed**
- **Affected Modules**: [src/main.rs](file:///home/akarta/Projects/SandyRusty/src/main.rs#L688-L788), [src/rng.rs](file:///home/akarta/Projects/SandyRusty/src/rng.rs#L14-L53)
- **Reviewed Commit**: `1a0c1cf985975bed93719756b6d146c3cbea176e`
- **Exact Locations**:
  - State save in [`cmd_train`](file:///home/akarta/Projects/SandyRusty/src/main.rs#L778-L786)
  - State resume in [`cmd_train`](file:///home/akarta/Projects/SandyRusty/src/main.rs#L688-L709)
- **Expected Behavior**:
  `checkpoint::TrainingState` includes `rng_state: ([u32; 16], [u32; 16], usize)` ([src/checkpoint.rs](file:///home/akarta/Projects/SandyRusty/src/checkpoint.rs#L314)) so that saving and resuming a training run preserves the exact ChaCha20 PRNG state alongside `WindowLoader` state.
- **Observed Behavior**:
  In [src/main.rs](file:///home/akarta/Projects/SandyRusty/src/main.rs#L782), `cmd_train` hardcodes:
  ```rust
  rng_state: ([0u32; 16], [0u32; 16], 0),
  ```
  And in [src/main.rs](file:///home/akarta/Projects/SandyRusty/src/main.rs#L688-L709), `state.rng_state` is ignored on `--resume`. Furthermore, `Rng` in [src/rng.rs](file:///home/akarta/Projects/SandyRusty/src/rng.rs#L14-L19) has private fields (`state`, `buf`, `idx`, `cached_normal`) and does not expose `dump_state()` / `from_state()` methods (note: `WindowLoader` uses its own internal `Rng::new(seed ^ epoch)` per epoch, which is why `test_n2_training_state_resume_identical_to_uninterrupted` in [tests/test_training_resume.rs](file:///home/akarta/Projects/SandyRusty/tests/test_training_resume.rs#L76) passed even with a dummy `rng_state`).
- **Trigger Condition**:
  Running `sandy_rust train --save-state <path>` and resuming with `sandy_rust train --resume <path>`.
- **Impact**:
  The `rng_state` field in `RTRN` checkpoints written by the CLI is always zeroed rather than capturing actual PRNG state.
- **Fix Direction**:
  Add `pub fn state(&self) -> ([u32; 16], [u32; 16], usize)` and `pub fn from_state(...) -> Self` to `Rng` in [src/rng.rs](file:///home/akarta/Projects/SandyRusty/src/rng.rs#L14-L53), and wire them into `cmd_train` when saving and resuming `TrainingState`.
- **Regression Check**:
  Add a unit test in `src/rng.rs` verifying that dumping and restoring `Rng` mid-block (`idx != 16`) produces an identical subsequent random stream, and verify `cmd_train` round-trips a non-zero `rng_state`.

---

### [F-08] CLI `verify` Default Gate List Omits Extensions `GX1` through `GX8`
- **Priority**: **P2**
- **Confidence**: **Confirmed**
- **Affected Modules**: [src/main.rs](file:///home/akarta/Projects/SandyRusty/src/main.rs#L115-L120)
- **Reviewed Commit**: `1a0c1cf985975bed93719756b6d146c3cbea176e`
- **Exact Location**: [`cmd_verify`](file:///home/akarta/Projects/SandyRusty/src/main.rs#L116)
- **Expected Behavior**:
  Running `sandy_rust verify` with no `--gates` flag (or `--gates all`) should run all implemented gates listed in [docs/evidence_manifest.md](file:///home/akarta/Projects/SandyRusty/docs/evidence_manifest.md#L11-L52): `G1`–`G14`, `GX1`–`GX8`, `GX12E`, `GX13`, `GX14`, `GX15`, and `GX17`.
- **Observed Behavior**:
  Line 116 of [src/main.rs](file:///home/akarta/Projects/SandyRusty/src/main.rs#L116) defines:
  ```rust
  let default_gates = "g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14,gx12e,gx13,gx14,gx15,gx17";
  ```
  `gx1,gx2,gx3,gx4,gx5,gx6,gx7,gx8` are omitted from `default_gates`, even though `gates::run` ([src/gates.rs](file:///home/akarta/Projects/SandyRusty/src/gates.rs#L769-L776)) implements and dispatches all eight of them.
- **Trigger Condition**:
  Running `cargo run --release -- verify` or `cargo run --release -- verify --gates all`.
- **Impact**:
  Default verification runs skip 8 of the 13 extension gates (`GX1`–`GX8`).
- **Fix Direction**:
  Add `gx1,gx2,gx3,gx4,gx5,gx6,gx7,gx8` (and optionally `gx17_0..gx17_9`) to `default_gates` in [src/main.rs](file:///home/akarta/Projects/SandyRusty/src/main.rs#L116).
- **Regression Check**:
  Add a test in `tests/test_evidence.rs` asserting that the default gate list covers every gate ID registered in `gates::run`.

---

### [F-09] `AdapterTarget::dims` Ignores `n_kv_heads` for `Wk` and `Wv`, Breaking LoRA on GQA/MQA and Imported Models
- **Priority**: **P2**
- **Confidence**: **Confirmed**
- **Affected Modules**: [src/adapter/mod.rs](file:///home/akarta/Projects/SandyRusty/src/adapter/mod.rs#L40-L49)
- **Reviewed Commit**: `1a0c1cf985975bed93719756b6d146c3cbea176e`
- **Exact Location**: [`AdapterTarget::dims`](file:///home/akarta/Projects/SandyRusty/src/adapter/mod.rs#L40-L49)
- **Expected Behavior**:
  For `AdapterTarget::Wk` and `AdapterTarget::Wv`, the output dimension `d_out` is `cfg.n_kv_heads * cfg.head_dim()` and input dimension `d_in` is `cfg.d_model` (matching `wk` and `wv` shapes `[d_model, n_kv_heads * dh]` in `model::inventory` at [src/model.rs](file:///home/akarta/Projects/SandyRusty/src/model.rs#L69-L70)).
- **Observed Behavior**:
  `AdapterTarget::dims` ([src/adapter/mod.rs](file:///home/akarta/Projects/SandyRusty/src/adapter/mod.rs#L41-L44)) returns `(cfg.d_model, cfg.d_model)` for `Wq | Wk | Wv | Wo`:
  ```rust
  pub fn dims(&self, cfg: &Config) -> (usize, usize) {
      match self {
          AdapterTarget::Wq | AdapterTarget::Wk | AdapterTarget::Wv | AdapterTarget::Wo => {
              (cfg.d_model, cfg.d_model)
          }
  ```
- **Trigger Condition**:
  Creating and merging a `LoraAdapter` targeting `Wk` or `Wv` on any GQA/MQA model (`Config::gqa_v2()`, `Config::mqa_v2()`, or an imported GQA model such as `SmolLM2-135M` where `n_q_heads = 9, n_kv_heads = 3`).
- **Impact**:
  `LoraAdapter::merge` ([src/adapter/mod.rs](file:///home/akarta/Projects/SandyRusty/src/adapter/mod.rs#L365-L388)) fails with `AdapterError::ShapeMismatch` because `mat.d_out` (`cfg.d_model`) does not match `w_Data.len()` (`cfg.d_model * cfg.n_kv_heads * dh`).
- **Fix Direction**:
  Return `(cfg.n_q_heads * cfg.head_dim(), cfg.d_model)` for `Wq`, `(cfg.n_kv_heads * cfg.head_dim(), cfg.d_model)` for `Wk | Wv`, and `(cfg.d_model, cfg.n_q_heads * cfg.head_dim())` for `Wo`.
- **Regression Check**:
  Add a test in `tests/test_x15_lora.rs` that initializes and merges a `LoraAdapter` with `[AdapterTarget::Wq, AdapterTarget::Wk, AdapterTarget::Wv, AdapterTarget::Wo]` onto `Config::gqa_v2()` and `Config::mqa_v2()`.

---

### [F-10] `import_model` Hardcodes `batch_size: 2` on Imported Models, Violating the `MAX_IMPORT_BATCH_CAP = 1` Design Rule
- **Priority**: **P2**
- **Confidence**: **Confirmed**
- **Affected Modules**: [src/import/mod.rs](file:///home/akarta/Projects/SandyRusty/src/import/mod.rs#L1065), [src/config.rs](file:///home/akarta/Projects/SandyRusty/src/config.rs#L25), [src/config.rs](file:///home/akarta/Projects/SandyRusty/src/config.rs#L294)
- **Reviewed Commit**: `1a0c1cf985975bed93719756b6d146c3cbea176e`
- **Exact Locations**:
  - [`MAX_IMPORT_BATCH_CAP`](file:///home/akarta/Projects/SandyRusty/src/config.rs#L25) (`pub const MAX_IMPORT_BATCH_CAP: usize = 1;`)
  - [`Config::validate_import_profile`](file:///home/akarta/Projects/SandyRusty/src/config.rs#L294-L296)
  - [`parse_donor_config`](file:///home/akarta/Projects/SandyRusty/src/import/mod.rs#L1065)
- **Expected Behavior**:
  [src/config.rs](file:///home/akarta/Projects/SandyRusty/src/config.rs#L25) defines `pub const MAX_IMPORT_BATCH_CAP: usize = 1;` per spec (`specv3 §B3 D10`: single-sequence inference profile `batch_size == 1` for imported models). `Config::validate_import_profile` should enforce `MAX_IMPORT_BATCH_CAP`, and `parse_donor_config` should set `batch_size: 1`.
- **Observed Behavior**:
  `parse_donor_config` ([src/import/mod.rs](file:///home/akarta/Projects/SandyRusty/src/import/mod.rs#L1065)) sets `batch_size: 2`, and `Config::validate_import_profile` ([src/config.rs](file:///home/akarta/Projects/SandyRusty/src/config.rs#L294)) checks `self.batch_size > MAX_BATCH_CAP` (`8`) instead of `MAX_IMPORT_BATCH_CAP` (`1`), leaving `MAX_IMPORT_BATCH_CAP` unused.
- **Trigger Condition**:
  Calling `parse_donor_config` or `Config::validate()` with `ext_x17_import: true`.
- **Impact**:
  Minor discrepancy between the declared constant `MAX_IMPORT_BATCH_CAP` and actual validation behavior (note: changing `batch_size` in `parse_donor_config` will change the config fingerprint of synthetic RIMP tests if updated).
- **Fix Direction**:
  Either enforce `MAX_IMPORT_BATCH_CAP` and set `batch_size: 1` in `parse_donor_config` (updating synthetic RIMP test configs), or document why `MAX_BATCH_CAP` is used in `validate_import_profile`.
- **Regression Check**:
  Run `cargo test --test test_x17_import`.

---

### [F-11] CLI `lora train` Uses Synthetic Random Vectors Instead of Token Windows or Model Activations
- **Priority**: **P2**
- **Confidence**: **Confirmed**
- **Affected Modules**: [src/main.rs](file:///home/akarta/Projects/SandyRusty/src/main.rs#L614-L660)
- **Reviewed Commit**: `1a0c1cf985975bed93719756b6d146c3cbea176e`
- **Exact Location**: [`cmd_lora`](file:///home/akarta/Projects/SandyRusty/src/main.rs#L636-L654)
- **Expected Behavior**:
  `sandy_rust lora train` should either train the LoRA matrices end-to-end on token sequences from `--data` using the frozen base model's activations and cross-entropy loss gradients, or clearly document that the CLI command is a synthetic matrix-level smoke test.
- **Observed Behavior**:
  Lines 636–654 of [src/main.rs](file:///home/akarta/Projects/SandyRusty/src/main.rs#L636-L654) ignore token data completely and loop over each `LoraMatrix` in isolation, feeding random normal vectors `x` and `target` (`dy[i] = y[i] - target[i]`) directly into `mat.forward(&x)` and `mat.backward(&x, &dy)`.
- **Trigger Condition**:
  Running `sandy_rust lora train --checkpoint <ckpt> --out <adapter>`.
- **Impact**:
  `sandy_rust lora train` produces a synthetically perturbed `.lora` file rather than a task-adapted LoRA trained on language modeling data.
- **Fix Direction**:
  Wire `LoraAdapter` into the transformer forward/backward pass so `lora train --data <shard>` computes language modeling cross-entropy loss while keeping base weights frozen and updating only `(A, B)`.
- **Regression Check**:
  Add a test in `tests/test_x15_lora.rs` verifying that 20 steps of end-to-end LoRA training on a repeating token sequence decreases cross-entropy loss on that sequence while `base_model.fingerprint()` remains unchanged.

---

### [F-12] Unchecked `usize` Addition on Untrusted Tensor Offsets in `deserialize_rimp` and `checkpoint::parse`
- **Priority**: **P3**
- **Confidence**: **Confirmed**
- **Affected Modules**: [src/import/mod.rs](file:///home/akarta/Projects/SandyRusty/src/import/mod.rs#L1245-L1480), [src/checkpoint.rs](file:///home/akarta/Projects/SandyRusty/src/checkpoint.rs#L171-L176)
- **Reviewed Commit**: `1a0c1cf985975bed93719756b6d146c3cbea176e`
- **Exact Locations**:
  - [`deserialize_rimp`](file:///home/akarta/Projects/SandyRusty/src/import/mod.rs#L1245) (minimum header length check uses `13 * 4` instead of `14 * 4 + 8`)
  - [`deserialize_rimp`](file:///home/akarta/Projects/SandyRusty/src/import/mod.rs#L1474-L1476):
    ```rust
    let tensor_start = payload_start + entry.off;
    let tensor_end = tensor_start + entry.len;
    if tensor_end > content_len { ... }
    ```
  - [`checkpoint::parse`](file:///home/akarta/Projects/SandyRusty/src/checkpoint.rs#L172-L173):
    ```rust
    let start = payload_start + off;
    let end = start + len;
    if end > payload_end { ... }
    ```
- **Expected Behavior**:
  Deserializing untrusted binary buffers (even with a valid trailing SHA-256 checksum crafted by an attacker) must never panic on arithmetic overflow or wrap around `usize::MAX`.
- **Observed Behavior**:
  `entry.off` and `entry.len` are read as `u64` from the binary header and cast to `usize`. If `entry.off` or `entry.len` is near `usize::MAX`, `payload_start + entry.off` or `tensor_start + entry.len` panics with integer overflow in debug builds or wraps around the `> content_len` check in release builds (subsequently panicking on slice indexing `&data[tensor_start..tensor_end]`).
- **Trigger Condition**:
  Passing a crafted `RIMP` or `RCHK` buffer (with recomputed outer SHA-256 checksum and `off = u64::MAX`) to `deserialize_rimp` or `checkpoint::parse`.
- **Impact**:
  Denial-of-service panic instead of returning `Err(ImportError::OffsetOutOfBounds)` / `Err(CheckpointError::Truncated)` (violating the zero-panic parser contract).
- **Fix Direction**:
  Use `checked_add`:
  ```rust
  let tensor_start = payload_start.checked_add(entry.off)
      .ok_or_else(|| ImportError::OffsetOutOfBounds { name: entry.name.clone() })?;
  let tensor_end = tensor_start.checked_add(entry.len)
      .ok_or_else(|| ImportError::OffsetOutOfBounds { name: entry.name.clone() })?;
  ```
  And update the minimum header size check at [src/import/mod.rs](file:///home/akarta/Projects/SandyRusty/src/import/mod.rs#L1245) from `13 * 4` to `14 * 4 + 8`.
- **Regression Check**:
  Add a negative test case in `GX17-1` / `test_x17_import.rs` with `off = u64::MAX` and a valid outer SHA-256 digest, asserting `Err(ImportError::OffsetOutOfBounds)`.

---

### [F-13] Committed Gate Artifacts (`artifacts/*.json`) Reference Historical Revisions and Mixed Dirty States
- **Priority**: **P3**
- **Confidence**: **Confirmed**
- **Affected Modules**: `artifacts/*.json`, [docs/evidence_manifest.md](file:///home/akarta/Projects/SandyRusty/docs/evidence_manifest.md#L1-L214)
- **Reviewed Commit**: `1a0c1cf985975bed93719756b6d146c3cbea176e`
- **Exact Locations**:
  - `artifacts/G1.json`..`artifacts/G14.json` (generated at commit `7e67f5a30143311f935fdcaaf1812f0beae08161`)
  - `artifacts/GX1.json`..`artifacts/GX8.json` (generated across commits `eb88c857...` through `3d7cb55c...`)
- **Expected Behavior**:
  Per Rule 3 and [docs/evidence_manifest.md](file:///home/akarta/Projects/SandyRusty/docs/evidence_manifest.md#L3), committed gate artifacts should correspond to a single clean revision or be regenerated after all gate commits land.
- **Observed Behavior**:
  Because each extension `GX1`..`GX8` was committed sequentially with its own artifact, the JSON files in `artifacts/` record the parent commit active when each gate was run rather than a single unified revision.
- **Trigger Condition**:
  Inspecting `revision` and `dirty_state` across `artifacts/*.json`.
- **Impact**:
  Audit trail fragmentation across commits in `artifacts/`.
- **Fix Direction**:
  After fixing F-08 (adding `gx1..gx8` to `verify`), run `cargo run --release -- verify --out artifacts` on a clean working tree and commit the refreshed artifact bundle in a single commit.
- **Regression Check**:
  Verify `jq -r '.revision' artifacts/*.json | sort -u` yields a single commit hash.

---

## 4. Testing, Verification & Linter Summary

All existing unit tests, integration tests, gate suites, and static analysis checks were executed in the local sandbox environment on commit `1a0c1cf985975bed93719756b6d146c3cbea176e`.

### Commands Executed & Results

| Check / Suite | Command | Exit Code | Passed | Failed | Ignored / Skipped | Duration |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **Static Analysis (Clippy)** | `cargo clippy --all-targets` | `0` | — | `0` warnings / `0` errors | — | 1.46s |
| **Unit Tests (`src/lib.rs`)** | `cargo test --lib` | `0` | 40 | 0 | 0 | 5.34s |
| **Binary Tests (`src/main.rs`)** | `cargo test --bin sandy_rust` | `0` | 0 | 0 | 0 | 0.00s |
| **Crouzeix Math Fixtures** | `cargo test --test test_crouzeix_fixtures` | `0` | 6 | 0 | 0 | 0.00s |
| **Evidence & Gate Contracts** | `cargo test --test test_evidence` | `0` | 6 | 0 | 0 | 120.56s |
| **Tensor & Attention Primitives** | `cargo test --test test_primitives` | `0` | 9 | 0 | 0 | 0.26s |
| **Training Resume & Loader** | `cargo test --test test_training_resume` | `0` | 3 | 0 | 0 | 23.34s |
| **X1: GQA / MQA (`GX1`)** | `cargo test --test test_x1_gqa` | `0` | 8 | 0 | 0 | 10.78s |
| **X2: Untied Embeddings (`GX2`)** | `cargo test --test test_x2_untied` | `0` | 8 | 0 | 0 | 45.58s |
| **X3: QK-Norm (`GX3`)** | `cargo test --test test_x3_qknorm` | `0` | 7 | 0 | 0 | 3.69s |
| **X4: RoPE Scaling (`GX4`)** | `cargo test --test test_x4_rope` | `0` | 7 | 0 | 0 | 0.03s |
| **X5: CPU GEMM (`GX5`)** | `cargo test --test test_x5_gemm` | `0` | 7 | 0 | 0 | 0.33s |
| **X6: Scratch Arena (`GX6`)** | `cargo test --test test_x6_arena` | `0` | 8 | 0 | 0 | 0.02s |
| **X7: Chunked Loss (`GX7`)** | `cargo test --test test_x7_loss` | `0` | 8 | 0 | 0 | 8.45s |
| **X8: FlashAttention-2 (`GX8`)** | `cargo test --test test_x8_flash` | `0` | 9 | 0 | 0 | 0.02s |
| **X12E: Adaptive Sampling (`GX12E`)** | `cargo test --test test_x12e_sampling` | `0` | 8 | 0 | 0 | 7.50s |
| **X13: Activation Hooks (`GX13`)** | `cargo test --test test_x13_hooks` | `0` | 8 | 0 | 0 | 1.32s |
| **X14: Semantic Flip Eval (`GX14`)** | `cargo test --test test_x14_eval` | `0` | 7 | 0 | 0 | 0.02s |
| **X15: LoRA Adapters (`GX15`)** | `cargo test --test test_x15_lora` | `0` | 8 | 0 | 0 | 1.63s |
| **X17: Pretrained Import (`GX17`)** | `cargo test --test test_x17_import` | `0` | 29 | 0 | 0 | 0.95s |
| **Doc-Tests** | `cargo test --doc` | `0` | 0 | 0 | 0 | 0.00s |
| **TOTAL** | `cargo test` | **`0`** | **186** | **0** | **0** | **~229.8s** |

### Open Verification Gates
1. **Gate `G13` (`NOT_RUN`)**: External multi-seed corpus benchmark is intentionally classified as `NOT_RUN` when no external benchmark corpus is mounted ([src/gates.rs](file:///home/akarta/Projects/SandyRusty/src/gates.rs#L698-L725)).
2. **Gate `GX17-5` (`NOT_RUN`)**: Offline real-model parity gate against `HuggingFaceTB/SmolLM2-135M` (`models/smollm2_135m.rimp` + `fixtures/x17_smollm2_135m.json`) returns `NOT_RUN` when the 540 MB donor weights and fixture JSON are not present on disk ([src/import/mod.rs](file:///home/akarta/Projects/SandyRusty/src/import/mod.rs#L3191-L3229)). Synthetic LLaMA safetensors parity is verified by `GX17-4`.

---

## 5. Better Features & Architectural Improvements Roadmap

Below is a prioritized list of concrete architectural improvements and high-value features to elevate `SandyRusty` from a verified reference implementation into a fast, cohesive, end-to-end research and inference engine while preserving its **zero-dependency (`std`-only)** and **`#![forbid(unsafe_code)]`** guarantees.

### Tier 1: High-Priority Architectural Unification & Performance Wins

1. **Unify Extension Wiring Across `Model::forward`, `cache::forward_step`, and `autodiff` (Fixes F-01, F-02, F-03)**
   - **Why**: Currently, `src/model.rs`, `src/cache.rs`, `src/autodiff.rs`, and `src/loss/mod.rs` each maintain separate hand-written forward/backward loops with different subset support for X1–X8 and X17.
   - **Improvement**:
     - Route all matrix multiplications through a unified helper `ops::linear(x, w, cfg)` that automatically dispatches to `gemm::gemm` / `gemm::gemm_trans_b` when `cfg.ext_x5_gemm` is enabled.
     - Route all RoPE rotations through `rope_ext::compute_frequencies` + `rope_with_freqs` + `yarn_temperature_scale` when `cfg.ext_x4_rope` is enabled.
     - Route multi-head attention through `flash::flash_attention` (and `flash_attention_backward` in `autodiff`) when `cfg.ext_x8_flash_attn` is enabled and `HookSite::AttnPattern` is not hooked.
     - Fix the parameter-offset indexing in `autodiff.rs` and `loss/mod.rs` to support the 11-tensor-per-layer layout of `ext_x3_qk_norm`.

2. **KV-Cached Autoregressive Generation & Zero-Copy Decode Step (Fixes F-04, F-05)**
   - **Why**: `generation::generate` currently re-runs `model.forward(&seq, None)` from token 0 at every step ($O(N \cdot T^2)$), and `cache::forward_step` re-hashes all model weights via SHA-256 and transposes the `[vocab_size, d_model]` matrix on every token.
   - **Improvement**:
     - Store `fingerprint: [u8; 32]` directly on `Model` at construction time so `model.fingerprint()` is $O(1)$.
     - Replace `matmul(&x, &head_mat.transpose2d()?)` with `gemm_trans_b(&x, head_mat)` (computing $x W^T$ directly without allocating or transposing $W$).
     - Write `k_rot` and `v_cur` directly into `cache.k[l_idx]` and `cache.v[l_idx]` at row `pos` before running attention over `&cache.k[l_idx].data()[0..(pos + 1) * kv_dim]`, eliminating per-step KV prefix cloning.
     - Switch `generation::generate` (and `import::run_parity_check`) to use `KVCache` prefill + incremental `forward_step`, yielding a **20x–100x inference speedup** on imported models like `SmolLM2-135M`.

3. **Pre-Transposed Weight Layout or `matmul_nt` Everywhere**
   - **Why**: Even in `Model::forward`, `head_mat.transpose2d()` allocates a temporary `[d_model, vocab_size]` copy of the embedding matrix on every call. Furthermore, standard row-major `matmul(A, B)` has stride-`N` memory access on `B`, whereas `matmul_nt(A, B_transposed)` streams both row `A[i, :]` and row `B_transposed[j, :]` contiguously (`stride == 1`), enabling automatic LLVM SIMD vectorization (`AVX2` / `AVX-512` / `NEON`) even under `#![forbid(unsafe_code)]`.
   - **Improvement**: Provide a first-class `ops::matmul_nt(a, b)` (where `b` has shape `[N, K]`) with `chunks_exact(8)` inner dot-product loops so the compiler auto-vectorizes every projection without runtime transposes.

### Tier 2: Training, Fine-Tuning & Evaluation Features

4. **End-to-End Token-Level LoRA Fine-Tuning &Cached Inference with Unmerged LoRA (Fixes F-09, F-11)**
   - **Why**: `src/adapter/mod.rs` currently provides `LoraMatrix::forward`/`backward` and `LoraAdapter::merge`, while `sandy_rust lora train` runs a synthetic random-vector loop.
   - **Improvement**:
     - Fix `AdapterTarget::dims` for GQA (`Wk`, `Wv` -> `n_kv_heads * dh`).
     - Add `autodiff::lora_loss_and_grad(cfg, base_model, adapter, tokens, ...)` that runs the forward pass with frozen base weights plus active LoRA adapters and backpropagates gradients strictly into `LoraMatrix::{a, b}` (avoiding computing full `dW` for frozen base matrices).
     - Wire `sandy_rust lora train --data <shard.rtok>` to train LoRA adapters on real token corpora (including imported `.rimp` models!).

5. **Gradient Accumulation & Micro-Batching in `cmd_train`**
   - **Why**: Currently `cmd_train` processes a single sequence window per optimizer step (`batch_size = 1` window per `loss_and_grad` call), even though `Config` has a `batch_size` field.
   - **Improvement**: Support `--grad-accum-steps <K>` (or loop over `cfg.batch_size` windows per step), accumulating gradients into `grads` before calling `opt.step_with_clipping`. Combine with `ext_x7_chunked_loss` and `ext_x6_arena` for bounded-memory multi-sequence training.

6. **Real Model Evaluation in `eval flip` & Perplexity/BPC Evaluation on Imported Models (Fixes F-06)**
   - **Why**: `sandy_rust eval flip` currently copies `oracle_base_answer` into `PredictionResult`, and `sandy_rust eval` only loads baseline `RCHK` checkpoints with `ByteTokenizer`.
   - **Improvement**:
     - Allow `sandy_rust eval` and `sandy_rust eval flip` to accept either `--checkpoint <rchk>` or `--rimp <rimp> --tokenizer <tokenizer.json>`.
     - In `eval flip`, tokenize `base_prompt` and `variant_prompt`, run greedy decoding with `KVCache`, and score actual model predictions against the oracle answers.

### Tier 3: Inference & Model Ecosystem Features (Pure-Rust, Zero-Dependency)

7. **Weight-Only Quantization (`Q8_0` / `Q4_0`) for Imported `.rimp` Models**
   - **Why**: Converting `SmolLM2-135M` or `SmolLM2-360M` to FP32 triples its disk and RAM footprint compared to `Q8_0` / `Q4_0` and makes large-vocab/MLP projections memory-bandwidth bound on CPU.
   - **Improvement**: Add a `RIMP` v2 tensor dtype tag for block-quantized `Q8_0` (block size 32: 1 `f16`/`f32` scale + 32 `i8` weights) with fused dequantize-dot-product kernels in pure safe Rust. This cuts memory bandwidth by **4x** while remaining 100% `std`-only and `#![forbid(unsafe_code)]`.

8. **Streaming Token Generation & Chat Template Formatter in CLI `generate`**
   - **Why**: `sandy_rust generate` currently buffers all generated tokens until `max_new_tokens` completes before printing. Moreover, imported instruct models (`SmolLM2-135M-Instruct`) expect ChatML formatting (`<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`).
   - **Improvement**:
     - Stream decoded UTF-8 tokens to `stdout` incrementally on each decode step using `Utf8StreamDecoder` (or incremental BPE byte buffer) and `std::io::Write::flush`.
     - Add an optional `--chat` flag and `--stop-token <id|str>` list to `generate` for instruct/chat models.

9. **Expanded Donor Architecture Support in X17 Importer (Qwen2 / Llama-3.2 / Gemma-2)**
   - **Why**: `src/import/mod.rs` already implements `F16`/`BF16`/`F32` safetensors parsing, GQA, RoPE row permutation, and Llama-3 regex BPE pre-tokenization, while `src/attention/qknorm.rs` and `src/attention/rope_ext.rs` implement QK-Norm and RoPE scaling.
   - **Improvement**:
     - Connect `rope_scaling` in `parse_donor_config` ([src/import/mod.rs](file:///home/akarta/Projects/SandyRusty/src/import/mod.rs#L973-L977)) to `Config::rope_scaling` (`ext_x4_rope`) instead of rejecting non-null `rope_scaling`.
     - Connect `q_norm` / `k_norm` in `remap_safetensors_to_model` ([src/import/mod.rs](file:///home/akarta/Projects/SandyRusty/src/import/mod.rs#L1691-L1696)) to `ext_x3_qk_norm` instead of rejecting them.
     - Optional attention bias support (`q_proj.bias`, `k_proj.bias`, `v_proj.bias`) to unlock `Qwen2.5-0.5B` import alongside `SmolLM2` and `Llama-3.2`.

10. **Speculative Decoding (Draft + Target `.rimp` Verification)**
    - **Why**: Since `Model::forward` already computes causal logits for a sequence of tokens in parallel and `KVCache` supports arbitrary position steps, `SandyRusty` has all building blocks for exact speculative decoding.
    - **Improvement**: Allow pairing a tiny draft model (or n-gram prompt lookup) with a target `.rimp` model using lossless rejection sampling, verifying $K$ draft tokens in a single batched `gemm` forward pass.
