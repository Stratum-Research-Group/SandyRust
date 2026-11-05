# Decision Record (specv3.md §1)

Recorded 2026-10-07 as repository-declared choices for the recommended baseline (specv3 §1).

| ID | Decision | Choice |
| :-- | :-- | :-- |
| D1 | Transformer task | **A.** Autoregressive decoder-only LM |
| D2 | Rust boundary | **R1.** Strict CPU pure Rust. Zero C/C++/CUDA in repo code or build dependencies (`build.rs`). Disclosed exceptions: Rust compiler, linker, `std`/OS interfaces, platform runtime. |
| D3 | Numerical ownership | **A.** Project-owned tensors and custom reverse-mode autodiff |
| D4 | Release scope | **IT.** Inference + training at tiny reference scale |
| D5 | Resource budget | Tiny profile: 4 GiB host RAM, >=4 GiB system headroom, T=64, B=2, 4 threads, <=500 steps, <=10 MB dataset; no GPU |
| D6 | Weight compatibility | **D6-R1.** Baseline remains random-init scratch training. Optional extension X17 may load external pretrained weights into a separate, explicitly labelled import profile; imported checkpoints are never mislabelled baseline. |
| D7 | LoRA adaptation scope | **D7-FP32 Approved.** Plain FP32 LoRA is implemented as optional parameter-efficient fine-tuning extension X15 (Gate GX15), maintaining the invariant that base weights ($P = 886,272$) remain frozen. The baseline reference model operates independently and satisfies gates G1–G14 without requiring LoRA or extensions. QLoRA, quantized base weights, and GRPO or RL remain deferred. |
| D8 | Amend D6 to D6-R1 | Approved. Baseline remains scratch-trained ($P = 886,272$). Extension X17 operates under separate, explicitly identified import profile. |
| D9 | Source model | SmolLM2-135M (base, Apache-2.0). |
| D10 | Import memory profile | 1 GiB total FP32 parameter budget, 512 MiB single-tensor cap for import profile. Baseline cap remains 128 MiB. |
| D11 | Tooling boundary | Zero external crates in repo; pure Rust std-only. Offline reference fixture generator script allowed outside repository. |
| D12 | Evaluation reporting & gate standards | **D12-R1.** Adopt rigorous reporting invariants from literature audits (Shahriar & Mastoi 2026, Regmi et al. 2026, Lin et al. 2026): (1) Any future structured-output / tool-calling gate must evaluate and report raw parseability separately from recovery-parser success; (2) Evaluation harnesses must verify prompt-format invariance and audit tokenizer fertility across combining marks; (3) Future training runs must mandate matched-compute controls and defect catalogues. |
| D13 | Associative Fast-Weight Memory | **D13-X16 Approved.** Extension X16 (Associative Fast-Weight Memory) approved as optional extension (Gate GX16). Feature key: `ext_x16_memory`. Baseline parameter count $P = 886,272$ remains untouched when disabled. Zero-gate ($g=0$) guarantees exact baseline equivalence. |
| D14 | BPE Tokenizer & Safe Buffered Streaming | **D14-X9 Approved.** Extension X9 (BPE Tokenizer & Buffered Shard Streaming) approved as optional modular extension (Gate GX9). Safe `std`-only buffered streaming with strict size cap, file immutability pre-check, and atomic cursor rewind replaces unsafe mmap dependencies. Byte-level BPE subword pair merges expand vocabulary ($V > 259$) while strictly isolating special tokens (BOS=256, EOS=257, PAD=258). |
| D15 | Mixed Precision & Dynamic Loss Scaling | **D15-X10 Approved.** Extension X10 (Mixed Precision FP16/BF16 Simulation & Dynamic Loss Scaling) approved as optional modular extension (Gate GX10). Stores authoritative FP32 master weights and optimizer moments while simulating FP16/BF16 GEMMs. Reductions, RMSNorm, and loss execute in FP32. On non-finite gradients, step is skipped, weights and moments are frozen, data loader cursor advances (tokens consumed), loss scale is halved, and 5 consecutive skips trigger `ExcessiveConsecutiveSkips(5)`. |
| D16 | Paged KV-Cache Management | **D16-X11 Approved.** Extension X11 (Paged KV-Cache Management) approved as optional modular extension (Gate GX11). Implements fixed block paging ($B_{\text{page}} = 16$), logical-to-physical block table mapping, transactional reserve-before-append with atomic rollback on pool exhaustion, generational multi-session isolation, and leak-free page recycling without rebasing live sessions. |

Consequences: Base model passes all baseline gates independently. All 17 modular extensions (X1–X11, X12-E, X13–X17) are 100% complete and verified, leaving base model invariants untouched ($P = 886,272$). Baseline reference model remains byte-identical. Structured output and training evaluations adhere to strict split reporting under D12.
