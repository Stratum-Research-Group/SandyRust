# Decision Record (specv3.md §1)

Recorded 2026-10-07 as repository-declared choices for the recommended baseline (specv3 §1).

| ID | Decision | Choice |
| :-- | :-- | :-- |
| D1 | Transformer task | **A.** Autoregressive decoder-only LM |
| D2 | Rust boundary | **R1.** Strict CPU pure Rust. Zero C/C++/CUDA in repo code or build dependencies (`build.rs`). Disclosed exceptions: Rust compiler, linker, `std`/OS interfaces, platform runtime. |
| D3 | Numerical ownership | **A.** Project-owned tensors and custom reverse-mode autodiff |
| D4 | Release scope | **IT.** Inference + training at tiny reference scale |
| D5 | Resource budget | Tiny profile: 4 GiB host RAM, >=4 GiB system headroom, T=64, B=2, 4 threads, <=500 steps, <=10 MB dataset; no GPU |
| D6 | Weight compatibility | **No.** Random seeded init, train from scratch |
| D7 | LoRA adaptation scope | **D7-FP32 Proposed / Extension (Subject to confirmation).** Plain FP32 LoRA is implemented as optional parameter-efficient fine-tuning extension X15 (Gate GX15), maintaining the invariant that base weights ($P = 886,272$) remain frozen. The baseline reference model operates independently and satisfies gates G1–G14 without requiring LoRA or extensions. QLoRA, quantized base weights, and GRPO or RL remain deferred. |

Consequences: Base model passes all baseline gates independently. Extensions X12-E (adaptive sampling), X13 (activation steering), X14 (flip-rate evaluation), and X15 (FP32 LoRA) are modular extensions leaving base model invariants untouched. GPU spike (G13) is `NOT RUN`; no external weight loading.
