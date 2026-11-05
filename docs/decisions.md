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
| D7 | LoRA adaptation scope | **D7-FP32 Proposed / Extension (Subject to confirmation).** Plain FP32 LoRA is implemented as optional parameter-efficient fine-tuning extension X15 (Gate GX15), maintaining the invariant that base weights ($P = 886,272$) remain frozen. The baseline reference model operates independently and satisfies gates G1–G14 without requiring LoRA or extensions. QLoRA, quantized base weights, and GRPO or RL remain deferred. |
| D8 | Amend D6 to D6-R1 | Approved. Baseline remains scratch-trained ($P = 886,272$). Extension X17 operates under separate, explicitly identified import profile. |
| D9 | Source model | SmolLM2-135M (base, Apache-2.0). |
| D10 | Import memory profile | 1 GiB total FP32 parameter budget, 512 MiB single-tensor cap for import profile. Baseline cap remains 128 MiB. |
| D11 | Tooling boundary | Zero external crates in repo; pure Rust std-only. Offline reference fixture generator script allowed outside repository. |
| D12 | Evaluation reporting & gate standards | **D12-R1.** Adopt rigorous reporting invariants from literature audits (Shahriar & Mastoi 2026, Regmi et al. 2026, Lin et al. 2026): (1) Any future structured-output / tool-calling gate must evaluate and report raw parseability separately from recovery-parser success; (2) Evaluation harnesses must verify prompt-format invariance and audit tokenizer fertility across combining marks; (3) Future training runs must mandate matched-compute controls and defect catalogues. |

Consequences: Base model passes all baseline gates independently. Extensions X12-E (adaptive sampling), X13 (activation steering), X14 (flip-rate evaluation), X15 (FP32 LoRA), and X17 (external weight import) are modular extensions leaving base model invariants untouched. Baseline reference model remains byte-identical. Structured output and training evaluations adhere to strict split reporting under D12.
