# Dependency & License Audit Report

**Specification Reference:** `specv3.md` §1, §13  
**Audited Date:** 2026-10-07  
**Auditor:** SandyRust Verification Engine  
**Policy Baseline:** Decision D2-R1 (Strict CPU Pure Rust)

## 1. Direct & Transitive Dependencies Inventory

| Crate Name | Pinned Version | Upstream Source | Enabled Features | Target Platforms | MSRV | License Expression | Build Script Audit | Network / Telemetry Audit |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `sandy_rust` (root) | 0.1.0 | Local workspace | Default | `x86_64`, `aarch64` | Rust 1.82.0 (2021) | MIT OR Apache-2.0 | Zero `build.rs` scripts | Zero telemetry, zero network calls |

## 2. External Third-Party Dependencies

* **Total External Crates:** `0` (Zero direct or transitive third-party dependencies).
* **Native Build Scripts:** None (`build.rs` absent).
* **C / C++ / CUDA Sources:** Strictly zero in codebase and build tree.
* **Disclosed System Runtime Linkage:**
  - Standard Rust `std` OS runtime interfaces.
  - Linux host dynamically loaded platform libraries (verified via `ldd target/release/sandy_rust`): `linux-vdso.so.1`, `libgcc_s.so.1`, `libm.so.6`, `libc.so.6`, `/lib64/ld-linux-x86-64.so.2`.

## 3. Compliance Verdict

* **Decision D2-R1:** **COMPLIANT** (100% pure Rust implementation, zero external crates, zero native build scripts).
* **Gate G1 Quality Acceptance:** **PASS** (Zero unauthorized native symbols, zero non-disclosed external dependencies).
* **License Compatibility:** **COMPLIANT** (Permissive dual-license MIT OR Apache-2.0).
