//! SandyRust: standalone pure-Rust decoder-only transformer (specv3.md).
pub mod attention;
pub mod autodiff;
pub mod cache;
pub mod checkpoint;
pub mod config;
pub mod data;
pub mod gates;
pub mod generation;
pub mod hooks;
pub mod model;
pub mod optimizer;
pub mod rng;
pub mod sha256;
pub mod tensor;
pub mod tokenizer;
