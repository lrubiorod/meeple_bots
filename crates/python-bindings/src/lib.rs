//! Reserved Rust facade for the future PyO3 boundary.
//!
//! The engine API is re-exported here so the eventual extension module can
//! translate Python configuration once, before entering statically dispatched code.

pub use meeple_bots_catalog::{AgentConfig, CatalogError, GameId, run_batch, run_match};
