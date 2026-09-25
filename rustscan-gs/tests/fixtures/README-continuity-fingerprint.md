# Continuity fingerprint fixture

`HISTORICAL_DEFAULT_CONTINUITY_HASH` in
`rustscan-gs/src/training/checkpoint.rs` (`fingerprint_tests`) is the blake3
hex of struct-order JSON for `TrainingConfig::default()` with `iterations=0`
and **no** `profiler` field — matching `hash_training_config` at commit
`701d051` (clone + zero iterations + `serde_json::to_vec(&TrainingConfig)`).

Do not regenerate the expected hash via `serde_json::Value` key order.
