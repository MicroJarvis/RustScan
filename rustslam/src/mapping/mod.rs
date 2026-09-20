//! Mapping module
//!
//! This module handles the mapping thread in the SLAM system.
//! It includes:
//! - Local Mapping: processes keyframes, triangulates points, runs local BA

pub mod local_mapping;

pub use local_mapping::{LocalMapping, LocalMappingConfig, MappingState};

#[cfg(test)]
mod tests {
    #[test]
    fn test_mapping_module_imports() {
        use crate::mapping::{LocalMapping, LocalMappingConfig};

        let config = LocalMappingConfig::default();
        let _mapping = LocalMapping::new(config);
    }
}
