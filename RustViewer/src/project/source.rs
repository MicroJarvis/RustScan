use std::fs::{self, File};
use std::io::{self, Read, Seek};
use std::path::PathBuf;
use std::time::UNIX_EPOCH;

use serde::{Deserialize, Serialize};

#[cfg(target_os = "macos")]
use objc2::rc::Retained;
#[cfg(target_os = "macos")]
use objc2::runtime::Bool;
#[cfg(target_os = "macos")]
use objc2_foundation::{
    NSData, NSString, NSURLBookmarkCreationOptions, NSURLBookmarkResolutionOptions, NSURL,
};

use super::SourceOwnership;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ImportedSourceRecord {
    pub(crate) schema_version: u32,
    pub(crate) ownership: SourceOwnership,
    pub(crate) identity: String,
    pub(crate) frames: Vec<ImportedSourceFrame>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ImportedSourceFrame {
    pub(crate) id: u32,
    pub(crate) source_name: String,
    pub(crate) original_path: String,
    pub(crate) managed_copy: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct SourceBookmark {
    schema_version: u32,
    canonical_paths: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    macos_security_scoped: Option<Vec<Vec<u8>>>,
}

impl SourceBookmark {
    pub(crate) fn from_canonical_paths(
        canonical_paths: Vec<String>,
    ) -> Result<Self, SourceBookmarkError> {
        #[cfg(target_os = "macos")]
        let macos_security_scoped = Some(
            canonical_paths
                .iter()
                .map(|path| create_macos_security_scoped_bookmark(path))
                .collect::<Result<Vec<_>, _>>()?,
        );
        #[cfg(not(target_os = "macos"))]
        let macos_security_scoped = None;
        Ok(Self {
            schema_version: 1,
            canonical_paths,
            macos_security_scoped,
        })
    }

    pub(crate) fn encode(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }

    pub(crate) fn decode(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(bytes)
    }

    /// Executes `operation` while every resolved macOS security scope remains active.
    ///
    /// Callers must perform all filesystem work on referenced paths from inside this callback.
    pub(crate) fn with_resolved_paths<T>(
        &self,
        operation: impl FnOnce(&[String]) -> T,
    ) -> Result<T, SourceBookmarkError> {
        #[cfg(target_os = "macos")]
        if let Some(bookmarks) = &self.macos_security_scoped {
            let guards = bookmarks
                .iter()
                .map(|bookmark| resolve_macos_security_scoped_bookmark(bookmark))
                .collect::<Result<Vec<_>, _>>()?;
            let paths = guards
                .iter()
                .map(|guard| guard.path.clone())
                .collect::<Vec<_>>();
            return Ok(operation(&paths));
        }
        Ok(operation(&self.canonical_paths))
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum SourceBookmarkError {
    #[error("bookmark serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
    #[cfg(target_os = "macos")]
    #[error("macOS security-scoped bookmark failed: {0}")]
    Macos(String),
    #[cfg(target_os = "macos")]
    #[error("macOS security-scoped bookmark is stale; relink or reimport the source")]
    Stale,
}

#[cfg(target_os = "macos")]
fn create_macos_security_scoped_bookmark(path: &str) -> Result<Vec<u8>, SourceBookmarkError> {
    let path = NSString::from_str(path);
    let url = NSURL::fileURLWithPath(&path);
    url.bookmarkDataWithOptions_includingResourceValuesForKeys_relativeToURL_error(
        NSURLBookmarkCreationOptions::WithSecurityScope
            | NSURLBookmarkCreationOptions::SecurityScopeAllowOnlyReadAccess,
        None,
        None,
    )
    .map(|data| data.to_vec())
    .map_err(|error| SourceBookmarkError::Macos(error.to_string()))
}

#[cfg(target_os = "macos")]
struct SecurityScopedPath {
    url: Retained<NSURL>,
    path: String,
}

#[cfg(target_os = "macos")]
impl Drop for SecurityScopedPath {
    fn drop(&mut self) {
        unsafe { self.url.stopAccessingSecurityScopedResource() };
    }
}

#[cfg(target_os = "macos")]
fn resolve_macos_security_scoped_bookmark(
    bookmark: &[u8],
) -> Result<SecurityScopedPath, SourceBookmarkError> {
    let data = NSData::with_bytes(bookmark);
    let mut stale = Bool::NO;
    let url = unsafe {
        NSURL::URLByResolvingBookmarkData_options_relativeToURL_bookmarkDataIsStale_error(
            &data,
            NSURLBookmarkResolutionOptions::WithSecurityScope
                | NSURLBookmarkResolutionOptions::WithoutUI,
            None,
            &mut stale,
        )
    }
    .map_err(|error| SourceBookmarkError::Macos(error.to_string()))?;
    if stale.as_bool() {
        return Err(SourceBookmarkError::Stale);
    }
    if !unsafe { url.startAccessingSecurityScopedResource() } {
        return Err(SourceBookmarkError::Macos(
            "security-scoped bookmark access could not be started".to_owned(),
        ));
    }
    let path = match url.path().map(|path| path.to_string()) {
        Some(path) => path,
        None => {
            unsafe { url.stopAccessingSecurityScopedResource() };
            return Err(SourceBookmarkError::Macos(
                "bookmark did not resolve to a file path".to_owned(),
            ));
        }
    };
    Ok(SecurityScopedPath { url, path })
}

pub(crate) fn image_sequence_identity(sources: &[(String, PathBuf)]) -> io::Result<String> {
    let mut hasher = blake3::Hasher::new();
    for (name, path) in sources {
        let metadata = fs::metadata(path)?;
        hasher.update(name.as_bytes());
        hasher.update(&metadata.len().to_le_bytes());
        let modified_ns = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .map(|duration| duration.as_nanos())
            .unwrap_or_default();
        hasher.update(&modified_ns.to_le_bytes());
        let mut file = File::open(path)?;
        let mut first = vec![0_u8; metadata.len().min(64 * 1024) as usize];
        file.read_exact(&mut first)?;
        hasher.update(&first);
        if metadata.len() > 64 * 1024 {
            file.seek(io::SeekFrom::End(-(64 * 1024_i64)))?;
            let mut last = vec![0_u8; 64 * 1024];
            file.read_exact(&mut last)?;
            hasher.update(&last);
        }
    }
    Ok(hasher.finalize().to_hex().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolved_paths_remain_usable_for_the_duration_of_the_callback() {
        let temporary = tempfile::tempdir().unwrap();
        let source = temporary.path().join("frame1.png");
        fs::write(&source, b"fixture image data").unwrap();
        let canonical = fs::canonicalize(&source)
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let bookmark = SourceBookmark::from_canonical_paths(vec![canonical]).unwrap();

        let bytes = bookmark
            .with_resolved_paths(|paths| fs::read(&paths[0]))
            .unwrap()
            .unwrap();

        assert_eq!(bytes, b"fixture image data");
    }
}
