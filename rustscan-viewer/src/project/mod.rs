pub(crate) mod artifacts;
mod events;
mod library;
mod manifest;
mod session;
pub(crate) mod source;
mod state;
mod store;

pub use library::{
    cleanup_delete_tombstone, list_summaries, ProjectLibraryError, ProjectSummary,
    ProjectSummaryEntry, ProjectSummaryStatus,
};
pub use manifest::{
    ArtifactRef, ArtifactValidationError, CompatibilityRecord, ImportConfigSnapshot,
    KeyframeSelectionMode, PnpConfigSnapshot, ProjectErrorRecord, ProjectLease, ProjectManifest,
    ProjectManifestValidationError, ProjectStage, SfmConfigSnapshot, SourceKind, SourceOwnership,
    SourceSpec, StageRecord, StageState, SuggestedAction, PROJECT_SCHEMA_VERSION,
};
pub use session::{ProjectSessionSummary, ProjectStagePresentation};
pub use state::{ChangeKind, ProjectStateError};
pub use store::{ProjectCreateRequest, ProjectStore, ProjectStoreError, ProjectStoreWarning};
