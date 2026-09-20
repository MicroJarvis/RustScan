use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};

use rustix::fs::{self as rustix_fs, FileType, Mode, OFlags};
use serde::Serialize;

use super::ProjectStage;

const LOGS_DIRECTORY: &str = "Logs";
const EVENTS_FILE: &str = "events.jsonl";

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ProjectEvent {
    pub kind: &'static str,
    pub stage: ProjectStage,
    pub attempt: u32,
    pub unix_ms: u64,
}

impl ProjectEvent {
    pub(crate) fn new(kind: &'static str, stage: ProjectStage, attempt: u32, unix_ms: u64) -> Self {
        Self {
            kind,
            stage,
            attempt,
            unix_ms,
        }
    }
}

pub(crate) fn append(root_directory: &File, event: &ProjectEvent) -> io::Result<()> {
    let logs = File::from(
        rustix_fs::openat(
            root_directory,
            LOGS_DIRECTORY,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(io::Error::from)?,
    );

    let mut record = serde_json::to_vec(event)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    record.push(b'\n');

    let needs_separator = match rustix_fs::openat(
        &logs,
        EVENTS_FILE,
        OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
        Mode::empty(),
    ) {
        Ok(opened) => {
            let mut existing = File::from(opened);
            let metadata = existing.metadata()?;
            if !metadata.is_file() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "project event log is not a regular file",
                ));
            }
            if metadata.len() == 0 {
                false
            } else {
                existing.seek(SeekFrom::End(-1))?;
                let mut last = [0_u8; 1];
                existing.read_exact(&mut last)?;
                last[0] != b'\n'
            }
        }
        Err(error) if error == rustix::io::Errno::NOENT => false,
        Err(error) => return Err(io::Error::from(error)),
    };

    let opened = rustix_fs::openat(
        &logs,
        EVENTS_FILE,
        OFlags::WRONLY | OFlags::CREATE | OFlags::APPEND | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::RUSR | Mode::WUSR,
    )
    .map_err(io::Error::from)?;
    let metadata = rustix_fs::fstat(&opened).map_err(io::Error::from)?;
    if !FileType::from_raw_mode(metadata.st_mode).is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "project event log is not a regular file",
        ));
    }
    let mut file = File::from(opened);
    if needs_separator {
        file.write_all(b"\n")?;
    }
    file.write_all(&record)?;
    file.flush()?;
    file.sync_data()
}
