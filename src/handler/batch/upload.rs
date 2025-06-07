use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::multipart::Field;
use axum::{extract::Multipart, http::StatusCode, response::IntoResponse, Json};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::fs::{self, File};
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

const UPLOAD_DIR: &str = "uploads";
const MAX_FILE_SIZE: usize = 100 * 1024 * 1024; // 100MB

#[derive(Debug, Serialize, Deserialize)]
pub struct UploadResponse {
    id: String,
    object: &'static str,
    bytes: usize,
    created_at: u64,
    filename: String,
    purpose: String,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

type Result<T> = std::result::Result<T, UploadError>;

#[derive(Debug, Error)]
enum UploadError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Multipart parsing error: {0}")]
    Multipart(#[from] axum::extract::multipart::MultipartError),

    #[error("File size exceeds maximum limit of {max_size} bytes")]
    FileSizeExceeded { max_size: usize },

    #[error("No file provided in the request")]
    NoFileProvided,

    #[error("Invalid or missing filename")]
    InvalidFileName,

    #[error("Failed to get current timestamp")]
    TimestampError,
}

impl IntoResponse for UploadError {
    fn into_response(self) -> axum::response::Response {
        let status = match self {
            UploadError::Io(_) => StatusCode::INTERNAL_SERVER_ERROR,
            UploadError::Multipart(_) => StatusCode::BAD_REQUEST,
            UploadError::FileSizeExceeded { .. } => StatusCode::PAYLOAD_TOO_LARGE,
            UploadError::NoFileProvided => StatusCode::BAD_REQUEST,
            UploadError::InvalidFileName => StatusCode::BAD_REQUEST,
            UploadError::TimestampError => StatusCode::INTERNAL_SERVER_ERROR,
        };

        (status, Json(ErrorResponse { error: self.to_string() })).into_response()
    }
}

pub async fn upload_file(mut multipart: Multipart) -> impl IntoResponse {
    match handle_upload(&mut multipart).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(e) => e.into_response(),
    }
}

async fn handle_upload(multipart: &mut Multipart) -> Result<UploadResponse> {
    let uuid = Uuid::new_v4();

    // Ensure upload directory exists
    fs::create_dir_all(UPLOAD_DIR).await?;

    let mut filename = None;
    let mut total_bytes = 0usize;

    while let Some(field) = multipart.next_field().await? {
        let field_name = field.name().unwrap_or_default();

        if field_name == "file" {
            let field_filename = field.file_name().ok_or(UploadError::InvalidFileName)?.to_owned();

            if field_filename.is_empty() {
                return Err(UploadError::InvalidFileName);
            }

            filename = Some(field_filename);

            let file_path = create_file_path(&uuid)?;
            total_bytes = save_field_to_file(field, &file_path).await?;

            break; // Only process the first file field
        }
    }

    let filename = filename.ok_or(UploadError::NoFileProvided)?;

    let response = UploadResponse {
        id: format!("file-{}", uuid),
        object: "file",
        bytes: total_bytes,
        created_at: get_current_timestamp()?,
        filename,
        purpose: "batch".to_owned(),
    };

    Ok(response)
}

fn create_file_path(uuid: &Uuid) -> Result<PathBuf> {
    let mut file_path = PathBuf::from(UPLOAD_DIR);
    file_path.push(format!("{}.jsonl", uuid));
    Ok(file_path)
}

async fn save_field_to_file(mut field: Field<'_>, file_path: &PathBuf) -> Result<usize> {
    let mut file = File::create(file_path).await?;
    let mut total_bytes = 0usize;

    // Process the field in chunks
    while let Some(chunk) = field.chunk().await? {
        // Check file size limit
        if total_bytes + chunk.len() > MAX_FILE_SIZE {
            // Clean up the partial file
            let _ = fs::remove_file(file_path).await;
            return Err(UploadError::FileSizeExceeded { max_size: MAX_FILE_SIZE });
        }

        file.write_all(&chunk).await?;
        total_bytes += chunk.len();
    }

    // Ensure all data is written to disk
    file.flush().await?;

    Ok(total_bytes)
}

fn get_current_timestamp() -> Result<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|_| UploadError::TimestampError)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_file_path() {
        let uuid = Uuid::new_v4();
        let path = create_file_path(&uuid).unwrap();

        assert!(path.to_string_lossy().contains(&uuid.to_string()));
        assert!(path.to_string_lossy().ends_with(".jsonl"));
    }

    #[test]
    fn test_get_current_timestamp() {
        let timestamp = get_current_timestamp().unwrap();
        assert!(timestamp > 0);
    }
}
