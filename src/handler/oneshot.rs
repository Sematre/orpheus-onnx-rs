/// HTTP endpoint for text-to-speech synthesis using SNAC.
///
/// Accepts text input and voice selection, generates audio tokens through LLM,
/// processes them through SNAC decoder, and streams WAV audio back to client.
use std::pin::Pin;

use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::Response;
use axum::Json;
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{error, info, warn};

use crate::media::wav::{create_wav_header, create_wav_header_with_size};
use crate::processing::audio_frame::{AudioFrameError, AudioFrameStream};
use crate::processing::snac_processor::{ContextualFrameProcessor, SnacError};
use crate::processing::token_generator::{TokenError, TokenGenerator};
use crate::AppState;

// Channel size for backpressure control - prevents memory bloat
const CHANNEL_BUFFER_SIZE: usize = 100;
const BYTES_PER_SAMPLE: usize = 2; // 16-bit PCM
const DEFAULT_RESPONSE_FORMAT: &str = "wav";

type Result<T> = std::result::Result<T, SpeechError>;
type AudioStream = Pin<Box<dyn Stream<Item = std::result::Result<Vec<u8>, std::io::Error>> + Send>>;

/// Comprehensive error types for the speech synthesis pipeline.
#[derive(Debug, Error)]
pub enum SpeechError {
    #[error("Invalid request: {message}")]
    InvalidRequest { message: String },

    #[error("Token generation failed: {0}")]
    TokenGeneration(#[from] TokenError),

    #[error("Audio frame processing failed: {0}")]
    AudioFrame(#[from] AudioFrameError),

    #[error("SNAC processing failed: {0}")]
    SnacProcessing(#[from] SnacError),

    #[error("Audio encoding failed: {0}")]
    AudioEncoding(#[from] std::io::Error),

    #[error("Channel communication failed")]
    ChannelClosed,

    #[error("Stream processing failed: {message}")]
    StreamProcessing { message: String },
}

impl SpeechError {
    /// Convert errors to API-friendly JSON responses.
    fn to_error_response(&self) -> ErrorResponse {
        let (message, error_type) = match self {
            SpeechError::InvalidRequest { message } => (message.clone(), "invalid_request_error"),
            SpeechError::TokenGeneration(_) => ("Token generation failed".to_string(), "processing_error"),
            SpeechError::AudioFrame(_) => ("Audio frame processing failed".to_string(), "processing_error"),
            SpeechError::SnacProcessing(_) => ("Audio processing failed".to_string(), "processing_error"),
            SpeechError::AudioEncoding(_) => ("Audio encoding failed".to_string(), "encoding_error"),
            SpeechError::ChannelClosed => ("Internal communication error".to_string(), "server_error"),
            SpeechError::StreamProcessing { .. } => ("Stream processing failed".to_string(), "processing_error"),
        };

        ErrorResponse {
            error: ErrorDetail {
                message,
                error_type: error_type.to_string(),
            },
        }
    }

    /// Map errors to appropriate HTTP status codes.
    fn to_status_code(&self) -> StatusCode {
        match self {
            SpeechError::InvalidRequest { .. } => StatusCode::BAD_REQUEST,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

/// Request payload for speech synthesis.
#[derive(Debug, Serialize, Deserialize)]
pub struct SpeechRequest {
    #[serde(default)]
    pub model: Option<String>, // Optional model override

    pub input: String, // Text to synthesize
    pub voice: String, // Voice identifier (e.g., "tara", "leah")

    #[serde(default)]
    pub response_format: Option<String>, // Output format (only "wav" supported)
}

impl SpeechRequest {
    fn validate(&self) -> Result<()> {
        // Text must have content
        if self.input.trim().is_empty() {
            return Err(SpeechError::InvalidRequest {
                message: "Input text cannot be empty".to_string(),
            });
        }

        // Voice must be specified
        if self.voice.trim().is_empty() {
            return Err(SpeechError::InvalidRequest {
                message: "Voice cannot be empty".to_string(),
            });
        }

        // Currently only WAV format is supported
        if let Some(format) = &self.response_format {
            if format != "wav" {
                return Err(SpeechError::InvalidRequest {
                    message: format!("Unsupported response format: {}", format),
                });
            }
        }

        Ok(())
    }

    fn response_format(&self) -> &str {
        self.response_format.as_deref().unwrap_or(DEFAULT_RESPONSE_FORMAT)
    }
}

/// Standard error response format.
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    pub message: String,

    #[serde(rename = "type")]
    pub error_type: String,
}

/// Track audio generation statistics for logging.
#[derive(Debug)]
struct AudioStreamMetrics {
    total_samples: u32,
    data_size: u32,
    frames_processed: u32,
}

impl AudioStreamMetrics {
    fn new() -> Self {
        Self {
            total_samples: 0,
            data_size: 0,
            frames_processed: 0,
        }
    }

    fn update_with_samples(&mut self, samples: &[i16]) {
        self.total_samples += samples.len() as u32;
        self.data_size += (samples.len() * BYTES_PER_SAMPLE) as u32;
        self.frames_processed += 1;
    }
}

/// Main endpoint handler for speech synthesis requests.
pub async fn generate_speech(State(state): State<AppState>, Json(request): Json<SpeechRequest>) -> std::result::Result<Response, (StatusCode, Json<ErrorResponse>)> {
    // Validate request parameters
    if let Err(e) = request.validate() {
        error!("Request validation failed: {e}");
        return Err((e.to_status_code(), Json(e.to_error_response())));
    }

    info!("Generating speech for voice '{}' with {} chars", request.voice, request.input.len());

    // Create streaming response body
    match create_audio_stream(state, request).await {
        Ok(stream) => {
            // Build HTTP response with appropriate headers for WAV streaming
            let response = Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "audio/wav")
                .header(header::CACHE_CONTROL, "no-cache") // Prevent caching of generated audio
                .header("X-Response-Format", "wav")
                .body(axum::body::Body::from_stream(stream))
                .map_err(|e| {
                    error!("Failed to build response: {e}");
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(ErrorResponse {
                            error: ErrorDetail {
                                message: "Failed to build response".to_string(),
                                error_type: "server_error".to_string(),
                            },
                        }),
                    )
                })?;

            Ok(response)
        }
        Err(e) => {
            error!("Error generating speech: {e}");
            Err((e.to_status_code(), Json(e.to_error_response())))
        }
    }
}

/// Set up the audio generation pipeline with channel-based streaming.
async fn create_audio_stream(state: AppState, request: SpeechRequest) -> Result<AudioStream> {
    // Create channel for streaming audio chunks from generator task to HTTP response
    let (tx, rx) = mpsc::channel::<std::result::Result<Vec<u8>, std::io::Error>>(CHANNEL_BUFFER_SIZE);

    // Spawn background task to generate audio
    // This allows immediate HTTP response with streaming body
    let tx_clone = tx.clone();
    tokio::spawn(async move {
        if let Err(e) = generate_and_stream_audio(state, request, tx_clone).await {
            error!("Error in audio generation task: {e}");
            // Convert our error to io::Error for channel compatibility
            let io_error = std::io::Error::other(format!("Audio generation failed: {}", e));
            let _ = tx.send(Err(io_error)).await;
        }
    });

    // Convert channel receiver to Stream for axum body
    Ok(Box::pin(ReceiverStream::new(rx)))
}

/// Core audio generation pipeline running in background task.
async fn generate_and_stream_audio(state: AppState, request: SpeechRequest, tx: mpsc::Sender<std::result::Result<Vec<u8>, std::io::Error>>) -> Result<()> {
    // Initialize processing components
    let mut contextual_processor = ContextualFrameProcessor::new(state.snac_processor);
    let generator = TokenGenerator::new(state.client)?;
    let mut metrics = AudioStreamMetrics::new();

    // Send WAV header first - client needs this to start playing audio
    send_wav_header(&tx).await?;

    // Start token generation from LLM
    let token_stream = generator.generate_token_stream(&request.voice, &request.input).await?;

    // Convert token stream to audio frame stream
    let mut audio_frame_stream = AudioFrameStream::new(token_stream);

    // Process frames and stream audio chunks
    process_audio_frames(&mut audio_frame_stream, &mut contextual_processor, &tx, &mut metrics).await?;

    info!(
        "Audio generation completed: {} samples, {} bytes, {} frames",
        metrics.total_samples, metrics.data_size, metrics.frames_processed
    );

    Ok(())
}

/// Send WAV header with placeholder size (streaming doesn't know final size).
async fn send_wav_header(tx: &mpsc::Sender<std::result::Result<Vec<u8>, std::io::Error>>) -> Result<()> {
    // Create WAV header with max size placeholder
    let wav_header = create_wav_header().map_err(SpeechError::AudioEncoding)?;
    tx.send(Ok(wav_header)).await.map_err(|_| SpeechError::ChannelClosed)?;

    Ok(())
}

/// Main processing loop - converts frames to audio and streams to client.
async fn process_audio_frames(
    audio_frame_stream: &mut AudioFrameStream<impl Stream<Item = std::result::Result<u32, TokenError>> + Unpin>,
    contextual_processor: &mut ContextualFrameProcessor,
    tx: &mpsc::Sender<std::result::Result<Vec<u8>, std::io::Error>>,
    metrics: &mut AudioStreamMetrics,
) -> Result<()> {
    while let Some(frame_result) = audio_frame_stream.next().await {
        match frame_result {
            Ok(frame) => {
                // Process frame through SNAC decoder
                if let Some(samples) = contextual_processor.process_frame(frame)? {
                    // Convert 16-bit samples to bytes for WAV format
                    let audio_data = convert_samples_to_bytes(&samples);
                    metrics.update_with_samples(&samples);

                    // Stream chunk to client
                    if tx.send(Ok(audio_data)).await.is_err() {
                        // Client disconnected - stop processing
                        warn!("Client disconnected during audio streaming");
                        break;
                    }
                }
            }

            Err(AudioFrameError::TokenOutOfRange { token, index, computed_id, .. }) => {
                // Token encoding errors are recoverable - log and continue
                // This handles occasional bad tokens from LLM
                debug!("Ignoring invalid token {token} at index {index}: computed ID {computed_id} is out of valid range, continuing with next token",);

                continue;
            }

            Err(e) => {
                // Other errors are fatal
                return Err(e.into());
            }
        }
    }

    Ok(())
}

/// Convert 16-bit PCM samples to little-endian bytes for WAV format.
fn convert_samples_to_bytes(samples: &[i16]) -> Vec<u8> {
    let mut audio_data = Vec::with_capacity(samples.len() * BYTES_PER_SAMPLE);
    for &sample in samples {
        // WAV uses little-endian byte order
        audio_data.extend_from_slice(&sample.to_le_bytes());
    }

    audio_data
}

/// Alternative: Send corrected WAV header with actual size after generation.
/// Note: This doesn't work well with HTTP streaming since headers must come first.
#[allow(dead_code)]
async fn send_corrected_wav_header(tx: &mpsc::Sender<std::result::Result<Vec<u8>, std::io::Error>>, data_size: u32) -> Result<()> {
    let corrected_header = create_wav_header_with_size(data_size).map_err(SpeechError::AudioEncoding)?;

    tx.send(Ok(corrected_header)).await.map_err(|_| SpeechError::ChannelClosed)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speech_request_validation() {
        // Valid request
        let valid_request = SpeechRequest {
            model: Some("test-model".to_string()),
            input: "Hello world".to_string(),
            voice: "coral".to_string(),
            response_format: Some("wav".to_string()),
        };
        assert!(valid_request.validate().is_ok());

        // Empty input
        let empty_input = SpeechRequest {
            model: None,
            input: "   ".to_string(),
            voice: "coral".to_string(),
            response_format: None,
        };
        assert!(matches!(empty_input.validate(), Err(SpeechError::InvalidRequest { .. })));

        // Empty voice
        let empty_voice = SpeechRequest {
            model: None,
            input: "Hello".to_string(),
            voice: "".to_string(),
            response_format: None,
        };
        assert!(matches!(empty_voice.validate(), Err(SpeechError::InvalidRequest { .. })));

        // Invalid format
        let invalid_format = SpeechRequest {
            model: None,
            input: "Hello".to_string(),
            voice: "coral".to_string(),
            response_format: Some("mp3".to_string()),
        };
        assert!(matches!(invalid_format.validate(), Err(SpeechError::InvalidRequest { .. })));
    }

    #[test]
    fn test_audio_stream_metrics() {
        let mut metrics = AudioStreamMetrics::new();
        assert_eq!(metrics.total_samples, 0);
        assert_eq!(metrics.data_size, 0);
        assert_eq!(metrics.frames_processed, 0);

        let samples = vec![100i16, 200i16, 300i16];
        metrics.update_with_samples(&samples);

        assert_eq!(metrics.total_samples, 3);
        assert_eq!(metrics.data_size, 6); // 3 samples * 2 bytes each
        assert_eq!(metrics.frames_processed, 1);
    }

    #[test]
    fn test_convert_samples_to_bytes() {
        let samples = vec![0x1234i16, 0x5678i16];
        let bytes = convert_samples_to_bytes(&samples);

        // Should be little-endian encoded
        assert_eq!(bytes, vec![0x34, 0x12, 0x78, 0x56]);
    }

    #[test]
    fn test_error_to_response_conversion() {
        let error = SpeechError::InvalidRequest {
            message: "Test error".to_string(),
        };

        let response = error.to_error_response();
        assert_eq!(response.error.message, "Test error");
        assert_eq!(response.error.error_type, "invalid_request_error");
        assert_eq!(error.to_status_code(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_speech_endpoint_integration() {
        // This test would need a proper AppState setup
        // Skipping actual HTTP test for now due to dependencies

        let request = SpeechRequest {
            model: Some("test-model".to_string()),
            input: "Hello world".to_string(),
            voice: "coral".to_string(),
            response_format: Some("wav".to_string()),
        };

        // Test validation
        assert!(request.validate().is_ok());
        assert_eq!(request.response_format(), "wav");
    }
}
