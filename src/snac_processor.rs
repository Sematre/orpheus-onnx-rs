/// SNAC (Speech and Audio Codec) decoder for processing audio frames through ONNX models.
///
/// Provides both single-frame and streaming decode capabilities with context windows.
use ndarray::{s, Array2, CowArray, IxDyn};
use ort::{environment::Environment, session::SessionBuilder, value::Value};
use std::path::PathBuf;
use std::sync::Arc;
use thiserror::Error;
use tracing::{debug, info};

use crate::audio_frame::AudioFrame;

// SNAC decoder needs 4 frames of context for high-quality output
const SNAC_CONTEXT_FRAMES: usize = 4;

// Audio slice bounds - samples 2048-4096 correspond to the newest frame
const AUDIO_SLICE_START: usize = 2048;
const AUDIO_SLICE_END: usize = 4096;

// Scale factor for float [-1.0, 1.0] to 16-bit PCM conversion
const AUDIO_SCALE_FACTOR: f32 = i16::MAX as f32;

type Result<T> = std::result::Result<T, SnacError>;

/// Errors that can occur during SNAC processing.
#[derive(Debug, Error)]
pub enum SnacError {
    #[error("ONNX environment creation failed: {0}")]
    EnvironmentCreation(ort::OrtError),

    #[error("Failed to load SNAC decoder model from {path:?}: {source}")]
    ModelLoad {
        path: PathBuf,
        #[source]
        source: ort::OrtError,
    },

    #[error("Failed to create ndarray for {code_type} with length {length}")]
    ArrayCreation { code_type: CodeType, length: usize },

    #[error("Failed to create ONNX tensor for {code_type}")]
    TensorCreation { code_type: CodeType },

    #[error("SNAC decoder inference failed: {0}")]
    InferenceFailed(ort::OrtError),

    #[error("Expected {expected} outputs from decoder, got {actual}")]
    UnexpectedOutputCount { expected: usize, actual: usize },

    #[error("Failed to extract f32 tensor from decoder output")]
    TensorExtraction,

    #[error("Expected 3D output tensor, got {dimensions}D with shape {shape:?}")]
    InvalidTensorShape { dimensions: usize, shape: Vec<usize> },

    #[error("Cannot process empty frame list")]
    EmptyFrameList,

    #[error("Audio slice [{start}:{end}] is invalid for audio length {length}")]
    InvalidAudioSlice { start: usize, end: usize, length: usize },
}

/// SNAC code hierarchy levels.
#[derive(Debug, Clone, Copy)]
pub enum CodeType {
    Codes0, // 1 code per frame
    Codes1, // 2 codes per frame
    Codes2, // 4 codes per frame
}

impl std::fmt::Display for CodeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CodeType::Codes0 => write!(f, "codes_0"),
            CodeType::Codes1 => write!(f, "codes_1"),
            CodeType::Codes2 => write!(f, "codes_2"),
        }
    }
}

/// SNAC decoder that converts encoded frames to PCM audio using ONNX.
pub struct SnacProcessor {
    decoder_session: ort::session::Session,
    _environment: Arc<Environment>, // Must outlive session, hence the Arc
}

impl SnacProcessor {
    /// Loads a SNAC decoder model from the specified path.
    pub fn new(decoder_path: &PathBuf) -> Result<Self> {
        let environment = Arc::new(Environment::builder().with_name("snac_decoder").build().map_err(SnacError::EnvironmentCreation)?);

        let decoder_session = SessionBuilder::new(&environment)
            .map_err(SnacError::EnvironmentCreation)?
            .with_model_from_file(decoder_path)
            .map_err(|e| SnacError::ModelLoad {
                path: decoder_path.clone(),
                source: e,
            })?;

        info!("SNAC decoder loaded from: {:?}", decoder_path);

        Ok(SnacProcessor {
            decoder_session,
            _environment: environment,
        })
    }

    /// Decodes audio frames into 16-bit PCM samples.
    pub fn process_frames(&self, frames: &[AudioFrame]) -> Result<Vec<i16>> {
        if frames.is_empty() {
            return Err(SnacError::EmptyFrameList);
        }

        // Extract codes from frames and convert u32 -> i64 for ONNX
        let (codes0, codes1, codes2) = self.collect_codes(frames);
        let (array0, array1, array2) = self.create_arrays(codes0, codes1, codes2)?;

        // Convert to dynamic arrays with copy-on-write semantics for ONNX efficiency
        let cow0: CowArray<i64, IxDyn> = array0.into_dyn().into();
        let cow1: CowArray<i64, IxDyn> = array1.into_dyn().into();
        let cow2: CowArray<i64, IxDyn> = array2.into_dyn().into();

        // Create ONNX tensors - the model expects 3 separate code inputs
        let tensor0 = Value::from_array(self.decoder_session.allocator(), &cow0).map_err(|_| SnacError::TensorCreation { code_type: CodeType::Codes0 })?;
        let tensor1 = Value::from_array(self.decoder_session.allocator(), &cow1).map_err(|_| SnacError::TensorCreation { code_type: CodeType::Codes1 })?;
        let tensor2 = Value::from_array(self.decoder_session.allocator(), &cow2).map_err(|_| SnacError::TensorCreation { code_type: CodeType::Codes2 })?;

        // Run inference - SNAC decoder combines all 3 code levels to reconstruct audio
        let inputs = vec![tensor0, tensor1, tensor2];
        let mut outputs = self.decoder_session.run(inputs).map_err(SnacError::InferenceFailed)?;

        // SNAC decoder should only produce one output tensor
        if outputs.len() != 1 {
            return Err(SnacError::UnexpectedOutputCount {
                expected: 1,
                actual: outputs.len(),
            });
        }

        self.extract_audio_samples(outputs.pop().unwrap())
    }

    /// Concatenates codes from frames into separate vectors by level.
    fn collect_codes(&self, frames: &[AudioFrame]) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
        // Calculate total lengths based on code hierarchy:
        // - codes_0: 1 per frame
        // - codes_1: 2 per frame
        // - codes_2: 4 per frame
        let total_len0 = frames.len();
        let total_len1 = frames.len() * 2;
        let total_len2 = frames.len() * 4;

        let mut codes0 = Vec::with_capacity(total_len0);
        let mut codes1 = Vec::with_capacity(total_len1);
        let mut codes2 = Vec::with_capacity(total_len2);

        // Flatten all frames' codes into contiguous vectors, converting u32 to i64
        for frame in frames {
            codes0.extend(frame.codes_0.iter().map(|&x| x as i64));
            codes1.extend(frame.codes_1.iter().map(|&x| x as i64));
            codes2.extend(frame.codes_2.iter().map(|&x| x as i64));
        }

        (codes0, codes1, codes2)
    }

    /// Creates 2D arrays for ONNX input tensors.
    fn create_arrays(&self, codes0: Vec<i64>, codes1: Vec<i64>, codes2: Vec<i64>) -> Result<(Array2<i64>, Array2<i64>, Array2<i64>)> {
        let len0 = codes0.len();
        let len1 = codes1.len();
        let len2 = codes2.len();

        // Shape is (batch_size=1, sequence_length) for each code level
        // ONNX expects batch dimension even for single inference
        let array0 = Array2::from_shape_vec((1, len0), codes0).map_err(|_| SnacError::ArrayCreation {
            code_type: CodeType::Codes0,
            length: len0,
        })?;

        let array1 = Array2::from_shape_vec((1, len1), codes1).map_err(|_| SnacError::ArrayCreation {
            code_type: CodeType::Codes1,
            length: len1,
        })?;

        let array2 = Array2::from_shape_vec((1, len2), codes2).map_err(|_| SnacError::ArrayCreation {
            code_type: CodeType::Codes2,
            length: len2,
        })?;

        Ok((array0, array1, array2))
    }

    /// Extracts PCM samples from the decoder output tensor.
    fn extract_audio_samples(&self, output: Value) -> Result<Vec<i16>> {
        let output_tensor = output.try_extract::<f32>().map_err(|_| SnacError::TensorExtraction)?;

        let tensor_view = output_tensor.view();
        let shape = tensor_view.shape();

        debug!("SNAC decoder output shape: {shape:?}");

        self.validate_tensor_shape(shape)?;

        // Extract [batch=0, channel=0, all_samples] - we only process mono audio
        // SNAC outputs normalized float audio that we need to scale to 16-bit range
        let audio_samples: Vec<i16> = tensor_view.slice(s![0, 0, ..]).iter().map(|&sample| self.scale_and_clamp_sample(sample)).collect();

        Ok(audio_samples)
    }

    /// Ensures output tensor is 3D [batch, channels, samples].
    fn validate_tensor_shape(&self, shape: &[usize]) -> Result<()> {
        if shape.len() != 3 {
            return Err(SnacError::InvalidTensorShape {
                dimensions: shape.len(),
                shape: shape.to_vec(),
            });
        }

        Ok(())
    }

    /// Converts float sample to 16-bit PCM with clamping.
    #[inline]
    fn scale_and_clamp_sample(&self, sample: f32) -> i16 {
        // SNAC outputs normalized [-1.0, 1.0], scale to [-32768, 32767]
        let scaled = sample * AUDIO_SCALE_FACTOR;

        // Clamp to prevent integer overflow from out-of-range float values
        scaled.clamp(i16::MIN as f32, i16::MAX as f32) as i16
    }
}

/// Streaming processor with sliding window context for continuous audio generation.
pub struct ContextualFrameProcessor {
    context_buffer: Vec<AudioFrame>,
    snac_processor: Arc<SnacProcessor>, // Arc for thread-safe sharing
}

impl ContextualFrameProcessor {
    /// Creates a processor with empty context buffer.
    pub fn new(snac_processor: Arc<SnacProcessor>) -> Self {
        Self {
            context_buffer: Vec::with_capacity(SNAC_CONTEXT_FRAMES),
            snac_processor,
        }
    }

    /// Processes a frame with context, returning audio for the new frame only.
    /// Returns None until sufficient context (4 frames) is available.
    pub fn process_frame(&mut self, new_frame: AudioFrame) -> Result<Option<Vec<i16>>> {
        self.update_context_buffer(new_frame);

        // Need full context window before we can generate quality audio
        if self.has_sufficient_context() {
            let audio_slice = self.process_with_context()?;
            Ok(Some(audio_slice))
        } else {
            debug!("Insufficient context: {}/{} frames", self.context_buffer.len(), SNAC_CONTEXT_FRAMES);
            Ok(None)
        }
    }

    /// Adds frame to buffer, maintaining sliding window of SNAC_CONTEXT_FRAMES.
    fn update_context_buffer(&mut self, new_frame: AudioFrame) {
        self.context_buffer.push(new_frame);

        // Keep only the most recent SNAC_CONTEXT_FRAMES frames
        // This creates a sliding window effect for streaming
        if self.context_buffer.len() > SNAC_CONTEXT_FRAMES {
            self.context_buffer.drain(..self.context_buffer.len() - SNAC_CONTEXT_FRAMES);
        }
    }

    /// Checks if we have enough frames for processing.
    fn has_sufficient_context(&self) -> bool {
        self.context_buffer.len() >= SNAC_CONTEXT_FRAMES
    }

    /// Processes all context frames and extracts the new audio slice.
    fn process_with_context(&self) -> Result<Vec<i16>> {
        // Process all 4 frames together - SNAC uses the context to improve quality
        let full_audio = self.snac_processor.process_frames(&self.context_buffer)?;

        debug!("Full audio length: {} samples", full_audio.len());

        // But we only want the audio for the newest frame to avoid duplicates
        self.extract_new_audio_slice(&full_audio)
    }

    /// Extracts audio samples corresponding to the newest frame.
    fn extract_new_audio_slice(&self, full_audio: &[i16]) -> Result<Vec<i16>> {
        // Clamp indices to avoid out-of-bounds if audio is shorter than expected
        let start_idx = AUDIO_SLICE_START.min(full_audio.len());
        let end_idx = AUDIO_SLICE_END.min(full_audio.len());

        if start_idx >= end_idx {
            return Err(SnacError::InvalidAudioSlice {
                start: AUDIO_SLICE_START,
                end: AUDIO_SLICE_END,
                length: full_audio.len(),
            });
        }

        // Extract samples [2048:4096] which corresponds to the newest frame
        // This avoids outputting duplicate audio from previous frames
        let audio_slice = full_audio[start_idx..end_idx].to_vec();

        debug!("Extracted {} new audio samples from slice [{}:{}]", audio_slice.len(), start_idx, end_idx);

        Ok(audio_slice)
    }

    /// Returns current buffer size.
    pub fn context_size(&self) -> usize {
        self.context_buffer.len()
    }

    /// Checks if processor has enough context to generate audio.
    pub fn is_ready(&self) -> bool {
        self.has_sufficient_context()
    }

    /// Clears the context buffer.
    pub fn reset(&mut self) {
        self.context_buffer.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_snac_processing() -> Result<()> {
        let decoder_path = PathBuf::from("path/to/snac_decoder.onnx");

        if decoder_path.exists() {
            let processor = SnacProcessor::new(&decoder_path)?;

            let test_frame = AudioFrame {
                codes_0: [100],
                codes_1: [200, 300],
                codes_2: [400, 500, 600, 700],
            };

            let samples = processor.process_frames(&[test_frame])?;
            assert!(!samples.is_empty());
            println!("Generated {} audio samples", samples.len());
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_contextual_processor_context_management() -> Result<()> {
        let decoder_path = PathBuf::from("path/to/snac_decoder.onnx");
        if decoder_path.exists() {
            let processor = Arc::new(SnacProcessor::new(&decoder_path)?);
            let contextual = ContextualFrameProcessor::new(processor);

            assert_eq!(contextual.context_size(), 0);
            assert!(!contextual.is_ready());
        }

        Ok(())
    }
}
