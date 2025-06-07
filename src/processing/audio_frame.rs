/// Streaming audio frame assembler that converts token numbers into structured audio frames.
///
/// Takes a stream of token numbers and groups them into AudioFrames with hierarchical
/// code structure (1 + 2 + 4 = 7 tokens per frame).
use std::ops::RangeInclusive;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures::{Stream, StreamExt};
use thiserror::Error;
use tracing::warn;

use crate::processing::token_generator::TokenError;

// Token ID calculation: token_number - 10 - (position * 4096)
// This reverses the encoding: original_id + 10 + (position * 4096) = token_number
const TOKEN_BASE_OFFSET: i32 = 10;
const TOKEN_INDEX_MULTIPLIER: i32 = 4096;
// Valid SNAC token IDs must be in [0, 4096] range
const VALID_TOKEN_RANGE: RangeInclusive<i32> = 0..=4096;

// SNAC (Orpheus variant) uses hierarchical encoding: 1 coarse + 2 medium + 4 fine = 7 tokens/frame
const CODES_0_COUNT: usize = 1;
const CODES_1_COUNT: usize = 2;
const CODES_2_COUNT: usize = 4;
const TOKENS_PER_FRAME: usize = CODES_0_COUNT + CODES_1_COUNT + CODES_2_COUNT;

type Result<T> = std::result::Result<T, AudioFrameError>;

/// Errors that can occur during frame assembly.
#[derive(Debug, Error)]
pub enum AudioFrameError {
    #[error("Token generation error: {0}")]
    TokenGeneration(#[from] TokenError),

    #[error("Invalid token {token} at index {index}: computed ID {computed_id} is out of valid range [{min}..{max}]")]
    TokenOutOfRange { token: u32, index: usize, computed_id: i32, min: i32, max: i32 },

    #[error("Stream ended with incomplete frame: {received}/{expected} tokens")]
    IncompleteFrame { received: usize, expected: usize },

    #[error("Frame assembly failed: missing token at position {position}")]
    MissingToken { position: usize },
}

/// Structured audio frame with hierarchical SNAC codes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioFrame {
    pub codes_0: [i32; CODES_0_COUNT], // Coarsest temporal resolution
    pub codes_1: [i32; CODES_1_COUNT], // Medium temporal resolution
    pub codes_2: [i32; CODES_2_COUNT], // Finest temporal resolution
}

impl AudioFrame {
    /// Create a new AudioFrame from a complete set of 7 tokens.
    fn from_tokens(tokens: [i32; TOKENS_PER_FRAME]) -> Self {
        // Token order is specific: codes are interleaved for temporal alignment
        // Pattern: [codes_0[0], codes_1[0], codes_2[0], codes_2[1], codes_1[1], codes_2[2], codes_2[3]]
        Self {
            codes_0: [tokens[0]],                                  // Position 0
            codes_1: [tokens[1], tokens[4]],                       // Positions 1, 4 (spread out)
            codes_2: [tokens[2], tokens[3], tokens[5], tokens[6]], // Positions 2, 3, 5, 6
        }
    }
}

/// Buffer for accumulating tokens until a complete frame is formed.
#[derive(Debug)]
struct FrameBuffer {
    tokens: [Option<i32>; TOKENS_PER_FRAME],
    position: usize,
}

impl FrameBuffer {
    fn new() -> Self {
        Self {
            tokens: [None; TOKENS_PER_FRAME],
            position: 0,
        }
    }

    fn add_token(&mut self, token: i32) -> bool {
        if self.position < TOKENS_PER_FRAME {
            self.tokens[self.position] = Some(token);
            self.position += 1;
            // Return true when frame is complete
            self.position == TOKENS_PER_FRAME
        } else {
            false // Shouldn't happen, but handle gracefully
        }
    }

    fn is_complete(&self) -> bool {
        self.position == TOKENS_PER_FRAME
    }

    fn reset(&mut self) {
        self.tokens = [None; TOKENS_PER_FRAME];
        self.position = 0;
    }

    fn try_into_audio_frame(self) -> Result<AudioFrame> {
        if !self.is_complete() {
            return Err(AudioFrameError::IncompleteFrame {
                received: self.position,
                expected: TOKENS_PER_FRAME,
            });
        }

        // Convert Option<i32> array to i32 array, checking for gaps
        let mut tokens = [0i32; TOKENS_PER_FRAME];
        for (i, token_opt) in self.tokens.iter().enumerate() {
            match token_opt {
                Some(token) => tokens[i] = *token,
                None => return Err(AudioFrameError::MissingToken { position: i }),
            }
        }

        Ok(AudioFrame::from_tokens(tokens))
    }
}

/// Stream adapter that converts token numbers into AudioFrames.
pub struct AudioFrameStream<S> {
    token_stream: S,
    frame_buffer: FrameBuffer,
    token_index: usize, // Global token counter for position calculation
}

impl<S> AudioFrameStream<S>
where
    S: Stream<Item = std::result::Result<u32, TokenError>> + Unpin,
{
    pub fn new(token_stream: S) -> Self {
        Self {
            token_stream,
            frame_buffer: FrameBuffer::new(),
            token_index: 0,
        }
    }

    /// Convert a token number to a token ID using the specified algorithm.
    fn convert_token_to_id(&self, token_number: u32, index: usize) -> Result<i32> {
        // Reverse the encoding: token_number = original_id + 10 + (position_in_frame * 4096)
        // So: original_id = token_number - 10 - (position_in_frame * 4096)
        let position_in_frame = (index % TOKENS_PER_FRAME) as i32;
        let token_id = (token_number as i32) - TOKEN_BASE_OFFSET - (position_in_frame * TOKEN_INDEX_MULTIPLIER);

        // SNAC token IDs must be in valid range
        if !VALID_TOKEN_RANGE.contains(&token_id) {
            return Err(AudioFrameError::TokenOutOfRange {
                token: token_number,
                index,
                computed_id: token_id,
                min: *VALID_TOKEN_RANGE.start(),
                max: *VALID_TOKEN_RANGE.end(),
            });
        }

        Ok(token_id)
    }

    /// Process a single token and potentially return a complete frame.
    fn process_token(&mut self, token_number: u32) -> Result<Option<AudioFrame>> {
        // Decode the token number back to original SNAC token ID
        let token_id = self.convert_token_to_id(token_number, self.token_index)?;

        // Add to buffer and check if frame is complete
        let frame_complete = self.frame_buffer.add_token(token_id);
        self.token_index += 1;

        if frame_complete {
            // Take ownership of buffer and reset for next frame
            let completed_buffer = std::mem::replace(&mut self.frame_buffer, FrameBuffer::new());
            let audio_frame = completed_buffer.try_into_audio_frame()?;

            Ok(Some(audio_frame))
        } else {
            Ok(None) // Need more tokens
        }
    }

    /// Handle stream end, checking for incomplete frames.
    fn handle_stream_end(&self) -> Option<Result<AudioFrame>> {
        if self.frame_buffer.position > 0 {
            // Log incomplete frame but don't error - common at stream end
            warn!("Stream ended with incomplete frame ({}/{} tokens)", self.frame_buffer.position, TOKENS_PER_FRAME);

            // Could return error if strict frame boundaries required:
            // Some(Err(AudioFrameError::IncompleteFrame {
            //     received: self.frame_buffer.position,
            //     expected: TOKENS_PER_FRAME,
            // }))
        }

        None
    }
}

impl<S> Stream for AudioFrameStream<S>
where
    S: Stream<Item = std::result::Result<u32, TokenError>> + Unpin,
{
    type Item = Result<AudioFrame>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            match self.token_stream.poll_next_unpin(cx) {
                Poll::Ready(Some(Ok(token_number))) => {
                    match self.process_token(token_number) {
                        Ok(Some(frame)) => return Poll::Ready(Some(Ok(frame))),
                        Ok(None) => continue, // Buffer not full yet
                        Err(e) => return Poll::Ready(Some(Err(e))),
                    }
                }
                Poll::Ready(Some(Err(e))) => {
                    // Propagate token stream errors
                    return Poll::Ready(Some(Err(AudioFrameError::TokenGeneration(e))));
                }
                Poll::Ready(None) => {
                    // Stream ended - check for incomplete frames
                    return Poll::Ready(self.handle_stream_end());
                }
                Poll::Pending => {
                    return Poll::Pending;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;

    #[test]
    fn test_audio_frame_creation() {
        let tokens = [100, 200, 300, 400, 500, 600, 700];
        let frame = AudioFrame::from_tokens(tokens);

        assert_eq!(frame.codes_0, [100]);
        assert_eq!(frame.codes_1, [200, 500]);
        assert_eq!(frame.codes_2, [300, 400, 600, 700]);
    }

    #[test]
    fn test_frame_buffer() {
        let mut buffer = FrameBuffer::new();

        assert!(!buffer.is_complete());

        // Add 6 tokens
        for i in 0..6 {
            assert!(!buffer.add_token(i));
        }

        // 7th token should complete the frame
        assert!(buffer.add_token(6));
        assert!(buffer.is_complete());

        let frame = buffer.try_into_audio_frame().unwrap();
        assert_eq!(frame.codes_0, [0]);
        assert_eq!(frame.codes_1, [1, 4]);
        assert_eq!(frame.codes_2, [2, 3, 5, 6]);
    }

    #[test]
    fn test_token_conversion() {
        let stream = AudioFrameStream::new(stream::empty());

        // Test first token: 4106 - 10 - (0 * 4096) = 4096
        let result = stream.convert_token_to_id(4106, 0);
        assert_eq!(result.unwrap(), 4096);

        // Test out of range token
        let result = stream.convert_token_to_id(10000, 0);
        assert!(matches!(result, Err(AudioFrameError::TokenOutOfRange { .. })));
    }

    #[tokio::test]
    async fn test_audio_frame_stream() {
        use futures::stream::StreamExt;

        // Create tokens for one frame with position-based encoding
        let tokens = vec![
            Ok(4106),  // 4096 + 10 + (0 * 4096)
            Ok(4107),  // 4097 + 10 + (1 * 4096) = 4107
            Ok(4108),  // 4098 + 10 + (2 * 4096) = 8202
            Ok(4109),  // 4099 + 10 + (3 * 4096) = 12307
            Ok(20487), // 4101 + 10 + (4 * 4096) = 20487
            Ok(20488), // 4102 + 10 + (5 * 4096) = 24598
            Ok(20489), // 4103 + 10 + (6 * 4096) = 28713
        ];
        let token_stream = stream::iter(tokens);

        let mut frame_stream = AudioFrameStream::new(token_stream);

        if let Some(Ok(frame)) = frame_stream.next().await {
            // Verify hierarchical structure
            assert_eq!(frame.codes_0.len(), 1);
            assert_eq!(frame.codes_1.len(), 2);
            assert_eq!(frame.codes_2.len(), 4);
        } else {
            panic!("Expected a valid frame");
        }
    }
}
