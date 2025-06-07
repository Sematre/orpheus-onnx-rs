/// Streaming token generator that extracts custom token numbers from LLM completion API.
///
/// Connects to a local LLM API (like LM Studio) and streams responses, extracting
/// custom tokens in the format <custom_token_123> from the generated text.
use futures::Stream;
use regex::Regex;
use reqwest::Client;
use serde_json::{json, Value};
use std::pin::Pin;
use std::task::{Context, Poll};
use thiserror::Error;
use tracing::debug;

// Default to local LM Studio API endpoint
const DEFAULT_API_URL: &str = "http://127.0.0.1:1234/v1/completions";
const DEFAULT_MODEL: &str = "legraphista/Orpheus";

// Regex pattern to find tokens like <custom_token_123> in the LLM output
const CUSTOM_TOKEN_PATTERN: &str = r"<custom_token_(\d+)>";
// SSE (Server-Sent Events) format prefix
const DATA_PREFIX: &str = "data: ";
// Standard SSE end-of-stream marker
const STREAM_END_MARKER: &str = "[DONE]";

type Result<T> = std::result::Result<T, TokenError>;

/// Errors that can occur during token generation and streaming.
#[derive(Debug, Error)]
pub enum TokenError {
    #[error("HTTP request failed with status {status}: {message}")]
    HttpRequest { status: reqwest::StatusCode, message: String },

    #[error("Request error: {0}")]
    Request(#[from] reqwest::Error),

    #[error("Regex compilation failed: {0}")]
    RegexCompilation(#[from] regex::Error),

    #[error("Stream error: {0}")]
    Stream(reqwest::Error),

    #[error("Failed to parse token number '{token}': {source}")]
    TokenParsing {
        token: String,
        #[source]
        source: std::num::ParseIntError,
    },

    #[error("JSON parsing failed: {0}")]
    JsonParsing(#[from] serde_json::Error),

    #[error("Invalid UTF-8 in stream data")]
    InvalidUtf8,
}

/// Configuration for the token generation API.
#[derive(Debug, Clone)]
pub struct ApiConfig {
    pub url: String,
    pub model: String,
    pub temperature: Option<f32>,
    pub stop_sequence: String,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            url: DEFAULT_API_URL.to_string(),
            model: DEFAULT_MODEL.to_string(),
            temperature: None,
            stop_sequence: "<|begin_of_text|>".to_string(),
        }
    }
}

/// Main token generator that manages API communication and token extraction.
pub struct TokenGenerator {
    client: Client,
    custom_token_regex: Regex,
    config: ApiConfig,
}

/// Stream implementation that extracts token numbers from SSE response.
pub struct TokenNumberStream {
    response_stream: Pin<Box<dyn Stream<Item = std::result::Result<bytes::Bytes, reqwest::Error>> + Send>>,
    buffer: String,
    regex: Regex,
}

impl Stream for TokenNumberStream {
    type Item = Result<u32>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Always check buffer first - might have multiple tokens from previous chunk
        if let Some(result) = self.process_buffer_lines() {
            return Poll::Ready(Some(result));
        }

        // Buffer empty, need more data from network
        self.poll_response_stream(cx)
    }
}

impl TokenNumberStream {
    /// Process complete lines in the buffer and return the first token found.
    fn process_buffer_lines(&mut self) -> Option<Result<u32>> {
        // SSE format uses newlines to separate events
        while let Some(line_end) = self.buffer.find('\n') {
            let line = self.extract_line(line_end);

            match self.process_line(&line) {
                LineResult::Token(token) => return Some(Ok(token)),
                LineResult::EndOfStream => return None,
                LineResult::Continue => continue,
                LineResult::Error(err) => return Some(Err(err)),
            }
        }

        None
    }

    /// Extract a line from the buffer and update the buffer.
    fn extract_line(&mut self, line_end: usize) -> String {
        let line = self.buffer[..line_end].trim().to_string();
        // Remove processed line including newline
        self.buffer.drain(..line_end + 1);

        line
    }

    /// Process a single SSE line and return the result.
    fn process_line(&self, line: &str) -> LineResult {
        // SSE data lines start with "data: "
        let data_str = match line.strip_prefix(DATA_PREFIX) {
            Some(s) => s.trim(),
            None => return LineResult::Continue, // Not a data line, skip
        };

        // Check for end of stream marker
        if data_str == STREAM_END_MARKER {
            return LineResult::EndOfStream;
        }

        // Parse JSON from the data line
        let Ok(data) = serde_json::from_str::<Value>(data_str) else {
            return LineResult::Continue; // Skip malformed JSON, don't fail stream
        };

        match self.extract_text_from_json(&data) {
            Some(text) => self.extract_token_from_text(text),
            None => LineResult::Continue,
        }
    }

    /// Extract text from standard OpenAI-style completion response.
    fn extract_text_from_json<'a>(&self, data: &'a Value) -> Option<&'a str> {
        // Navigate JSON: {"choices": [{"text": "..."}]}
        data.get("choices")?.as_array()?.first()?.get("text")?.as_str()
    }

    /// Extract custom token number from text using regex.
    fn extract_token_from_text(&self, text: &str) -> LineResult {
        // Look for patterns like <custom_token_123>
        for cap in self.regex.captures_iter(text) {
            let Some(token) = cap.get(1) else {
                continue;
            };

            // Parse the numeric part
            return match token.as_str().parse::<u32>() {
                Ok(num) => LineResult::Token(num),
                Err(e) => LineResult::Error(TokenError::TokenParsing {
                    token: token.as_str().to_string(),
                    source: e,
                }),
            };
        }

        LineResult::Continue
    }

    /// Poll the response stream for more data.
    fn poll_response_stream(&mut self, cx: &mut Context<'_>) -> Poll<Option<Result<u32>>> {
        match self.response_stream.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => {
                if let Err(e) = self.append_chunk_to_buffer(chunk) {
                    return Poll::Ready(Some(Err(e)));
                }

                // New data in buffer, recursively check for tokens
                // This ensures we process all tokens in the chunk
                let mut pinned = std::pin::pin!(self);
                pinned.as_mut().poll_next(cx)
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(TokenError::Stream(e)))),
            Poll::Ready(None) => Poll::Ready(None), // Stream ended
            Poll::Pending => Poll::Pending,
        }
    }

    /// Append chunk to buffer, handling UTF-8 conversion.
    fn append_chunk_to_buffer(&mut self, chunk: bytes::Bytes) -> Result<()> {
        // Use lossy conversion to handle partial UTF-8 sequences at chunk boundaries
        self.buffer.push_str(String::from_utf8_lossy(&chunk).as_ref());
        Ok(())
    }
}

/// Internal result type for line processing.
#[derive(Debug)]
enum LineResult {
    Token(u32),
    EndOfStream,
    Continue,
    Error(TokenError),
}

impl TokenGenerator {
    pub fn new(client: Client) -> Result<Self> {
        Self::with_config(client, ApiConfig::default())
    }

    pub fn with_config(client: Client, config: ApiConfig) -> Result<Self> {
        // Pre-compile regex for efficiency
        let custom_token_regex = Regex::new(CUSTOM_TOKEN_PATTERN)?;

        Ok(TokenGenerator {
            client,
            custom_token_regex,
            config,
        })
    }

    /// Generate a stream of custom token numbers for the given speaker and text.
    pub async fn generate_token_stream(&self, speaker: &str, text: &str) -> Result<impl Stream<Item = Result<u32>>> {
        let formatted_prompt = self.format_prompt(text, speaker);
        debug!("Generating token stream for: {speaker} - {text}");

        let response = self.send_completion_request(&formatted_prompt).await?;
        // Convert to bytes stream for efficient chunk processing
        let response_stream = Box::pin(response.bytes_stream());

        Ok(TokenNumberStream {
            response_stream,
            buffer: String::new(),
            regex: self.custom_token_regex.clone(),
        })
    }

    /// Send completion request to the API.
    async fn send_completion_request(&self, prompt: &str) -> Result<reqwest::Response> {
        let mut payload = json!({
            "prompt": prompt,
            "stream": true,  // Enable SSE streaming
            "model": self.config.model,
            "stop": self.config.stop_sequence,
        });

        // Temperature is optional - controls randomness
        if let Some(temp) = self.config.temperature {
            payload["temperature"] = json!(temp);
        }

        let response = self
            .client
            .post(&self.config.url)
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream") // Request SSE format
            .json(&payload)
            .send()
            .await?;

        // Handle non-200 responses gracefully
        if !response.status().is_success() {
            let status = response.status();
            let message = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());

            return Err(TokenError::HttpRequest { status, message });
        }

        Ok(response)
    }

    /// Format the prompt with speaker and text.
    fn format_prompt(&self, text: &str, speaker: &str) -> String {
        // Custom prompt format for audio token generation
        format!("<|audio|>{speaker}: {text}<|eot_id|>")
    }

    /// Update the API configuration.
    pub fn update_config(&mut self, config: ApiConfig) {
        self.config = config;
    }

    /// Get the current API configuration.
    pub fn config(&self) -> &ApiConfig {
        &self.config
    }
}

/// Builder pattern for easier configuration.
pub struct TokenGeneratorBuilder {
    client: Option<Client>,
    config: ApiConfig,
}

impl TokenGeneratorBuilder {
    pub fn new() -> Self {
        Self {
            client: None,
            config: ApiConfig::default(),
        }
    }

    pub fn client(mut self, client: Client) -> Self {
        self.client = Some(client);
        self
    }

    pub fn api_url(mut self, url: impl Into<String>) -> Self {
        self.config.url = url.into();
        self
    }

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.config.model = model.into();
        self
    }

    pub fn temperature(mut self, temp: f32) -> Self {
        self.config.temperature = Some(temp);
        self
    }

    pub fn stop_sequence(mut self, stop: impl Into<String>) -> Self {
        self.config.stop_sequence = stop.into();
        self
    }

    pub fn build(self) -> Result<TokenGenerator> {
        // Use default client if none provided
        let client = self.client.unwrap_or_default();
        TokenGenerator::with_config(client, self.config)
    }
}

impl Default for TokenGeneratorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_generator_creation() -> Result<()> {
        let client = Client::new();
        let generator = TokenGenerator::new(client)?;
        assert_eq!(generator.config().model, DEFAULT_MODEL);
        Ok(())
    }

    #[test]
    fn test_builder_pattern() -> Result<()> {
        let generator = TokenGeneratorBuilder::new()
            .api_url("http://localhost:8080/v1/completions")
            .model("custom-model")
            .temperature(0.5)
            .build()?;

        assert_eq!(generator.config().url, "http://localhost:8080/v1/completions");
        assert_eq!(generator.config().model, "custom-model");
        assert_eq!(generator.config().temperature, Some(0.5));
        Ok(())
    }

    #[test]
    fn test_prompt_formatting() -> Result<()> {
        let client = Client::new();
        let generator = TokenGenerator::new(client)?;
        let formatted = generator.format_prompt("Hello world", "Alice");
        assert_eq!(formatted, "<|audio|>Alice: Hello world<|eot_id|>");
        Ok(())
    }
}
