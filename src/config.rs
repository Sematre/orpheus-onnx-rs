use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Main configuration structure
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Config {
    pub server: ServerConfig,
    pub models: ModelConfig,
    pub token_generator: TokenGeneratorConfig,
    pub audio: AudioConfig,
    pub streaming: StreamingConfig,
    pub logging: LoggingConfig,
    pub api: ApiConfig,
    pub voices: HashMap<String, VoiceConfig>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ServerConfig {
    pub bind_addr: String,

    #[serde(default = "default_workers")]
    pub workers: usize,

    #[serde(default)]
    pub cors: CorsConfig,
}

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct CorsConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,

    pub allow_origins: Vec<String>,
    pub allow_methods: Vec<String>,
    pub allow_headers: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ModelConfig {
    pub snac_decoder_path: PathBuf,

    #[serde(default = "default_true")]
    pub cache_models: bool,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct TokenGeneratorConfig {
    pub api_url: String,
    pub model: String,
    pub temperature: Option<f32>,
    pub stop_sequence: String,

    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,

    #[serde(default = "default_retry_attempts")]
    pub retry_attempts: u32,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct AudioConfig {
    pub sample_rate: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
    pub context_frames: usize,
    pub audio_slice_start: usize,
    pub audio_slice_end: usize,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct StreamingConfig {
    pub channel_buffer_size: usize,
    pub chunk_size: usize,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct LoggingConfig {
    pub level: String,
    pub format: String,
    pub file: Option<PathBuf>,
    pub rotation: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ApiConfig {
    pub max_request_size: String,
    pub request_timeout_secs: u64,

    #[serde(default)]
    pub rate_limiting: RateLimitConfig,
}

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct RateLimitConfig {
    pub enabled: bool,
    pub requests_per_minute: u32,
    pub burst_size: u32,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct VoiceConfig {
    pub description: String,
    pub custom_temperature: Option<f32>,
}

// Default value functions for serde
fn default_workers() -> usize {
    0
}
fn default_true() -> bool {
    true
}
fn default_timeout() -> u64 {
    30
}
fn default_retry_attempts() -> u32 {
    3
}

impl Config {
    /// Load configuration from file
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&contents)?;

        config.validate()?;
        Ok(config)
    }

    /// Load configuration with environment variable override support
    pub fn from_file_with_env(path: impl AsRef<Path>) -> Result<Self> {
        let mut config = Self::from_file(path)?;

        // Override with environment variables
        if let Ok(bind_addr) = std::env::var("TTS_BIND_ADDR") {
            config.server.bind_addr = bind_addr;
        }

        if let Ok(api_url) = std::env::var("TTS_TOKEN_API_URL") {
            config.token_generator.api_url = api_url;
        }

        if let Ok(log_level) = std::env::var("TTS_LOG_LEVEL") {
            config.logging.level = log_level;
        }

        Ok(config)
    }

    /// Validate configuration values
    fn validate(&self) -> Result<()> {
        // Validate model path exists
        if !self.models.snac_decoder_path.exists() {
            anyhow::bail!("SNAC decoder model not found at {:?}", self.models.snac_decoder_path);
        }

        // Validate audio settings
        if self.audio.audio_slice_start >= self.audio.audio_slice_end {
            anyhow::bail!("audio_slice_start must be less than audio_slice_end");
        }

        // Validate log level
        let valid_levels = ["trace", "debug", "info", "warn", "error"];
        if !valid_levels.contains(&self.logging.level.as_str()) {
            anyhow::bail!("Invalid log level: {}", self.logging.level);
        }

        Ok(())
    }
}
