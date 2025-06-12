use anyhow::Result;
use ort::ExecutionProvider;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;

/// Main configuration structure
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Config {
    pub server: ServerConfig,
    pub audio: AudioConfig,
    pub streaming: StreamingConfig,
    pub logging: LoggingConfig,
    pub api: ApiConfig,
    pub providers: HashMap<String, ProviderConfig>,
    pub models: HashMap<String, ModelConfig>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ServerConfig {
    pub bind_addr: String,

    #[serde(default)]
    pub cors: CorsConfig,
}

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct CorsConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,

    #[serde(default)]
    pub allow_origins: Vec<String>,

    #[serde(default)]
    pub allow_methods: Vec<String>,

    #[serde(default)]
    pub allow_headers: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct AudioConfig {
    pub snac_decoder_path: PathBuf,

    #[serde(default)]
    pub execution_provider: OnnxExecutionProvider,
}

/// ONNX execution provider options
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
#[serde(rename_all = "lowercase")]
pub enum OnnxExecutionProvider {
    #[default]
    Cpu,
    Cuda,
    TensorRT,
    DirectML,
    CoreML,
    ROCm,
}

impl OnnxExecutionProvider {
    /// Convert to ort::ExecutionProvider
    pub fn to_ort_provider(&self) -> ExecutionProvider {
        match self {
            OnnxExecutionProvider::Cpu => ExecutionProvider::CPU(Default::default()),
            OnnxExecutionProvider::Cuda => ExecutionProvider::CUDA(Default::default()),
            OnnxExecutionProvider::TensorRT => ExecutionProvider::TensorRT(Default::default()),
            OnnxExecutionProvider::DirectML => ExecutionProvider::DirectML(Default::default()),
            OnnxExecutionProvider::CoreML => ExecutionProvider::CoreML(Default::default()),
            OnnxExecutionProvider::ROCm => ExecutionProvider::ROCm(Default::default()),
        }
    }
}

impl FromStr for OnnxExecutionProvider {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "cpu" => Ok(OnnxExecutionProvider::Cpu),
            "cuda" => Ok(OnnxExecutionProvider::Cuda),
            "tensorrt" => Ok(OnnxExecutionProvider::TensorRT),
            "directml" => Ok(OnnxExecutionProvider::DirectML),
            "coreml" => Ok(OnnxExecutionProvider::CoreML),
            "rocm" => Ok(OnnxExecutionProvider::ROCm),
            _ => Err(anyhow::anyhow!(
                "Invalid execution provider: {s}. Valid options are: cpu, cuda, tensorrt, directml, coreml, rocm"
            )),
        }
    }
}

impl fmt::Display for OnnxExecutionProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OnnxExecutionProvider::Cpu => write!(f, "CPU"),
            OnnxExecutionProvider::Cuda => write!(f, "CUDA"),
            OnnxExecutionProvider::TensorRT => write!(f, "TensorRT"),
            OnnxExecutionProvider::DirectML => write!(f, "DirectML"),
            OnnxExecutionProvider::CoreML => write!(f, "CoreML"),
            OnnxExecutionProvider::ROCm => write!(f, "ROCm"),
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct StreamingConfig {
    pub channel_buffer_size: usize,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct LoggingConfig {
    pub level: String,
    pub format: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ApiConfig {
    pub max_request_size: String,
    pub request_timeout_secs: u64,
    pub default_model: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ProviderConfig {
    pub api_url: String,

    #[serde(default)]
    pub api_key: String,

    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,

    #[serde(default = "default_retry_attempts")]
    pub retry_attempts: u32,

    #[serde(default = "default_concurrent_requests")]
    pub concurrent_requests: u32,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ModelConfig {
    pub provider: String,
    pub id: String,
    pub temperature: f32,
    pub max_context_length: u32,
    pub stop_sequence: String,
    pub description: String,
    pub voices: HashMap<String, VoiceConfig>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct VoiceConfig {
    pub description: String,
    pub languages: Vec<String>,
}

// Default value functions for serde
fn default_true() -> bool {
    true
}

fn default_timeout() -> u64 {
    30
}

fn default_retry_attempts() -> u32 {
    3
}

fn default_concurrent_requests() -> u32 {
    1
}

impl Config {
    /// Load configuration from a directory containing TOML files
    pub fn from_dir(dir_path: impl AsRef<Path>) -> Result<Self> {
        let dir_path = dir_path.as_ref();

        if !dir_path.exists() {
            anyhow::bail!("Configuration directory not found: {:?}", dir_path);
        }

        if !dir_path.is_dir() {
            anyhow::bail!("Path is not a directory: {:?}", dir_path);
        }

        // Read all TOML files in alphabetical order
        let mut toml_files = std::fs::read_dir(dir_path)?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.extension().and_then(|ext| ext.to_str()).map(|ext| ext.eq_ignore_ascii_case("toml")).unwrap_or(false))
            .collect::<Vec<_>>();

        toml_files.sort();

        if toml_files.is_empty() {
            anyhow::bail!("No TOML files found in directory: {:?}", dir_path);
        }

        // Start with empty config content
        let mut merged_content = String::new();

        // Read and merge all TOML files
        for file_path in &toml_files {
            let file_content = std::fs::read_to_string(file_path).map_err(|e| anyhow::anyhow!("Failed to read {:?}: {}", file_path, e))?;

            merged_content.push_str(&file_content);
            merged_content.push('\n');
        }

        // Parse the merged TOML content
        let config: Config = toml::from_str(&merged_content).map_err(|e| anyhow::anyhow!("Failed to parse merged TOML configuration: {}", e))?;

        config.validate()?;
        Ok(config)
    }

    /// Load configuration from a directory with environment variable override support
    pub fn from_dir_with_env(dir_path: impl AsRef<Path>) -> Result<Self> {
        let mut config = Self::from_dir(dir_path)?;

        // Override with environment variables
        if let Ok(bind_addr) = std::env::var("TTS_BIND_ADDR") {
            config.server.bind_addr = bind_addr;
        }

        if let Ok(log_level) = std::env::var("TTS_LOG_LEVEL") {
            config.logging.level = log_level;
        }

        if let Ok(execution_provider) = std::env::var("TTS_EXECUTION_PROVIDER") {
            config.audio.execution_provider = execution_provider
                .parse()
                .map_err(|_| anyhow::anyhow!("Invalid execution provider: {}", execution_provider))?;
        }

        Ok(config)
    }

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

        if let Ok(log_level) = std::env::var("TTS_LOG_LEVEL") {
            config.logging.level = log_level;
        }

        if let Ok(execution_provider) = std::env::var("TTS_EXECUTION_PROVIDER") {
            config.audio.execution_provider = execution_provider
                .parse()
                .map_err(|_| anyhow::anyhow!("Invalid execution provider: {}", execution_provider))?;
        }

        Ok(config)
    }

    /// Validate configuration values
    fn validate(&self) -> Result<()> {
        // Validate audio model path exists
        if !self.audio.snac_decoder_path.exists() {
            anyhow::bail!("SNAC decoder model not found at {:?}", self.audio.snac_decoder_path);
        }

        // Validate log level
        let valid_levels = ["trace", "debug", "info", "warn", "error"];
        if !valid_levels.contains(&self.logging.level.as_str()) {
            anyhow::bail!("Invalid log level: {}", self.logging.level);
        }

        // Validate that the default model exists in models config
        if !self.models.contains_key(&self.api.default_model) {
            anyhow::bail!("Default model '{}' not found in models configuration", self.api.default_model);
        }

        // Validate that each model references a valid provider
        for (model_name, model_config) in &self.models {
            if !self.providers.contains_key(&model_config.provider) {
                anyhow::bail!("Model '{}' references unknown provider '{}'", model_name, model_config.provider);
            }
        }

        Ok(())
    }
}
