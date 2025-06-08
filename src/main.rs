use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use axum::routing::{get, post};
use axum::Router;
use tokio::net::TcpListener;
use tower::ServiceBuilder;
use tower_http::cors::{AllowHeaders, AllowMethods, AllowOrigin, CorsLayer};
use tracing::info;
use tracing_subscriber::EnvFilter;

use crate::config::Config;
use crate::handler::batch::upload::upload_file;
use crate::handler::oneshot::generate_speech;
use crate::processing::snac_processor::SnacProcessor;

mod config;
mod handler;
mod media;
mod processing;

// Application state with configuration
#[derive(Clone)]
struct AppState {
    client: reqwest::Client,
    snac_processor: Arc<SnacProcessor>,
    config: Arc<Config>,
}

impl AppState {
    fn from_config(config: Config) -> anyhow::Result<Self> {
        // Create HTTP client with no timeout
        let client = reqwest::Client::new();

        // Load SNAC processor with configured path from audio config
        let snac_processor = Arc::new(SnacProcessor::new(&config.audio.snac_decoder_path)?);

        Ok(Self {
            client,
            snac_processor,
            config: Arc::new(config),
        })
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    // Load configuration
    let config_dir = std::env::var("TTS_CONFIG_DIR").unwrap_or_else(|_| "./config".to_string());
    let config_dir_path: &Path = config_dir.as_ref();
    let config = Config::from_dir_with_env(config_dir_path)?;

    // Initialize logging based on config
    init_logging(&config.logging)?;

    info!(
        "Starting TTS service with config from: \"{}\"",
        config_dir_path.canonicalize()?.to_str().unwrap_or(&config_dir)
    );

    println!("{config:#?}");

    // Create app state
    let state = AppState::from_config(config.clone())?;

    // Build CORS layer from config
    let cors_layer = build_cors_layer(&config.server.cors);

    // Build the router
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/v1/audio/speech", post(generate_speech))
        .route("/v1/files", post(upload_file))
        .layer(ServiceBuilder::new().layer(cors_layer))
        .with_state(state);

    // Start the server with configured address
    let listener = TcpListener::bind(&config.server.bind_addr).await?;
    info!("Server running on http://{}", listener.local_addr()?);

    tokio::select! {
        res = axum::serve(listener, app) => res,
        res = tokio::signal::ctrl_c() => res,
    }?;

    info!("Server shutdown gracefully");
    Ok(())
}

/// Health check endpoint
async fn health_check() -> &'static str {
    "OK"
}

/// Initialize logging based on configuration
fn init_logging(config: &crate::config::LoggingConfig) -> anyhow::Result<()> {
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&config.level));

    let subscriber = tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_target(false)
        .with_thread_ids(true)
        .with_line_number(false);

    // Configure output format
    match config.format.as_str() {
        "json" => {
            subscriber.json().init();
        }
        "compact" => {
            subscriber.compact().init();
        }
        _ => {
            // Default to pretty format
            subscriber.init();
        }
    }

    Ok(())
}

/// Build CORS layer from configuration
fn build_cors_layer(config: &crate::config::CorsConfig) -> CorsLayer {
    if !config.enabled {
        return CorsLayer::new();
    }

    let mut cors = CorsLayer::new();

    // Configure allowed origins
    if config.allow_origins.iter().any(|o| o == "*") {
        cors = cors.allow_origin(AllowOrigin::any());
    } else {
        let origins: Vec<_> = config.allow_origins.iter().filter_map(|origin| origin.parse().ok()).collect();
        cors = cors.allow_origin(origins);
    }

    // Configure allowed methods
    if config.allow_methods.is_empty() {
        cors = cors.allow_methods(AllowMethods::any());
    } else {
        let methods: Vec<_> = config.allow_methods.iter().filter_map(|origin| origin.parse().ok()).collect();
        cors = cors.allow_methods(methods);
    }

    // Configure allowed headers
    if config.allow_headers.is_empty() {
        cors = cors.allow_headers(AllowHeaders::any());
    } else {
        let headers: Vec<_> = config.allow_headers.iter().filter_map(|origin| origin.parse().ok()).collect();
        cors = cors.allow_headers(headers);
    }

    cors
}
