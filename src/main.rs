use std::sync::Arc;
use std::time::Duration;

use axum::routing::post;
use axum::Router;
use std::path::PathBuf;
use tokio::net::TcpListener;
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;
use tracing::info;
use tracing_subscriber::EnvFilter;

use crate::handler::batch::upload::upload_file;
use crate::handler::oneshot::generate_speech;
use crate::snac_processor::SnacProcessor;

mod audio;
mod audio_frame;
mod handler;
mod snac_processor;
mod token_generator;

// Application state
#[derive(Clone)]
struct AppState {
    client: reqwest::Client,
    snac_processor: Arc<SnacProcessor>,
}

impl AppState {
    async fn new() -> anyhow::Result<Self> {
        let client = reqwest::Client::builder().timeout(Duration::from_secs(30)).build()?;

        let decoder_path = PathBuf::from("models/snac_decoder.onnx");
        let snac_processor = Arc::new(SnacProcessor::new(&decoder_path)?);

        Ok(Self { client, snac_processor })
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with_target(false)
        .with_thread_ids(true)
        .with_line_number(false)
        .init();

    // Create app state
    let state = AppState::new().await?;

    // Build the router
    let app = Router::new()
        .route("/v1/audio/speech", post(generate_speech))
        .route("/v1/files", post(upload_file))
        .layer(ServiceBuilder::new().layer(CorsLayer::permissive()))
        .with_state(state);

    // Start the server
    let listener = TcpListener::bind("[::]:3000").await?;
    info!("Server running on http://localhost:3000");

    tokio::select! {
        res = axum::serve(listener, app) => res,
        res = tokio::signal::ctrl_c() => res,
    }?;

    Ok(())
}
