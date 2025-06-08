# Orpheus-ONNX-rs

A high-performance Text-to-Speech server using ONNX for optimized inference. A Rust reimplementation of [Orpheus-FastAPI](https://github.com/Lex-au/Orpheus-FastAPI).

## Overview

This project reimplements the Orpheus TTS pipeline in Rust, leveraging ONNX Runtime for efficient neural audio synthesis. It converts text to speech by generating tokens through an external LLM server and then converting those tokens to audio using the SNAC model.

## Features

- **Fast inference** with ONNX Runtime
- **Streaming audio** generation for low latency
- **Multiple voices** with configurable parameters
- **RESTful API** compatible with OpenAI's TTS endpoint
- **Configurable** via TOML with environment variable overrides
- **Production-ready** with proper error handling and logging

## Architecture

- **Token Generation**: Connects to external OpenAI API-compatible server running the [Orpheus-3b model](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) or a compatible model, like [lex-au/Orpheus-3b-FT-Q8_0.gguf](https://huggingface.co/lex-au/Orpheus-3b-FT-Q8_0.gguf)
- **Audio Synthesis**: Uses [SNAC 24kHz ONNX model](https://huggingface.co/onnx-community/snac_24khz-ONNX) for token-to-audio conversion
- **Runtime**: Built with Tokio for async processing and ONNX Runtime for ML inference

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Sematre/orpheus-onnx-rs.git
cd orpheus-onnx-rs

# Download the SNAC model (52.6 MB)
mkdir -p models
wget -O models/snac_decoder.onnx https://huggingface.co/onnx-community/snac_24khz-ONNX/resolve/main/onnx/decoder_model.onnx

# Start your LLM server (e.g., with LM Studio on port 1234)
# Or if using Ollama, update config.toml to use port 11434
ollama pull hf.co/lex-au/Orpheus-3b-FT-Q8_0.gguf:Q8_0

# Build and run
cargo run --release
```

## Configuration

The project includes a default `config.toml` file with sensible defaults. You can modify it to suit your needs:

- **Server settings** - Port, bind address, worker threads
- **Model paths** - Location of SNAC decoder model
- **Token generator** - LLM API endpoint and parameters
- **Audio settings** - Sample rate, channels, quality settings
- **Voice profiles** - Configure different voices with custom parameters

See `config.toml` for all available options.

```toml
# Example: Custom provider
[providers.remote-api]
api_url = "https://api.example.com/v1/completions"
api_key = "${API_KEY}"                             # Use environment variable
timeout_secs = 60
retry_attempts = 5
max_context_length = 4096

# Example: Custom multilingual model
[models.multilingual-v2]
provider = "local-lm-studio"
model_string = "custom/multilingual-tts-v2.gguf"
temperature = 0.7
stop_sequence = "<|begin_of_text|>"
description = "Custom multilingual TTS model"

[models.multilingual-v2.voices.alex]
description = "Neutral, clear, polyglot"
languages = ["en", "es", "fr", "de"]     # This voice can speak 4 languages

[models.multilingual-v2.voices.yuki]
description = "Female, soft"
languages = ["ja", "en"]     # This voice can speak Japanese and English

[models.multilingual-v2.voices.maria]
description = "Female, warm"
languages = ["es"]           # This voice only speaks Spanish

[models.multilingual-v2.voices.zhang]
description = "Male, professional"
languages = ["zh", "en"]           # Bilingual Mandarin/English voice

```

### Environment Variables

Override configuration values using environment variables:

- `TTS_CONFIG` - Path to configuration file (default: `config.toml`)
- `TTS_BIND_ADDR` - Server bind address
- `TTS_TOKEN_API_URL` - LLM API endpoint
- `TTS_LOG_LEVEL` - Logging level (trace, debug, info, warn, error)

### Docker

```yaml
# docker-compose.yml example
services:
  tts:
    build: .
    environment:
      - TTS_CONFIG=/app/config.toml  # Uses the default config
      - TTS_BIND_ADDR=0.0.0.0:3000
      - TTS_TOKEN_API_URL=http://llm:1234/v1/completions
    volumes:
      - ./models:/app/models
      - ./config.toml:/app/config.toml:ro  # Mount config as read-only
    ports:
      - "3000:3000"
```

## API Usage

### Generate Speech

```bash
curl http://localhost:3000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "orpheus",
    "input": "Today is a wonderful day to build something people love!",
    "voice": "tara",
    "response_format": "wav"
  }' > output.wav
```

### Available Voices

- `tara` - Female, conversational, clear
- `sage` - Female, warm, gentle
- Configure voices by updating `config.toml`

### API Parameters

- `model` (optional): Model identifier
- `input` (required): Text to synthesize
- `voice` (required): Voice identifier
- `response_format` (optional): Output format (currently only "wav")

## Dependencies

Key dependencies include:
- `ort` - ONNX Runtime bindings for Rust
- `ndarray` - Multi-dimensional array processing
- `tokio` - Async runtime
- `axum` - Web framework
- `reqwest` - HTTP client for LLM API communication

## Models

This TTS system uses a two-stage pipeline combining a language model for text-to-token generation and a neural audio codec for token-to-audio synthesis.

### Language Model (Text -> Tokens)

**Orpheus 3B** by Canopy Labs is a fine-tuned Llama 3B model that generates SNAC audio tokens from text input.

- **Base model**: [canopylabs/orpheus-3b-0.1-ft](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft)
- **Recommended quantization**: [lex-au/Orpheus-3b-FT-Q8_0.gguf](https://huggingface.co/lex-au/Orpheus-3b-FT-Q8_0.gguf) (Q8_0 quantization maintains quality while reducing size)
- **Model size**: ~3.2GB (Q8_0)

The model outputs tokens in a specific format that encode hierarchical audio information across multiple temporal resolutions.

### Audio Codec (Tokens -> Audio)

**SNAC (Multi-Scale Neural Audio Codec)** decodes the tokens into 24kHz audio waveforms.

- **ONNX model**: [onnx-community/snac_24khz-ONNX](https://huggingface.co/onnx-community/snac_24khz-ONNX)
- **Original model**: [hubertsiuzdak/snac_24khz](https://huggingface.co/hubertsiuzdak/snac_24khz)
- **Model size**: 52.6MB (ONNX decoder)
- **Architecture**: Neural audio codec with multi-scale residual vector quantization (RVQ)
- **Token structure**: 7 tokens per frame (1 coarse + 2 medium + 4 fine)
- **Quality**: 24kHz sample rate, suitable for speech synthesis

### How They Work Together

1. **Text Input**: Orpheus model generates a sequence of tokens
2. **Token Stream**: Tokens are grouped into frames of 7 tokens each
3. **Frame Processing**: 4-frame sliding window maintains temporal coherence
4. **Audio Output**: SNAC decoder converts frames to PCM audio at 24kHz

The hierarchical token structure (1+2+4) allows SNAC to capture speech at different temporal resolutions, resulting in natural-sounding synthesis.

## Performance

- **Latency**: First audio chunk typically arrives within 200-500ms
- **Throughput**: Can handle multiple concurrent requests
- **Memory**: ~200MB base + model size
- **CPU**: Runs efficiently on modern CPUs without GPU

## Development

```bash
# Run with debug logging
RUST_LOG=debug cargo run

# Run tests
cargo test
```

## Credits

This project is based on [Orpheus-FastAPI](https://github.com/Lex-au/Orpheus-FastAPI) (Apache-2.0 license) by Lex-au.

## License

Distributed under the **Apache 2.0 License**. See `LICENSE` for more information.
