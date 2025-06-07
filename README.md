# Orpheus-ONNX-rs

A Text-to-Speech server using ONNX for optimized inference. A Rust recreation of [Orpheus-FastAPI](https://github.com/Lex-au/Orpheus-FastAPI).

## Overview

This project reimplements the Orpheus TTS pipeline in Rust, leveraging ONNX Runtime for efficient neural audio synthesis. It converts text to speech by generating tokens through an external LLM server and then converting those tokens to audio using the SNAC model.

## Architecture

- **Token Generation**: Connects to external OpenAI API-compatible server running the [Orpheus-3b model](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) or a compatible model, like [lex-au/Orpheus-3b-FT-Q8_0.gguf](https://huggingface.co/lex-au/Orpheus-3b-FT-Q8_0.gguf).
- **Audio Synthesis**: Uses [SNAC 24kHz ONNX model](https://huggingface.co/onnx-community/snac_24khz-ONNX) for token-to-audio conversion
- **Runtime**: Built with Tokio for async processing and ONNX Runtime for ML inference

## Setup

### Download Orpheus TTS Model

```bash
ollama pull hf.co/lex-au/Orpheus-3b-FT-Q8_0.gguf:Q8_0
```

https://huggingface.co/lex-au/Orpheus-3b-FT-Q8_0.gguf

### Download SNAC Decoder Model

Download the SNAC decoder model and place it in the `models/` directory:

```bash
# Create models directory
mkdir -p models

# Download the SNAC decoder ONNX model (52.6 MB)
wget -O models/snac_decoder.onnx https://huggingface.co/onnx-community/snac_24khz-ONNX/resolve/main/onnx/decoder_model.onnx
```

Or download manually from: [https://huggingface.co/onnx-community/snac_24khz-ONNX/blob/main/onnx/decoder_model.onnx](https://huggingface.co/onnx-community/snac_24khz-ONNX/blob/main/onnx/decoder_model.onnx)

## Dependencies

Key dependencies include:
- `ort` - ONNX Runtime bindings for Rust
- `ndarray` - Multi-dimensional array processing
- `tokio` - Async runtime
- `reqwest` - HTTP client for LLM API communication

## Usage

To generate speech from text using the API, run the following `curl` command:

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

## Models

### Tokenization

[Orpheus 3b](https://canopylabs.ai/model-releases) by Canopy Labs uses Llama-3b as the backbone. Recommended quantization: [lex-au/Orpheus-3b-FT-Q8_0.gguf](https://huggingface.co/lex-au/Orpheus-3b-FT-Q8_0.gguf) (quantized from [canopylabs/orpheus-3b-0.1-ft](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft)).

### Audio Synthesis 

[SNAC (Multi-Scale Neural Audio Codec)](https://github.com/hubertsiuzdak/snac) encodes audio into hierarchical tokens. Orpheus generates 7 tokens per frame. 4 frames (1 new frame + 3 sliding window) are fed into the SNAC decoder for high-quality output. Recommended model: [onnx-community/snac_24khz-ONNX](https://huggingface.co/onnx-community/snac_24khz-ONNX) (converted from [hubertsiuzdak/snac_24khz](https://huggingface.co/hubertsiuzdak/snac_24khz)).

## Credits

This project is based on [Orpheus-FastAPI](https://github.com/Lex-au/Orpheus-FastAPI) (Apache-2.0 license) by Lex-au.

## License

Distributed under the **Apache 2.0 License**. See ``LICENSE`` for more information.
