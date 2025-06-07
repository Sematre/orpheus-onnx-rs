use std::io::{Cursor, Write};

pub fn create_wav_header() -> std::io::Result<Vec<u8>> {
    // Create a minimal WAV header for streaming
    // We'll update the size fields later
    create_wav_header_with_size(0)
}

pub fn create_wav_header_with_size(data_size: u32) -> std::io::Result<Vec<u8>> {
    let mut header = Vec::with_capacity(44);
    let mut cursor = Cursor::new(&mut header);

    // WAV file header
    cursor.write_all(b"RIFF")?; // ChunkID
    cursor.write_all(&(36 + data_size).to_le_bytes())?; // ChunkSize
    cursor.write_all(b"WAVE")?; // Format

    // Format sub-chunk
    cursor.write_all(b"fmt ")?; // Subchunk1ID
    cursor.write_all(&16u32.to_le_bytes())?; // Subchunk1Size (PCM)
    cursor.write_all(&1u16.to_le_bytes())?; // AudioFormat (PCM)
    cursor.write_all(&1u16.to_le_bytes())?; // NumChannels (Mono)
    cursor.write_all(&24000u32.to_le_bytes())?; // SampleRate (24kHz)
    cursor.write_all(&48000u32.to_le_bytes())?; // ByteRate (SampleRate * NumChannels * BitsPerSample/8)
    cursor.write_all(&2u16.to_le_bytes())?; // BlockAlign (NumChannels * BitsPerSample/8)
    cursor.write_all(&16u16.to_le_bytes())?; // BitsPerSample

    // Data sub-chunk
    cursor.write_all(b"data")?; // Subchunk2ID
    cursor.write_all(&data_size.to_le_bytes())?; // Subchunk2Size

    Ok(header)
}
