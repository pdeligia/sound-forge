---
name: convert-audio
description: Convert between audio formats and resample to a target sample rate. Use this skill when the user wants to convert WAV to M4A/OGG/FLAC/MP3 for game integration, change sample rates, or convert between mono and stereo.
---

# convert-audio

A Python tool for converting between audio formats, resampling, and changing channel layout (mono/stereo).

## How to Run
```bash
uv run convert-audio <input.wav> [options]
```

## Arguments
| Argument | Required | Description |
|----------|----------|-------------|
| `input` | Yes | Input audio file |

## Options
| Flag | Default | Description |
|------|---------|-------------|
| `--format FORMAT` | (from extension) | Output format: `wav`, `flac`, `ogg`, `mp3`, or `m4a` |
| `--sample-rate N` | (same as input) | Target sample rate in Hz (e.g. 44100, 48000) |
| `--channels MODE` | (same as input) | Convert to `mono` or `stereo` |
| `--output FILE` | `./tmp/convert_audio/output.<format>` | Output file path |

## Supported Formats
| Format | Extension | Type | Backend |
|--------|-----------|------|---------|
| WAV | `.wav` | Uncompressed PCM | soundfile |
| FLAC | `.flac` | Lossless compressed | soundfile |
| OGG Vorbis | `.ogg` | Lossy compressed | soundfile |
| MP3 | `.mp3` | Lossy compressed | torchaudio (requires ffmpeg) |
| AAC/M4A | `.m4a` | Lossy, iOS-optimized | macOS afconvert |

## Behavior
- Format is inferred from `--output` extension, or set explicitly with `--format`
- Channel conversion: stereo → mono averages both channels; mono → stereo duplicates the channel
- Resampling uses torchaudio's high-quality resampler
- Reports: output duration, sample rate, channels, peak/RMS levels, file size, and compression ratio

## Examples

### Convert WAV to OGG for game use
```bash
uv run convert-audio soundtrack.wav --format ogg
```

### Convert to FLAC with resampling to 44.1kHz
```bash
uv run convert-audio output.wav --format flac --sample-rate 44100
```

### Convert stereo music to mono for mobile
```bash
uv run convert-audio music.wav --channels mono --output music_mono.wav
```

### Convert SFX to OGG at 44.1kHz mono (optimized for games)
```bash
uv run convert-audio sfx.wav --format ogg --sample-rate 44100 --channels mono
```

### Convert to M4A/AAC for iOS games (recommended for SpriteKit)
```bash
uv run convert-audio soundtrack.wav --format m4a --sample-rate 44100
```

### Convert to MP3 (requires ffmpeg installed)
```bash
uv run convert-audio soundtrack.wav --format mp3 --output soundtrack.mp3
```

## Related Tools
- `gen-music` — Generate music (outputs WAV by default)
- `gen-sfx` — Generate sound effects (outputs WAV by default)
- `analyze-audio` — Check audio properties before/after conversion
- `normalize-audio` — Normalize loudness before converting
