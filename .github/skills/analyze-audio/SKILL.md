---
name: analyze-audio
description: Analyze audio using AI to generate natural language descriptions and a detailed timeline. Use this skill when the user wants to understand what a music track sounds like, identify instruments, mood, or structural changes.
---

# analyze-audio

A Python tool that uses Qwen2-Audio AI to listen to an audio file and describe it in natural language — like a human music analyst would.

## How to Run
```bash
uv run analyze-audio <input-audio> [options]
```

## Arguments
| Argument | Required | Description |
|----------|----------|-------------|
| `input` | Yes | Input audio file (WAV, MP3, M4A, AAC, or OGG; compressed formats require `ffmpeg`) |

## Options
| Flag | Default | Description |
|------|---------|-------------|
| `--sections N` | auto | Number of fixed sections to analyze instead of auto-detecting natural boundaries |
| `--output FILE` | (none) | Save analysis to a text file |

## Behavior
- Generates a detailed paragraph describing the track (genre, mood, instruments, tempo, evolution)
- Detects natural section boundaries from energy and spectral changes, then describes each section
- Converts compressed inputs to 16kHz mono before sending them to the model
- Uses Qwen2-Audio-7B-Instruct (first run downloads ~15GB model)
- Runs on Apple Silicon (MPS), CUDA, or CPU

## Examples

### Describe a soundtrack
```bash
uv run analyze-audio tmp/forest_theme.wav
```

### Analyze with custom section count
```bash
uv run analyze-audio tmp/boss_fight.wav --sections 6
```

### Save analysis to file
```bash
uv run analyze-audio tmp/town.wav --output tmp/town_analysis.txt
```

## Related Tools
- `gen-music` — Generate music from text descriptions
- `convert-audio` — Convert formats before analysis
- `trim-audio` — Trim audio before analysis
