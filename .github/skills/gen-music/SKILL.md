---
name: gen-music
description: Generate music from a text prompt using MusicGen AI. Use this skill when the user wants to create soundtrack clips, background music, or melodic audio assets for games from a text description.
---

# gen-music

A Python tool for generating music from a text prompt using Meta's MusicGen model (via HuggingFace Transformers).

## How to Run
```bash
uv run gen-music "<prompt>" [options]
```

## Arguments
| Argument | Required | Description |
|----------|----------|-------------|
| `prompt` | Yes | Text description of the music to generate |

## Options
| Flag | Default | Description |
|------|---------|-------------|
| `--duration N` | `10` | Duration in seconds |
| `--model-size SIZE` | `small` | Model size: `small` (300M), `medium` (1.5B), or `large` (3.3B) |
| `--temperature N` | `1.0` | Sampling temperature — higher = more creative/random |
| `--output FILE` | `./tmp/gen_music/output.wav` | Output WAV file path |
| `--no-highpass` | off | Disable the 150Hz high-pass filter (on by default to remove low-frequency hum) |

## Behavior
- Loads the MusicGen model (auto-downloads on first use, cached afterward)
- Generates audio tokens from the text prompt using classifier-free guidance (scale 3.0)
- Applies a 150Hz Butterworth high-pass filter by default to clean up AI-generated rumble
- Outputs a WAV file with detailed stats: duration, peak/RMS levels, frequency spectrum, per-second dynamics, and performance metrics
- Auto-detects MPS (Apple Silicon), CUDA, or CPU

## Models Used
| Model | Size | Description |
|-------|------|-------------|
| `facebook/musicgen-small` | ~300M | Fast, good for prototyping |
| `facebook/musicgen-medium` | ~1.5B | Balanced quality/speed |
| `facebook/musicgen-large` | ~3.3B | Best quality, slower |

## Examples

### Generate a short ambient loop
```bash
uv run gen-music "calm ambient fantasy dungeon music with soft echoes" --duration 15
```

### Generate epic battle music with a larger model
```bash
uv run gen-music "intense orchestral battle theme with drums and brass" --duration 20 --model-size medium
```

### More creative output with higher temperature
```bash
uv run gen-music "mysterious cave exploration music" --temperature 1.3
```

### Skip the high-pass filter (keep deep bass)
```bash
uv run gen-music "deep bass drone ambient soundscape" --no-highpass --duration 30
```

## Related Tools
- `loop-music` — Post-process a track for seamless looping
- `extend-music` — Continue/extend a generated clip
- `mix-audio` — Layer and mix multiple tracks together
- `convert-audio` — Convert the output WAV to M4A/OGG/FLAC/MP3
