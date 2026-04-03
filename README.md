# SoundForge 🔊 - Forge your sounds using AI

[![SoundForge Banner](assets/image.png)](https://github.com/pdeligia/sound-forge)

[![CI](https://github.com/pdeligia/sound-forge/actions/workflows/ci.yml/badge.svg)](https://github.com/pdeligia/sound-forge/actions/workflows/ci.yml)

An agentic toolkit for generating game audio assets using AI.

## Requirements
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for dependency management

## Setup
```bash
uv sync
```

## Tools

| Tool | Description |
|------|-------------|
| `analyze-audio` | Analyze audio using AI to generate natural language descriptions and timeline |
| `convert-audio` | Convert between audio formats (WAV, FLAC, OGG, MP3, M4A) and resample |
| `gen-music` | Generate music from a text prompt using MusicGen |
| `gen-sfx` | Generate sound effects from a text prompt using Stable Audio Open |
| `loop-music` | Post-process a track for seamless looping (crossfade, fade, or trim modes) |
| `trim-audio` | Trim seconds from the start and/or end of an audio clip |

> **Copilot skills** for each tool are available in [`.github/skills/`](.github/skills/) — these provide detailed usage docs, options, examples, and related tool references.
