---
name: gen-sfx
description: Generate sound effects from a text prompt using AudioGen AI. Use this skill when the user wants to create weapon sounds, monster cries, environmental audio, footsteps, explosions, or any non-musical audio assets for games from a text description.
---

# gen-sfx

A Python tool for generating sound effects from a text prompt using Meta's AudioGen model (via HuggingFace Transformers).

## How to Run
```bash
uv run gen-sfx "<prompt>" [options]
```

## Arguments
| Argument | Required | Description |
|----------|----------|-------------|
| `prompt` | Yes | Text description of the sound effect to generate |

## Options
| Flag | Default | Description |
|------|---------|-------------|
| `--duration N` | `3` | Duration in seconds |
| `--model-size SIZE` | `medium` | Model size: `medium` (currently the only AudioGen size available) |
| `--temperature N` | `1.0` | Sampling temperature — higher = more varied |
| `--guidance-scale N` | `3.0` | Classifier-free guidance — higher = closer to prompt |
| `--top-k N` | `250` | Top-k sampling |
| `--top-p N` | `0.92` | Nucleus sampling threshold |
| `--highpass N` | `0` | High-pass filter cutoff in Hz, 0 to disable |
| `--count N` | `1` | Generate N variants and keep the best |
| `--output FILE` | `./tmp/gen_sfx/output.wav` | Output WAV file path |

## Behavior
- Loads the AudioGen model (auto-downloads on first use, cached afterward)
- Generates audio from the text prompt using classifier-free guidance (scale 3.0)
- Default duration is 3 seconds (typical for game SFX; increase for longer ambient effects)
- When using `--count`, generates multiple variants and auto-selects the best based on mid+high frequency energy (most perceptually clear)
- Outputs a WAV file with detailed stats: duration, peak/RMS levels, frequency spectrum, and performance metrics
- Auto-detects MPS (Apple Silicon), CUDA, or CPU

## Models Used
| Model | Size | Description |
|-------|------|-------------|
| `facebook/audiogen-medium` | ~1.5B | General-purpose sound effect generation |

## Examples

### Weapon sounds
```bash
uv run gen-sfx "sword slash with metallic ring" --duration 2
uv run gen-sfx "bow and arrow being released" --duration 2
uv run gen-sfx "heavy hammer impact on stone" --duration 2
uv run gen-sfx "magical energy blast firing" --duration 3
```

### Monster and creature sounds
```bash
uv run gen-sfx "deep growling monster roar" --duration 3
uv run gen-sfx "small creature screech and hiss" --duration 2
uv run gen-sfx "dragon breathing fire" --duration 4
uv run gen-sfx "wolf howling in the distance" --duration 5
```

### Item and UI sounds
```bash
uv run gen-sfx "glass bottle potion drinking gulp" --duration 2
uv run gen-sfx "coins clinking and dropping" --duration 2
uv run gen-sfx "treasure chest opening with creak" --duration 3
uv run gen-sfx "magical sparkle chime" --duration 2
```

### Environment and ambience
```bash
uv run gen-sfx "footsteps on gravel path" --duration 4
uv run gen-sfx "wooden door creaking open slowly" --duration 3
uv run gen-sfx "crackling campfire" --duration 5
uv run gen-sfx "thunder rumbling in the distance" --duration 4
```

### Generate multiple variants and pick the best
```bash
uv run gen-sfx "explosion with debris" --count 3 --duration 3
```

## Tips for Good Prompts
- Be specific about materials: "metal sword" vs just "sword"
- Describe the action: "slashing", "clanging", "whooshing"
- Add context: "in a cave", "echoing", "muffled"
- Keep prompts concise but descriptive
- For game SFX, shorter durations (1-4s) generally work better

## Related Tools
- `trim-audio` — Trim silence or unwanted parts from the start/end
- `convert-audio` — Convert the output WAV to OGG/MP3/FLAC/M4A for game engine integration
- `gen-music` — Generate background music and soundtrack clips
