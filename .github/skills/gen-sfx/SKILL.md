---
name: gen-sfx
description: Generate sound effects from a text prompt using Stable Audio Open AI. Use this skill when the user wants to create weapon sounds, monster cries, environmental audio, footsteps, explosions, or any non-musical audio assets for games from a text description.
---

# gen-sfx

A Python tool for generating sound effects from a text prompt using Stable Audio Open (via HuggingFace Diffusers).

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
| `--duration N` | `3` | Duration in seconds (max 47) |
| `--steps N` | `100` | Inference steps — higher = better quality, slower |
| `--guidance-scale N` | `7.0` | Classifier-free guidance — higher = closer to prompt |
| `--negative-prompt TEXT` | `low quality, noise, distortion` | Negative prompt to steer away from |
| `--seed N` | random | Random seed for reproducibility |
| `--highpass N` | `0` | High-pass filter cutoff in Hz, 0 to disable |
| `--count N` | `1` | Generate N variants and keep the best by RMS energy |
| `--output FILE` | `./tmp/gen_sfx/output.wav` | Output WAV file path |

## Behavior
- Loads the Stable Audio Open pipeline (auto-downloads on first use, cached afterward)
- Generates 44.1kHz audio via latent diffusion from the text prompt
- Default duration is 3 seconds (typical for game SFX; increase for longer ambient effects)
- When using `--count`, generates multiple variants and auto-selects the one with highest RMS energy (fullest sound)
- Outputs a 44.1kHz mono WAV file with detailed stats: duration, peak/RMS levels, frequency spectrum, and performance metrics
- Auto-detects MPS (Apple Silicon), CUDA, or CPU

## Models Used
| Model | Description |
|-------|-------------|
| `stabilityai/stable-audio-open-1.0` | 44.1kHz stereo, up to 47s, high quality |

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

### Reproducible output with a seed
```bash
uv run gen-sfx "sword clash" --seed 42 --duration 2
```

## Tips for Good Prompts
- Be specific about materials: "metal sword" vs just "sword"
- Describe the action: "slashing", "clanging", "whooshing"
- Add context: "in a cave", "echoing", "muffled"
- Keep prompts concise but descriptive
- Use `--negative-prompt` to avoid unwanted qualities
- For game SFX, shorter durations (1-4s) generally work better

## Related Tools
- `trim-audio` — Trim silence or unwanted parts from the start/end
- `convert-audio` — Convert the output WAV to OGG/MP3/FLAC/M4A for game engine integration
- `gen-music` — Generate background music and soundtrack clips
