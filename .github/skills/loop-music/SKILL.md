---
name: loop-music
description: Post-process a music track for seamless looping via crossfade, fade-through-silence, or region trimming. Use this skill when the user wants to make a soundtrack clip loop seamlessly for use as game background music.
---

# loop-music

A Python tool that post-processes a music track for seamless looping using one of three strategies.

## How to Run
```bash
uv run loop-music <input.wav> [options]
```

## Arguments
| Argument | Required | Description |
|----------|----------|-------------|
| `input` | Yes | Input audio file |

## Options
| Flag | Default | Description |
|------|---------|-------------|
| `--mode MODE` | `crossfade` | Loop strategy: `crossfade`, `fade`, or `trim` |
| `--crossfade-ms N` | `2000` | Fade/crossfade duration in ms, or `auto` to detect ideal duration |
| `--silence-ms N` | `0` | Silence gap in ms between loops (used with `--mode fade`) |
| `--repeat N` | `0` | Concatenate the loop N times into the output for testing |
| `--output FILE` | `./tmp/loop_music/output.wav` | Output WAV file path |

## Modes

### `crossfade` (default)
Blends the end of the track into the beginning using an equal-power crossfade. Keeps the full track length. Works best when the start and end of the track sound similar.

### `fade`
Fades out at the end and fades in at the start, transitioning through silence. Uses equal-power (sinusoidal) curves for natural volume changes. Works on any track regardless of how different the start and end sound. Use `--silence-ms` to add a gap of silence between loops.

### `trim`
Scans the entire track for the two points where the audio most naturally matches (using normalized cross-correlation), then trims to just that region and applies a crossfade at the seam. Produces shorter loops but with the best possible match. Reports a match score with quality rating.

## Auto Crossfade
Use `--crossfade-ms auto` to automatically try multiple durations (250ms–8000ms) and pick the one with the smoothest seam.

## Examples

### Fade mode with silence gap (best for varied tracks)
```bash
uv run loop-music soundtrack.wav --mode fade --crossfade-ms 5000 --silence-ms 2000 --repeat 3
```

### Crossfade with auto-detected duration
```bash
uv run loop-music soundtrack.wav --crossfade-ms auto --repeat 3
```

### Trim to best matching region
```bash
uv run loop-music soundtrack.wav --mode trim --crossfade-ms auto
```

### Quick test with 3x repeat
```bash
uv run loop-music soundtrack.wav --mode fade --crossfade-ms 3000 --repeat 3
```

## Tips
- For tracks where the start and end sound very different, use `--mode fade`
- Use `--repeat 3` to generate a longer file and listen for seams
- Longer `--crossfade-ms` values give smoother transitions
- `--silence-ms 1000-3000` adds a natural breathing pause between loops

## Related Tools
- `gen-music` — Generate the source track from a text prompt
- `convert-audio` — Convert the looped WAV to M4A/OGG for game integration
- `analyze-audio` — Check audio stats before and after looping
