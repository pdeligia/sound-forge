---
name: trim-audio
description: Trim seconds from the start and/or end of an audio clip. Use this skill when the user wants to cut unwanted portions from the beginning or end of a music track or sound effect.
---

# trim-audio

A Python tool that trims a specified number of seconds from the start and/or end of an audio file.

## How to Run
```bash
uv run trim-audio <input.wav> [options]
```

## Arguments
| Argument | Required | Description |
|----------|----------|-------------|
| `input` | Yes | Input audio file |

## Options
| Flag | Default | Description |
|------|---------|-------------|
| `--start N` | `0` | Seconds to trim from the start |
| `--end N` | `0` | Seconds to trim from the end |
| `--output FILE` | `./tmp/trim_audio/output.wav` | Output WAV file path |

## Behavior
- Removes the specified duration from the start and/or end of the audio
- Works with both mono and stereo audio
- At least one of `--start` or `--end` must be specified
- Reports: original duration, trimmed duration, amount removed, peak level, and file size

## Examples

### Trim 2 seconds from the end
```bash
uv run trim-audio soundtrack.wav --end 2
```

### Trim 1 second from the start
```bash
uv run trim-audio soundtrack.wav --start 1
```

### Trim both start and end
```bash
uv run trim-audio soundtrack.wav --start 0.5 --end 3
```

### Trim and save to a specific path
```bash
uv run trim-audio soundtrack.wav --end 2 --output trimmed.wav
```

## Related Tools
- `gen-music` — Generate music (may need trimming)
- `loop-music` — Make the trimmed track loop seamlessly
- `convert-audio` — Convert the trimmed audio to M4A/OGG for game use
