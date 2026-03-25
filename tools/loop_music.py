#!/usr/bin/env python3
"""Post-process a music track for seamless looping."""

import argparse
import os
import time

import numpy as np
from rich.table import Table

from tools.lib.audio_utils import load_audio, save_audio, compute_peak, compute_rms, get_duration, remove_dc_offset
from tools.lib.console import console


def _n_samples(audio):
    """Return number of samples (works for both mono 1D and stereo 2D arrays)."""
    return audio.shape[-1] if audio.ndim > 1 else audio.shape[0]


def _slice(audio, start, end=None):
    """Slice audio samples along the time axis (works for mono and stereo)."""
    if audio.ndim > 1:
        return audio[:, start:end]
    return audio[start:end]


def find_best_loop_point(audio, sample_rate, min_loop_seconds=3.0):
    """Find the best point to start the loop-back by comparing the end with earlier segments.

    Uses cross-correlation on short windows to find where the end of the track
    most closely matches an earlier segment, giving the smoothest possible loop.
    Works with both mono (1D) and stereo (2D) audio by mixing down for analysis.
    """
    # Mix to mono for analysis if stereo
    mono = np.mean(audio, axis=0) if audio.ndim > 1 else audio
    n = len(mono)

    window_size = int(0.5 * sample_rate)  # 0.5s comparison window
    min_offset = int(min_loop_seconds * sample_rate)
    end_window = mono[-window_size:]

    best_score = -np.inf
    best_pos = min_offset

    # Slide through the track looking for the best match
    step = sample_rate // 10  # check every 0.1s
    for pos in range(min_offset, n - window_size * 2, step):
        candidate = mono[pos:pos + window_size]
        # Normalized cross-correlation
        norm = np.sqrt(np.sum(end_window**2) * np.sum(candidate**2))
        if norm < 1e-8:
            continue
        score = float(np.sum(end_window * candidate) / norm)
        if score > best_score:
            best_score = score
            best_pos = pos

    return best_pos, best_score


def _ncc(a, b):
    """Normalized cross-correlation between two equal-length windows."""
    norm = np.sqrt(np.sum(a**2) * np.sum(b**2))
    if norm < 1e-8:
        return -np.inf
    return float(np.sum(a * b) / norm)


def find_best_loop_region(audio, sample_rate, min_loop_seconds=3.0):
    """Find the best pair of points (start, end) where the audio naturally matches.

    Scans all pairs of positions at least min_loop_seconds apart and finds
    the two points where the surrounding audio is most similar — giving a
    natural loop region that can be trimmed out with minimal crossfade.
    """
    mono = np.mean(audio, axis=0) if audio.ndim > 1 else audio
    n = len(mono)

    window_size = int(0.5 * sample_rate)
    min_gap = int(min_loop_seconds * sample_rate)
    step = sample_rate // 10  # check every 0.1s

    # Build list of candidate positions (avoid very start/end for cleaner matches)
    margin = int(0.5 * sample_rate)
    positions = list(range(margin, n - margin - window_size, step))

    best_score = -np.inf
    best_start = margin
    best_end = n - margin

    for i, p1 in enumerate(positions):
        w1 = mono[p1:p1 + window_size]
        for p2 in positions[i + 1:]:
            if p2 - p1 < min_gap:
                continue
            w2 = mono[p2:p2 + window_size]
            score = _ncc(w1, w2)
            if score > best_score:
                best_score = score
                best_start = p1
                best_end = p2

    return best_start, best_end, best_score


def crossfade_loop(audio, crossfade_samples):
    """Create a seamless loop using overlap-add crossfade.

    Overlaps the tail of the track with the head so the loop point is truly
    seamless: adjacent samples in the original audio stay adjacent across the
    seam.  Output is shorter by crossfade_samples (the head and tail collapse
    into one crossfade region).
    Uses equal-power (sinusoidal) curves to prevent energy dips in the blend.
    Works with both mono (1D) and stereo (2D) audio.
    """
    n = _n_samples(audio)
    fade_len = min(crossfade_samples, n // 2)

    # Equal-power curves: cos²+sin²=1 preserves energy through the crossfade
    t = np.linspace(0.0, np.pi / 2, fade_len, dtype=np.float32)
    fade_out = np.cos(t)
    fade_in = np.sin(t)

    if audio.ndim > 1:
        head = audio[:, :fade_len]
        body = audio[:, fade_len:n - fade_len]
        tail = audio[:, n - fade_len:]
        xfade = tail * fade_out + head * fade_in
        return np.concatenate([body, xfade], axis=1)
    else:
        head = audio[:fade_len]
        body = audio[fade_len:n - fade_len]
        tail = audio[n - fade_len:]
        xfade = tail * fade_out + head * fade_in
        return np.concatenate([body, xfade])


def fade_loop(audio, fade_samples, silence_samples=0):
    """Create a loop by fading in at the start and fading out at the end.

    The transition passes through a silence gap, so it works on any track
    regardless of how different the start and end sound.
    Uses an equal-power (sinusoidal) curve for natural-sounding volume change.
    """
    n = _n_samples(audio)
    fade_len = min(fade_samples, n // 2)

    result = audio.copy()

    # Equal-power fade curves (sine/cosine) sound more natural than linear
    t = np.linspace(0.0, np.pi / 2, fade_len, dtype=np.float32)
    fade_in_curve = np.sin(t)
    fade_out_curve = np.cos(t)

    if audio.ndim > 1:
        result[:, :fade_len] = result[:, :fade_len] * fade_in_curve
        result[:, -fade_len:] = result[:, -fade_len:] * fade_out_curve
    else:
        result[:fade_len] = result[:fade_len] * fade_in_curve
        result[-fade_len:] = result[-fade_len:] * fade_out_curve

    # Insert silence gap between loops
    if silence_samples > 0:
        if audio.ndim > 1:
            gap = np.zeros((audio.shape[0], silence_samples), dtype=np.float32)
            result = np.concatenate([result, gap], axis=1)
        else:
            gap = np.zeros(silence_samples, dtype=np.float32)
            result = np.concatenate([result, gap])

    return result


def measure_seam_quality(looped, sample_rate):
    """Measure how smooth the loop seam is by comparing the last and first few ms."""
    mono = np.mean(looped, axis=0) if looped.ndim > 1 else looped
    window = int(0.01 * sample_rate)  # 10ms comparison window
    end_seg = mono[-window:]
    start_seg = mono[:window]
    diff = np.sqrt(np.mean((end_seg - start_seg) ** 2))
    rms = np.sqrt(np.mean(mono ** 2))
    # Lower ratio = smoother seam
    return diff / rms if rms > 1e-8 else float("inf")


def find_best_crossfade(audio, sample_rate):
    """Try multiple crossfade durations and return the one with the smoothest seam."""
    candidates_ms = [250, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 8000]
    best_ms = candidates_ms[0]
    best_quality = float("inf")

    for ms in candidates_ms:
        cf_samples = int(ms * sample_rate / 1000)
        looped = crossfade_loop(audio, cf_samples)
        quality = measure_seam_quality(looped, sample_rate)
        if quality < best_quality:
            best_quality = quality
            best_ms = ms

    return best_ms


def main():
    parser = argparse.ArgumentParser(description="Post-process a music track for seamless looping.")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("--mode", choices=["crossfade", "fade", "trim"], default="crossfade",
                        help="Loop strategy: crossfade (blend end→start), fade (fade out then fade in through silence), "
                             "trim (find best matching region and cut) (default: crossfade)")
    parser.add_argument("--crossfade-ms", default="2000",
                        help="Crossfade/fade duration in ms, or 'auto' to detect ideal duration (default: 2000)")
    parser.add_argument("--silence-ms", type=int, default=0,
                        help="Silence gap in ms between loops, used with --mode fade (default: 0)")
    parser.add_argument("--repeat", type=int, default=0, metavar="N",
                        help="Concatenate the loop N times into the output for testing")
    parser.add_argument("--output", default=None, help="Output file path (default: ./tmp/loop_music/output.wav)")
    args = parser.parse_args()

    output = args.output or os.path.join(".", "tmp", "loop_music", "output.wav")
    os.makedirs(os.path.dirname(output), exist_ok=True)

    console.print()
    console.print("[bold cyan]🔁 loop-music[/bold cyan]")
    console.print()

    auto_crossfade = args.crossfade_ms.lower() == "auto"
    crossfade_ms = 0 if auto_crossfade else int(args.crossfade_ms)

    mode_labels = {"crossfade": "crossfade", "fade": "fade", "trim": "trim"}
    params = Table(show_header=False, show_edge=False, pad_edge=False, padding=(0, 2))
    params.add_column(style="bold")
    params.add_column()
    params.add_row("Input", f"[cyan]{args.input}[/cyan]")
    params.add_row("Mode", f"[cyan]{mode_labels[args.mode]}[/cyan]")
    params.add_row("Fade/crossfade", "[cyan]auto[/cyan]" if auto_crossfade else f"[cyan]{crossfade_ms}ms[/cyan]")
    if args.mode == "fade" and args.silence_ms > 0:
        params.add_row("Silence gap", f"[cyan]{args.silence_ms}ms[/cyan]")
    params.add_row("Output", f"[cyan]{output}[/cyan]")
    console.print(params)
    console.print()

    console.print("  Loading audio...")
    audio, sample_rate = load_audio(args.input)
    audio = remove_dc_offset(audio)
    original_duration = get_duration(audio, sample_rate)
    is_stereo = audio.ndim == 2
    console.print(f"  Loaded [cyan]{original_duration:.1f}s[/cyan] at [cyan]{sample_rate}Hz[/cyan]"
                  f" ({'stereo' if is_stereo else 'mono'})")

    match_score = None

    if args.mode == "trim":
        console.print("  Scanning for best loop region...")
        t0 = time.time()
        region_start, region_end, match_score = find_best_loop_region(audio, sample_rate)
        search_time = time.time() - t0
        start_time = region_start / sample_rate
        end_time = region_end / sample_rate

        console.print(f"  Best region [cyan]{start_time:.2f}s[/cyan] → [cyan]{end_time:.2f}s[/cyan]"
                      f" (similarity: [cyan]{match_score:.3f}[/cyan])")

        audio = _slice(audio, region_start, region_end)

        if auto_crossfade:
            console.print("  Finding ideal crossfade...")
            crossfade_ms = find_best_crossfade(audio, sample_rate)
            console.print(f"  Ideal crossfade: [cyan]{crossfade_ms}ms[/cyan]")

        crossfade_samples = int(crossfade_ms * sample_rate / 1000)
        console.print("  Applying crossfade at seam...")
        looped = crossfade_loop(audio, crossfade_samples)

    elif args.mode == "fade":
        if auto_crossfade:
            # For fade mode, pick a sensible default based on track length
            crossfade_ms = min(3000, int(original_duration * 1000 * 0.1))
            crossfade_ms = max(crossfade_ms, 500)
            console.print(f"  Auto fade duration: [cyan]{crossfade_ms}ms[/cyan]")

        fade_samples = int(crossfade_ms * sample_rate / 1000)
        silence_samples = int(args.silence_ms * sample_rate / 1000)
        console.print("  Applying fade-out/fade-in...")
        looped = fade_loop(audio, fade_samples, silence_samples)

    else:  # crossfade
        if auto_crossfade:
            console.print("  Finding ideal crossfade...")
            crossfade_ms = find_best_crossfade(audio, sample_rate)
            console.print(f"  Ideal crossfade: [cyan]{crossfade_ms}ms[/cyan]")

        crossfade_samples = int(crossfade_ms * sample_rate / 1000)
        console.print("  Applying crossfade...")
        looped = crossfade_loop(audio, crossfade_samples)

    loop_duration = get_duration(looped, sample_rate)

    if args.repeat > 0:
        if looped.ndim > 1:
            looped = np.concatenate([looped] * args.repeat, axis=1)
        else:
            looped = np.concatenate([looped] * args.repeat)

    save_audio(output, looped, sample_rate)
    abs_output = os.path.abspath(output)
    file_size = os.path.getsize(abs_output)

    peak = compute_peak(looped)
    rms = compute_rms(looped)
    db_peak = 20 * np.log10(peak) if peak > 0 else -np.inf

    console.print()
    console.print("  [bold dim]Loop Properties[/bold dim]")
    console.print("  [dim]────────────────[/dim]")

    stats = Table(show_header=False, show_edge=False, pad_edge=False, padding=(0, 2))
    stats.add_column(style="bold")
    stats.add_column()
    stats.add_row("Original", f"[cyan]{original_duration:.1f}s[/cyan]")
    stats.add_row("Loop length", f"[cyan]{loop_duration:.1f}s[/cyan]")
    stats.add_row("Fade/crossfade", f"[cyan]{crossfade_ms}ms[/cyan]" + (" (auto)" if auto_crossfade else ""))
    stats.add_row("Peak", f"[cyan]{peak:.3f}[/cyan] ({db_peak:.1f} dB)")
    stats.add_row("File size", f"[cyan]{file_size / 1024:.0f} KB[/cyan]")
    console.print(stats)

    if match_score is not None:
        quality = "excellent" if match_score > 0.8 else "good" if match_score > 0.6 else "fair" if match_score > 0.4 else "poor"
        console.print()
        console.print(f"  Trim match: [bold]{quality}[/bold] ([cyan]{match_score:.3f}[/cyan])")
        if match_score < 0.5:
            console.print("  [yellow]Tip: try a longer source track for better region matching[/yellow]")

    console.print()
    console.print("[bold green]✓ Done[/bold green]")
    console.print()
    console.print(f"file://{abs_output}")
    console.print()


if __name__ == "__main__":
    main()
