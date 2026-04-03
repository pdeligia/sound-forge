#!/usr/bin/env python3
"""Post-process a music track for seamless looping."""

import argparse
import os

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


def _ncc(a, b):
    """Normalized cross-correlation between two equal-length windows."""
    norm = np.sqrt(np.sum(a**2) * np.sum(b**2))
    if norm < 1e-8:
        return -np.inf
    return float(np.sum(a * b) / norm)


def find_best_loop_region(audio, sample_rate, min_loop_seconds=3.0, top_n=1, window_seconds=0.1, step_seconds=None):
    """Find diverse, high-quality loop region candidates.

    Scans all pairs of positions at least min_loop_seconds apart. Collects a
    large pool of high-scoring pairs, then greedily selects top_n that are
    spread across different regions and lengths (at least 2s apart in both
    start and end, or at least 3s different in length).

    Returns a list of (start, end, score) tuples sorted by score descending.
    """
    import heapq

    mono = np.mean(audio, axis=0) if audio.ndim > 1 else audio
    n = len(mono)

    window_size = int(window_seconds * sample_rate)
    min_gap = int(min_loop_seconds * sample_rate)
    if step_seconds is not None:
        step = max(1, int(step_seconds * sample_rate))
    else:
        step = max(1, window_size // 5)  # default: 1/5 of window size

    # Build list of candidate positions (avoid very start/end for cleaner matches)
    margin = int(0.5 * sample_rate)
    positions = list(range(margin, n - margin - window_size, step))

    # Collect a large pool of the best raw matches
    pool_size = top_n * 200
    heap = []  # (score, start, end)

    for i, p1 in enumerate(positions):
        w1 = mono[p1:p1 + window_size]
        for p2 in positions[i + 1:]:
            if p2 - p1 < min_gap:
                continue
            w2 = mono[p2:p2 + window_size]
            score = _ncc(w1, w2)
            if len(heap) < pool_size:
                heapq.heappush(heap, (score, p1, p2))
            elif score > heap[0][0]:
                heapq.heapreplace(heap, (score, p1, p2))

    # Sort pool by score descending
    pool = sorted(heap, key=lambda x: -x[0])

    # Greedy diversity selection: reject candidates too similar to already-picked
    # ones. Two regions are "too similar" if they overlap more than 70% of the
    # shorter region AND their lengths are within 30% of each other.
    selected = []
    for score, start, end in pool:
        length = end - start
        is_diverse = True
        for s_start, s_end, _ in selected:
            s_length = s_end - s_start
            overlap_start = max(start, s_start)
            overlap_end = min(end, s_end)
            overlap = max(0, overlap_end - overlap_start)
            shorter = min(length, s_length)
            longer = max(length, s_length)
            high_overlap = overlap > 0.70 * shorter
            similar_length = shorter > 0.70 * longer
            if high_overlap and similar_length:
                is_diverse = False
                break
        if is_diverse:
            selected.append((start, end, score))
            if len(selected) >= top_n:
                break

    return selected


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
                        help="Loop strategy: crossfade (equal-power blend end→start), "
                             "fade (fade out then fade in through silence), "
                             "trim (find best matching region and cut) "
                             "(default: crossfade)")
    parser.add_argument("--crossfade-ms", default="2000",
                        help="Crossfade/fade duration in ms, or 'auto' to detect ideal duration (default: 2000)")
    parser.add_argument("--silence-ms", type=int, default=0,
                        help="Silence gap in ms between loops, used with --mode fade (default: 0)")
    parser.add_argument("--repeat", type=int, default=0, metavar="N",
                        help="Concatenate the loop N times into the output for testing")
    parser.add_argument("--top", type=int, default=10, metavar="N",
                        help="Number of trim candidates to show (default: 10)")
    parser.add_argument("--pick", default="A", metavar="LETTER",
                        help="Which trim candidate to use, e.g. A, B, C (default: A)")
    parser.add_argument("--min-loop", type=float, default=0, metavar="SECS",
                        help="Minimum loop length in seconds for trim mode (default: auto = 1/3 of track)")
    parser.add_argument("--window", type=float, default=0.5, metavar="SECS",
                        help="Comparison window size in seconds for trim mode (default: 0.5)")
    parser.add_argument("--step", type=float, default=0.1, metavar="SECS",
                        help="Scan step size in seconds for trim mode (default: 0.1)")
    parser.add_argument("--output", default=None, help="Output file path (default: ./tmp/loop_music/output.wav)")
    args = parser.parse_args()

    output = args.output or os.path.join(".", "tmp", "loop_music", "output.wav")
    os.makedirs(os.path.dirname(output), exist_ok=True)

    console.print()
    console.print("[bold cyan]🔁 loop-music[/bold cyan]")
    console.print()

    auto_crossfade = args.crossfade_ms.lower() == "auto"
    crossfade_ms = 0 if auto_crossfade else int(args.crossfade_ms)

    params = Table(show_header=False, show_edge=False, pad_edge=False, padding=(0, 2))
    params.add_column(style="bold")
    params.add_column()
    params.add_row("Input", f"[cyan]{args.input}[/cyan]")
    params.add_row("Mode", f"[cyan]{args.mode}[/cyan]")
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
        # Default min-loop scales with track length: at least 1/3 of original, minimum 3s
        min_loop = args.min_loop
        if min_loop <= 0:
            min_loop = max(3.0, original_duration / 3)
        console.print(f"  Scanning for best loop regions (min length: [cyan]{min_loop:.0f}s[/cyan])...")
        candidates = find_best_loop_region(audio, sample_rate, min_loop_seconds=min_loop,
                                            top_n=args.top, window_seconds=args.window,
                                            step_seconds=args.step)

        if not candidates:
            console.print("  [red]No suitable loop regions found[/red]")
            return

        # Display candidates table
        console.print()
        ctable = Table(title="Trim Candidates", show_edge=False, pad_edge=False, padding=(0, 2))
        ctable.add_column("", style="bold cyan")
        ctable.add_column("Start", justify="right")
        ctable.add_column("End", justify="right")
        ctable.add_column("Length", justify="right")
        ctable.add_column("Score", justify="right")

        labels = [chr(ord("A") + i) for i in range(len(candidates))]
        for label, (cstart, cend, cscore) in zip(labels, candidates):
            length = (cend - cstart) / sample_rate
            ctable.add_row(
                label,
                f"{cstart / sample_rate:.2f}s",
                f"{cend / sample_rate:.2f}s",
                f"{length:.1f}s",
                f"{cscore:.3f}",
            )

        console.print(ctable)
        console.print()

        # Select the picked candidate
        pick_idx = ord(args.pick.upper()) - ord("A")
        if pick_idx < 0 or pick_idx >= len(candidates):
            console.print(f"  [red]Invalid pick '{args.pick}' — valid: {labels[0]}–{labels[-1]}[/red]")
            return

        region_start, region_end, match_score = candidates[pick_idx]
        start_time = region_start / sample_rate
        end_time = region_end / sample_rate
        loop_length = end_time - start_time

        console.print(f"  Using [bold cyan]{labels[pick_idx]}[/bold cyan]: "
                      f"[cyan]{start_time:.2f}s[/cyan] → [cyan]{end_time:.2f}s[/cyan] "
                      f"([cyan]{loop_length:.1f}s[/cyan], score [cyan]{match_score:.3f}[/cyan])")

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
