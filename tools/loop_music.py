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


def find_bridge(audio, sample_rate, window_seconds=0.5, step_seconds=0.1, max_hops=4, top_n=10):
    """Find the best chain of windows to bridge from track end → track start.

    Uses dynamic programming to find paths of 1–max_hops windows where:
      - First window matches the track's end
      - Last window matches the track's start
      - Each consecutive pair flows naturally (high NCC between neighbors)

    Returns top_n candidates as (path, avg_score) where path is a list of
    sample positions and avg_score is the average NCC across all hops (0–1).
    """
    mono = np.mean(audio, axis=0) if audio.ndim > 1 else audio
    n = len(mono)

    window = int(window_seconds * sample_rate)
    step = max(1, int(step_seconds * sample_rate))
    margin = window  # avoid matching the actual start/end

    # Signatures
    end_sig = mono[n - window:]
    start_sig = mono[:window]

    # Score every position against end and start signatures
    positions = list(range(margin, n - margin - window, step))
    end_scores = {}    # how well position matches track end
    start_scores = {}  # how well position matches track start

    for p in positions:
        chunk = mono[p:p + window]
        end_scores[p] = _ncc(chunk, end_sig)
        start_scores[p] = _ncc(chunk, start_sig)

    # Precompute flow scores between positions (how well p1's end flows into p2's start)
    # Only compute for positions where p2 > p1 (forward in time)
    # Use the overlap: last half of window at p1 vs first half of window at p2
    half = window // 2

    def flow_score(p1, p2):
        seg1 = mono[p1 + half:p1 + window]
        seg2 = mono[p2:p2 + half]
        return _ncc(seg1, seg2)

    # Dynamic programming: best path of length k ending at each position
    # State: (position) -> (best_avg_score, path)
    # For k=1: score = avg(end_score, start_score) — single hop
    # For k>1: score = avg(end_score[first], flow_scores..., start_score[last])

    all_candidates = []  # (avg_score, path)

    # 1-hop: single window bridges end→start directly
    for p in positions:
        avg = (end_scores[p] + start_scores[p]) / 2
        all_candidates.append((avg, [p]))

    # Multi-hop: use beam search to keep it tractable
    beam_width = 50
    # Current beam: list of (total_score, num_edges, path)
    # total_score = end_score[first] + sum(flow_scores)
    # At the end we add start_score[last] and divide by (num_edges + 2)

    # Initialize beam with best entry points
    beam = []
    for p in positions:
        beam.append((end_scores[p], 1, [p]))
    # Keep top beam_width by score
    beam.sort(key=lambda x: -x[0] / x[1])
    beam = beam[:beam_width]

    for hop in range(2, max_hops + 1):
        next_beam = []
        for total, num_edges, path in beam:
            last_pos = path[-1]
            for p in positions:
                # Must be forward and non-overlapping
                if p <= last_pos + window:
                    continue
                fs = flow_score(last_pos, p)
                new_total = total + fs
                new_edges = num_edges + 1
                next_beam.append((new_total, new_edges, path + [p]))

        if not next_beam:
            break

        # Keep top beam_width
        next_beam.sort(key=lambda x: -x[0] / x[1])
        next_beam = next_beam[:beam_width]
        beam = next_beam

        # Score complete paths (add start_score for the last position)
        for total, num_edges, path in beam:
            last_pos = path[-1]
            avg = (total + start_scores[last_pos]) / (num_edges + 1)
            all_candidates.append((avg, path))

    # Sort by average score descending
    all_candidates.sort(key=lambda x: -x[0])

    # Diversity selection based on path overlap
    selected = []
    for avg_score, path in all_candidates:
        path_set = set()
        for p in path:
            for s in range(p, p + window, step):
                path_set.add(s)

        is_diverse = True
        for _, sel_path in selected:
            sel_set = set()
            for p in sel_path:
                for s in range(p, p + window, step):
                    sel_set.add(s)
            overlap = len(path_set & sel_set)
            shorter = min(len(path_set), len(sel_set))
            if shorter > 0 and overlap > 0.50 * shorter:
                is_diverse = False
                break

        if is_diverse:
            selected.append((avg_score, path))
            if len(selected) >= top_n:
                break

    return selected


def build_bridge_loop(audio, bridge_chunks, crossfade_samples):
    """Build a loop by chaining bridge chunks between track end and start.

    bridge_chunks is a list of audio arrays to chain. Crossfades are applied
    at every seam: track→chunk1, chunk1→chunk2, ..., chunkN→track(loop).
    """
    fade_len = crossfade_samples

    t = np.linspace(0.0, np.pi / 2, fade_len, dtype=np.float32)
    fade_out = np.cos(t)
    fade_in = np.sin(t)

    def _xfade(a_tail, b_head):
        """Equal-power crossfade between end of a and start of b."""
        return a_tail * fade_out + b_head * fade_in

    # Build the chain: track_body + xfade(track_end, chunk1) + chunk1_body + ... + xfade(chunkN, track_start)
    pieces = []
    is_stereo = audio.ndim > 1
    n = _n_samples(audio)

    # Track body (without first and last fade_len)
    if is_stereo:
        pieces.append(audio[:, fade_len:n - fade_len])
    else:
        pieces.append(audio[fade_len:n - fade_len])

    # Chain: track_end → chunks → track_start
    prev_tail = audio[:, -fade_len:] if is_stereo else audio[-fade_len:]

    for chunk in bridge_chunks:
        cn = _n_samples(chunk)
        chunk_fade = min(fade_len, cn // 2)

        if is_stereo:
            chunk_head = chunk[:, :chunk_fade]
            chunk_body = chunk[:, chunk_fade:cn - chunk_fade]
            chunk_tail = chunk[:, -chunk_fade:]
        else:
            chunk_head = chunk[:chunk_fade]
            chunk_body = chunk[chunk_fade:cn - chunk_fade]
            chunk_tail = chunk[-chunk_fade:]

        # Crossfade previous tail → this chunk's head
        xf = _xfade(prev_tail, chunk_head)
        pieces.append(xf)
        pieces.append(chunk_body)
        prev_tail = chunk_tail

    # Final crossfade: last chunk tail → track start (loop point)
    track_head = audio[:, :fade_len] if is_stereo else audio[:fade_len]
    pieces.append(_xfade(prev_tail, track_head))

    if is_stereo:
        return np.concatenate(pieces, axis=1)
    else:
        return np.concatenate(pieces)


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
    parser.add_argument("--mode", choices=["crossfade", "fade", "trim", "bridge", "trim-bridge"], default="crossfade",
                        help="Loop strategy: crossfade (equal-power blend end→start), "
                             "fade (fade out then fade in through silence), "
                             "trim (find best matching region and cut), "
                             "bridge (clone internal section as loop transition), "
                             "trim-bridge (trim to best region then bridge the seam) "
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

    mode_labels = {"crossfade": "crossfade", "fade": "fade", "trim": "trim", "bridge": "bridge", "trim-bridge": "trim-bridge"}
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
        # Default min-loop scales with track length: at least 1/3 of original, minimum 3s
        min_loop = args.min_loop
        if min_loop <= 0:
            min_loop = max(3.0, original_duration / 3)
        console.print(f"  Scanning for best loop regions (min length: [cyan]{min_loop:.0f}s[/cyan])...")
        t0 = time.time()
        candidates = find_best_loop_region(audio, sample_rate, min_loop_seconds=min_loop,
                                            top_n=args.top, window_seconds=args.window,
                                            step_seconds=args.step)
        search_time = time.time() - t0

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

    elif args.mode == "bridge":
        console.print(f"  Scanning for bridge chains (window: [cyan]{args.window}s[/cyan], step: [cyan]{args.step}s[/cyan])...")
        t0 = time.time()
        candidates = find_bridge(audio, sample_rate, window_seconds=args.window,
                                 step_seconds=args.step, top_n=args.top)
        search_time = time.time() - t0

        if not candidates:
            console.print("  [red]No suitable bridge chains found[/red]")
            return

        window_samples = int(args.window * sample_rate)

        # Display candidates table
        console.print()
        ctable = Table(title="Bridge Candidates", show_edge=False, pad_edge=False, padding=(0, 2))
        ctable.add_column("", style="bold cyan")
        ctable.add_column("Hops", justify="right")
        ctable.add_column("Path", justify="left")
        ctable.add_column("Added", justify="right")
        ctable.add_column("Score", justify="right")

        labels = [chr(ord("A") + i) for i in range(len(candidates))]
        for label, (cscore, path) in zip(labels, candidates):
            hops = len(path)
            path_str = " → ".join(f"{p / sample_rate:.1f}s" for p in path)
            total_added = len(path) * window_samples / sample_rate
            ctable.add_row(label, str(hops), path_str, f"{total_added:.1f}s", f"{cscore:.3f}")

        console.print(ctable)
        console.print()

        # Select the picked candidate
        pick_idx = ord(args.pick.upper()) - ord("A")
        if pick_idx < 0 or pick_idx >= len(candidates):
            console.print(f"  [red]Invalid pick '{args.pick}' — valid: {labels[0]}–{labels[-1]}[/red]")
            return

        match_score, path = candidates[pick_idx]
        hops = len(path)
        console.print(f"  Using [bold cyan]{labels[pick_idx]}[/bold cyan]: "
                      f"{hops} hop{'s' if hops > 1 else ''}, score [cyan]{match_score:.3f}[/cyan]")

        # Extract bridge chunks
        bridge_chunks = [_slice(audio, p, p + window_samples) for p in path]

        if auto_crossfade:
            crossfade_ms = min(500, int(args.window * 1000 * 0.25))
            crossfade_ms = max(crossfade_ms, 100)
            console.print(f"  Auto crossfade: [cyan]{crossfade_ms}ms[/cyan]")

        crossfade_samples = int(crossfade_ms * sample_rate / 1000)
        console.print("  Building bridge loop...")
        looped = build_bridge_loop(audio, bridge_chunks, crossfade_samples)

    elif args.mode == "trim-bridge":
        # Step 1: Trim — find best loop region
        min_loop = args.min_loop
        if min_loop <= 0:
            min_loop = max(3.0, original_duration / 3)
        console.print(f"  [bold]Step 1: Trim[/bold] — scanning for loop regions (min: [cyan]{min_loop:.0f}s[/cyan])...")
        trim_candidates = find_best_loop_region(audio, sample_rate, min_loop_seconds=min_loop,
                                                 top_n=args.top, window_seconds=args.window,
                                                 step_seconds=args.step)

        if not trim_candidates:
            console.print("  [red]No suitable loop regions found[/red]")
            return

        console.print()
        ctable = Table(title="Trim Candidates", show_edge=False, pad_edge=False, padding=(0, 2))
        ctable.add_column("", style="bold cyan")
        ctable.add_column("Start", justify="right")
        ctable.add_column("End", justify="right")
        ctable.add_column("Length", justify="right")
        ctable.add_column("Score", justify="right")

        labels = [chr(ord("A") + i) for i in range(len(trim_candidates))]
        for label, (cstart, cend, cscore) in zip(labels, trim_candidates):
            length = (cend - cstart) / sample_rate
            ctable.add_row(label, f"{cstart / sample_rate:.2f}s", f"{cend / sample_rate:.2f}s",
                           f"{length:.1f}s", f"{cscore:.3f}")

        console.print(ctable)
        console.print()

        pick_idx = ord(args.pick.upper()) - ord("A")
        if pick_idx < 0 or pick_idx >= len(trim_candidates):
            console.print(f"  [red]Invalid pick '{args.pick}' — valid: {labels[0]}–{labels[-1]}[/red]")
            return

        region_start, region_end, trim_score = trim_candidates[pick_idx]
        trimmed = _slice(audio, region_start, region_end)
        trim_dur = (region_end - region_start) / sample_rate
        console.print(f"  Using [bold cyan]{labels[pick_idx]}[/bold cyan]: "
                      f"[cyan]{region_start / sample_rate:.2f}s[/cyan] → [cyan]{region_end / sample_rate:.2f}s[/cyan] "
                      f"([cyan]{trim_dur:.1f}s[/cyan], trim score [cyan]{trim_score:.3f}[/cyan])")

        # Step 2: Bridge — find best bridge chain within the trimmed region
        console.print()
        console.print("  [bold]Step 2: Bridge[/bold] — scanning for bridge chains...")
        bridge_candidates = find_bridge(trimmed, sample_rate, window_seconds=args.window,
                                        step_seconds=args.step, top_n=5)

        window_samples = int(args.window * sample_rate)

        if bridge_candidates:
            bridge_score, path = bridge_candidates[0]
            hops = len(path)
            path_str = " → ".join(f"{p / sample_rate:.1f}s" for p in path)
            console.print(f"  Best bridge: {hops} hop{'s' if hops > 1 else ''} ({path_str}), "
                          f"score [cyan]{bridge_score:.3f}[/cyan]")

            bridge_chunks = [_slice(trimmed, p, p + window_samples) for p in path]
            match_score = (trim_score + bridge_score) / 2

            if auto_crossfade:
                crossfade_ms = min(500, int(args.window * 1000 * 0.25))
                crossfade_ms = max(crossfade_ms, 100)
                console.print(f"  Auto crossfade: [cyan]{crossfade_ms}ms[/cyan]")

            crossfade_samples = int(crossfade_ms * sample_rate / 1000)
            console.print("  Building bridge loop...")
            looped = build_bridge_loop(trimmed, bridge_chunks, crossfade_samples)

            crossfade_samples = int(crossfade_ms * sample_rate / 1000)
            console.print("  Building bridge loop...")
            looped = build_bridge_loop(trimmed, bridge_audio, crossfade_samples)
        else:
            console.print("  [yellow]No bridge found — falling back to crossfade[/yellow]")
            match_score = trim_score
            if auto_crossfade:
                crossfade_ms = find_best_crossfade(trimmed, sample_rate)
                console.print(f"  Auto crossfade: [cyan]{crossfade_ms}ms[/cyan]")
            crossfade_samples = int(crossfade_ms * sample_rate / 1000)
            looped = crossfade_loop(trimmed, crossfade_samples)

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
