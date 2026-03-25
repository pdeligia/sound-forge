#!/usr/bin/env python3
"""Trim audio from the start and/or end of a clip."""

import argparse
import os

import numpy as np
from rich.table import Table

from tools.lib.audio_utils import load_audio, save_audio, compute_peak, compute_rms, get_duration
from tools.lib.console import console


def trim_audio(audio, sample_rate, start_seconds=0.0, end_seconds=0.0):
    """Trim audio by removing time from the start and/or end.

    Args:
        audio: 1D (mono) or 2D (channels, samples) array.
        sample_rate: Sample rate in Hz.
        start_seconds: Seconds to remove from the start.
        end_seconds: Seconds to remove from the end.

    Returns:
        Trimmed audio array.
    """
    n = audio.shape[-1] if audio.ndim > 1 else audio.shape[0]
    start_samples = int(start_seconds * sample_rate)
    end_samples = int(end_seconds * sample_rate)

    start_samples = min(start_samples, n)
    end_samples = min(end_samples, n - start_samples)

    end_idx = n - end_samples if end_samples > 0 else n

    if audio.ndim > 1:
        return audio[:, start_samples:end_idx]
    return audio[start_samples:end_idx]


def main():
    parser = argparse.ArgumentParser(description="Trim audio from the start and/or end of a clip.")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("--start", type=float, default=0.0,
                        help="Seconds to trim from the start (default: 0)")
    parser.add_argument("--end", type=float, default=0.0,
                        help="Seconds to trim from the end (default: 0)")
    parser.add_argument("--output", default=None,
                        help="Output file path (default: ./tmp/trim_audio/output.wav)")
    args = parser.parse_args()

    if args.start == 0.0 and args.end == 0.0:
        console.print("[red]Error: specify --start and/or --end to trim[/red]")
        raise SystemExit(1)

    output = args.output or os.path.join(".", "tmp", "trim_audio", "output.wav")
    os.makedirs(os.path.dirname(output), exist_ok=True)

    console.print()
    console.print("[bold cyan]✂️  trim-audio[/bold cyan]")
    console.print()

    params = Table(show_header=False, show_edge=False, pad_edge=False, padding=(0, 2))
    params.add_column(style="bold")
    params.add_column()
    params.add_row("Input", f"[cyan]{args.input}[/cyan]")
    if args.start > 0:
        params.add_row("Trim start", f"[cyan]{args.start}s[/cyan]")
    if args.end > 0:
        params.add_row("Trim end", f"[cyan]{args.end}s[/cyan]")
    params.add_row("Output", f"[cyan]{output}[/cyan]")
    console.print(params)
    console.print()

    console.print("  Loading audio...")
    audio, sample_rate = load_audio(args.input)
    original_duration = get_duration(audio, sample_rate)
    is_stereo = audio.ndim == 2
    console.print(f"  Loaded [cyan]{original_duration:.1f}s[/cyan] at [cyan]{sample_rate}Hz[/cyan]"
                  f" ({'stereo' if is_stereo else 'mono'})")

    console.print("  Trimming...")
    trimmed = trim_audio(audio, sample_rate, args.start, args.end)
    trimmed_duration = get_duration(trimmed, sample_rate)
    removed = original_duration - trimmed_duration

    save_audio(output, trimmed, sample_rate)
    abs_output = os.path.abspath(output)
    file_size = os.path.getsize(abs_output)

    peak = compute_peak(trimmed)
    rms = compute_rms(trimmed)
    db_peak = 20 * np.log10(peak) if peak > 0 else -np.inf

    console.print()
    console.print("  [bold dim]Output Properties[/bold dim]")
    console.print("  [dim]────────────────[/dim]")

    stats = Table(show_header=False, show_edge=False, pad_edge=False, padding=(0, 2))
    stats.add_column(style="bold")
    stats.add_column()
    stats.add_row("Original", f"[cyan]{original_duration:.1f}s[/cyan]")
    stats.add_row("Trimmed", f"[cyan]{trimmed_duration:.1f}s[/cyan]")
    stats.add_row("Removed", f"[cyan]{removed:.1f}s[/cyan]")
    stats.add_row("Peak", f"[cyan]{peak:.3f}[/cyan] ({db_peak:.1f} dB)")
    stats.add_row("File size", f"[cyan]{file_size / 1024:.0f} KB[/cyan]")
    console.print(stats)

    console.print()
    console.print("[bold green]✓ Done[/bold green]")
    console.print()
    console.print(f"file://{abs_output}")
    console.print()


if __name__ == "__main__":
    main()
