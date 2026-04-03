#!/usr/bin/env python3
"""Generate sound effects from a text prompt using Stable Audio Open."""

import argparse
import os
import time

import numpy as np
import torch
from rich.table import Table

from tools.lib.audio_utils import save_audio, compute_peak, compute_rms, highpass_filter
from tools.lib.console import console, run_with_hf_fallback
from tools.lib.model_utils import load_stable_audio


def analyze_spectrum(audio, sample_rate):
    """Analyze frequency spectrum and return band data."""
    fft = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1 / sample_rate)
    magnitudes = np.abs(fft)
    total_energy = np.sum(magnitudes**2)

    bands = [
        (0, 250, "Low", "rumble, thuds, explosions"),
        (250, 2000, "Mid", "body, impacts, voices"),
        (2000, 6000, "High-mid", "clarity, attacks, crackle"),
        (6000, sample_rate // 2, "High", "sizzle, air, shimmer"),
    ]
    band_data = []
    for lo, hi, name, desc in bands:
        mask = (freqs >= lo) & (freqs < hi)
        energy = np.sum(magnitudes[mask]**2)
        pct = 100 * energy / total_energy if total_energy > 0 else 0
        band_data.append((lo, hi, name, desc, pct))

    return band_data


def to_mono_numpy_audio(audio):
    """Convert Stable Audio output to a mono float32 NumPy array."""
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().float().cpu().numpy()

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    return audio


def print_analysis(audio, sample_rate, highpass_hz, file_size, gen_time, device):
    """Print rich audio analysis tables."""
    duration = len(audio) / sample_rate
    peak = compute_peak(audio)
    rms = compute_rms(audio)
    db_peak = 20 * np.log10(peak) if peak > 0 else -np.inf
    db_rms = 20 * np.log10(rms) if rms > 0 else -np.inf
    band_data = analyze_spectrum(audio, sample_rate)

    console.print()
    console.print("  [bold dim]Audio Properties[/bold dim]")
    console.print("  [dim]────────────────[/dim]")

    stats = Table(show_header=False, show_edge=False, pad_edge=False, padding=(0, 2))
    stats.add_column(style="bold")
    stats.add_column()
    stats.add_row("Duration", f"[cyan]{duration:.1f}s[/cyan]")
    stats.add_row("Sample rate", f"[cyan]{sample_rate}Hz[/cyan]")
    stats.add_row("Peak", f"[cyan]{peak:.3f}[/cyan] ({db_peak:.1f} dB)")
    stats.add_row("RMS", f"[cyan]{rms:.4f}[/cyan] ({db_rms:.1f} dB)")
    if highpass_hz:
        stats.add_row("Highpass", f"[cyan]{highpass_hz}Hz[/cyan]")
    stats.add_row("File size", f"[cyan]{file_size / 1024:.0f} KB[/cyan]")
    console.print(stats)

    console.print()
    console.print("  [bold dim]Frequency Spectrum[/bold dim]")
    console.print("  [dim]────────────────[/dim]")

    freq_table = Table(show_edge=False, pad_edge=False, padding=(0, 2))
    freq_table.add_column("Band", style="bold")
    freq_table.add_column("Range", justify="right")
    freq_table.add_column("Energy", justify="right")
    freq_table.add_column("", min_width=20)
    freq_table.add_column("Character", style="dim")
    for lo, hi, name, desc, pct in band_data:
        bar_len = int(pct / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        color = "red" if pct > 60 else "yellow" if pct > 40 else "green"
        flag = " ⚠️" if pct > 60 else ""
        freq_table.add_row(name, f"{lo}-{hi}Hz", f"[{color}]{pct:.1f}%{flag}[/{color}]", f"[{color}]{bar}[/{color}]", desc)
    console.print(freq_table)

    console.print()
    console.print("  [bold dim]Performance[/bold dim]")
    console.print("  [dim]────────────────[/dim]")

    perf = Table(show_header=False, show_edge=False, pad_edge=False, padding=(0, 2))
    perf.add_column(style="bold")
    perf.add_column()
    perf.add_row("Generation time", f"[cyan]{gen_time:.1f}s[/cyan]")
    perf.add_row("Realtime factor", f"[cyan]{duration / gen_time:.1f}x[/cyan]")
    perf.add_row("Device", f"[cyan]{device}[/cyan]")
    console.print(perf)


def main():
    parser = argparse.ArgumentParser(description="Generate sound effects from a text prompt using Stable Audio Open.")
    parser.add_argument("prompt", help="Text description of the sound effect to generate")
    parser.add_argument("--duration", type=float, default=3.0,
                        help="Duration in seconds (default: 3)")
    parser.add_argument("--steps", type=int, default=100,
                        help="Number of inference steps — higher = better quality, slower (default: 100)")
    parser.add_argument("--guidance-scale", type=float, default=7.0,
                        help="Classifier-free guidance — higher = closer to prompt (default: 7.0)")
    parser.add_argument("--negative-prompt", default="low quality, noise, distortion",
                        help="Negative prompt to steer away from (default: 'low quality, noise, distortion')")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (default: random)")
    parser.add_argument("--highpass", type=float, default=0,
                        help="High-pass filter cutoff in Hz, 0 to disable (default: 0)")
    parser.add_argument("--count", type=int, default=1,
                        help="Generate N variants and keep the best by RMS energy (default: 1)")
    parser.add_argument("--output", default=None, help="Output file path (default: ./tmp/gen_sfx/output.wav)")
    args = parser.parse_args()

    output = args.output or os.path.join(".", "tmp", "gen_sfx", "output.wav")
    os.makedirs(os.path.dirname(output), exist_ok=True)

    console.print()
    console.print("[bold cyan]💥 gen-sfx[/bold cyan]")
    console.print()

    params = Table(show_header=False, show_edge=False, pad_edge=False, padding=(0, 2))
    params.add_column(style="bold")
    params.add_column()
    params.add_row("Prompt", f"[cyan]{args.prompt}[/cyan]")
    params.add_row("Duration", f"[cyan]{args.duration}s[/cyan]")
    params.add_row("Model", "[cyan]stable-audio-open-1.0[/cyan]")
    params.add_row("Steps", f"[cyan]{args.steps}[/cyan]")
    params.add_row("Guidance", f"[cyan]{args.guidance_scale}[/cyan]")
    if args.seed is not None:
        params.add_row("Seed", f"[cyan]{args.seed}[/cyan]")
    if args.highpass > 0:
        params.add_row("Highpass", f"[cyan]{args.highpass}Hz[/cyan]")
    if args.count > 1:
        params.add_row("Variants", f"[cyan]{args.count}[/cyan]")
    params.add_row("Output", f"[cyan]{output}[/cyan]")
    console.print(params)
    console.print()

    console.print("  Loading model...")
    t0 = time.time()
    pipe = run_with_hf_fallback(load_stable_audio)
    load_time = time.time() - t0
    console.print(f"  Model loaded in [cyan]{load_time:.1f}s[/cyan]")

    sample_rate = int(pipe.vae.sampling_rate)
    device = str(pipe.device)
    generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed is not None else None

    results = []
    for i in range(args.count):
        label = f" ({i + 1}/{args.count})" if args.count > 1 else ""
        console.print(f"  Generating{label}...")
        t0 = time.time()
        result = pipe(
            args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            audio_end_in_s=args.duration,
            guidance_scale=args.guidance_scale,
            generator=generator,
        )
        gen_time = time.time() - t0

        audio = to_mono_numpy_audio(result.audios[0])
        if args.highpass > 0:
            audio = highpass_filter(audio, sample_rate, cutoff_hz=args.highpass)

        rms = compute_rms(audio)
        results.append({"audio": audio, "gen_time": gen_time, "rms": rms})

    if args.count > 1:
        console.print()
        console.print("  [bold dim]Variant Ranking[/bold dim]")
        console.print("  [dim]────────────────[/dim]")

        rank_table = Table(show_edge=False, pad_edge=False, padding=(0, 2))
        rank_table.add_column("#", justify="right", style="bold")
        rank_table.add_column("Peak", justify="right")
        rank_table.add_column("RMS", justify="right")
        rank_table.add_column("Time", justify="right")
        rank_table.add_column("")

        best_idx = int(np.argmax([r["rms"] for r in results]))

        for i, r in enumerate(results):
            peak = compute_peak(r["audio"])
            marker = " [bold green]← best[/bold green]" if i == best_idx else ""
            rank_table.add_row(
                str(i + 1), f"{peak:.3f}", f"{r['rms']:.4f}",
                f"{r['gen_time']:.1f}s", marker,
            )
            variant_path = output.replace(".wav", f"_{i + 1}.wav")
            save_audio(variant_path, r["audio"], sample_rate)

        console.print(rank_table)
        console.print(f"  All variants saved as [cyan]{output.replace('.wav', '_N.wav')}[/cyan]")

        audio = results[best_idx]["audio"]
        gen_time = results[best_idx]["gen_time"]
    else:
        audio = results[0]["audio"]
        gen_time = results[0]["gen_time"]

    save_audio(output, audio, sample_rate)
    abs_output = os.path.abspath(output)
    file_size = os.path.getsize(abs_output)

    print_analysis(audio, sample_rate, args.highpass if args.highpass > 0 else None,
                   file_size, gen_time, device)

    console.print()
    console.print("[bold green]✓ Done[/bold green]")
    console.print()
    console.print(f"file://{abs_output}")
    console.print()


if __name__ == "__main__":
    main()
