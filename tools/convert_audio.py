#!/usr/bin/env python3
"""Convert between audio formats and resample to target sample rate."""

import argparse
import os
import time

import numpy as np
from rich.table import Table

from tools.lib.audio_utils import load_audio, save_audio, compute_peak, compute_rms, resample, get_duration
from tools.lib.console import console


SUPPORTED_FORMATS = {
    ".wav": "WAV (uncompressed PCM)",
    ".flac": "FLAC (lossless)",
    ".ogg": "OGG Vorbis (lossy)",
    ".mp3": "MP3 (lossy)",
    ".m4a": "AAC/M4A (lossy, iOS-optimized)",
}


def _n_samples_static(data: np.ndarray) -> int:
    """Return number of samples (works for both mono 1D and stereo 2D arrays)."""
    return data.shape[-1] if data.ndim > 1 else data.shape[0]


def to_mono(data: np.ndarray) -> np.ndarray:
    """Down-mix multi-channel audio to mono by averaging channels."""
    if data.ndim == 1:
        return data
    return np.mean(data, axis=0).astype(np.float32)


def to_stereo(data: np.ndarray) -> np.ndarray:
    """Up-mix mono to stereo by duplicating the channel."""
    if data.ndim == 2:
        return data
    return np.stack([data, data])


def save_with_format(path: str, data: np.ndarray, sample_rate: int) -> None:
    """Save audio in the format determined by the file extension."""
    ext = os.path.splitext(path)[1].lower()

    if ext in (".wav", ".flac", ".ogg"):
        import soundfile as sf

        subtype_map = {".wav": "PCM_16", ".flac": "PCM_16", ".ogg": "VORBIS"}
        format_map = {".wav": "WAV", ".flac": "FLAC", ".ogg": "OGG"}
        write_data = data.T if data.ndim == 2 else data
        sf.write(path, write_data, sample_rate,
                 format=format_map[ext], subtype=subtype_map[ext])
    elif ext == ".mp3":
        import torch
        import torchaudio

        tensor = torch.from_numpy(data)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        try:
            torchaudio.save(path, tensor, sample_rate, format="mp3")
        except RuntimeError as exc:
            raise RuntimeError(
                f"MP3 encoding failed ({exc}). "
                "Ensure ffmpeg is installed: brew install ffmpeg"
            ) from exc
    elif ext == ".m4a":
        import subprocess
        import tempfile

        # AAC encoders add priming silence (~2112 samples at 44100Hz) at the
        # start of the file. For seamless looping, we prepend samples from the
        # end of the audio so that after decoding the priming region contains
        # real audio instead of silence.
        AAC_PRIMING_SAMPLES = 2112
        priming = min(AAC_PRIMING_SAMPLES, _n_samples_static(data) // 2)
        if priming > 0:
            if data.ndim == 2:
                data = np.concatenate([data[:, -priming:], data], axis=1)
            else:
                data = np.concatenate([data[-priming:], data])

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            import soundfile as sf
            write_data = data.T if data.ndim == 2 else data
            sf.write(tmp_path, write_data, sample_rate, format="WAV", subtype="PCM_16")
            subprocess.run(
                ["afconvert", tmp_path, path, "-d", "aac", "-f", "m4af", "-b", "256000"],
                check=True, capture_output=True,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "M4A/AAC encoding requires macOS afconvert (not available on this platform)"
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"M4A/AAC encoding failed: {exc.stderr.decode()}") from exc
        finally:
            os.unlink(tmp_path)
    else:
        raise ValueError(f"Unsupported output format: {ext}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert between audio formats and resample to target sample rate.")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("--format", choices=["wav", "flac", "ogg", "mp3", "m4a"], default=None,
                        help="Output format (inferred from --output extension if not set)")
    parser.add_argument("--sample-rate", type=int, default=None,
                        help="Target sample rate in Hz (e.g. 44100, 48000)")
    parser.add_argument("--channels", choices=["mono", "stereo"], default=None,
                        help="Convert to mono or stereo")
    parser.add_argument("--output", default=None,
                        help="Output file path (default: ./tmp/convert_audio/output.<format>)")
    args = parser.parse_args()

    # Determine output format
    if args.format:
        out_ext = f".{args.format}"
    elif args.output:
        out_ext = os.path.splitext(args.output)[1].lower()
        if out_ext not in SUPPORTED_FORMATS:
            console.print(f"[red]Error: unsupported format '{out_ext}'. "
                          f"Supported: {', '.join(SUPPORTED_FORMATS)}[/red]")
            raise SystemExit(1)
    else:
        out_ext = ".wav"

    output = args.output or os.path.join(".", "tmp", "convert_audio", f"output{out_ext}")
    os.makedirs(os.path.dirname(output), exist_ok=True)

    console.print()
    console.print("[bold cyan]🔄 convert-audio[/bold cyan]")
    console.print()

    params = Table(show_header=False, show_edge=False, pad_edge=False, padding=(0, 2))
    params.add_column(style="bold")
    params.add_column()
    params.add_row("Input", f"[cyan]{args.input}[/cyan]")
    params.add_row("Format", f"[cyan]{SUPPORTED_FORMATS[out_ext]}[/cyan]")
    if args.sample_rate:
        params.add_row("Sample rate", f"[cyan]{args.sample_rate}Hz[/cyan]")
    if args.channels:
        params.add_row("Channels", f"[cyan]{args.channels}[/cyan]")
    params.add_row("Output", f"[cyan]{output}[/cyan]")
    console.print(params)
    console.print()

    # Load input
    console.print("  Loading audio...")
    audio, sample_rate = load_audio(args.input)
    input_duration = get_duration(audio, sample_rate)
    is_stereo = audio.ndim == 2
    console.print(f"  Loaded [cyan]{input_duration:.1f}s[/cyan] at [cyan]{sample_rate}Hz[/cyan]"
                  f" ({'stereo' if is_stereo else 'mono'})")

    # Channel conversion
    if args.channels == "mono" and is_stereo:
        console.print("  Converting to mono...")
        audio = to_mono(audio)
    elif args.channels == "stereo" and not is_stereo:
        console.print("  Converting to stereo...")
        audio = to_stereo(audio)

    # Resample
    target_sr = args.sample_rate or sample_rate
    if target_sr != sample_rate:
        console.print(f"  Resampling [cyan]{sample_rate}Hz[/cyan] → [cyan]{target_sr}Hz[/cyan]...")
        t0 = time.time()
        audio = resample(audio, sample_rate, target_sr)
        resample_time = time.time() - t0
        console.print(f"  Resampled in [cyan]{resample_time:.1f}s[/cyan]")
        sample_rate = target_sr

    # Save
    console.print(f"  Writing {SUPPORTED_FORMATS[out_ext]}...")
    t0 = time.time()
    save_with_format(output, audio, sample_rate)
    write_time = time.time() - t0

    abs_output = os.path.abspath(output)
    file_size = os.path.getsize(abs_output)
    input_size = os.path.getsize(os.path.abspath(args.input))

    peak = compute_peak(audio)
    rms = compute_rms(audio)
    db_peak = 20 * np.log10(peak) if peak > 0 else -np.inf
    db_rms = 20 * np.log10(rms) if rms > 0 else -np.inf
    output_duration = get_duration(audio, sample_rate)
    out_channels = "stereo" if audio.ndim == 2 else "mono"

    console.print()
    console.print("  [bold dim]Output Properties[/bold dim]")
    console.print("  [dim]────────────────[/dim]")

    stats = Table(show_header=False, show_edge=False, pad_edge=False, padding=(0, 2))
    stats.add_column(style="bold")
    stats.add_column()
    stats.add_row("Duration", f"[cyan]{output_duration:.1f}s[/cyan]")
    stats.add_row("Sample rate", f"[cyan]{sample_rate}Hz[/cyan]")
    stats.add_row("Channels", f"[cyan]{out_channels}[/cyan]")
    stats.add_row("Peak", f"[cyan]{peak:.3f}[/cyan] ({db_peak:.1f} dB)")
    stats.add_row("RMS", f"[cyan]{rms:.4f}[/cyan] ({db_rms:.1f} dB)")
    stats.add_row("File size", f"[cyan]{file_size / 1024:.0f} KB[/cyan]")
    if input_size > 0:
        ratio = file_size / input_size
        label = "Compression" if ratio < 1.0 else "Size change"
        color = "green" if ratio < 1.0 else "yellow"
        stats.add_row(label, f"[{color}]{ratio:.1%}[/{color}] of original")
    stats.add_row("Write time", f"[cyan]{write_time:.2f}s[/cyan]")
    console.print(stats)

    console.print()
    console.print("[bold green]✓ Done[/bold green]")
    console.print()
    console.print(f"file://{abs_output}")
    console.print()


if __name__ == "__main__":
    main()
