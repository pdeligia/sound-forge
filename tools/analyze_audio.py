#!/usr/bin/env python3
"""Analyze audio using AI to generate natural language descriptions."""

import argparse
import os
import time

import numpy as np
import soundfile as sf
import torch
from rich.table import Table

from tools.lib.console import console, run_with_hf_fallback
from tools.lib.model_utils import load_qwen2_audio


def detect_sections(audio_mono, sample_rate, min_section=3.0, sensitivity=1.5):
    """Detect natural section boundaries using energy and spectral changes.

    Returns a list of (start_sec, end_sec) tuples for each detected section.
    """
    n = len(audio_mono)
    duration = n / sample_rate

    hop = int(0.5 * sample_rate)
    win = int(1.0 * sample_rate)

    times = []
    features = []

    for start in range(0, n - win, hop):
        chunk = audio_mono[start:start + win]
        t = (start + win / 2) / sample_rate

        # RMS energy
        rms = np.sqrt(np.mean(chunk ** 2))

        # Spectral centroid (brightness)
        fft = np.abs(np.fft.rfft(chunk))
        freqs = np.fft.rfftfreq(len(chunk), 1 / sample_rate)
        centroid = np.sum(freqs * fft) / (np.sum(fft) + 1e-8)

        times.append(t)
        features.append((rms, centroid))

    if len(features) < 3:
        return [(0, duration)]

    rms_vals = np.array([f[0] for f in features])
    cent_vals = np.array([f[1] for f in features])

    # Normalize to 0-1
    rms_norm = (rms_vals - rms_vals.min()) / (rms_vals.max() - rms_vals.min() + 1e-8)
    cent_norm = (cent_vals - cent_vals.min()) / (cent_vals.max() - cent_vals.min() + 1e-8)

    combined = 0.6 * rms_norm + 0.4 * cent_norm
    deriv = np.abs(np.diff(combined))

    threshold = np.mean(deriv) + sensitivity * np.std(deriv)
    min_gap = int(min_section / 0.5)

    # Find change points above threshold with minimum gap
    change_indices = []
    for i, d in enumerate(deriv):
        if d > threshold:
            if not change_indices or (i - change_indices[-1]) >= min_gap:
                change_indices.append(i)

    # Build section boundaries
    boundaries = [0.0]
    for idx in change_indices:
        boundaries.append(times[idx])
    boundaries.append(duration)

    sections = []
    for i in range(len(boundaries) - 1):
        sections.append((boundaries[i], boundaries[i + 1]))

    return sections


def prepare_audio_for_model(path: str) -> np.ndarray:
    """Load and resample audio to 16kHz mono for the model."""
    import subprocess
    import tempfile

    # soundfile can't read MP3/M4A — convert via ffmpeg first
    ext = os.path.splitext(path)[1].lower()
    if ext in (".mp3", ".m4a", ".aac", ".ogg"):
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_wav.close()
        try:
            subprocess.run(["ffmpeg", "-y", "-i", path, "-ar", "16000", "-ac", "1",
                            tmp_wav.name], capture_output=True, check=True)
            data, sr = sf.read(tmp_wav.name, dtype="float32")
            return data  # already 16kHz mono
        finally:
            os.unlink(tmp_wav.name)

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    # Mix to mono (soundfile returns samples x channels)
    waveform = np.mean(data, axis=1)
    # Resample to 16kHz if needed
    if sr != 16000:
        import torchaudio.transforms as T
        tensor = torch.from_numpy(waveform).unsqueeze(0)
        resampler = T.Resample(sr, 16000)
        waveform = resampler(tensor)[0].numpy()
    return waveform


def ask_model(model, processor, audio_array, prompt):
    """Send audio + text prompt to the model and get a text response."""
    conversation = [
        {"role": "system", "content": "You are a professional music analyst. Be detailed and specific."},
        {"role": "user", "content": [
            {"type": "audio", "audio_url": "audio.wav"},
            {"type": "text", "text": prompt},
        ]},
    ]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=[audio_array], sampling_rate=16000,
                       return_tensors="pt", padding=True)
    inputs = inputs.to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512)

    # Decode only the new tokens (skip input tokens)
    input_len = inputs["input_ids"].shape[-1]
    response = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
    return response.strip()


def main():
    parser = argparse.ArgumentParser(description="Analyze audio using AI to generate natural language descriptions.")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("--sections", type=int, default=0, metavar="N",
                        help="Split into N equal sections instead of auto-detecting natural boundaries")
    parser.add_argument("--output", default=None, help="Save analysis to a text file")
    args = parser.parse_args()

    output = args.output
    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    start_time = time.time()

    console.print()
    console.print("[bold cyan]🔍 analyze-audio[/bold cyan]")
    console.print()

    params = Table(show_header=False, show_edge=False, pad_edge=False, padding=(0, 2))
    params.add_column(style="bold")
    params.add_column()
    params.add_row("Input", f"[cyan]{args.input}[/cyan]")
    if output:
        params.add_row("Output", f"[cyan]{output}[/cyan]")
    console.print(params)
    console.print()

    # Prepare audio (16kHz mono for model) — also handles common compressed formats.
    console.print("  Loading audio...")
    audio_16k = prepare_audio_for_model(args.input)
    duration = len(audio_16k) / 16000

    # Get file info for display without loading uncompressed inputs twice.
    ext = os.path.splitext(args.input)[1].lower()
    if ext in (".mp3", ".m4a", ".aac", ".ogg"):
        is_stereo = False  # converted to mono
        sample_rate = 16000
    else:
        info = sf.info(args.input)
        is_stereo = info.channels > 1
        sample_rate = info.samplerate
        duration = info.frames / info.samplerate
    console.print(f"  Loaded [cyan]{duration:.1f}s[/cyan] at [cyan]{sample_rate}Hz[/cyan]"
                  f" ({'stereo' if is_stereo else 'mono'})")

    # Load model
    console.print("  Loading Qwen2-Audio model...")
    t0 = time.time()
    model, processor = run_with_hf_fallback(load_qwen2_audio)
    load_time = time.time() - t0
    console.print(f"  Model loaded in [cyan]{load_time:.1f}s[/cyan] on [cyan]{model.device}[/cyan]")

    # Overall description
    console.print()
    console.print("  [bold]Generating description...[/bold]")
    description = ask_model(model, processor, audio_16k,
                            "Describe this music track in a detailed paragraph. Include the genre, mood, "
                            "instrumentation, tempo, key characteristics, and how the piece evolves over time. "
                            "Mention any vocals, spoken words, singing, chanting, or human voice if present. "
                            "Write as if describing it to someone who hasn't heard it.")

    console.print()
    console.print("  [bold dim]Description[/bold dim]")
    console.print("  [dim]────────────────[/dim]")
    console.print()
    # Word-wrap the description
    for line in description.split("\n"):
        console.print(f"  {line}")

    # Section-by-section analysis — detect natural boundaries
    if args.sections > 0:
        # Fixed sections if explicitly requested
        sec_dur = duration / args.sections
        detected_sections = [(i * sec_dur, min((i + 1) * sec_dur, duration))
                             for i in range(args.sections)]
        console.print(f"\n  [bold]Analyzing {args.sections} fixed sections...[/bold]")
    else:
        # Dynamic detection from energy + spectral changes
        console.print("\n  [bold]Detecting section boundaries...[/bold]")
        detected_sections = detect_sections(audio_16k, 16000)
        console.print(f"  Found [cyan]{len(detected_sections)}[/cyan] natural sections")

    sections = []
    for i, (start_sec, end_sec) in enumerate(detected_sections):
        start_sample = int(start_sec * 16000)
        end_sample = int(end_sec * 16000)
        section_audio = audio_16k[start_sample:end_sample]

        if len(section_audio) < 1600:  # less than 0.1s
            continue

        section_desc = ask_model(model, processor, section_audio,
                                 "In one concise sentence, describe what is happening in this audio segment. "
                                 "Mention instruments, dynamics, mood changes, vocals, speech, singing, "
                                 "chanting, or any human voice. Note any notable or unusual events.")
        sections.append((start_sec, end_sec, section_desc))
        console.print(f"  Section {i + 1}/{len(detected_sections)} done")

    # Display sections table
    console.print()
    console.print("  [bold dim]Timeline[/bold dim]")
    console.print("  [dim]────────────────[/dim]")
    console.print()

    stbl = Table(show_edge=False, pad_edge=False, padding=(0, 2))
    stbl.add_column("Time", style="bold cyan", justify="right")
    stbl.add_column("Description")

    for start_sec, end_sec, desc in sections:
        time_label = f"{start_sec:.1f}s – {end_sec:.1f}s"
        stbl.add_row(time_label, desc)

    console.print(stbl)

    # Save to file if requested
    if output:
        with open(output, "w") as f:
            f.write(f"# Audio Analysis: {os.path.basename(args.input)}\n\n")
            f.write(f"Duration: {duration:.1f}s | Sample Rate: {sample_rate}Hz | "
                    f"{'Stereo' if is_stereo else 'Mono'}\n\n")
            f.write("## Description\n\n")
            f.write(description + "\n\n")
            f.write("## Timeline\n\n")
            for start_sec, end_sec, desc in sections:
                f.write(f"- **{start_sec:.1f}s – {end_sec:.1f}s**: {desc}\n")
            f.write("\n")
        console.print()
        console.print(f"  Saved to [cyan]{os.path.abspath(output)}[/cyan]")

    total_time = time.time() - start_time
    console.print()
    console.print(f"  Analysis completed in [cyan]{total_time:.1f}s[/cyan]")
    console.print()
    console.print("[bold green]✓ Done[/bold green]")
    console.print()


if __name__ == "__main__":
    main()
