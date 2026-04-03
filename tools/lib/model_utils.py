"""Model loading utilities for SoundForge tools."""

import torch


def get_device() -> str:
    """Return the best available device (mps for Apple Silicon, cuda, or cpu)."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_musicgen(model_size: str = "small"):
    """Load a MusicGen model via transformers.

    Args:
        model_size: One of "small" (300M), "medium" (1.5B), or "large" (3.3B).

    Returns:
        A tuple of (model, processor) ready for generation.
    """
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    model_name = f"facebook/musicgen-{model_size}"
    processor = AutoProcessor.from_pretrained(model_name)
    model = MusicgenForConditionalGeneration.from_pretrained(model_name)
    model = model.to(get_device())
    return model, processor


def load_qwen2_audio():
    """Load Qwen2-Audio-7B-Instruct for audio understanding.

    Returns:
        A tuple of (model, processor) ready for audio analysis.
        Audio input must be 16kHz mono. Outputs text descriptions.
    """
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

    model_name = "Qwen/Qwen2-Audio-7B-Instruct"
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, trust_remote_code=True
    )
    model = model.to(get_device())
    return model, processor


def load_stable_audio(model_name: str = "stabilityai/stable-audio-open-1.0"):
    """Load a Stable Audio Open pipeline via diffusers for sound effect generation.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        The StableAudioPipeline ready for generation.
    """
    from diffusers import StableAudioPipeline, DPMSolverMultistepScheduler

    device = get_device()
    dtype = torch.float32 if device == "cpu" else torch.float16
    pipe = StableAudioPipeline.from_pretrained(model_name, torch_dtype=dtype)

    # The default CosineDPMSolverMultistepScheduler uses torchsde for SDE sampling,
    # which hits a RecursionError on MPS (Apple Silicon). Swap to a non-SDE scheduler.
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    pipe = pipe.to(device)
    return pipe
