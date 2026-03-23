"""AudioCraft model loading utilities for SoundForge tools (via HuggingFace transformers)."""

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


def load_audiogen(model_size: str = "medium"):
    """Load an AudioGen model via transformers.

    Args:
        model_size: Model size (currently only "medium" is available).

    Returns:
        A tuple of (model, processor) ready for generation.
    """
    from transformers import AutoProcessor, AutoModelForTextToWaveform

    model_name = f"facebook/audiogen-{model_size}"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForTextToWaveform.from_pretrained(model_name)
    model = model.to(get_device())
    return model, processor
