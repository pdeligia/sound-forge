"""Rich console utilities for SoundForge tools."""

import os
import sys

from rich.console import Console

_force_terminal = sys.stdout.isatty()
console = Console(force_terminal=_force_terminal)


def enable_hf_offline():
    """Enable Hugging Face offline mode."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        import huggingface_hub.constants as hf_constants
        hf_constants.HF_HUB_OFFLINE = True
    except (ImportError, AttributeError):
        pass


def disable_hf_offline():
    """Disable Hugging Face offline mode."""
    os.environ.pop("HF_HUB_OFFLINE", None)
    try:
        import huggingface_hub.constants as hf_constants
        hf_constants.HF_HUB_OFFLINE = False
    except (ImportError, AttributeError):
        pass


def run_with_hf_fallback(fn, *args, **kwargs):
    """Run a function that loads HF models, retrying with download on cache miss."""
    enable_hf_offline()
    try:
        return fn(*args, **kwargs)
    except (OSError, ValueError):
        console.print("  [dim]Model not cached, downloading...[/dim]")
        disable_hf_offline()
        return fn(*args, **kwargs)
