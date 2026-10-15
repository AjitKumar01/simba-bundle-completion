"""Machine-independent device selector: cuda > mps > cpu. Override with FORCE_DEVICE=cpu|mps|cuda."""
import os, torch

def pick_device() -> torch.device:
    forced = os.environ.get("FORCE_DEVICE", "").strip().lower()
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = pick_device()
