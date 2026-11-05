"""Expose the repository-root modules under their VQA package name."""
from pathlib import Path

__path__.append(str(Path(__file__).resolve().parents[2]))
