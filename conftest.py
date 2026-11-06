"""Test bootstrap.

All runtime paths in SpriteMind (data/, models/) are project-relative, so the
suite must run from the repository root regardless of where pytest is invoked.
"""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)
