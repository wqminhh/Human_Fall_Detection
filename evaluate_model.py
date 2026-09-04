#!/usr/bin/env python3
"""Compatibility wrapper for the nested Human-Fall-Detection project."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent
REPO_DIR = WORKSPACE_ROOT / "Human-Fall-Detection-master"
SCRIPT = REPO_DIR / "evaluate_model.py"

if not SCRIPT.exists():
    SCRIPT = WORKSPACE_ROOT / "evaluate_model.py"
    REPO_DIR = WORKSPACE_ROOT

if not SCRIPT.exists():
    raise FileNotFoundError(
        f"Project script not found under {WORKSPACE_ROOT}. "
        "Make sure the repository is checked out in the workspace root or in a nested folder."
    )

normalized_args: list[str] = []
for arg in sys.argv[1:]:
    candidate = Path(arg)
    if candidate.is_absolute():
        normalized_args.append(arg)
        continue

    if candidate.exists() and not arg.startswith("-"):
        normalized_args.append(str(candidate.resolve()))
        continue

    root_candidate = (WORKSPACE_ROOT / candidate).resolve()
    if root_candidate.exists():
        normalized_args.append(str(root_candidate))
        continue

    repo_candidate = (REPO_DIR / candidate).resolve()
    if repo_candidate.exists():
        normalized_args.append(str(repo_candidate))
        continue

    normalized_args.append(arg)

os.chdir(REPO_DIR)
raise SystemExit(subprocess.call([sys.executable, str(SCRIPT), *normalized_args]))
