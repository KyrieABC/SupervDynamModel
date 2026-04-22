from __future__ import annotations

import json
from pathlib import Path

# A set of lightweight, helper functions for handling file I/O operations specifically for JSON data

# Reads a JSON file from disk and parses it into a Python dictionary
# It uses encoding="utf-8", which is a best practice to ensure compatibility across different operating systems (Windows, Linux, macOS) that might default to different text encodings
def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# path.parent.mkdir(parents=True, exist_ok=True) is the standout feature here. 
# If the directory leading to the file doesn't exist yet, it automatically creates the full path tree
def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        # The indent=2 argument ensures the resulting JSON file is "pretty-printed" (human-readable) rather than saved as a single, dense line
        json.dump(data, f, indent=2)

# This is a clean, reusable wrapper for creating directories
def ensure_dir(path: str | Path) -> None:
    # exist_ok=True: Prevents the program from crashing if the directory already exists
    Path(path).mkdir(parents=True, exist_ok=True)
