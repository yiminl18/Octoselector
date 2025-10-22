"""
Small utilities for sampling records from a JSON dataset.

- load_json(path): read JSON (UTF-8).
- select_random_datapoints(input_file, output_file, num_samples, seed_value):
  write a deterministic random subset to disk and return it.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, List, Union


def load_json(file_path: Union[str, Path]) -> Any:
    """Load JSON from a file path."""
    p = Path(file_path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def select_random_datapoints(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    num_samples: int,
    seed_value: int = 42,
) -> List[Any]:
    """Select `num_samples` unique items from top-level JSON array and save to `output_file`."""
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")

    data = load_json(input_file)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a top-level list/array")

    if num_samples > len(data):
        raise ValueError("num_samples exceeds dataset size")

    if seed_value is not None:
        random.seed(seed_value)

    samples = random.sample(data, num_samples)

    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    return samples