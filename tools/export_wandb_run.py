#!/usr/bin/env python3
"""Export a W&B run locally for offline debugging.

Examples:
  python tools/export_wandb_run.py \
    https://wandb.ai/entity/project/runs/run_id \
    --out wandb_exports/my_run \
    --download-files

  python tools/export_wandb_run.py entity/project/run_id --out wandb_exports/my_run
"""

from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import wandb


DEFAULT_PATTERNS = (
    "output.log",
    "requirements.txt",
    "config.yaml",
    "wandb-metadata.json",
    "wandb-summary.json",
    "media/videos/*",
    "media/images/*",
)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    try:
        return dict(value)
    except (TypeError, ValueError):
        return str(value)


def normalize_run_path(run: str) -> str:
    if run.startswith("http://") or run.startswith("https://"):
        parsed = urlparse(run)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 4 and parts[-2] == "runs":
            return f"{parts[-4]}/{parts[-3]}/{parts[-1]}"
        if len(parts) >= 3:
            return "/".join(parts[-3:])
        raise ValueError(f"Cannot parse W&B run URL: {run}")
    return run.strip("/")


def matches_any(name: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def safe_attr(obj, name: str):
    try:
        return getattr(obj, name)
    except AttributeError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", help="W&B run path entity/project/run_id or run URL")
    parser.add_argument("--out", default=None, help="Output directory")
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--download-files", action="store_true")
    parser.add_argument(
        "--file-pattern",
        action="append",
        default=[],
        help="Extra file glob to download, e.g. 'media/videos/*'",
    )
    args = parser.parse_args()

    run_path = normalize_run_path(args.run)
    out_dir = Path(args.out or Path("wandb_exports") / run_path.replace("/", "__"))
    out_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    run = api.run(run_path)

    print(f"Exporting history for {run_path} ...")
    history = list(run.scan_history(page_size=args.page_size))
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    with (out_dir / "summary.json").open("w") as f:
        json.dump(dict(run.summary), f, indent=2, default=_json_default)

    with (out_dir / "config.json").open("w") as f:
        json.dump(dict(run.config), f, indent=2, default=_json_default)

    metadata = {
        "name": safe_attr(run, "name"),
        "id": safe_attr(run, "id"),
        "path": safe_attr(run, "path"),
        "url": safe_attr(run, "url"),
        "state": safe_attr(run, "state"),
        "created_at": str(safe_attr(run, "created_at")),
        "updated_at": str(safe_attr(run, "updated_at")),
    }
    with (out_dir / "run_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2, default=_json_default)

    if args.download_files:
        patterns = (*DEFAULT_PATTERNS, *tuple(args.file_pattern))
        files_dir = out_dir / "files"
        files_dir.mkdir(exist_ok=True)
        print("Downloading selected run files ...")
        for run_file in run.files():
            if matches_any(run_file.name, patterns):
                target = files_dir / run_file.name
                target.parent.mkdir(parents=True, exist_ok=True)
                run_file.download(root=files_dir, replace=True)

    print(f"Done. Exported to: {out_dir}")
    print("Most useful files for analysis: history.csv, summary.json, config.json, files/output.log")


if __name__ == "__main__":
    main()
