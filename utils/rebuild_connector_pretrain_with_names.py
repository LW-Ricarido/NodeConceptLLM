"""Safely rebuild all_connector_pretrain with dataset provenance.

The output is written to a sibling temporary directory and renamed only after
all component datasets have been concatenated and saved successfully. The old
combined dataset is never modified or overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

import pyarrow as pa
from datasets import Dataset, concatenate_datasets, load_from_disk


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source_dir", required=True, help="Directory whose children are source datasets")
    parser.add_argument("--output_dir", required=True, help="New combined load_from_disk directory")
    parser.add_argument(
        "--strip_suffix",
        default="_pretrain",
        help="Strip this suffix from child directory names; pass an empty string to preserve exact names",
    )
    return parser.parse_args()


def dataset_name(path: Path, suffix: str) -> str:
    if suffix and path.name.endswith(suffix):
        return path.name[: -len(suffix)]
    return path.name


def rebuild(source_dir: Path, output_dir: Path, strip_suffix: str) -> None:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite {output_dir}. Choose a new --output_dir, verify it, then update training commands."
        )
    if not source_dir.is_dir():
        raise NotADirectoryError(source_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f".{output_dir.name}.building-{uuid.uuid4().hex}"
    if temporary.exists():
        raise FileExistsError(temporary)

    components = []
    manifest = []
    for child in sorted(source_dir.iterdir()):
        if not child.is_dir() or not (child / "state.json").exists():
            continue
        current = load_from_disk(str(child))
        if not isinstance(current, Dataset):
            raise TypeError(f"Expected Dataset at {child}, got {type(current).__name__}")
        name = dataset_name(child, strip_suffix)
        if not name:
            raise ValueError(f"Empty dataset name derived from {child.name!r}")
        if "dataset_name" in current.column_names:
            current = current.remove_columns("dataset_name")
        # Existing Arrow columns stay zero-copy; only this small provenance column
        # is allocated before the final combined dataset is written once.
        current = current.add_column("dataset_name", pa.repeat(pa.scalar(name), len(current)))
        components.append(current)
        manifest.append({"source": str(child), "dataset_name": name, "rows": len(current)})

    if not components:
        raise ValueError(f"No load_from_disk Dataset children found below {source_dir}")
    combined = concatenate_datasets(components)
    try:
        combined.save_to_disk(str(temporary))
        with open(temporary / "provenance_manifest.json", "w", encoding="utf-8") as handle:
            json.dump(
                {"source_dir": str(source_dir), "total_rows": len(combined), "components": manifest},
                handle,
                indent=2,
                sort_keys=True,
            )
        # Sibling rename is atomic on the same filesystem. output_dir was checked
        # absent above, so the original combined dataset cannot be replaced.
        os.rename(temporary, output_dir)
    except BaseException:
        print(f"Build did not complete. Partial data was left at {temporary} for inspection/removal.")
        raise
    print(f"Saved {len(combined)} rows with dataset_name provenance to {output_dir}")


if __name__ == "__main__":
    arguments = parse_args()
    rebuild(Path(arguments.source_dir), Path(arguments.output_dir), arguments.strip_suffix)
