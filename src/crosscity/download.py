from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path


# Canonical Google Drive files referenced by the official DCRNN repository.
DATASETS = {
    "metr_la": {
        "url": "https://drive.google.com/uc?export=download&id=1pAGRfzMx6K9WWsfDcD1NMbIif0T0saFC",
        "archive": "METR-LA.zip",
    },
    "pems_bay": {
        "url": "https://drive.google.com/uc?export=download&id=1wD-mHlqAb2mtHOe_68fZvDh1K6P3GQ7o",
        "archive": "PEMS-BAY.zip",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(city: str, destination: str | Path = "data/raw") -> dict[str, str]:
    """Download then unpack a dataset. Never silently overwrites an existing archive."""
    if city not in DATASETS:
        raise ValueError(f"unknown city {city!r}")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    spec = DATASETS[city]
    archive = destination / spec["archive"]
    if not archive.exists():
        urllib.request.urlretrieve(spec["url"], archive)
    shutil.unpack_archive(archive, destination)
    checksum = sha256(archive)
    (archive.with_suffix(archive.suffix + ".sha256")).write_text(checksum + "\n", encoding="utf-8")
    return {"archive": str(archive), "sha256": checksum}

