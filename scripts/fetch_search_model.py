"""Downloads the search model into the Docker image at build time (see the Dockerfile).

nomic-embed-text v1.5 (Apache-2.0) in its quantised ONNX form. The revision is pinned and each
file is checked against its SHA-256, so every build gets exactly the same model.

Usage: python fetch_search_model.py <target folder>
"""

import hashlib
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "nomic-ai/nomic-embed-text-v1.5"
REVISION = "e9b6763023c676ca8431644204f50c2b100d9aab"
FILES = {
    "onnx/model_quantized.onnx": "b4342336debaea79de872370664b0aaeb67dea4605513d00ee236ea871a81f27",
    "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(target: Path) -> None:
    for name, expected in FILES.items():
        path = Path(hf_hub_download(REPO, name, revision=REVISION, local_dir=target))
        actual = sha256(path)
        if actual != expected:
            sys.exit(f"{name}: SHA-256 is {actual}, expected {expected}")
        print(f"{name}: {path.stat().st_size // 1_000_000} MB, checksum ok")
    shutil.rmtree(target / ".cache", ignore_errors=True)  # download bookkeeping, not needed at runtime


if __name__ == "__main__":
    main(Path(sys.argv[1]))
