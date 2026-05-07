"""Download bitmind/celeb-a-hq parquet shards and decode to 30,000 PNGs.

Output: ${DATA_ROOT}/original_data/celeba/{00000..29999}.png
"""
import io
import os
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

REPO_ID = "bitmind/celeb-a-hq"
NUM_SHARDS = 6
OUT_DIR = Path("${DATA_ROOT}/original_data/celeba")
CACHE_DIR = Path("${DATA_ROOT}/original_data/_hf_cache")

OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_image_column(schema):
    for name in schema.names:
        f = schema.field(name)
        t = f.type
        if name.lower() in {"image", "img", "picture"}:
            return name
        if str(t) in {"binary", "large_binary"}:
            return name
        # HF Image feature serializes as struct<bytes: binary, path: string>
        if hasattr(t, "num_fields") and any(
            t.field(i).name == "bytes" for i in range(t.num_fields)
        ):
            return name
    return None


def extract_bytes(cell):
    if isinstance(cell, dict):
        b = cell.get("bytes")
        if b is not None:
            return b
        p = cell.get("path")
        if p:
            return Path(p).read_bytes()
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    raise TypeError(f"unsupported image cell type: {type(cell)}")


def main():
    log(f"target: {OUT_DIR}")
    log(f"cache:  {CACHE_DIR}")

    shard_paths = []
    for i in range(NUM_SHARDS):
        fname = f"data/train-{i:05d}-of-{NUM_SHARDS:05d}.parquet"
        log(f"downloading {fname}")
        p = hf_hub_download(
            repo_id=REPO_ID,
            filename=fname,
            repo_type="dataset",
            cache_dir=str(CACHE_DIR),
        )
        size_mb = os.path.getsize(p) / 1e6
        log(f"  -> {p} ({size_mb:.1f} MB)")
        shard_paths.append(p)

    idx = 0
    for shard_path in shard_paths:
        log(f"decoding {shard_path}")
        pf = pq.ParquetFile(shard_path)
        col_name = find_image_column(pf.schema_arrow)
        if col_name is None:
            raise RuntimeError(f"no image column in {pf.schema_arrow}")
        log(f"  image column = {col_name!r}, num_rows = {pf.metadata.num_rows}")

        for batch in pf.iter_batches(batch_size=64, columns=[col_name]):
            col = batch.column(0).to_pylist()
            for cell in col:
                raw = extract_bytes(cell)
                img = Image.open(io.BytesIO(raw))
                if img.mode != "RGB":
                    img = img.convert("RGB")
                if img.size != (1024, 1024):
                    raise RuntimeError(
                        f"unexpected size {img.size} at idx {idx}"
                    )
                out = OUT_DIR / f"{idx:05d}.png"
                img.save(out, format="PNG", optimize=False, compress_level=1)
                idx += 1
                if idx % 500 == 0:
                    log(f"  saved {idx} images")

    log(f"done. total saved: {idx}")
    if idx != 30000:
        log(f"WARNING: expected 30000 images, got {idx}")
        sys.exit(2)


if __name__ == "__main__":
    main()
