"""Build a compact, lossless cache of the CubeLearn HAR dataset from HAR_data.zip.

The released samples are (2, 20, 128, 12, 256) float64 = 125.8 MB each, 181.2 GB
unpacked. Two observations shrink that ~6x with no loss of information:

  * Antennas 8-11 are never read by any of the 10 released models, so we keep 0:7.
  * The values are integer ADC counts stored as float64. int16 holds them exactly.

Result is (2, 20, 128, 8, 256) int16 = 20.97 MB per sample, ~30.2 GB total. The
integrality assumption is asserted per file, so a violation fails loudly rather
than silently truncating.

Resumable: existing outputs of the right size are skipped.
"""
import argparse
import io
import os
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np

N_ANTENNAS = 8          # models only ever use the 8-wide row
EXPECTED_SHAPE = (2, 20, 128, 12, 256)
INT16_MIN, INT16_MAX = -32768, 32767


def convert(task):
    zip_path, name, out_path = task
    expected_bytes = 2 * 20 * 128 * N_ANTENNAS * 256 * 2  # int16
    if os.path.exists(out_path) and os.path.getsize(out_path) >= expected_bytes:
        return name, 0, "skipped"

    raw = zipfile.ZipFile(zip_path).read(name)
    a = np.load(io.BytesIO(raw))
    if a.shape != EXPECTED_SHAPE:
        return name, 0, f"FAIL unexpected shape {a.shape}"

    lo, hi = a.min(), a.max()
    if lo < INT16_MIN or hi > INT16_MAX:
        return name, 0, f"FAIL out of int16 range [{lo}, {hi}]"
    if not np.array_equal(a, np.rint(a)):
        return name, 0, "FAIL values are not integral; int16 would lose data"

    out = np.ascontiguousarray(a[:, :, :, :N_ANTENNAS, :]).astype(np.int16)
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as fh:        # file handle: np.save would append ".npy"
        np.save(fh, out)
    os.replace(tmp, out_path)          # atomic, so a killed job stays resumable
    return name, out.nbytes, "ok"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", default=os.environ.get("CUBELEARN_ARCHIVE", "data/HAR_data.zip"))
    p.add_argument("--out", default=os.environ.get("CUBELEARN_CACHE", "data/har_cache"))
    p.add_argument("--workers", type=int, default=32)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    names = sorted(n for n in zipfile.ZipFile(args.zip).namelist() if n.endswith(".npy"))
    print(f"{len(names)} members in {args.zip}", flush=True)

    tasks = [(args.zip, n, os.path.join(args.out, os.path.basename(n))) for n in names]

    t0 = time.time()
    done = skipped = failed = 0
    total_bytes = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for name, nbytes, status in ex.map(convert, tasks):
            if status == "ok":
                done += 1
                total_bytes += nbytes
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                print(f"  {name}: {status}", file=sys.stderr, flush=True)
            n = done + skipped + failed
            if n % 100 == 0 or n == len(tasks):
                el = time.time() - t0
                print(f"  {n}/{len(tasks)}  {el:6.1f}s  "
                      f"({n / el:.1f}/s, eta {(len(tasks) - n) / max(n / el, 1e-9):5.0f}s)",
                      flush=True)

    print(f"\nconverted {done}, skipped {skipped}, failed {failed} "
          f"in {time.time() - t0:.1f}s -> {total_bytes / 1e9:.1f} GB written")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
