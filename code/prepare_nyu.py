"""NYU-Depth V2 (Eigen test split, 654 images) preparation.

Input : data/nyu/val-*.tar  (webdataset shards, each sample = val/official/XXXXX.h5)
Output: data/nyu_eigen/images/XXXXX.png  (RGB, 640x480)
        data/nyu_eigen/depths/XXXXX.npy  (float32 metres, 640x480)
        data/nyu_eigen/index.txt
"""
import io
import os
import tarfile

import cv2
import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAR_DIR = os.path.join(ROOT, "data", "nyu")
OUT_DIR = os.path.join(ROOT, "data", "nyu_eigen")
IMG_DIR = os.path.join(OUT_DIR, "images")
DEP_DIR = os.path.join(OUT_DIR, "depths")


def probe(path):
    """Print h5 structure of the first sample so the format can be verified."""
    with tarfile.open(path, "r|") as t:
        for m in t:
            if not m.name.endswith(".h5"):
                continue
            f = t.extractfile(m)
            with h5py.File(io.BytesIO(f.read()), "r") as h:
                print("sample:", m.name)
                for k in h.keys():
                    d = h[k]
                    print("  ", k, d.shape, d.dtype,
                          "min=%.4f max=%.4f" % (float(np.min(d[:])), float(np.max(d[:]))))
            return


def convert(tar_path, index_lines):
    with tarfile.open(tar_path, "r|") as t:
        for m in t:
            if not m.name.endswith(".h5"):
                continue
            key = os.path.splitext(os.path.basename(m.name))[0]
            raw = t.extractfile(m).read()
            with h5py.File(io.BytesIO(raw), "r") as h:
                rgb = np.asarray(h["rgb"])
                depth = np.asarray(h["depth"], dtype=np.float32)
            if rgb.shape[0] == 3:  # official h5 stores CHW
                rgb = np.transpose(rgb, (1, 2, 0))
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if not ok:
                raise RuntimeError("encode failed: " + key)
            buf.tofile(os.path.join(IMG_DIR, key + ".png"))
            np.save(os.path.join(DEP_DIR, key + ".npy"), depth)
            index_lines.append(key)


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(DEP_DIR, exist_ok=True)
    tars = sorted(
        os.path.join(TAR_DIR, f) for f in os.listdir(TAR_DIR) if f.endswith(".tar")
    )
    print("shards:", tars)
    probe(tars[0])
    index = []
    for tp in tars:
        convert(tp, index)
        print("done", os.path.basename(tp), "total", len(index))
    with open(os.path.join(OUT_DIR, "index.txt"), "w", encoding="utf8") as f:
        f.write("\n".join(index))
    print("TOTAL", len(index))


if __name__ == "__main__":
    main()
