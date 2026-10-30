"""Make a serving dir: the FP8 checkpoint with mtp.* replaced by trained weights (re-quantized like the original).

Files without mtp.* tensors are hard-linked (copied if --src and --out are on different filesystems); shards that
hold mtp.* tensors are rewritten. An existing file at a rewritten path is unlinked first, so a hard link to the
original checkpoint is never written through.
"""
import argparse, errno, json, os, shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--mtp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--shards-only", action="store_true", help="only rewrite the mtp shards; the caller linked the rest")
args = ap.parse_args()


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError as e:
        if e.errno not in (errno.EXDEV, errno.EPERM, errno.EMLINK):
            raise
        shutil.copy2(src, dst)


idx = json.load(open(f"{args.src}/model.safetensors.index.json"))["weight_map"]
mtp_shards = {f for k, f in idx.items() if k.startswith("mtp.")}
with safe_open(args.mtp, "pt") as fh:
    new = {k: fh.get_tensor(k) for k in fh.keys()}
os.makedirs(args.out, exist_ok=True)
if not args.shards_only:
    for name in os.listdir(args.src):
        if name in mtp_shards or os.path.exists(f"{args.out}/{name}"):
            continue
        if os.path.isdir(f"{args.src}/{name}"):
            shutil.copytree(f"{args.src}/{name}", f"{args.out}/{name}", copy_function=link_or_copy)
        else:
            link_or_copy(f"{args.src}/{name}", f"{args.out}/{name}")

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
for shard in mtp_shards:
    out, replaced = {}, 0
    with safe_open(f"{args.src}/{shard}", "pt") as fh:
        keys = list(fh.keys())
        for k in keys:
            t = fh.get_tensor(k)
            if not k.startswith("mtp."):
                out[k] = t
            elif k.endswith(".weight_scale"):
                continue
            elif t.dtype == torch.float8_e4m3fn:
                w = new[k].float()
                scale = (w.abs().amax(-1, keepdim=True) / FP8_MAX).clamp(min=1e-12)
                out[k] = (w / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
                out[k + "_scale"] = scale.to(fh.get_slice(k + "_scale").get_dtype() == "BF16" and torch.bfloat16 or torch.float32)
                replaced += 1
            else:
                assert new[k].shape == t.shape, k
                out[k] = new[k].to(t.dtype)
                replaced += 1
    assert set(out) == set(keys), set(keys) ^ set(out)
    dst = f"{args.out}/{shard}"
    if os.path.lexists(dst):
        os.unlink(dst)
    save_file(out, dst, metadata={"format": "pt"})
    print(shard, "replaced", replaced, "of", sum(k.startswith("mtp.") and not k.endswith("_scale") for k in keys))
