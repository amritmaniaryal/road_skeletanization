"""Corruption model for realistic road-mask noise.

These functions simulate what a real road-extraction model or noisy source
would produce, so a skeletonizer has to work on imperfect inputs rather than
perfectly clean blobs. Everything is a pure function on a binary numpy mask
(H, W) with values in {0, 1}.

    corrupt(mask, cfg, seed)

applies a randomized subset of corruptors at a given intensity.
"""

import random

import numpy as np
from scipy import ndimage


def _sample_foreground(mask, rng):
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    i = rng.randrange(len(ys))
    return ys[i], xs[i]


def add_gaps(mask, rng, count=None, gap_radius=3):
    """Punch small holes along the road network (tree/car/cloud occlusion)."""
    out = mask.copy()
    if count is None:
        count = int(round(mask.sum() * 0.002))
    for _ in range(count):
        center = _sample_foreground(out, rng)
        if center is None:
            break
        r = rng.randint(1, gap_radius)
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        disc = (yy ** 2 + xx ** 2) <= r ** 2
        y0, x0 = center
        y_slice = slice(max(0, y0 - r), y0 + r + 1)
        x_slice = slice(max(0, x0 - r), x0 + r + 1)
        patch = disc[
            max(0, r - y0): disc.shape[0] - max(0, (y0 + r + 1) - mask.shape[0]),
            max(0, r - x0): disc.shape[1] - max(0, (x0 + r + 1) - mask.shape[1]),
        ]
        out[y_slice, x_slice] = np.logical_and(
            out[y_slice, x_slice], np.logical_not(patch)
        )
    return out


def jitter_edges(mask, rng, radius=1):
    """Roughen boundaries with random dilation/erosion (imperfect segmenter)."""
    out = mask
    n = rng.randint(1, 2)
    for _ in range(n):
        r = rng.randint(1, radius)
        if rng.random() < 0.5:
            out = ndimage.binary_erosion(out, iterations=r)
        else:
            out = ndimage.binary_dilation(out, iterations=r)
    return out


def spurious_blobs(mask, rng, count=None, max_radius=8):
    """Add random disconnected false-positive blobs."""
    out = mask.copy()
    if count is None:
        count = int(round(mask.sum() * 0.001)) + 1
    h, w = out.shape
    for _ in range(count):
        r = rng.randint(1, max_radius)
        y0, x0 = rng.randint(0, h - 1), rng.randint(0, w - 1)
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        disc = (yy ** 2 + xx ** 2) <= r ** 2
        y_slice = slice(max(0, y0 - r), y0 + r + 1)
        x_slice = slice(max(0, x0 - r), x0 + r + 1)
        patch = disc[
            max(0, r - y0): disc.shape[0] - max(0, (y0 + r + 1) - h),
            max(0, r - x0): disc.shape[1] - max(0, (x0 + r + 1) - w),
        ]
        out[y_slice, x_slice] = np.logical_or(out[y_slice, x_slice], patch)
    return out


def partial_occlusion(mask, rng, count=None, block=16):
    """Erase random square patches (shadows, buildings, clouds)."""
    out = mask.copy()
    if count is None:
        count = int(round(mask.sum() / (block ** 2) * 0.05)) + 1
    h, w = out.shape
    for _ in range(count):
        y0 = rng.randint(0, max(1, h - block))
        x0 = rng.randint(0, max(1, w - block))
        out[y0:y0 + block, x0:x0 + block] = 0
    return out


def width_variation(mask, rng, radius=2):
    """Local thinning/thickening to mimic inconsistent detector widths."""
    return jitter_edges(mask, rng, radius=radius)


def corrupt(mask, cfg, seed):
    """Apply a randomized set of corruptors to a binary mask.

    cfg: dict with
        enabled:   list of corruptor names to potentially apply
                   (default: all of them)
        intensity: float 0.0 (clean) .. 1.0 (heavy). Scales how often
                   corruptors trigger and how aggressive they are.
    seed: random seed for reproducibility.
    Returns the corrupted binary mask.
    """
    intensity = cfg.get("intensity", 0.0)
    if intensity <= 0.0:
        return np.asarray(mask, dtype=np.uint8).copy()

    rng = random.Random(seed)
    enabled = cfg.get(
        "enabled",
        ["add_gaps", "jitter_edges", "spurious_blobs",
         "partial_occlusion", "width_variation"],
    )

    out = mask.copy().astype(bool)
    for name in enabled:
        if rng.random() > intensity * 0.8:
            continue
        fn = globals()[name]
        kwargs = {}
        if name in ("add_gaps", "spurious_blobs", "partial_occlusion"):
            base = {"add_gaps": 0.0015, "spurious_blobs": 0.0008,
                    "partial_occlusion": 0.03}
            count = max(1, int(round(out.sum() * base[name] * intensity)))
            kwargs["count"] = count
        if name == "jitter_edges":
            kwargs["radius"] = max(1, int(round(1.2 * intensity)))
        if name == "width_variation":
            kwargs["radius"] = max(1, int(round(1.2 * intensity)))
        out = fn(out, rng, **kwargs)

    return out.astype(np.uint8)
