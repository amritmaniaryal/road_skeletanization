"""Classical, non-learned skeletonization baselines.

Each takes a binary mask (H, W) in {0, 1} and returns a binary 1px skeleton.
These are the deterministic algorithms our U-Net is compared against —
including the 'divide by 2' medial-axis baseline the user guessed would
match the network on clean data.
"""

import numpy as np
from skimage.morphology import medial_axis, skeletonize


def _thin(x):
    """Ensure the output is a strict 1px skeleton (some methods emit 2px)."""
    return skeletonize(x > 0).astype(np.uint8)


def medial_axis_baseline(mask):
    """Distance-transform-based medial axis = the generalized 'divide by 2'.

    Computes, for every foreground pixel, its distance to the nearest edge;
    the ridge of maximal distance is the shape's centerline.
    """
    return _thin(medial_axis(mask > 0))


def skeletonize_baseline(mask):
    """Classic morphological thinning: peel boundary pixels until 1px lines.

    Known to be brittle to noise — a single bump spawns spurious branches.
    """
    return _thin(skeletonize(mask > 0))


def distance_ridge(mask, threshold=0.5):
    """Explicit distance-transform ridge baseline.

    Distance to nearest edge, normalized; keep pixels whose distance is at
    least `threshold` of the local max, then thin.
    """
    from scipy import ndimage

    dist = ndimage.distance_transform_edt(mask > 0)
    if dist.max() == 0:
        return np.zeros_like(mask)
    ridge = dist >= (dist.max() * threshold)
    return _thin(ridge)


BASELINES = {
    "medial_axis": medial_axis_baseline,
    "skeletonize": skeletonize_baseline,
    "distance_ridge": distance_ridge,
}
