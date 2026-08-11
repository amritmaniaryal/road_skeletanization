"""Preprocess the Massachusetts Roads validation set into our RoadDataset format.

The Mnih Massachusetts Roads dataset ships 1500x1500 binary road masks (real
road networks) plus a vector shapefile of the road centerlines used to build
them. We:

  1. read each mask's georeferencing from its paired satellite TIFF (EPSG:26986,
     1 m/pixel — the masks align 1:1 with the sat images),
  2. tile the 1500x1500 mask into 256x256 patches (stride 256, keeping patches
     with road content),
  3. rasterize the shapefile centerlines clipped to each patch into a 1px
     skeleton ground truth,
  4. write image_*/target_* PNGs that RoadDataset already understands.

Output: data/mass_roads/patches/
Usage:
    python -m src.make_mass_data
"""

import glob
import os

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.coords import BoundingBox
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from tqdm import tqdm

SAT_DIR = "data/mass_roads/valid/sat"
MAP_DIR = "data/mass_roads/valid/map"
SHAPE = "data/mass_roads/shape/massachusetts-roads_corrected.shp"
OUT_DIR = "data/mass_roads/patches"

PATCH = 256
STRIDE = 256
MIN_ROAD_PX = 50  # skip patches with almost no road


def clip_centerlines(gdf, bounds, transform, shape=(PATCH, PATCH)):
    """Rasterize shapefile centerlines within bounds at 1px in image frame."""
    clip = gdf.cx[bounds.left:bounds.right, bounds.bottom:bounds.top]
    geoms = [(g, 1) for g in clip.geometry if g is not None and not g.is_empty]
    if not geoms:
        return np.zeros(shape, dtype=np.uint8)
    return rasterize(geoms, out_shape=shape, transform=transform, fill=0, dtype=np.uint8)


def main():
    gdf = gpd.read_file(SHAPE)
    os.makedirs(OUT_DIR, exist_ok=True)

    n_written = 0
    for sat_fp in tqdm(sorted(glob.glob(os.path.join(SAT_DIR, "*.tiff")))):
        stem = os.path.basename(sat_fp).replace(".tiff", "")
        map_fp = os.path.join(MAP_DIR, stem + ".tif")
        if not os.path.exists(map_fp):
            continue

        with rasterio.open(sat_fp) as ds:
            bounds = ds.bounds
            size = ds.width
        mask = np.array(Image.open(map_fp)) > 0

        for y0 in range(0, size - PATCH + 1, STRIDE):
            for x0 in range(0, size - PATCH + 1, STRIDE):
                patch = mask[y0:y0 + PATCH, x0:x0 + PATCH]
                if patch.sum() < MIN_ROAD_PX:
                    continue

                # pixel (0,0) -> top-left geographic corner
                x_lo = bounds.left + x0
                y_hi = bounds.top - y0
                bb = BoundingBox(left=x_lo, bottom=y_hi - PATCH,
                                 right=x_lo + PATCH, top=y_hi)
                transform = from_bounds(bb.left, bb.bottom, bb.right, bb.top, PATCH, PATCH)
                sk = clip_centerlines(gdf, bb, transform)

                img_fp = os.path.join(OUT_DIR, f"image_{n_written:05d}.png")
                tgt_fp = os.path.join(OUT_DIR, f"target_{n_written:05d}.png")
                Image.fromarray((patch * 255).astype(np.uint8)).save(img_fp)
                Image.fromarray((sk > 0).astype(np.uint8) * 255).save(tgt_fp)
                n_written += 1

    print(f"Wrote {n_written} patches to {OUT_DIR}/")


if __name__ == "__main__":
    main()
