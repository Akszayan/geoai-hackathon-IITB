# scripts/visualize_overlay.py
"""
Create an overlay PNG of ortho + colored mask for quick visual checks.

Usage:
python -u scripts/visualize_overlay.py --ortho raw_data/orthos/PB/28996_NADALA_ORTHO/28996_NADALA_ORTHO.tif \
    --mask outputs/predictions/28996_NADALA_ORTHO_pred_mask.tif --out outputs/figs/overlay.png
"""
import argparse
from pathlib import Path
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ortho", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--bbox", nargs=4, type=int, default=None, help="x_min y_min x_max y_max to crop for quick view")
    return p.parse_args()

def color_map():
    # index -> RGBA (0..1)
    return {
        0: (0.0,0.0,0.0,0.0),    # transparent background
        1: (0.8,0.2,0.2,0.6),    # red-ish roof
        2: (0.2,0.8,0.2,0.6),    # green
        3: (0.9,0.6,0.1,0.6),    # orange
        4: (0.4,0.4,0.9,0.6),    # blue
        5: (0.2,0.6,0.9,0.6),    # cyan (water)
    }

def main():
    args = parse_args()
    ortho = rasterio.open(args.ortho)
    maskf = rasterio.open(args.mask)

    if args.bbox:
        x0,y0,x1,y1 = args.bbox
        w = x1 - x0; h = y1 - y0
        ortho_arr = np.moveaxis(ortho.read([1,2,3], window=rasterio.windows.Window(x0,y0,w,h)), 0, 2)
        mask_arr = maskf.read(1, window=rasterio.windows.Window(x0,y0,w,h))
    else:
        # sample a central crop to keep image size reasonable
        H,W = ortho.height, ortho.width
        cx = W//2; cy = H//2
        sz = min(2000, W, H)
        x0 = max(0, cx - sz//2); y0 = max(0, cy - sz//2)
        w = sz; h = sz
        ortho_arr = np.moveaxis(ortho.read([1,2,3], window=rasterio.windows.Window(x0,y0,w,h)), 0, 2)
        mask_arr = maskf.read(1, window=rasterio.windows.Window(x0,y0,w,h))

    cmap = color_map()
    overlay = np.zeros((mask_arr.shape[0], mask_arr.shape[1], 4), dtype=float)
    for cls, col in cmap.items():
        maskc = (mask_arr == cls)
        overlay[maskc] = col

    fig, ax = plt.subplots(1,1, figsize=(12,12))
    ax.imshow(np.clip(ortho_arr/255.0, 0,1))
    ax.imshow(overlay)
    ax.axis('off')

    # legend
    patches = [mpatches.Patch(color=cmap[k][:3], label=f"class {k}") for k in sorted(cmap.keys()) if k!=0]
    ax.legend(handles=patches, loc='lower left')

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outp, bbox_inches='tight', dpi=150)
    print("Saved overlay:", outp)

if __name__ == "__main__":
    main()
