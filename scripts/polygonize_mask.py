# scripts/polygonize_mask.py
"""
Polygonize mask TIFF to per-class shapefiles.
Usage:
python -u scripts/polygonize_mask.py --mask outputs/predictions/28996_NADALA_ORTHO_pred_mask.tif --out outputs/vectors/
"""
import argparse
from pathlib import Path
import rasterio
from rasterio import features
import geopandas as gpd
from shapely.geometry import shape
import numpy as np 
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mask", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min_area", type=float, default=50.0)  # area threshold in map units (meters^2)
    return p.parse_args()

def main():
    args = parse_args()
    mask_p = Path(args.mask)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(mask_p) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs

    classes = sorted(np.unique(mask))
    all_polys = []
    for cls in classes:
        if cls == 0:
            # optionally skip background if you want
            pass
        mask_bin = (mask == cls).astype('uint8')
        results = []
        for geom, val in features.shapes(mask_bin, mask=mask_bin, transform=transform):
            if val == 0:
                continue
            geom_shp = shape(geom)
            # area filter here optional; preserve for postprocess step
            results.append({'geometry': geom_shp, 'class': int(cls)})

        if results:
            gdf = gpd.GeoDataFrame(results, crs=crs)
            cls_path = out_dir / f"class_{int(cls)}.geojson"
            gdf.to_file(cls_path, driver="GeoJSON")
            print("Saved", cls_path)
            all_polys.append(gdf)

    if all_polys:
        merged = gpd.GeoDataFrame(pd.concat(all_polys, ignore_index=True), crs=all_polys[0].crs)
        merged_path = out_dir / "all_classes.geojson"
        merged.to_file(merged_path, driver="GeoJSON")
        print("Saved merged", merged_path)

if __name__ == "__main__":
    main()
