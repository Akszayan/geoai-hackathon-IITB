# scripts/postprocess_polygons.py
"""
Clean and dissolve polygons per class.
Usage:
python -u scripts/postprocess_polygons.py --vectors outputs/vectors/ --out outputs/vectors_clean/ --min_area 100
"""
import argparse
from pathlib import Path
import geopandas as gpd
from shapely.ops import unary_union

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vectors", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min_area", type=float, default=100.0)
    return p.parse_args()

def main():
    args = parse_args()
    in_dir = Path(args.vectors)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    files = list(in_dir.glob("class_*.geojson"))
    for f in files:
        gdf = gpd.read_file(f)
        # area filter (area in CRS units)
        gdf['area'] = gdf.geometry.area
        gdf = gdf[gdf['area'] >= args.min_area]
        if gdf.empty:
            continue
        # dissolve all geometries of this class into single MultiPolygon, then explode
        dissolved = gpd.GeoSeries([unary_union(gdf.geometry.values)], crs=gdf.crs)
        out_gdf = gpd.GeoDataFrame({'geometry': dissolved})
        out_path = out_dir / f"{f.stem}_clean.geojson"
        out_gdf.to_file(out_path, driver="GeoJSON")
        print("Saved cleaned:", out_path)

if __name__ == "__main__":
    main()
