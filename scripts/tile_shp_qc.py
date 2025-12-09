
"""
tile_shp_qc.py
Comprehensive pre-rasterization QA for tiles + shapefiles.

Usage:
    python tile_shp_qc.py \
        --tiles_dir "outputs/tiles/28996_NADALA_ORTHO" \
        --shp_dirs "raw_data/shp/PB_32643/buildings" "raw_data/shp/PB_32643/water" \
        --sample_tiles 8 \
        --max_sample_polygons 200

What it does (read-only):
 - Scans tiles and shapefile folders
 - Prints raster stats per tile (shape, bands, means/stds, all-zero)
 - Prints vector stats per shapefile (CRS, feature count, geom types, invalid geometries)
 - Checks CRS matches between tiles and shapefiles
 - Computes bbox overlap and percent coverage
 - For a small set of tiles, rasterizes vectors in memory at tile transform/resolution and reports
   coverage percentages per class and warnings about tiny polygons / slivers
 - Heuristically checks whether band 4 looks like NIR vs alpha (by constant or saturated values)
 - Summarizes problematic tiles (all-zero, tiny edge tiles, mismatched CRS, no-overlap)

Note: This script only prints diagnostics. It does not modify any files.
"""

import os
import sys
import argparse
import glob
import json
import math
from collections import defaultdict, Counter

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import box, mapping
from shapely.ops import unary_union
from tqdm import tqdm
import pandas as pd

def discover_tiles(tiles_dir):
    patterns = ["**/*.tif", "**/*.tiff"]
    paths = []
    for p in patterns:
        paths.extend(glob.glob(os.path.join(tiles_dir, p), recursive=True))
    paths = sorted(paths)
    return paths

def discover_shapefiles(shp_dirs):
    paths = []
    for d in shp_dirs:
        if os.path.isdir(d):
            # accept a folder containing shapefile components OR a direct .shp
            shp_files = glob.glob(os.path.join(d, "*.shp"))
            if shp_files:
                paths.extend(sorted(shp_files))
            else:
                # maybe user passed a folder that itself is a shapefile dir
                # no-op
                pass
        elif os.path.isfile(d) and d.lower().endswith(".shp"):
            paths.append(d)
    return sorted(paths)

def raster_basic_stats(path):
    out = {}
    try:
        with rasterio.open(path) as src:
            out['path'] = path
            out['width'] = src.width
            out['height'] = src.height
            out['count'] = src.count
            out['dtype'] = str(src.dtypes[0]) if src.count>0 else 'unknown'
            out['crs'] = src.crs.to_string() if src.crs else None
            out['transform'] = src.transform
            out['bounds'] = src.bounds
            out['nodata'] = src.nodata
            out['filesize'] = os.path.getsize(path)
            # read overview: small read of the first band as thumbnail to compute stats cheaply
            # read full raster if small
            if src.width * src.height <= 512*512:
                arr = src.read()  # (bands, H, W)
                arr = np.array(arr)
                out['band_means'] = list(np.nanmean(arr.reshape(arr.shape[0], -1), axis=1))
                out['band_stds'] = list(np.nanstd(arr.reshape(arr.shape[0], -1), axis=1))
                nonzeros = [(np.count_nonzero(arr[i]) / arr[i].size) for i in range(arr.shape[0])]
                out['band_nonzero_frac'] = list(nonzeros)
                out['all_zero'] = all([x == 0 for x in out['band_means']])
            else:
                # sample center patch (256x256) for stats
                win = Window(src.width//2 - 128, src.height//2 - 128, 256, 256)
                arr = src.read(window=win)
                out['band_means'] = list(np.nanmean(arr.reshape(arr.shape[0], -1), axis=1))
                out['band_stds'] = list(np.nanstd(arr.reshape(arr.shape[0], -1), axis=1))
                nonzeros = [(np.count_nonzero(arr[i]) / arr[i].size) for i in range(arr.shape[0])]
                out['band_nonzero_frac'] = list(nonzeros)
                out['all_zero'] = all([x == 0 for x in out['band_means']])
            # simple heuristics
            out['alpha_like_band4'] = False
            if out['count'] >= 4:
                b4 = out['band_means'][3]
                b4std = out['band_stds'][3]
                # if band4 mean very near 255 or std == 0, suspect alpha/saturation
                if (b4 >= 250 and b4std < 1.0) or (b4std == 0):
                    out['alpha_like_band4'] = True
            return out
    except Exception as e:
        return {'path': path, 'error': str(e)}

def vector_basic_stats(shp_path, max_preview=200):
    out = {}
    try:
        gdf = gpd.read_file(shp_path)
        out['path'] = shp_path
        out['crs'] = gdf.crs.to_string() if gdf.crs else None
        out['count'] = len(gdf)
        # geometry types distribution
        types = gdf.geom_type.value_counts().to_dict()
        out['geom_types'] = types
        # validity check (sample or all)
        invalid_mask = ~gdf.is_valid
        out['invalid_count'] = int(invalid_mask.sum())
        # bounding box
        out['bounds'] = gdf.total_bounds.tolist()  # minx,miny,maxx,maxy
        # simple attributes summary
        out['columns'] = list(gdf.columns)
        out['preview_head'] = gdf.head(min(max_preview, len(gdf))).iloc[:, :5].to_dict(orient='records') if len(gdf)>0 else []
        return out
    except Exception as e:
        return {'path': shp_path, 'error': str(e)}

def bbox_overlap_pct(tile_bounds, shp_bounds):
    # both in same CRS (assumes caller ensured CRS check)
    tminx, tminy, tmaxx, tmaxy = tile_bounds
    sminx, sminy, smaxx, smaxy = shp_bounds
    tile_box = box(tminx, tminy, tmaxx, tmaxy)
    shp_box = box(sminx, sminy, smaxx, smaxy)
    inter = tile_box.intersection(shp_box).area
    tile_area = tile_box.area
    if tile_area == 0:
        return 0.0
    return float(inter / tile_area)

def rasterize_coverage_for_tile(tile_path, shp_paths_and_labels, max_polys=500):
    """
    For a single tile, rasterize the provided shapefiles (list of (shp_path,label))
    onto the tile grid and compute coverage fraction per label.
    Returns dict: {label: fraction_covered}
    """
    coverage = {}
    try:
        with rasterio.open(tile_path) as src:
            transform = src.transform
            width = src.width
            height = src.height
            tile_bounds = src.bounds
            # prepare shapes list per label
            for shp_path, label in shp_paths_and_labels:
                gdf = gpd.read_file(shp_path)
                if gdf.crs != src.crs:
                    try:
                        gdf = gdf.to_crs(src.crs)
                    except Exception as e:
                        coverage[label] = {'error': f'failed reproject: {e}'}
                        continue
                if len(gdf) == 0:
                    coverage[label] = {'fraction': 0.0, 'polygons': 0}
                    continue
                # optionally limit to features intersecting tile bbox
                bbox_poly = box(*tile_bounds)
                sel = gdf[gdf.intersects(bbox_poly)]
                if len(sel) == 0:
                    coverage[label] = {'fraction': 0.0, 'polygons': 0}
                    continue
                # take sample of polygons if too many
                if len(sel) > max_polys:
                    sel = sel.sample(max_polys)
                shapes = ((geom, 1) for geom in sel.geometry)
                mask = rasterize(shapes, out_shape=(height, width), transform=transform, fill=0, all_touched=False, dtype='uint8')
                frac = float(np.count_nonzero(mask) / (width * height))
                coverage[label] = {'fraction': frac, 'polygons': len(sel)}
            return coverage
    except Exception as e:
        return {'error': str(e)}

def quick_ndvi_check(tile_path):
    """
    If there are >=4 bands, compute a tiny center-sample NDVI = (B4 - B1) / (B4 + B1)
    as a heuristic to see whether band 4 behaves like NIR.
    Returns mean_ndvi, std_ndvi or None if not computable.
    """
    try:
        with rasterio.open(tile_path) as src:
            if src.count < 4:
                return None
            # sample a small window at center
            cx = src.width // 2
            cy = src.height // 2
            w = min(128, cx)
            h = min(128, cy)
            win = Window(cx - w//2, cy - h//2, w, h)
            arr = src.read(window=win).astype('float32')  # (bands, H, W)
            red = arr[0]
            b4 = arr[3]
            denom = (b4 + red)
            valid = denom != 0
            ndvi = np.zeros_like(red, dtype='float32')
            ndvi[valid] = (b4[valid] - red[valid]) / denom[valid]
            # sanity measures
            mean_ndvi = float(np.nanmean(ndvi))
            std_ndvi = float(np.nanstd(ndvi))
            # water tends to negative-ish ndvi; veg positive. If ndvi values are near 0 always, band4 may not be NIR.
            return {'mean_ndvi': mean_ndvi, 'std_ndvi': std_ndvi}
    except Exception:
        return None

def summarize(tiles_info, shp_info, tile_shp_overlap, coverage_samples, args):
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    print(f"Scanned {len(tiles_info)} tile files.")
    tot_all_zero = sum(1 for t in tiles_info if t.get('all_zero'))
    edge_tiles = sum(1 for t in tiles_info if t.get('width',0) != args.tile_size or t.get('height',0) != args.tile_size)
    print(f"All-zero tiles: {tot_all_zero} (these should be excluded from training).")
    print(f"Edge / non-{args.tile_size} tiles: {edge_tiles} (consider padding or dropping).")
    # list a few examples
    if tot_all_zero > 0:
        print("Examples of all-zero tiles (first 5):")
        for t in [x for x in tiles_info if x.get('all_zero')][:5]:
            print("  -", t['path'])
    # CRS summary
    tile_crs_counts = Counter(t.get('crs') for t in tiles_info)
    shp_crs_counts = Counter(s.get('crs') for s in shp_info)
    print("\nTile CRS distribution (top):")
    for k, v in tile_crs_counts.most_common():
        print(f"  {k}: {v}")
    print("Shapefile CRS distribution (top):")
    for k, v in shp_crs_counts.most_common():
        print(f"  {k}: {v}")
    # overlaps
    print("\nTile <-> Shapefile bbox overlap (sample):")
    for key, val in tile_shp_overlap.items():
        # key: (tile,shp)
        tbase = os.path.basename(key[0])
        sbase = os.path.basename(key[1])
        print(f"  Tile {tbase} vs SHP {sbase}: overlap fraction of tile bbox = {val:.3f}")
    # coverage samples
    print("\nSampled in-memory rasterization coverage (per-class):")
    for tile, cov in coverage_samples.items():
        print("  Tile:", os.path.basename(tile))
        for label, info in cov.items():
            if 'error' in info:
                print(f"    {label}: ERROR {info['error']}")
            else:
                print(f"    {label}: polygons={info['polygons']}, coverage_fraction={info['fraction']:.4f}")
                if info['fraction'] < 0.001 and info['polygons']>0:
                    print("      >> VERY SMALL COVERAGE -> possible tiny polygons / poor alignment / scale mismatch")
    # NDVI heuristics
    ndvi_ok = [t for t in tiles_info if t.get('ndvi')]
    if ndvi_ok:
        print("\nNDVI heuristic (center patch) results (first 5):")
        for entry in ndvi_ok[:5]:
            print(f"  {os.path.basename(entry['path'])}: mean_ndvi={entry['ndvi']['mean_ndvi']:.3f}, std={entry['ndvi']['std_ndvi']:.3f}")
            if abs(entry['ndvi']['mean_ndvi']) < 0.02 and entry.get('alpha_like_band4'):
                print("    >> Band4 looks like ALPHA or not true NIR (mean NDVI ~ 0 and band4 saturated).")
    # suggestions
    print("\nRECOMMENDATIONS:")
    print(" - Exclude all-zero tiles from training. (Blank tiles add noise.)")
    print(" - Reproject any shapefile whose CRS != tile CRS. Use the tile CRS as canonical for rasterization.")
    print(" - For small edge tiles (non-512), either pad to the canonical tile size or exclude from training; include them for inference if needed.")
    print(" - If many shapefile polygons intersect tiles but coverage_fraction is near 0, inspect alignment and polygon sizes (may need buffering).")
    print(" - If band 4 appears alpha-like (mean≈255 && std≈0), DO NOT use it as NIR. Confirm band order before using NDVI/MNDWI.")
    print(" - Use an area threshold post-rasterization to drop tiny objects (e.g., < 9-25 pixels).")
    print("="*80 + "\n")

def main(argv=None):
    p = argparse.ArgumentParser(description="Tile & Shapefile pre-mask QA")
    p.add_argument("--tiles_dir", required=True, help="Directory with tile .tif files (may contain subfolders)")
    p.add_argument("--shp_dirs", nargs="+", required=True, help="One or more shapefile directories or direct .shp paths. Pass building and water shapefiles.")
    p.add_argument("--sample_tiles", type=int, default=6, help="Number of tiles to sample for in-memory rasterization coverage")
    p.add_argument("--max_sample_polygons", type=int, default=250, help="Max polygons to rasterize per class in a tile sample")
    p.add_argument("--tile_size", type=int, default=512, help="Canonical tile size expected (512)")
    args = p.parse_args(argv)

    tiles = discover_tiles(args.tiles_dir)
    if not tiles:
        print("No tile files found in:", args.tiles_dir)
        return
    shps = discover_shapefiles(args.shp_dirs)
    if not shps:
        print("No shapefiles found in:", args.shp_dirs)
        return

    print(f"Found {len(tiles)} tile files and {len(shps)} shapefiles.")
    print("Scanning a few tiles for basic statistics (fast mode)...")
    tiles_info = []
    for t in tqdm(tiles[:min(200, len(tiles))]):
        info = raster_basic_stats(t)
        tiles_info.append(info)

    print("Scanning shapefiles for basic statistics...")
    shp_info = []
    for s in shps:
        info = vector_basic_stats(s)
        shp_info.append(info)

    # build tile-shp bbox overlap sample (first N)
    tile_shp_overlap = {}
    sampled_tiles_for_overlap = tiles[:min(40, len(tiles))]
    for t in sampled_tiles_for_overlap:
        try:
            with rasterio.open(t) as src:
                tb = src.bounds
                tcrs = src.crs.to_string() if src.crs else None
                for s in shps:
                    try:
                        g = gpd.read_file(s)
                        if g.crs is None:
                            # cannot compute reliably
                            overlap = 0.0
                        else:
                            # reproject shapefile bounds to tile CRS if needed
                            if g.crs != src.crs:
                                g2 = g.to_crs(src.crs)
                                sb = g2.total_bounds
                            else:
                                sb = g.total_bounds
                            overlap = bbox_overlap_pct(tb, sb)
                        tile_shp_overlap[(t, s)] = overlap
                    except Exception as e:
                        tile_shp_overlap[(t, s)] = 0.0
        except Exception as e:
            tile_shp_overlap[(t, s)] = 0.0

    # sample a few tiles for in-memory rasterization coverage
    sample_tiles = []
    # prefer positive, non-all-zero tiles
    pos_tiles = [t for t in tiles_info if (not t.get('all_zero')) and (not t.get('error'))]
    if len(pos_tiles) >= args.sample_tiles:
        sample_tiles = [t['path'] for t in pos_tiles[:args.sample_tiles]]
    else:
        sample_tiles = [t['path'] for t in tiles_info[:args.sample_tiles]]

    shp_pairs = []
    # label shapefiles sensibly by directory name or basename
    for s in shps:
        label = os.path.splitext(os.path.basename(s))[0]
        shp_pairs.append((s, label))

    coverage_samples = {}
    for t in sample_tiles:
        cov = rasterize_coverage_for_tile(t, shp_pairs, max_polys=args.max_sample_polygons)
        coverage_samples[t] = cov

    # compute NDVI heuristics for first few tiles
    for tinfo in tiles_info:
        nd = quick_ndvi_check(tinfo['path']) if not tinfo.get('error') else None
        tinfo['ndvi'] = nd

    summarize(tiles_info, shp_info, tile_shp_overlap, coverage_samples, args)

if __name__ == "__main__":
    main()
