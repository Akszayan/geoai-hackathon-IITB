# scripts/generate_tiles_npy.py
"""
Resumable tile + mask generator for PB orthos with progress prints and geometry fixes.

Run: python -u "scripts/generate_tiles_npy.py"
"""
import json
import os
import random
import warnings
from pathlib import Path
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
import geopandas as gpd
import pandas as pd
from shapely.geometry import mapping
from shapely.ops import unary_union

# ---------------- CONFIG ----------------
PROJECT_ROOT = Path(".").resolve()
ORTHO_DIR = PROJECT_ROOT / "raw_data" / "orthos" / "PB"
BUILD_SHP = PROJECT_ROOT / "raw_data" / "shp" / "PB_32643" / "buildings" / "Built_Up_Area_typ.shp"
WATER_SHP = PROJECT_ROOT / "raw_data" / "shp" / "PB_32643" / "water" / "Water_Body.shp"
MAPPING_JSON = PROJECT_ROOT / "configs" / "pb_roof_mapping.json"

OUT_DIR = PROJECT_ROOT / "data"
IM_TILES_DIR = OUT_DIR / "tiles" / "images_npy"
MASK_TILES_DIR = OUT_DIR / "tiles" / "masks_npy"
META_DIR = OUT_DIR / "meta"

TILE_SIZE = 512
OVERLAP = 64
STRIDE = TILE_SIZE - OVERLAP
NEGATIVE_TILE_RATIO = 0.10  # keep 10% empty tiles
RANDOM_SEED = 42
MAX_MEAN_STD_SAMPLES = 500
PROGRESS_PRINT_EVERY = 200  # print progress every N tiles while processing an ortho

# create folders
for d in (IM_TILES_DIR, MASK_TILES_DIR, META_DIR):
    d.mkdir(parents=True, exist_ok=True)

# load mapping
with open(MAPPING_JSON, "r") as f:
    mapping = json.load(f)
WATER_CLASS_ID = int(mapping["water"])

# load vectors
buildings_gdf = gpd.read_file(BUILD_SHP)
water_gdf = gpd.read_file(WATER_SHP)

# attempt to fix invalid geometries (in-place)
def ensure_valid_gdf(gdf, name="layer"):
    invalid_count = (~gdf.is_valid).sum()
    if invalid_count > 0:
        print(f"[INFO] Fixing {invalid_count} invalid geometries in {name} using buffer(0)")
        fixed_geoms = []
        for geom in gdf.geometry:
            if geom is None:
                fixed_geoms.append(None)
                continue
            try:
                if not geom.is_valid:
                    g = geom.buffer(0)
                    if g.is_valid:
                        fixed_geoms.append(g)
                    else:
                        # last resort: attempt unary_union of the geometry
                        try:
                            u = unary_union([geom])
                            fixed_geoms.append(u)
                        except Exception:
                            fixed_geoms.append(None)
                else:
                    fixed_geoms.append(geom)
            except Exception:
                fixed_geoms.append(None)
        gdf = gdf.copy()
        gdf.geometry = fixed_geoms
        # drop null geometries
        gdf = gdf[~gdf.geometry.isnull()].reset_index(drop=True)
    return gdf

buildings_gdf = ensure_valid_gdf(buildings_gdf, "buildings")
water_gdf = ensure_valid_gdf(water_gdf, "water")

# ---------------- helpers ----------------
def rasterize_for_ortho(ortho_path):
    """Rasterize buildings and water to a full resolution mask aligned to ortho."""
    ortho_path = Path(ortho_path)
    with rasterio.open(ortho_path) as src:
        meta = src.meta.copy()
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs

    # reproject if needed
    b = buildings_gdf.to_crs(crs)
    w = water_gdf.to_crs(crs)

    # prepare building tuples: use Roof_type numeric codes directly (1..4)
    b_tuples = []
    for idx, row in b.iterrows():
        rt = row.get("Roof_type", None)
        try:
            code = int(rt)
        except Exception:
            continue
        if code > 0 and row.geometry is not None:
            try:
                b_tuples.append((row.geometry, int(code)))
            except Exception:
                # skip bad geometry
                continue

    w_tuples = []
    for geom in w.geometry:
        if geom is None:
            continue
        try:
            w_tuples.append((geom, WATER_CLASS_ID))
        except Exception:
            continue

    # Rasterize with try/except; catch invalid shape skips and continue after attempts to fix
    # First buildings
    if len(b_tuples) == 0 and len(w_tuples) == 0:
        mask = np.zeros((height, width), dtype=np.uint8)
        return mask, transform, crs, meta

    # rasterize buildings
    mask_build = np.zeros((height, width), dtype=np.uint8)
    if b_tuples:
        try:
            mask_build = rasterize(
                b_tuples,
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype="uint8",
                all_touched=False,
            )
        except Exception as e:
            print("[WARN] rasterize(buildings) failed, attempting to filter bad geometries:", e)
            # filter using mapping to safe geoms
            safe = []
            for geom, val in b_tuples:
                try:
                    mapping(geom)  # check
                    safe.append((geom, val))
                except Exception:
                    try:
                        g = geom.buffer(0)
                        mapping(g)
                        safe.append((g, val))
                    except Exception:
                        continue
            if safe:
                mask_build = rasterize(safe, out_shape=(height, width), transform=transform, fill=0, dtype="uint8", all_touched=False)
            else:
                mask_build = np.zeros((height, width), dtype=np.uint8)

    # rasterize water
    mask_water = np.zeros((height, width), dtype=np.uint8)
    if w_tuples:
        try:
            mask_water = rasterize(
                w_tuples,
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype="uint8",
                all_touched=False,
            )
        except Exception as e:
            print("[WARN] rasterize(water) failed, attempting to fix:", e)
            safe = []
            for geom, val in w_tuples:
                try:
                    mapping(geom)
                    safe.append((geom, val))
                except Exception:
                    try:
                        g = geom.buffer(0)
                        mapping(g)
                        safe.append((g, val))
                    except Exception:
                        continue
            if safe:
                mask_water = rasterize(safe, out_shape=(height, width), transform=transform, fill=0, dtype="uint8", all_touched=False)
            else:
                mask_water = np.zeros((height, width), dtype=np.uint8)

    # combine: buildings override water
    mask = mask_build.copy()
    mask[mask == 0] = mask_water[mask == 0]

    return mask, transform, crs, meta

def pad_to_tile(arr, tile_size=TILE_SIZE):
    h, w = arr.shape[:2]
    if h == tile_size and w == tile_size:
        return arr
    pad_h = tile_size - h
    pad_w = tile_size - w
    if arr.ndim == 3:
        return np.pad(arr, ((0,pad_h),(0,pad_w),(0,0)), constant_values=0)
    else:
        return np.pad(arr, ((0,pad_h),(0,pad_w)), constant_values=0)

def get_tile_id(stem, x, y):
    return f"{stem}_x{x}_y{y}"

# ---------------- main processing ----------------
def process_ortho(ortho_path: Path):
    stem = ortho_path.stem
    meta_csv = META_DIR / f"{stem}_tiles.csv"
    if meta_csv.exists():
        print(f"[SKIP] Meta exists for {stem}, skipping (resume).")
        return pd.read_csv(meta_csv)

    print(f"[RUN] Processing ortho: {ortho_path}")
    mask, transform, crs, _meta = rasterize_for_ortho(ortho_path)
    H, W = mask.shape
    total_tiles_est = ((H + STRIDE - 1) // STRIDE) * ((W + STRIDE - 1) // STRIDE)
    print(f" - Ortho size: {W} x {H} px -> estimated tiles: {total_tiles_est}")

    saved_tiles = []
    negative_candidates = []
    processed = 0
    saved_count = 0

    with rasterio.open(ortho_path) as src:
        for y in range(0, H, STRIDE):
            for x in range(0, W, STRIDE):
                processed += 1
                win_w = TILE_SIZE if x + TILE_SIZE <= W else W - x
                win_h = TILE_SIZE if y + TILE_SIZE <= H else H - y
                window = Window(x, y, win_w, win_h)
                try:
                    img = src.read([1,2,3], window=window)
                except Exception as e:
                    print(f"[ERR] reading window x{x} y{y}: {e}")
                    continue
                img = np.moveaxis(img, 0, 2)
                img = pad_to_tile(img)

                m = mask[y:y+TILE_SIZE, x:x+TILE_SIZE]
                m = pad_to_tile(m)

                has_label = (m != 0).any()
                tid = get_tile_id(stem, x, y)
                img_out = IM_TILES_DIR / f"{tid}.npy"
                mask_out = MASK_TILES_DIR / f"{tid}.npy"

                if has_label:
                    # save if missing
                    if not img_out.exists():
                        np.save(img_out, img.astype(np.uint8))
                    if not mask_out.exists():
                        np.save(mask_out, m.astype(np.uint8))
                    saved_tiles.append((tid, str(img_out), str(mask_out), None, x, y, True))
                    saved_count += 1
                else:
                    negative_candidates.append((tid, x, y))

                # progress print
                if processed % PROGRESS_PRINT_EVERY == 0 or processed == total_tiles_est:
                    pct = (processed/total_tiles_est)*100
                    print(f"   Processed {processed}/{total_tiles_est} tiles ({pct:.1f}%) - saved so far: {saved_count}")

    # keep fraction of negative tiles
    n_keep = int(len(negative_candidates) * NEGATIVE_TILE_RATIO)
    if n_keep > 0:
        random.Random(RANDOM_SEED).shuffle(negative_candidates)
        pick = negative_candidates[:n_keep]
        with rasterio.open(ortho_path) as src:
            for tid, x, y in pick:
                win_w = TILE_SIZE if x + TILE_SIZE <= W else W - x
                win_h = TILE_SIZE if y + TILE_SIZE <= H else H - y
                window = Window(x, y, win_w, win_h)
                img = src.read([1,2,3], window=window)
                img = np.moveaxis(img, 0, 2)
                img = pad_to_tile(img)
                img_out = IM_TILES_DIR / f"{tid}.npy"
                mask_out = MASK_TILES_DIR / f"{tid}.npy"
                if not img_out.exists():
                    np.save(img_out, img.astype(np.uint8))
                if not mask_out.exists():
                    np.save(mask_out, np.zeros((TILE_SIZE,TILE_SIZE), dtype=np.uint8))
                saved_tiles.append((tid, str(img_out), str(mask_out), None, x, y, False))
                saved_count += 1

    # write meta CSV
    df = pd.DataFrame(saved_tiles, columns=['tile_id','img_npy','mask_npy','gp_code','x','y','has_label'])
    df.to_csv(meta_csv, index=False)
    print(f"[DONE] ortho {stem} -> tiles saved: {len(df)} (including negatives kept). meta: {meta_csv}")
    return df

def main():
    random.seed(RANDOM_SEED)
    # find ortho files
    ortho_list = []
    for root, dirs, files in os.walk(ORTHO_DIR):
        for f in files:
            if f.lower().endswith(".tif"):
                ortho_list.append(Path(root) / f)
    ortho_list = sorted(ortho_list)
    if not ortho_list:
        print("[ERR] No PB orthos found under:", ORTHO_DIR)
        return
    print("[FOUND] ortho files:", ortho_list)

    all_meta_dfs = []
    for ortho in ortho_list:
        try:
            df = process_ortho(ortho)
            all_meta_dfs.append(df)
        except Exception as e:
            print(f"[ERR] processing {ortho}: {e}")

    # combine metas
    if all_meta_dfs:
        all_tiles = pd.concat(all_meta_dfs, ignore_index=True)
    else:
        all_tiles = pd.DataFrame(columns=['tile_id','img_npy','mask_npy','gp_code','x','y','has_label'])

    # simple train/val split: all into train for now if gp_code missing
    gp_codes = all_tiles['gp_code'].dropna().unique().tolist()
    random.Random(RANDOM_SEED).shuffle(gp_codes)
    cutoff = int(0.8 * len(gp_codes)) if gp_codes else 0
    train_gps = set(gp_codes[:cutoff])
    val_gps = set(gp_codes[cutoff:])
    train_rows = all_tiles[all_tiles['gp_code'].isin(train_gps)]
    val_rows = all_tiles[all_tiles['gp_code'].isin(val_gps)]
    null_rows = all_tiles[all_tiles['gp_code'].isna()]
    train_rows = pd.concat([train_rows, null_rows], ignore_index=True)

    train_list = train_rows['img_npy'].astype(str).tolist()
    val_list = val_rows['img_npy'].astype(str).tolist()
    with open(META_DIR / "train_tiles.txt", "w", encoding="utf8") as f:
        f.write("\n".join(train_list))
    with open(META_DIR / "val_tiles.txt", "w", encoding="utf8") as f:
        f.write("\n".join(val_list))

    print(f"[SUMMARY] Saved train_tiles: {len(train_list)}, val_tiles: {len(val_list)}")

    # compute mean/std (sample)
    sample_paths = train_list[:MAX_MEAN_STD_SAMPLES]
    if sample_paths:
        imgs = []
        for p in sample_paths:
            try:
                arr = np.load(p)
                imgs.append(arr.astype(np.float32) / 255.0)
            except Exception:
                continue
        if imgs:
            imgs = np.stack(imgs, axis=0)
            mean = imgs.mean(axis=(0,1,2))
            std = imgs.std(axis=(0,1,2))
            np.save(META_DIR / "mean.npy", mean)
            np.save(META_DIR / "std.npy", std)
            print("[DONE] Saved mean.npy and std.npy")
        else:
            print("[WARN] No valid sample images to compute mean/std.")
    else:
        print("[INFO] No train tiles found to compute mean/std.")

if __name__ == "__main__":
    main()
