# scripts/inspect_shp.py
import sys
import fiona
from pathlib import Path

if len(sys.argv) != 2:
    print("usage: python scripts/inspect_shp.py <shapefile.shp>")
    raise SystemExit(1)

p = Path(sys.argv[1])
with fiona.open(p) as src:
    print("SHP PATH:", str(p))
    print("crs:", src.crs)
    print("bounds:", src.bounds)
    print("feature count:", len(src))
