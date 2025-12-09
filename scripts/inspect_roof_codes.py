# scripts/inspect_roof_codes.py
import geopandas as gpd
pd = __import__('pandas')  # lightweight import so output is cleaner

b = gpd.read_file(r"raw_data\shp\PB_32643\buildings\Built_Up_Area_typ.shp")

# Show the counts (you already gave these, but keep for completeness)
print("=== Roof_type counts ===")
print(b['Roof_type'].value_counts(dropna=False))

# For each roof code, show:
#  - unique Area_Desc values (if any)
#  - top 5 sample rows with columns likely to be informative
cols = ['Roof_type','Area_Desc','Name','Built_Up_A','Area_Sqm','Village_Na','GP_Code']
for code in sorted(b['Roof_type'].unique()):
    print("\n\n>>> Roof_type =", code, " (n=", (b['Roof_type']==code).sum(), ")")
    sub = b[b['Roof_type']==code]
    # unique Area_Desc (if exists)
    if 'Area_Desc' in sub.columns:
        uniq = sub['Area_Desc'].dropna().unique()[:20]
        print("Sample unique Area_Desc values:", list(uniq))
    # show top 8 rows for inspection
    print("Sample rows (first 8):")
    print(sub[cols].head(8).to_string(index=False))
