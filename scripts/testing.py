import geopandas as gpd
pb_buildings = gpd.read_file(r"raw_data\shp\PB_32643\buildings\Built_Up_Area_typ.shp")
pb_water = gpd.read_file(r"raw_data\shp\PB_32643\water\Water_Body.shp")

print("Buildings CRS:", pb_buildings.crs)
print("Water CRS:", pb_water.crs)
print("Buildings fields:", list(pb_buildings.columns))
print("Water fields:", list(pb_water.columns))

# Unique roof codes and sample value counts:
print(pb_buildings['Roof_type'].value_counts(dropna=False).head(50))

# Check a few rows (roof types + description fields)
print(pb_buildings[['Roof_type','Area_Desc','Area_Sqm','Village_Na']].sample(10))

# Water body categories
print(pb_water['Water_Body'].value_counts(dropna=False))
print(pb_water[['Water_Body','Perenniali','Covered','Area_Sqm','Village_Na']].sample(10))
