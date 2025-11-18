import geopandas as gpd
from pyogrio import list_layers

path = "/mnt/12TB/Sujan/LRF_RC/NWM_Q_to_GH/Hypothesis/Checker/WBD_07_HU2_GPKG.gpkg"
print(list_layers(path))               # show all layer names

gdf = gpd.read_file(path, layer="WBDHU2")
print(gdf["huc2"].value_counts())
print(gdf.columns)
