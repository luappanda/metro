import geopandas as gpd
import os
from shapely.geometry import LineString

stations = gpd.read_file("GISFiles/selection.gpkg")
stations_sindex = stations.sindex

connections = []
matched_pairs = set()

for idx, geom in stations.iterrows():
    # Use the spatial index to find the nearest neighbor (excluding itself)
    nearest_idx_gen = stations_sindex.nearest(geom.geometry, return_all=False, exclusive=True)
    nearest_idx = int(nearest_idx_gen[1])
    pair=tuple(sorted((idx, nearest_idx)))
    if pair not in matched_pairs:
        closest_station = stations.geometry.iloc[nearest_idx]
        geom_point = geom.geometry.centroid
        closest_point = closest_station.centroid
        line = LineString([geom_point, closest_point])
        connections.append(line)   
        matched_pairs.add(pair)

# Create a new GeoDataFrame with the connecting lines
connections_gdf = gpd.GeoDataFrame(geometry=connections, crs=stations.crs)

# Save or plot the result
output_fp = os.getcwd() + "/GISFiles/connections.gpkg"
connections_gdf.to_file(output_fp, driver='GPKG')
connections_gdf.plot()