import geopandas as gpd
import os
import random
from shapely.geometry import LineString

stations = gpd.read_file("GISFiles/selection.gpkg")
connections = []

unused_station_ids = stations.index.values.tolist()
r_element = random.randint(0,len(unused_station_ids)-1)
beginning = unused_station_ids.pop(r_element)
while len(unused_station_ids) > 0:
    r_element = random.randint(0,len(unused_station_ids)-1)
    end = unused_station_ids.pop(r_element)
    beg_point = stations.geometry.iloc[beginning].centroid
    end_point = stations.geometry.iloc[end].centroid
    line = LineString([beg_point, end_point])
    connections.append(line)
    beginning=end

# Create a new GeoDataFrame with the connecting lines
connections_gdf = gpd.GeoDataFrame(geometry=connections, crs=stations.crs)

# Save or plot the result
output_fp = os.getcwd() + "/GISFiles/connections.gpkg"
connections_gdf.to_file(output_fp, driver='GPKG')
connections_gdf.plot()