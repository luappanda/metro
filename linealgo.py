import geopandas as gpd
import os
import numpy as np
from shapely.geometry import LineString
import networkx as nx

stations = gpd.read_file("GISFiles/output/best stations4.gpkg")
stations2 = gpd.read_file("GISFiles/output/best stations2.gpkg")
stations3 = gpd.read_file("GISFiles/output/best stations3.gpkg")
connections1=[]
connections2=[]
connections3=[]
connections=[]

points1 = [(point.x, point.y) for point in stations.geometry.centroid]
points2 = [(point.x, point.y) for point in stations2.geometry.centroid]
points3 = [(point.x, point.y) for point in stations3.geometry.centroid]


G = nx.complete_graph(len(points1))
for i, (x1, y1) in enumerate(points1):
    for j, (x2, y2) in enumerate(points1):
        if i != j:
            dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            G[i][j]['weight'] = dist


G2 = nx.complete_graph(len(points2))
for i, (x1, y1) in enumerate(points2):
    for j, (x2, y2) in enumerate(points2):
        if i != j:
            dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            G2[i][j]['weight'] = dist


G3 = nx.complete_graph(len(points3))
for i, (x1, y1) in enumerate(points3):
    for j, (x2, y2) in enumerate(points3):
        if i != j:
            dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            G3[i][j]['weight'] = dist

# Solve MST (tree)
mst = nx.minimum_spanning_tree(G, weight='weight')
print("MST edges:", mst.edges(data=True))

mst2 = nx.minimum_spanning_tree(G2, weight='weight')
print("MST2 edges:", mst2.edges(data=True))

mst3 = nx.minimum_spanning_tree(G3, weight='weight')
print("MST3 edges:", mst3.edges(data=True))


for edge in mst.edges:
    beg_point = stations.geometry.iloc[edge[0]].centroid
    end_point = stations.geometry.iloc[edge[1]].centroid
    line = LineString([beg_point, end_point])
    connections1.append(line)
    connections.append(line)

for edge in mst2.edges:
    beg_point = stations2.geometry.iloc[edge[0]].centroid
    end_point = stations2.geometry.iloc[edge[1]].centroid
    line = LineString([beg_point, end_point])
    connections2.append(line)
    connections.append(line)

for edge in mst3.edges:
    beg_point = stations3.geometry.iloc[edge[0]].centroid
    end_point = stations3.geometry.iloc[edge[1]].centroid
    line = LineString([beg_point, end_point])
    connections3.append(line)
    connections.append(line)

# Create a new GeoDataFrame with the connecting lines
connections1_gdf = gpd.GeoDataFrame(geometry=connections1, crs=stations.crs)
connections2_gdf = gpd.GeoDataFrame(geometry=connections2, crs=stations.crs)
connections3_gdf = gpd.GeoDataFrame(geometry=connections3, crs=stations.crs)
connections_gdf = gpd.GeoDataFrame(geometry=connections, crs=stations.crs)

# Save or plot the result
output_fp = os.getcwd() + "/GISFiles/output/connections1.gpkg"
connections1_gdf.to_file(output_fp, driver='GPKG')
output_fp = os.getcwd() + "/GISFiles/output/connections2.gpkg"
connections2_gdf.to_file(output_fp, driver='GPKG')
output_fp = os.getcwd() + "/GISFiles/output/connections3.gpkg"
connections3_gdf.to_file(output_fp, driver='GPKG')
output_fp = os.getcwd() + "/GISFiles/output/connections.gpkg"
connections_gdf.to_file(output_fp, driver='GPKG')