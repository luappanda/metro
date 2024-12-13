import geopandas as gpd
import os
import random
import math
import numpy as np
from shapely.geometry import LineString

stations = gpd.read_file("GISFiles/best stations4.gpkg")
stations2 = gpd.read_file("GISFiles/best stations2.gpkg")
stations3 = gpd.read_file("GISFiles/best stations3.gpkg")
connections = []
minrow = min(stations["row_index"])
mincol = min(stations["col_index"])
points = list()
minrow2 = min(stations2["row_index"])
mincol2 = min(stations2["col_index"])
points2 = list()
minrow3 = min(stations3["row_index"])
mincol3 = min(stations3["col_index"])
points3 = list()


# for idx in stations["row_index"]:
for index in range(len(stations)):
    x=int(stations["col_index"][index]-mincol)
    y=int(stations["row_index"][index]-minrow)
    points.append((y, x))

# for idx in stations["row_index"]:
for index in range(len(stations2)):
    x=int(stations2["col_index"][index]-mincol2)
    y=int(stations2["row_index"][index]-minrow2)
    points2.append((y, x))

# # for idx in stations["row_index"]:
for index in range(len(stations3)):
    x=int(stations3["col_index"][index]-mincol3)
    y=int(stations3["row_index"][index]-minrow3)
    points3.append((y, x))

def distance(p1,p2):
    return math.sqrt((p2[0]-p1[0])**2+(p2[1]-p2[1])**2)

def prim_mst(vertices):
    n=len(vertices) #Number of vertices
    selected = [False] * n
    min_edge = [(None, float('inf'))] * n  # (parent, weight)
    # min_edge[0] = (-1, 0)  # Start with the first vertex, with weight 0
    mst_edges = []  # List to store MST edges
    
    for _ in range(n):
        # Find the vertex with the smallest edge weight that isn't included in the MST
        u = min((i for i in range(n) if not selected[i]), key=lambda x: min_edge[x][1])
        selected[u] = True

        # If u has a parent, add the edge to the MST
        if min_edge[u][0] is not None:
            mst_edges.append((min_edge[u][0], u, min_edge[u][1]))

        # Update the minimum edge weights to connect other vertices to the MST
        for v in range(n):
            if not selected[v]:
                weight = distance(vertices[u], vertices[v])
                if weight < min_edge[v][1]:
                    min_edge[v] = (u, weight)

    return mst_edges

# Run the algorithm
mst = prim_mst(points)
mst2 = prim_mst(points2)
mst3 = prim_mst(points3)

for edge in mst:
    beg_point = stations.geometry.iloc[edge[0]].centroid
    end_point = stations.geometry.iloc[edge[1]].centroid
    line = LineString([beg_point, end_point])
    connections.append(line)

for edge in mst2:
    beg_point = stations2.geometry.iloc[edge[0]].centroid
    end_point = stations2.geometry.iloc[edge[1]].centroid
    line = LineString([beg_point, end_point])
    connections.append(line)

for edge in mst3:
    beg_point = stations3.geometry.iloc[edge[0]].centroid
    end_point = stations3.geometry.iloc[edge[1]].centroid
    line = LineString([beg_point, end_point])
    connections.append(line)

# Create a new GeoDataFrame with the connecting lines
connections_gdf = gpd.GeoDataFrame(geometry=connections, crs=stations.crs)

# Save or plot the result
output_fp = os.getcwd() + "/GISFiles/connections.gpkg"
connections_gdf.to_file(output_fp, driver='GPKG')
connections_gdf.plot()