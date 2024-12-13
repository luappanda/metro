import geopandas as gpd
from shapely.geometry import Point, Polygon
import numpy as np
import os
from tqdm import tqdm
import shapely.affinity
from concurrent.futures import ProcessPoolExecutor
import numpy as np


# Parameters
width = 3000  # Rectangle width in meters (1 km)
length = 50000  # Rectangle length in meters (example: 2 km)
orientations = np.linspace(0, 180, num=10)  # Angles to test (in degrees)

# Load Weighted Feasibilty Grid
grid_filepath = os.getcwd() + "/GISFiles/weighted grid.gpkg"  # Update the path if necessary
gdf = gpd.read_file(grid_filepath)


# # Ensure GeoDataFrame has geometry
# if 'geometry' not in gdf.columns:
#     gdf['geometry'] = gpd.points_from_xy(gdf['col'], gdf['row'])


# Project to a suitable CRS (e.g., UTM) for accurate distance calculations
if gdf.crs.is_geographic:  # Check if in lat/lon
    gdf = gdf.to_crs(epsg=3857)  # Convert to a projected CRS (Web Mercator)

# Grid search for the best rectangle
best_weight = -np.inf
best_rect = None
best_orientation = None


gdf['geometry'] = gdf.geometry.centroid
viable=gdf[gdf["TOTAL WEIGHTED FEASIBILITY"] > 0.75].reset_index(drop=True)


# Build a spatial index
spatial_index = gdf.sindex


# Function to process a single rectangle configuration
def process_rectangle(args):
    point, angle = args
    base_rect = Polygon([
        (-length / 2, -width / 2),
        (length / 2, -width / 2),
        (length / 2, width / 2),
        (-length / 2, width / 2)
    ])
    rotated_rect = shapely.affinity.rotate(base_rect, angle, origin=(0, 0))
    placed_rect = shapely.affinity.translate(rotated_rect, xoff=point.x, yoff=point.y)

    # Spatial index filtering
    possible_matches_idx = list(spatial_index.intersection(placed_rect.bounds))
    possible_matches = gdf.iloc[possible_matches_idx]

    # Calculate total weight within the rectangle
    total_weight = possible_matches[possible_matches.geometry.within(placed_rect)]['TOTAL WEIGHTED FEASIBILITY'].sum()

    return total_weight, placed_rect, angle


# Prepare inputs for parallel processing
args_list = [(point, angle) for point in viable.geometry for angle in orientations]

# Use ProcessPoolExecutor for parallel processing
print("Starting parallel computation...")
with ProcessPoolExecutor() as executor:
    results = list(tqdm(executor.map(process_rectangle, args_list), total=len(args_list)))

# Find the best result
best_weight, best_rect, best_orientation = max(results, key=lambda x: x[0])

# Output results
print(f"Best Weight: {best_weight}")
print(f"Best Orientation: {best_orientation}")
print(f"Best Rectangle Geometry: {best_rect}")



# Convert the placed rectangle to a GeoDataFrame
gdf = gpd.GeoDataFrame({'geometry': [best_rect]})

# Specify the CRS (e.g., EPSG:3857 for Web Mercator, change as per your data's CRS)
gdf.set_crs("EPSG:3857", allow_override=True, inplace=True)

output_fp = os.getcwd() + "/GISFiles/corridor.gpkg"
# Save the GeoDataFrame as a GeoPackage (.gpkg)
gdf.to_file(output_fp, driver="GPKG")



# Optional: Visualize
import matplotlib.pyplot as plt

gdf.plot(color='blue', markersize=5, label='Points')
gpd.GeoSeries([best_rect]).plot(ax=plt.gca(), edgecolor='red', alpha=0.5, label='Best Rectangle')
plt.legend()
plt.show()

