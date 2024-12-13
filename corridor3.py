import geopandas as gpd
from shapely.geometry import Point, Polygon
import numpy as np
import os
from tqdm import tqdm
import shapely.affinity
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import rasterio
from rasterio.mask import mask
import pandas as pd


# Parameters
width = 3000  # Rectangle width in meters (1 km)
length = 25000  # Rectangle length in meters (example: 2 km)
orientations = np.linspace(0, 180, num=20)  # Angles to test (in degrees)

W1 = 1
W2 = 100

# Load Weighted Feasibilty Grid
grid_filepath = os.getcwd() + "/GISFiles/weighted grid.gpkg"  # Update the path if necessary
gdf = gpd.read_file(grid_filepath)

# Load stations
stations_filepath = os.getcwd() + "/GISFiles/best stations.gpkg"
stations_gdf = gpd.read_file(stations_filepath)
stations2_filepath = os.getcwd() + "/GISFiles/best stations2.gpkg"
stations2_gdf = gpd.read_file(stations2_filepath)
total_stations = pd.concat([stations_gdf, stations2_gdf]).drop_duplicates()

# Load Corridor 1 Feasibilty Grid
cor_filepath = os.getcwd() + "/GISFiles/corridor.gpkg"  # Update the path if necessary
cor1 = gpd.read_file(cor_filepath)

# Load Corridor 2 Feasibilty Grid
cor2_filepath = os.getcwd() + "/GISFiles/corridor2.gpkg"  # Update the path if necessary
cor2 = gpd.read_file(cor2_filepath)

# Load raster file (assuming it's a GeoTIFF)
raster_filepath = os.getcwd() + "/GISFiles/output_population_raster2.tif"
raster = rasterio.open(raster_filepath)

# # Ensure GeoDataFrame has geometry
# if 'geometry' not in gdf.columns:
#     gdf['geometry'] = gpd.points_from_xy(gdf['col'], gdf['row'])


# Project to a suitable CRS (e.g., UTM) for accurate distance calculations
# if gdf.crs.is_geographic:  # Check if in lat/lon
gdf = gdf.to_crs(raster.crs)  # Convert to a projected CRS (Web Mercator)
cor1 = cor1.to_crs(raster.crs)
cor2 = cor2.to_crs(raster.crs)
total_stations = total_stations.to_crs(raster.crs)

# Grid search for the best rectangle
best_weight = -np.inf
best_rect = None
best_orientation = None

rect1=cor1.geometry.iloc[0]
rect2=cor2.geometry.iloc[0]
gdf['geometry'] = gdf.geometry.centroid
total_stations['geometry'] = total_stations.geometry.centroid
viable1=gdf[gdf["TOTAL WEIGHTED FEASIBILITY"] > 0.75].reset_index(drop=True)
# viable = viable1[viable1.geometry.within(rect1)]
# Spatial join: Find features in gdf1 that are within the polygon in gdf2
gdf_within_cor1 = gpd.sjoin(gdf, cor1, how="inner", predicate='within')

# Spatial join: Find features in gdf1 that are within the polygon in gdf3
gdf_within_cor2 = gpd.sjoin(gdf, cor2, how="inner", predicate='within')

viable = pd.concat([gdf_within_cor1, gdf_within_cor2]).drop_duplicates()
# print(viable)

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
    non_overlap_rect = placed_rect.difference(rect1).difference(rect2)
    # non_overlap_rect = non_overlap_rect.difference(rect2)
    
    if W1 > 0:
        # Spatial index filtering
        possible_matches_idx = list(spatial_index.intersection(non_overlap_rect.bounds))
        possible_matches = gdf.iloc[possible_matches_idx]

        # Calculate total weight within the rectangle
        total_weight = possible_matches[possible_matches.geometry.within(non_overlap_rect)]['TOTAL WEIGHTED FEASIBILITY'].sum(0)
    else:
        total_weight=0
    
    if W2 > 0:
        try:
            geoms = [non_overlap_rect.__geo_interface__]  # Convert Shapely geometry to GeoJSON-like format
            out_image, out_transform = mask(raster, geoms, crop=True)

            # Extract the pixel values
            # out_image is a 2D array where each element corresponds to a pixel's value
            pixel_values = out_image[0, :]  # Assuming single-band raster (first band)
            pixel_values = pixel_values[~np.isnan(pixel_values)]  # Remove NaNs

            # Get the sum of pixel values in the masked area (or other statistics if needed)
            total_pixel_value = pixel_values.sum() / 100
        except:
            total_pixel_value = 0
    else:
        total_pixel_value=0

    return total_weight*W1 + total_pixel_value*W2, placed_rect, angle


# Prepare inputs for parallel processing
args_list = [(point, angle) for point in total_stations.geometry for angle in orientations]

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
out_gdf = gpd.GeoDataFrame({'geometry': [best_rect]})

# Specify the CRS (e.g., EPSG:3857 for Web Mercator, change as per your data's CRS)
out_gdf.set_crs(raster.crs, allow_override=True, inplace=True)

output_fp = os.getcwd() + "/GISFiles/corridor3.gpkg"
# Save the GeoDataFrame as a GeoPackage (.gpkg)
out_gdf.to_file(output_fp, driver="GPKG")



# Optional: Visualize
import matplotlib.pyplot as plt

gdf.plot(color='blue', markersize=5, label='Points')
gpd.GeoSeries([best_rect]).plot(ax=plt.gca(), edgecolor='red', alpha=0.5, label='Best Rectangle')
plt.legend()
plt.show()

