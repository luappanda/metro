import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import box
from tqdm import tqdm 
import os

# Load the GeoDataFrame
population_filepath = os.getcwd() + "/GISFiles/block population.gpkg"
gdf = gpd.read_file(population_filepath)

# Reproject to UTM (e.g., UTM zone 33N)
gdf = gdf.to_crs("EPSG:26915")

# Define resolution in meters (e.g., 100 meters per pixel)
resolution = 150

# Get the bounding box and calculate the raster dimensions
bounds = gdf.total_bounds
width = int((bounds[2] - bounds[0]) / resolution)
height = int((bounds[3] - bounds[1]) / resolution)

# Create the transform for the raster
transform = from_origin(bounds[0], bounds[3], resolution, resolution)

# Create an empty raster (initialize with zeros)
raster = np.zeros((height, width), dtype=np.float32)

total_population = gdf['POP20'].sum()

## Function to rasterize each block and assign its population to the covered pixels
def rasterize_block(row):
    geometry = row['geometry']
    population = row['POP20']  # Replace with the actual population field
    
    # Perform rasterization and assign the entire block population to the covered pixels
    mask = rasterize([(geometry, 1)], out_shape=(height, width), transform=transform, fill=0, dtype='float32')
    
    # Calculate the number of pixels covered by the block
    num_pixels = np.sum(mask)  # Count of pixels where the block geometry intersects
    
    if num_pixels == 0:
        return None  # Skip if block doesn't cover any pixels
    
    # Distribute the population evenly across the covered pixels
    population_per_pixel = population / num_pixels
    raster_block = mask * population_per_pixel  # Assign population per pixel
    
    return raster_block

# Add rasterized blocks iteratively to avoid memory overflow
for _, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Rasterizing census blocks"):
    block_raster = rasterize_block(row)
    if block_raster is not None:
        raster = np.add(raster, block_raster)  # Add the block raster to the final raster


# Print the sum of the population in the grid (sum of the population field from the GeoDataFrame)
print(f"Total population in the grid (sum of all blocks): {total_population}")

# Print the sum of the rasterized pixel values
raster_sum = np.sum(raster)
print(f"Sum of rasterized pixel values (final raster): {raster_sum}")

output_filepath = os.getcwd() + "/GISFiles/output_population_raster2.tif"
# Save the raster to a file
with rasterio.open(output_filepath, 'w', driver='GTiff', height=height, width=width, count=1, dtype='float32', crs=gdf.crs, transform=rasterio.transform.from_bounds(*bounds, width, height)) as dst:
    dst.write(raster, 1)
