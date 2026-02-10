import json
from shapely.geometry import Polygon, shape, Point
import gdal
import numpy as np
import sys
import os

def Pixel2World(geoMatrix, i, j):
    """Convert pixel coordinates to world coordinates"""
    ulX = geoMatrix[0]
    ulY = geoMatrix[3]
    xDist = geoMatrix[1]
    yDist = geoMatrix[5]
    rtnX = geoMatrix[2]
    rtnY = geoMatrix[4]
    return (1.0 * i * xDist + ulX, -1.0 * j * yDist + ulY)

def main():
    if len(sys.argv) < 3:
        print("Usage: python distance_calculation.py <filename> <path>")
        sys.exit(1)
    
    fn = sys.argv[1]
    path = sys.argv[2]
    
    # Open raster files
    ds8_path = os.path.join(path, '8band', f'8band_{fn}.tif')
    ds3_path = os.path.join(path, '3band', f'3band_{fn}.tif')
    
    ds8 = gdal.Open(ds8_path)
    ds3 = gdal.Open(ds3_path)
    
    if ds8 is None or ds3 is None:
        print("Error: Could not open raster files")
        sys.exit(1)
    
    geoTrans = ds8.GetGeoTransform()
    
    # Load GeoJSON data
    geojson_path = os.path.join(path, 'vectorData', 'geoJson', f'{fn}_Geo.geojson')
    
    if not os.path.exists(geojson_path):
        print(f"Error: GeoJSON file not found at {geojson_path}")
        sys.exit(1)
    
    with open(geojson_path, 'r') as f:
        js = json.load(f)
    
    # Initialize distance array
    dist = np.zeros((ds8.RasterXSize, ds8.RasterYSize))
    
    print(f"Processing {ds8.RasterXSize}x{ds8.RasterYSize} pixels...")
    
    # Process each pixel
    for i in range(ds8.RasterXSize):
        for j in range(ds8.RasterYSize):
            point = Point(Pixel2World(geoTrans, i, j))
            pd = -100000.0
            
            for feature in js['features']:
                polygon = shape(feature['geometry'])
                newpd = point.distance(polygon.boundary)
                
                if not polygon.contains(point):
                    newpd = -1.0 * newpd
                
                if newpd > pd:
                    pd = newpd
            
            dist[i, j] = pd
        
        # Progress indicator
        if i % 100 == 0:
            print(f"Processed {i}/{ds8.RasterXSize} columns...")
    
    # Save results
    output_dir = os.path.join(path, 'CosmiQ_distance')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f'{fn}.distance.npy')
    np.save(output_path, dist)
    
    print(f"Distance calculation complete! Results saved to: {output_path}")

if __name__ == "__main__":
    main()
