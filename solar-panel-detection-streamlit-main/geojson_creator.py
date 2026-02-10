import os
import numpy as np
from osgeo import gdal, ogr, osr
from typing import Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeoJSONCreator:
    """Modern GeoJSON creation utility for building polygons"""
    
    def __init__(self, output_dir: str = './geojson'):
        self.output_dir = output_dir
        self._ensure_output_directory()
    
    def _ensure_output_directory(self):
        """Create output directory if it doesn't exist"""
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")
    
    def create_geojson(self, fn: str, cluster: np.ndarray, geom: Tuple, 
                     proj: str, layer_name: str = "BuildingID", 
                     connection_type: int = 8) -> str:
        """
        Create GeoJSON from cluster array
        
        Args:
            fn: Filename prefix
            cluster: 2D numpy array with cluster values
            geom: Geotransform tuple (6 elements)
            proj: Projection string (WKT or EPSG)
            layer_name: Name for the output layer
            connection_type: Pixel connectivity (4 or 8)
        
        Returns:
            Path to created GeoJSON file
        """
        
        try:
            # Validate inputs
            self._validate_inputs(cluster, geom, proj)
            
            # Create in-memory raster
            src_ds = self._create_memory_raster(cluster, geom, proj)
            
            # Create output GeoJSON
            output_path = self._create_vector_output(fn, layer_name)
            
            # Perform polygonization
            self._polygonize_raster(src_ds, output_path, layer_name, connection_type)
            
            # Clean up
            src_ds = None
            
            logger.info(f"✅ Successfully created GeoJSON: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Error creating GeoJSON: {e}")
            raise
    
    def _validate_inputs(self, cluster: np.ndarray, geom: Tuple, proj: str):
        """Validate input parameters"""
        if not isinstance(cluster, np.ndarray):
            raise ValueError("Cluster must be a numpy array")
        
        if len(cluster.shape) != 2:
            raise ValueError("Cluster must be a 2D array")
        
        if len(geom) != 6:
            raise ValueError("Geotransform must have 6 elements")
        
        if not proj:
            raise ValueError("Projection cannot be empty")
    
    def _create_memory_raster(self, cluster: np.ndarray, geom: Tuple, proj: str) -> gdal.Dataset:
        """Create in-memory raster dataset"""
        
        # Create memory driver
        memdrv = gdal.GetDriverByName('MEM')
        if memdrv is None:
            raise RuntimeError("Failed to create MEM driver")
        
        # Create in-memory dataset
        rows, cols = cluster.shape
        src_ds = memdrv.Create('', cols, rows, 1, gdal.GDT_Byte)
        
        if src_ds is None:
            raise RuntimeError("Failed to create in-memory dataset")
        
        # Set geotransform and projection
        src_ds.SetGeoTransform(geom)
        src_ds.SetProjection(proj)
        
        # Write array data
        band = src_ds.GetRasterBand(1)
        band.WriteArray(cluster)
        band.SetNoDataValue(0)  # Set no data value
        
        return src_ds
    
    def _create_vector_output(self, fn: str, layer_name: str) -> str:
        """Create vector output dataset"""
        
        # Generate output filename
        output_path = os.path.join(self.output_dir, f"{fn}{layer_name}.geojson")
        
        # Delete existing file if it exists
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                logger.info(f"Removed existing file: {output_path}")
            except Exception as e:
                logger.warning(f"Could not remove existing file: {e}")
        
        # Create GeoJSON driver
        drv = ogr.GetDriverByName("GeoJSON")
        if drv is None:
            raise RuntimeError("Failed to create GeoJSON driver")
        
        # Create data source
        dst_ds = drv.CreateDataSource(output_path)
        if dst_ds is None:
            raise RuntimeError(f"Failed to create output file: {output_path}")
        
        # Create layer
        srs = None  # Can be set to a spatial reference if needed
        dst_layer = dst_ds.CreateLayer(layer_name, srs=srs, geom_type=ogr.wkbPolygon)
        
        if dst_layer is None:
            raise RuntimeError(f"Failed to create layer: {layer_name}")
        
        # Create field for DN values
        field_defn = ogr.FieldDefn("DN", ogr.OFTInteger)
        dst_layer.CreateField(field_defn)
        
        # Clean up
        dst_ds = None
        
        return output_path
    
    def _polygonize_raster(self, src_ds: gdal.Dataset, output_path: str, 
                         layer_name: str, connection_type: int = 8):
        """Perform polygonization of raster to vector"""
        
        # Re-open the dataset for polygonization
        dst_ds = ogr.Open(output_path, 1)  # Update mode
        if dst_ds is None:
            raise RuntimeError(f"Failed to open output file: {output_path}")
        
        dst_layer = dst_ds.GetLayerByName(layer_name)
        if dst_layer is None:
            raise RuntimeError(f"Failed to get layer: {layer_name}")
        
        # Get raster band
        band = src_ds.GetRasterBand(1)
        
        # Set polygonization options
        options = [f'{connection_type}CONNECTED={connection_type}']
        
        # Perform polygonization
        result = gdal.Polygonize(
            band,           # Source band
            None,           # Mask band (None for no mask)
            dst_layer,      # Output layer
            0,              # Field index (0 for first field)
            options,         # Options
            callback=None     # Progress callback
        )
        
        if result != 0:
            raise RuntimeError("Polygonization failed")
        
        # Clean up
        dst_ds = None
        
        logger.info(f"Polygonization completed with {connection_type}-connected pixels")
    
    def batch_create_geojson(self, data_list: list) -> list:
        """Create multiple GeoJSON files from a list of data"""
        
        results = []
        for i, data in enumerate(data_list):
            try:
                fn = data.get('filename', f'cluster_{i}')
                cluster = data['cluster']
                geom = data['geotransform']
                proj = data['projection']
                layer_name = data.get('layer_name', 'BuildingID')
                
                output_path = self.create_geojson(fn, cluster, geom, proj, layer_name)
                results.append({'filename': fn, 'output_path': output_path, 'success': True})
                
            except Exception as e:
                logger.error(f"Failed to process {fn}: {e}")
                results.append({'filename': fn, 'output_path': None, 'success': False, 'error': str(e)})
        
        return results
    
    def get_geojson_info(self, geojson_path: str) -> dict:
        """Get information about created GeoJSON file"""
        
        if not os.path.exists(geojson_path):
            raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")
        
        # Open the GeoJSON file
        ds = ogr.Open(geojson_path)
        if ds is None:
            raise RuntimeError(f"Failed to open GeoJSON: {geojson_path}")
        
        layer = ds.GetLayer()
        feature_count = layer.GetFeatureCount()
        
        # Get extent
        extent = layer.GetExtent()
        
        # Get field information
        layer_def = layer.GetLayerDefn()
        field_count = layer_def.GetFieldCount()
        fields = []
        for i in range(field_count):
            field_def = layer_def.GetFieldDefn(i)
            fields.append({
                'name': field_def.GetName(),
                'type': field_def.GetTypeName(),
                'width': field_def.GetWidth(),
                'precision': field_def.GetPrecision()
            })
        
        # Clean up
        ds = None
        
        return {
            'file_path': geojson_path,
            'feature_count': feature_count,
            'extent': {
                'min_x': extent[0],
                'max_x': extent[1],
                'min_y': extent[2],
                'max_y': extent[3]
            },
            'fields': fields,
            'file_size': os.path.getsize(geojson_path)
        }

# Convenience function for backward compatibility
def CreateGeoJSON(fn: str, cluster: np.ndarray, geom: Tuple, proj: str, 
                 output_dir: str = './geojson', layer_name: str = "BuildingID") -> str:
    """
    Backward compatible function
    
    Args:
        fn: Filename prefix
        cluster: 2D numpy array with cluster values
        geom: Geotransform tuple (6 elements)
        proj: Projection string
        output_dir: Output directory for GeoJSON files
        layer_name: Name for the output layer
    
    Returns:
        Path to created GeoJSON file
    """
    
    creator = GeoJSONCreator(output_dir)
    return creator.create_geojson(fn, cluster, geom, proj, layer_name)

# Example usage
if __name__ == "__main__":
    # Example data
    cluster = np.array([
        [0, 0, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 1, 1],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 0, 0]
    ], dtype=np.uint8)
    
    # Example geotransform (top-left x, pixel width, rotation, top-left y, rotation, pixel height)
    geom = (440720.0, 10.0, 0.0, 3751320.0, 0.0, -10.0)
    
    # Example projection (WGS84 UTM Zone 10N)
    proj = 'PROJCS["WGS 84 / UTM zone 10N",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","4326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-123],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","32610"]]'
    
    print("🚀 Creating GeoJSON example...")
    
    try:
        # Create GeoJSON
        output_path = CreateGeoJSON("example", cluster, geom, proj)
        
        # Get info about created file
        creator = GeoJSONCreator()
        info = creator.get_geojson_info(output_path)
        
        print(f"✅ GeoJSON created successfully!")
        print(f"📁 Output path: {info['file_path']}")
        print(f"📊 Feature count: {info['feature_count']}")
        print(f"📏 Extent: {info['extent']}")
        print(f"📄 File size: {info['file_size']} bytes")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
