import os
import numpy as np
from osgeo import gdal, ogr, osr
from typing import Optional, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeoJSONFixer:
    """Modern GeoJSON fixing utility for buffering and filtering"""
    
    def __init__(self, input_dir: str = './geojson.full', output_dir: str = './geojson.full'):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.buffer_dir = os.path.join(output_dir, 'buffer')
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories"""
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.buffer_dir, exist_ok=True)
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Buffer directory: {self.buffer_dir}")
    
    def fix_geojson(self, fn: str, buffer_distance: float = 0.0, 
                   layer_name: str = "BuildingID", 
                   remove_zero_dn: bool = True) -> str:
        """
        Fix GeoJSON by buffering and filtering features
        
        Args:
            fn: Filename prefix
            buffer_distance: Buffer distance in map units (0.0 = no buffer)
            layer_name: Name of the layer to process
            remove_zero_dn: Whether to remove features with DN = 0
        
        Returns:
            Path to fixed GeoJSON file
        """
        
        try:
            # Input file path
            input_path = os.path.join(self.input_dir, f"{fn}{layer_name}.geojson")
            
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            # Load input GeoJSON
            src_ds = ogr.Open(input_path)
            if src_ds is None:
                raise RuntimeError(f"Failed to open input file: {input_path}")
            
            src_layer = src_ds.GetLayer()
            if src_layer is None:
                raise RuntimeError(f"Failed to get layer from: {input_path}")
            
            logger.info(f"Processing {src_layer.GetFeatureCount()} features from {input_path}")
            
            # Create output GeoJSON
            output_path = os.path.join(self.buffer_dir, f"buffer{fn}{layer_name}.geojson")
            
            # Remove existing output file
            if os.path.exists(output_path):
                os.remove(output_path)
                logger.info(f"Removed existing file: {output_path}")
            
            # Create output dataset
            dst_ds = self._create_output_dataset(output_path, layer_name)
            dst_layer = dst_ds.GetLayer()
            
            # Process features
            processed_count = 0
            skipped_count = 0
            
            for feature in src_layer:
                try:
                    clusternumber = feature.GetField("DN")
                    
                    # Skip features with DN = 0 if requested
                    if remove_zero_dn and clusternumber == 0:
                        skipped_count += 1
                        continue
                    
                    # Get geometry and apply buffer
                    geom = feature.GetGeometryRef()
                    if geom is None:
                        logger.warning(f"Feature {feature.GetFID()} has no geometry")
                        continue
                    
                    # Apply buffer if needed
                    if buffer_distance != 0.0:
                        buffered_geom = geom.Buffer(buffer_distance)
                        if buffered_geom is None or buffered_geom.IsEmpty():
                            logger.warning(f"Buffer failed for feature {feature.GetFID()}")
                            continue
                        geom = buffered_geom
                    
                    # Create new feature with buffered geometry
                    new_feature = ogr.Feature(dst_layer.GetLayerDefn())
                    new_feature.SetGeometry(geom)
                    new_feature.SetField("DN", clusternumber)
                    
                    # Create feature in output layer
                    dst_layer.CreateFeature(new_feature)
                    processed_count += 1
                    
                    # Clean up
                    new_feature = None
                    
                except Exception as e:
                    logger.error(f"Error processing feature {feature.GetFID()}: {e}")
                    continue
            
            # Clean up
            src_ds = None
            dst_ds = None
            
            logger.info(f"✅ Processed {processed_count} features, skipped {skipped_count}")
            logger.info(f"✅ Fixed GeoJSON saved: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Error fixing GeoJSON: {e}")
            raise
    
    def _create_output_dataset(self, output_path: str, layer_name: str) -> ogr.DataSource:
        """Create output GeoJSON dataset"""
        
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
        
        return dst_ds
    
    def batch_fix_geojson(self, filename_list: List[str], buffer_distance: float = 0.0,
                        layer_name: str = "BuildingID", remove_zero_dn: bool = True) -> List[dict]:
        """Fix multiple GeoJSON files"""
        
        results = []
        
        for fn in filename_list:
            try:
                output_path = self.fix_geojson(
                    fn=fn,
                    buffer_distance=buffer_distance,
                    layer_name=layer_name,
                    remove_zero_dn=remove_zero_dn
                )
                
                results.append({
                    'filename': fn,
                    'output_path': output_path,
                    'success': True,
                    'error': None
                })
                
            except Exception as e:
                logger.error(f"Failed to process {fn}: {e}")
                results.append({
                    'filename': fn,
                    'output_path': None,
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    def validate_geojson(self, geojson_path: str) -> dict:
        """Validate a GeoJSON file and return statistics"""
        
        if not os.path.exists(geojson_path):
            raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")
        
        # Open the GeoJSON file
        ds = ogr.Open(geojson_path)
        if ds is None:
            raise RuntimeError(f"Failed to open GeoJSON: {geojson_path}")
        
        layer = ds.GetLayer()
        feature_count = layer.GetFeatureCount()
        
        # Statistics
        dn_counts = {}
        valid_geometries = 0
        invalid_geometries = 0
        
        for feature in layer:
            dn = feature.GetField("DN")
            if dn is not None:
                dn_counts[dn] = dn_counts.get(dn, 0) + 1
            
            geom = feature.GetGeometryRef()
            if geom is not None and not geom.IsEmpty():
                valid_geometries += 1
            else:
                invalid_geometries += 1
        
        # Get extent
        extent = layer.GetExtent()
        
        # Clean up
        ds = None
        
        return {
            'file_path': geojson_path,
            'feature_count': feature_count,
            'valid_geometries': valid_geometries,
            'invalid_geometries': invalid_geometries,
            'dn_distribution': dn_counts,
            'extent': {
                'min_x': extent[0],
                'max_x': extent[1],
                'min_y': extent[2],
                'max_y': extent[3]
            },
            'file_size': os.path.getsize(geojson_path)
        }
    
    def merge_geojson_files(self, filename_list: List[str], output_filename: str,
                          layer_name: str = "BuildingID") -> str:
        """Merge multiple GeoJSON files into one"""
        
        output_path = os.path.join(self.output_dir, f"{output_filename}{layer_name}.geojson")
        
        # Remove existing output file
        if os.path.exists(output_path):
            os.remove(output_path)
        
        # Create output dataset
        dst_ds = self._create_output_dataset(output_path, layer_name)
        dst_layer = dst_ds.GetLayer()
        
        total_features = 0
        
        for fn in filename_list:
            input_path = os.path.join(self.buffer_dir, f"buffer{fn}{layer_name}.geojson")
            
            if not os.path.exists(input_path):
                logger.warning(f"Skipping missing file: {input_path}")
                continue
            
            # Open input file
            src_ds = ogr.Open(input_path)
            if src_ds is None:
                logger.warning(f"Failed to open: {input_path}")
                continue
            
            src_layer = src_ds.GetLayer()
            
            # Copy features
            for feature in src_layer:
                dst_layer.CreateFeature(feature)
                total_features += 1
            
            # Clean up
            src_ds = None
        
        # Clean up
        dst_ds = None
        
        logger.info(f"✅ Merged {total_features} features into {output_path}")
        return output_path

# Convenience function for backward compatibility
def FixGeoJSON(fn: str, buffer_distance: float = 0.0, input_dir: str = './geojson.full',
                output_dir: str = './geojson.full', layer_name: str = "BuildingID") -> str:
    """
    Backward compatible function
    
    Args:
        fn: Filename prefix
        buffer_distance: Buffer distance in map units
        input_dir: Input directory containing GeoJSON files
        output_dir: Output directory for fixed files
        layer_name: Name of the layer to process
    
    Returns:
        Path to fixed GeoJSON file
    """
    
    fixer = GeoJSONFixer(input_dir, output_dir)
    return fixer.fix_geojson(fn, buffer_distance, layer_name, remove_zero_dn=True)

# Example usage
if __name__ == "__main__":
    # Example: Create a test GeoJSON first
    from geojson_creator import CreateGeoJSON
    
    # Create test data
    cluster = np.array([
        [0, 0, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 1, 1],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 0, 0]
    ], dtype=np.uint8)
    
    geom = (440720.0, 10.0, 0.0, 3751320.0, 0.0, -10.0)
    proj = 'EPSG:32610'  # Simplified projection
    
    print("🚀 Creating test GeoJSON...")
    
    try:
        # Create original GeoJSON
        original_path = CreateGeoJSON("test", cluster, geom, proj, input_dir='./geojson.full')
        print(f"✅ Created original: {original_path}")
        
        # Fix the GeoJSON
        fixed_path = FixGeoJSON("test", buffer_distance=5.0)
        print(f"✅ Fixed GeoJSON: {fixed_path}")
        
        # Validate the fixed file
        fixer = GeoJSONFixer()
        validation = fixer.validate_geojson(fixed_path)
        
        print(f"\n📊 Validation Results:")
        print(f"  Features: {validation['feature_count']}")
        print(f"  Valid geometries: {validation['valid_geometries']}")
        print(f"  DN distribution: {validation['dn_distribution']}")
        print(f"  File size: {validation['file_size']} bytes")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
