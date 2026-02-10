// Google Earth Engine - Enhanced Sentinel-2 Chip Export
// Features: Cloud filtering, quality metrics, AOI support, metadata export

// ========================================
// CONFIGURATION PARAMETERS
// ========================================

// Date range (adjust as needed)
var startDate = '2025-11-11';      // ~90 days before Feb 9, 2026
var endDate   = '2026-02-09';      // today's date
var maxCloud  = 10;                // percent cloudiness max
var chipSize  = 256;               // pixels (square chip)
var pixelSize = 10;                // meters per pixel for Sentinel-2
var numChips  = 200;              // number of chips to export

// Export settings
var exportFolder = 'S2_chips';     // Drive folder name
var fileFormat = 'GeoTIFF';        // 'GeoTIFF' or 'TFRecord'
var includeMetadata = true;         // export metadata CSV

// Quality thresholds
var minNDVI = 0.2;               // minimum NDVI for vegetation areas
var maxNDVI = 0.8;               // maximum NDVI (avoid water bodies)
var minDataQuality = 0.7;          // minimum data quality score

// ========================================
// AOI CONFIGURATION (Optional)
// ========================================

// Option 1: Specific region (uncomment to use)
// var regionAOI = ee.Geometry.Rectangle([72.5, 18.4, 73.0, 19.0]); // Mumbai bbox
// var regionAOI = ee.Geometry.Rectangle([-74.3, 40.5, -73.7, 40.9]); // NYC bbox

// Option 2: Global sampling with exclusion zones
var globalBounds = ee.Geometry.BBox(-180, -60, 180, 80); // exclude poles
var excludePoles = true;

// ========================================
// DATA COLLECTION AND PROCESSING
// ========================================

// Build Sentinel-2 collection
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
         .filterDate(startDate, endDate)
         .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', maxCloud))
         .filter(ee.Filter.eq('MGRS_TILE', 'default')) // quality tiles
         .map(function(image) {
           // Add cloud mask
           var cloudMask = image.select(['QA60']).eq(0);
           return image.updateMask(cloudMask);
         });

// Apply AOI filter if specified
if (typeof regionAOI !== 'undefined') {
  s2 = s2.filterBounds(regionAOI);
  print('AOI Applied:', regionAOI.getInfo());
}

// Calculate quality metrics
function addQualityMetrics(image) {
  // Calculate NDVI
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  
  // Calculate data quality score
  var quality = image.select(['QA60']).eq(0).multiply(ndvi.gte(minNDVI).and(ndvi.lte(maxNDVI)));
  
  return image.addBands(ndvi).addBands(quality.rename('quality'));
}

// Apply quality metrics
s2 = s2.map(addQualityMetrics);

// Create median composite with quality weighting
var median = s2.median();
var quality = s2.select('quality').median();

// Visualization parameters
var visParams = {
  bands: ['B4', 'B3', 'B2'],
  min: 0,
  max: 3000,
  gamma: 1.4
};

// Create visualization layer
var rgb = median.visualize(visParams);

// ========================================
// SAMPLING STRATEGY
// ========================================

// Create quality mask for sampling
var validMask = quality.gte(minDataQuality).and(
  median.select('NDVI').gte(minNDVI).and(
    median.select('NDVI').lte(maxNDVI)
  )
);

// Generate stratified sampling grid
var samplingScale = pixelSize * chipSize * 2; // spacing between chips
var randomSeed = 42;

// Create sampling grid
var grid = ee.Image.random(randomSeed)
  .reproject('EPSG:3857', null, samplingScale)
  .multiply(1000)
  .floor()
  .mod(100);

// Generate sample points
var points = grid.stratifiedSample({
  numPoints: numChips * 3,  // oversample for quality filtering
  classBand: 'random',
  region: typeof regionAOI !== 'undefined' ? regionAOI : globalBounds,
  scale: samplingScale,
  geometries: true,
  seed: randomSeed,
  classValues: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  classPoints: [numChips * 3 / 10, numChips * 3 / 10, numChips * 3 / 10, 
                numChips * 3 / 10, numChips * 3 / 10, numChips * 3 / 10,
                numChips * 3 / 10, numChips * 3 / 10, numChips * 3 / 10, numChips * 3 / 10]
});

// Filter points by quality and valid data
var validPoints = points.filter(ee.Filter.and(
  ee.Filter.intersects('.geo', validMask.geometry()),
  ee.Filter.gte('NDVI', minNDVI),
  ee.Filter.lte('NDVI', maxNDVI)
)).limit(numChips);

// ========================================
// EXPORT FUNCTION
// ========================================

function exportChip(point, index) {
  var chipIndex = ee.Number(index).format();
  var geom = point.geometry().buffer(pixelSize * chipSize / 2).bounds();
  
  // Extract chip with all bands
  var chip = median.clip(geom);
  
  // Add quality bands to export
  var qualityChip = quality.clip(geom);
  var exportImage = chip.addBands(qualityChip);
  
  // Create filename
  var fileName = ee.String('s2_chip_').cat(chipIndex);
  
  // Export to Drive
  Export.image.toDrive({
    image: exportImage,
    description: fileName.getInfo(),
    folder: exportFolder,
    fileNamePrefix: fileName.getInfo(),
    region: geom,
    scale: pixelSize,
    maxPixels: 1e9,
    fileFormat: fileFormat,
    skipEmptyTiles: true
  });
  
  return point.set('exported', true);
}

// ========================================
// METADATA COLLECTION
// ========================================

function createMetadata(point, index) {
  var chipIndex = ee.Number(index).format();
  var geom = point.geometry();
  
  // Extract metadata for this chip
  var chip = median.clip(geom);
  var qualityChip = quality.clip(geom);
  
  // Calculate statistics
  var stats = chip.reduceRegion({
    reducer: ee.Reducer.mean().combine({
      reducer2: ee.Reducer.stdDev(),
      sharedInputs: true
    }),
    geometry: geom,
    scale: pixelSize,
    maxPixels: 1e6
  });
  
  // Add point information
  var metadata = ee.Feature(geom, {
    chip_id: chipIndex.getInfo(),
    longitude: geom.coordinates().get(0),
    latitude: geom.coordinates().get(1),
    ndvi_mean: stats.get('NDVI_mean'),
    ndvi_std: stats.get('NDVI_stdDev'),
    quality_mean: stats.get('quality_mean'),
    cloud_percentage: ee.Image(ee.ImageCollection('COPERNICUS/S2')
      .filterDate(startDate, endDate)
      .filterBounds(geom)
      .select(['CLOUDY_PIXEL_PERCENTAGE')
      .mean()
      .clip(geom)
      .reduceRegion({
        reducer: ee.Reducer.mean(),
        geometry: geom,
        scale: pixelSize,
        maxPixels: 1e6
      }).get('CLOUDY_PIXEL_PERCENTAGE')
  });
  
  return metadata;
}

// ========================================
// EXECUTE EXPORTS
// ========================================

// Get valid points as list
var validPointsList = validPoints.toList(numChips);

// Map over points and export
var exportList = validPointsList.map(function(point, index) {
  return exportChip(ee.Feature(point), index);
});

// Export metadata if enabled
if (includeMetadata) {
  var metadataList = validPointsList.map(function(point, index) {
    return createMetadata(ee.Feature(point), index);
  });
  
  var metadataCollection = ee.FeatureCollection(metadataList);
  
  Export.table.toDrive({
    collection: metadataCollection,
    description: 'S2_chip_metadata',
    folder: exportFolder,
    fileNamePrefix: 'S2_chip_metadata',
    fileFormat: 'CSV'
  });
}

// ========================================
// VISUALIZATION
// ========================================

// Add layers to map
Map.addLayer(rgb, {}, 'Sentinel-2 RGB Composite');
Map.addLayer(quality, {min: 0, max: 1, palette: ['red', 'yellow', 'green']}, 'Data Quality');
Map.addLayer(validPoints, {color: 'FF0000'}, 'Sample Points');

// Center map on first point or AOI
if (typeof regionAOI !== 'undefined') {
  Map.centerObject(regionAOI, 10);
} else {
  var firstPoint = ee.Feature(validPointsList.get(0));
  Map.centerObject(firstPoint, 10);
}

// ========================================
// PRINT INFORMATION
// ========================================

print('=== SENTINEL-2 CHIP EXPORT ===');
print('Start Date:', startDate);
print('End Date:', endDate);
print('Max Cloud %:', maxCloud);
print('Chip Size:', chipSize + 'x' + chipSize + ' pixels');
print('Pixel Size:', pixelSize + ' meters');
print('Number of Chips:', numChips);
print('Export Folder:', exportFolder);
print('Valid Points Found:', validPoints.size().getInfo());
print('Total Collection Size:', s2.size().getInfo());

if (typeof regionAOI !== 'undefined') {
  print('AOI:', regionAOI.getInfo());
}

// ========================================
// QUALITY CONTROL
// ========================================

// Function to check export status
function checkExports() {
  var taskList = Export.tasklist();
  print('Active Export Tasks:', taskList);
}

// Run quality check (uncomment to use)
// checkExports();

print('=== EXPORT TASKS CREATED ===');
print('Check the Tasks tab in Earth Engine to monitor progress');
print('Chips will be exported to your Google Drive folder:', exportFolder);
