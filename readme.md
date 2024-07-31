# Land Cover Data Processing for Haywood County, Tennessee

This repository contains Python scripts for processing, analyzing, and assessing the accuracy of land cover raster data for Haywood County, Tennessee, using 2023 data. The main script performs data processing tasks, while an additional script (`accuracy.py`) calculates and reports accuracy metrics.

## Background

Using spatial datasets found [here](https://croplandcros.scinet.usda.gov/), the objective is to process remote-sensed and classified field data. Landsat data is used to categorize land cover into specific crop categories. However, the categorization is unreliable in several areas. The main goal is to clean up these errors and create a readable spatial dataset.

### Area of Interest

The area of interest for this project is Haywood County, Tennessee.
## Requirements

To run the scripts, you need the following libraries:

- os
- logging
- geopandas
- rasterio
- numpy
- scipy
- matplotlib
- scikit-image
- scikit-learn
- pandas
- seaborn

You can install these libraries using `pip`:

```sh
pip install geopandas rasterio numpy scipy matplotlib scikit-image scikit-learn pandas seaborn
```

## Usage

### Data Processing Script (analysis.py)

## Usage

### 1. Load Vector and Raster Data

The script starts by loading vector data (e.g., county boundaries) and raster data (land cover data). The vector data is loaded using `geopandas` and the raster data is loaded using `rasterio`.

### 2. Identify Field Boundaries

Field boundaries are identified using Felzenszwalb's efficient graph-based image segmentation method from the `skimage` library. The raster data is normalized to a 0-1 range before applying the segmentation.

### 3. Clean Field Data

The field data is cleaned by applying a majority filter within each segment. This means that within each segment identified in the previous step, the most common value is assigned to all pixels in that segment.

### 4. Smooth Boundaries

Gaussian smoothing is applied to the cleaned data to smooth the field boundaries. This is done using the `gaussian_filter` function from `scipy.ndimage`.

### 5. Save Processed Data

The processed raster data is saved to a file using `rasterio`.

### 6. Visualize Results

The script visualizes the original, segmented, cleaned, and smoothed data to explain the changes at each step. The visual comparison is saved as an image file.

## Running the Script

You can run the script using the command:

```sh
python analysis.py
```


### Accuracy Assessment Script (accuracy.py)

The `accuracy.py` script calculates comprehensive accuracy metrics comparing the processed data with the original data. It performs the following tasks:

1. **Load Raster Data**: Loads both the original and processed raster data.
2. **Calculate Accuracy Metrics**: Computes various accuracy metrics including:
   - Overall accuracy
   - Kappa coefficient
   - Per-class precision, recall, and F1-score
   - Confusion matrix
3. **Visualize Confusion Matrix**: Generates a heatmap visualization of the confusion matrix.
4. **Save Results**: Saves the accuracy metrics to a CSV file and the confusion matrix plot as an image.

#### Running the Accuracy Script

You can run the accuracy script using the command:

```sh
python accuracy.py
```

Make sure your data files (`clipped.TIF` and `cleaned_smoothed_landcover.tif`) are in the correct directories as specified in the script.

#### Accuracy Output

The script generates two main outputs:

1. A CSV file (`accuracy_metrics.csv`) containing all calculated accuracy metrics.
2. A PNG image (`confusion_matrix.png`) visualizing the confusion matrix.

These files are saved in the `result` directory.

## Visualization

To make the visualization more effective, the following enhancements are included:
- **Colorbars**: Added to each subplot to understand the range of values.
- **Titles**: Each subplot has a descriptive title, and an overall title is added to the figure.
- **Difference Images**: Two additional subplots show the differences between the original and cleaned data, and between the cleaned and smoothed data, using a "hot" colormap to highlight differences.

### Example Visualization

Below is an example of the visualized results:

![Land Cover Data Processing Steps](result/landcover_processing_comparison.png)


Make sure your data files (`tl_rd22_47075_edges.shp` and `clipped.TIF`) are in the `data` directory.
## Conclusion

This repository provides a complete workflow for processing land cover raster data, from loading and segmentation to cleaning, smoothing, and visualization. Additionally, it includes a comprehensive accuracy assessment, allowing for a quantitative evaluation of the processing results. The accuracy metrics and confusion matrix visualization provide valuable insights into the performance of the land cover classification and the effects of the processing steps.