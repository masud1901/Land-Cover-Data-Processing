import os
import logging
import geopandas as gpd
import rasterio
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.segmentation import felzenszwalb


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Ensure the result directory exists
os.makedirs("result", exist_ok=True)


def load_vector_data(filepath):
    """Load vector data (e.g., county boundaries)."""
    logging.info(f"Loading vector data from {filepath}")
    return gpd.read_file(filepath)


def load_raster_data(filepath):
    """Load raster data and return the data, transform, and metadata."""
    logging.info(f"Loading raster data from {filepath}")
    with rasterio.open(filepath) as src:
        data = src.read(1)  # Read the first band
        transform = src.transform
        metadata = src.meta
    return data, transform, metadata


def identify_field_boundaries(crop_data, scale=100, sigma=0.5, min_size=50):
    """Identify field boundaries using Felzenszwalb's efficient graph based segmentation."""
    logging.info("Identifying field boundaries using Felzenszwalb segmentation")
    # Normalize the data to 0-1 range
    crop_data_normalized = (crop_data - np.min(crop_data)) / (
        np.max(crop_data) - np.min(crop_data)
    )
    segments = felzenszwalb(
        crop_data_normalized, scale=scale, sigma=sigma, min_size=min_size
    )
    return segments


def clean_field_data(crop_data, segments):
    """Clean field data by applying majority filter within each segment."""
    logging.info("Cleaning field data")
    cleaned_data = np.zeros_like(crop_data)

    for segment_id in np.unique(segments):
        mask = segments == segment_id
        values, counts = np.unique(crop_data[mask], return_counts=True)
        most_common = values[np.argmax(counts)]
        cleaned_data[mask] = most_common

    return cleaned_data


def smooth_boundaries(cleaned_data, sigma=0.5):
    """Apply Gaussian smoothing to field boundaries."""
    logging.info("Smoothing field boundaries")
    return ndimage.gaussian_filter(cleaned_data, sigma=sigma, order=0).astype(
        cleaned_data.dtype
    )


def save_raster_data(data, transform, metadata, output_path):
    """Save raster data to file."""
    logging.info(f"Saving raster data to {output_path}")
    metadata.update(
        {
            "driver": "GTiff",
            "height": data.shape[0],
            "width": data.shape[1],
            "transform": transform,
        }
    )
    with rasterio.open(output_path, "w", **metadata) as dest:
        dest.write(data, 1)


# def visualize_results(original, segments, cleaned, smoothed, output_path):
#     """Visualize original, segmented, cleaned, and smoothed data."""
#     logging.info(f"Visualizing results and saving to {output_path}")
#     fig, axs = plt.subplots(2, 2, figsize=(20, 20))
#     axs = axs.ravel()

#     axs[0].imshow(original, cmap="nipy_spectral")
#     axs[0].set_title("Original Land Cover Data")

#     axs[1].imshow(segments, cmap="nipy_spectral")
#     axs[1].set_title("Segmented Fields")

#     axs[2].imshow(cleaned, cmap="nipy_spectral")
#     axs[2].set_title("Cleaned Land Cover Data")

#     axs[3].imshow(smoothed, cmap="nipy_spectral")
#     axs[3].set_title("Smoothed Land Cover Data")

#     for ax in axs:
#         ax.axis("off")

#     plt.tight_layout()
#     plt.savefig(output_path)
#     plt.close()


def visualize_results(original, segments, cleaned, smoothed, output_path):
    """Visualize original, segmented, cleaned, and smoothed data."""
    logging.info(f"Visualizing results and saving to {output_path}")
    fig, axs = plt.subplots(3, 2, figsize=(20, 30))
    axs = axs.ravel()

    # Original Data
    im = axs[0].imshow(original, cmap="nipy_spectral")
    axs[0].set_title("Original Land Cover Data")
    fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)

    # Segmented Data
    im = axs[1].imshow(segments, cmap="nipy_spectral")
    axs[1].set_title("Segmented Fields")
    fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)

    # Cleaned Data
    im = axs[2].imshow(cleaned, cmap="nipy_spectral")
    axs[2].set_title("Cleaned Land Cover Data")
    fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)

    # Smoothed Data
    im = axs[3].imshow(smoothed, cmap="nipy_spectral")
    axs[3].set_title("Smoothed Land Cover Data")
    fig.colorbar(im, ax=axs[3], fraction=0.046, pad=0.04)

    # Difference between Original and Cleaned Data
    diff_cleaned = np.abs(original - cleaned)
    im = axs[4].imshow(diff_cleaned, cmap="hot")
    axs[4].set_title("Difference: Original vs Cleaned")
    fig.colorbar(im, ax=axs[4], fraction=0.046, pad=0.04)

    # Difference between Cleaned and Smoothed Data
    diff_smoothed = np.abs(cleaned - smoothed)
    im = axs[5].imshow(diff_smoothed, cmap="hot")
    axs[5].set_title("Difference: Cleaned vs Smoothed")
    fig.colorbar(im, ax=axs[5], fraction=0.046, pad=0.04)

    for ax in axs:
        ax.axis("off")

    plt.suptitle("Land Cover Data Processing Steps", fontsize=24)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    plt.close()


def main():
    try:
        # Load and prepare data
        # vector_data = load_vector_data("data/tl_rd22_47075_edges.shp")
        raster_data, raster_transform, metadata = load_raster_data("data/clipped.TIF")

        # Process data
        segments = identify_field_boundaries(
            raster_data, scale=100, sigma=0.2, min_size=100
        )
        cleaned_data = clean_field_data(raster_data, segments)
        smoothed_data = smooth_boundaries(cleaned_data, sigma=0.2)

        # Save results
        save_raster_data(
            smoothed_data,
            raster_transform,
            metadata,
            "result/cleaned_smoothed_landcover.tif",
        )

        # Visualize results
        visualize_results(
            raster_data,
            segments,
            cleaned_data,
            smoothed_data,
            "result/landcover_processing_comparison.png",
        )

        logging.info(f"Processed data saved with CRS: {metadata['crs']}")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        logging.exception("Exception details:")


if __name__ == "__main__":
    main()
