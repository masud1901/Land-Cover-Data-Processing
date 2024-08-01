import logging
import rasterio
from typing import Tuple
import numpy as np


def load_raster_data(filepath: str) -> Tuple[np.ndarray, dict]:
    """
    Load raster data from a file.

    :param filepath: Path to the raster file.
    :return: Tuple containing the raster data and metadata.
    """
    logging.info(f"Loading raster data from {filepath}")
    with rasterio.open(filepath) as src:
        data = src.read(1)
        metadata = src.meta
    return data, metadata


def safe_metric_calculation(
    metric_func, true_labels: np.ndarray, pred_labels: np.ndarray
) -> float:
    """
    Safely calculate a metric, handling any exceptions.

    :param metric_func: Metric function to calculate.
    :param true_labels: True labels.
    :param pred_labels: Predicted labels.
    :return: Calculated metric value or 0 in case of an error.
    """
    try:
        return metric_func(true_labels, pred_labels, zero_division=0)
    except Exception as e:
        logging.warning(f"Error calculating metric: {str(e)}. Returning 0.")
        return 0
