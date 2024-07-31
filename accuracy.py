import os
import logging
import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.exceptions import UndefinedMetricWarning
import warnings

# Filter out the specific warning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Hardcoded file paths
RASTER_FILE = "data/clipped.TIF"
CLEANED_SMOOTHED_FILE = "result/cleaned_smoothed_landcover.tif"
OUTPUT_DIR = "result"


def load_raster_data(filepath):
    """Load raster data and return the data and metadata."""
    logging.info(f"Loading raster data from {filepath}")
    with rasterio.open(filepath) as src:
        data = src.read(1)
        metadata = src.meta
    return data, metadata


def safe_metric_calculation(metric_func, true_labels, pred_labels):
    """Safely calculate a metric, returning 0 if there's an error."""
    try:
        return metric_func(true_labels, pred_labels, zero_division=0)
    except Exception as e:
        logging.warning(f"Error calculating metric: {str(e)}. Returning 0.")
        return 0


def calculate_accuracy(original_data, processed_data):
    """Calculate comprehensive accuracy metrics comparing processed data with the original data."""
    logging.info("Calculating accuracy metrics")
    original_data_flat = original_data.flatten()
    processed_data_flat = processed_data.flatten()

    conf_matrix = confusion_matrix(original_data_flat, processed_data_flat)
    overall_acc = accuracy_score(original_data_flat, processed_data_flat)
    kappa = cohen_kappa_score(original_data_flat, processed_data_flat)

    # Calculate per-class metrics
    classes = np.unique(np.concatenate((original_data_flat, processed_data_flat)))
    per_class_metrics = {}
    for cls in classes:
        per_class_metrics[cls] = {
            "precision": safe_metric_calculation(
                precision_score, original_data_flat == cls, processed_data_flat == cls
            ),
            "recall": safe_metric_calculation(
                recall_score, original_data_flat == cls, processed_data_flat == cls
            ),
            "f1_score": safe_metric_calculation(
                f1_score, original_data_flat == cls, processed_data_flat == cls
            ),
        }

    return {
        "confusion_matrix": conf_matrix,
        "overall_accuracy": overall_acc,
        "kappa": kappa,
        "per_class_metrics": per_class_metrics,
    }


def plot_confusion_matrix(conf_matrix, output_path):
    """Plot and save the confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(output_path)
    plt.close()


def main():
    try:
        # Load and prepare data
        original_data, _ = load_raster_data(RASTER_FILE)
        cleaned_smoothed_data, _ = load_raster_data(CLEANED_SMOOTHED_FILE)

        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Calculate accuracy
        accuracy_metrics = calculate_accuracy(original_data, cleaned_smoothed_data)

        # Log overall metrics
        logging.info(f"Overall Accuracy: {accuracy_metrics['overall_accuracy']}")
        logging.info(f"Kappa: {accuracy_metrics['kappa']}")

        # Prepare data for DataFrame
        df_data = {
            "Metric": ["Overall Accuracy", "Kappa"],
            "Value": [accuracy_metrics["overall_accuracy"], accuracy_metrics["kappa"]],
        }

        # Add per-class metrics to DataFrame
        for cls, metrics in accuracy_metrics["per_class_metrics"].items():
            for metric_name, value in metrics.items():
                df_data["Metric"].append(f"Class {cls} - {metric_name}")
                df_data["Value"].append(value)

        # Create and save DataFrame
        accuracy_df = pd.DataFrame(df_data)
        csv_path = os.path.join(OUTPUT_DIR, "accuracy_metrics.csv")
        accuracy_df.to_csv(csv_path, index=False)
        logging.info(f"Accuracy metrics saved to {csv_path}")

        # Plot and save confusion matrix
        cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
        plot_confusion_matrix(accuracy_metrics["confusion_matrix"], cm_path)
        logging.info(f"Confusion matrix plot saved to {cm_path}")

    except FileNotFoundError as fnf_error:
        logging.error(f"File not found: {fnf_error}")
    except RasterioIOError as rio_error:
        logging.error(f"Rasterio IO error: {rio_error}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        logging.exception("Exception details:")


if __name__ == "__main__":
    main()
