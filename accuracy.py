import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from rasterio.errors import RasterioIOError
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.exceptions import UndefinedMetricWarning
from utils import load_raster_data, safe_metric_calculation

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

RASTER_FILE = "data/clipped.TIF"
CLEANED_SMOOTHED_FILE = "result/cleaned_smoothed_landcover.tif"
OUTPUT_DIR = "result"


def calculate_accuracy(original_data: np.ndarray, processed_data: np.ndarray) -> dict:
    """
    Calculate accuracy metrics for the processed raster data.

    :param original_data: Original raster data.
    :param processed_data: Processed raster data.
    :return: Dictionary containing various accuracy metrics.
    """
    logging.info("Calculating accuracy metrics")
    original_data_flat = original_data.flatten()
    processed_data_flat = processed_data.flatten()

    original_classes = set(np.unique(original_data_flat))
    processed_classes = set(np.unique(processed_data_flat))
    valid_classes = sorted(list(original_classes.intersection(processed_classes)))

    if 0 in valid_classes:
        valid_classes.remove(0)

    conf_matrix = confusion_matrix(
        original_data_flat, processed_data_flat, labels=valid_classes
    )
    overall_acc = accuracy_score(original_data_flat, processed_data_flat)
    kappa = cohen_kappa_score(original_data_flat, processed_data_flat)

    per_class_metrics = {}
    for cls in valid_classes:
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
        "valid_classes": valid_classes,
    }


def plot_confusion_matrix(
    conf_matrix: np.ndarray, valid_classes: list, output_path: str
):
    """
    Plot and save the confusion matrix.

    :param conf_matrix: Confusion matrix.
    :param valid_classes: List of valid classes.
    :param output_path: Path to save the plot.
    """
    plt.figure(figsize=(15, 15))

    row_sums = conf_matrix.sum(axis=1)
    conf_matrix_norm = np.divide(
        conf_matrix.astype("float"),
        row_sums[:, np.newaxis],
        where=row_sums[:, np.newaxis] != 0,
        out=np.zeros_like(conf_matrix, dtype=float),
    )

    heatmap = sns.heatmap(
        conf_matrix_norm,
        annot=conf_matrix,
        fmt="d",
        cmap="Blues",
        square=True,
        cbar=False,
    )

    plt.title("Confusion Matrix", fontsize=16)
    plt.ylabel("True Label", fontsize=14)
    plt.xlabel("Predicted Label", fontsize=14)

    plt.xticks(
        np.arange(len(valid_classes)) + 0.5, valid_classes, rotation=45, ha="right"
    )
    plt.yticks(np.arange(len(valid_classes)) + 0.5, valid_classes, rotation=0)

    plt.tick_params(labelsize=12)

    cbar = plt.colorbar(heatmap.collections[0], shrink=0.8)
    cbar.set_label("Normalized Frequency", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
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

        # Plot and save confusion matrix if there are valid classes
        if accuracy_metrics["valid_classes"]:
            cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
            plot_confusion_matrix(
                accuracy_metrics["confusion_matrix"],
                accuracy_metrics["valid_classes"],
                cm_path,
            )
            logging.info(f"Confusion matrix plot saved to {cm_path}")
        else:
            logging.warning("No valid classes found. Skipping confusion matrix plot.")

    except FileNotFoundError as fnf_error:
        logging.error(f"File not found: {fnf_error}")
    except RasterioIOError as rio_error:
        logging.error(f"Rasterio IO error: {rio_error}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        logging.exception("Exception details:")


if __name__ == "__main__":
    main()
