# import os
# import json
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     accuracy_score, precision_score, recall_score, f1_score, 
#     confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
# )
# from sklearn.preprocessing import label_binarize
# from prediction import load_prediction_model, predict_image
# from utils import load_class_mapping

# def prepare_validation_dataset(ground_truth_csv, image_directory):
#     """
#     Recreates the exact validation split used in original Kaggle training.
#     """
#     from pathlib import Path
#     import pandas as pd
    
#     ground_truth_csv = str(ground_truth_csv).strip().strip('"').strip("'")
#     image_directory = str(image_directory).strip().strip('"').strip("'")
    
#     csv_path = Path(ground_truth_csv).expanduser()
#     img_dir = Path(image_directory).expanduser()
    
#     df = pd.read_csv(str(csv_path))
    
#     mel_df = df[df["MEL"] == 1.0].copy()
#     nv_df = df[df["NV"] == 1.0].head(4600).copy()
#     bcc_df = df[df["BCC"] == 1.0].copy()

#     mel_df["class"] = "Melanoma"
#     nv_df["class"] = "Nevus"
#     bcc_df["class"] = "BCC"

#     selected_df = pd.concat([mel_df, nv_df, bcc_df], ignore_index=True)
#     data = selected_df[["image", "class"]].copy()

#     class_to_idx = {"BCC": 0, "Melanoma": 1, "Nevus": 2}
#     data["label"] = data["class"].map(class_to_idx)
    
#     import os
#     data["image_path"] = data["image"].apply(lambda x: str(img_dir / f"{x}.jpg"))

#     _, val_df = train_test_split(
#         data,
#         test_size=0.20,
#         stratify=data["label"],
#         random_state=42
#     )
    
#     return val_df

# def evaluate_model(model=None, ground_truth_csv=None, image_directory=None, output_dir="metrics"):
#     """
#     Evaluates the model and generates all metrics and artifacts.
#     """
#     from pathlib import Path
    
#     if not ground_truth_csv or not image_directory:
#         raise ValueError("Ground truth CSV and image directory must be provided.")
        
#     ground_truth_csv = ground_truth_csv.strip().strip('"').strip("'")
#     image_directory = image_directory.strip().strip('"').strip("'")
    
#     csv_path = Path(ground_truth_csv).expanduser()
#     img_dir = Path(image_directory).expanduser()
    
#     if not csv_path.exists() or not csv_path.is_file():
#         raise FileNotFoundError(f"Ground truth CSV not found: {csv_path}")
#     if not img_dir.exists() or not img_dir.is_dir():
#         raise FileNotFoundError(f"Image directory not found: {img_dir}")
        
#     os.makedirs(output_dir, exist_ok=True)
    
#     if model is None:
#         model = load_prediction_model()
        
#     classes = load_class_mapping()
#     val_df = prepare_validation_dataset(ground_truth_csv, image_directory)
    
#     true_labels = []
#     predicted_labels = []
#     probabilities = []
    
#     print(f"Starting evaluation on {len(val_df)} validation images...")
    
#     for idx, row in val_df.iterrows():
#         img_path = row["image_path"]
#         true_label = row["class"]
        
#         if os.path.exists(img_path):
#             try:
#                 res = predict_image(img_path, model=model, classes=classes)
#                 pred_class = res["predicted_class"]
                
#                 probs_dict = res["probabilities"]
#                 probs_arr = [probs_dict.get(c, 0.0) / 100.0 for c in classes]
                
#                 true_labels.append(true_label)
#                 predicted_labels.append(pred_class)
#                 probabilities.append(probs_arr)
#             except Exception as e:
#                 print(f"Error evaluating {img_path}: {e}")
#         else:
#             print(f"Image not found: {img_path}")
            
#     if not true_labels:
#         raise ValueError("No valid predictions were made. Check image paths.")
        
#     # Sklearn metrics
#     accuracy = float(accuracy_score(true_labels, predicted_labels))
#     macro_precision = float(precision_score(true_labels, predicted_labels, average='macro', labels=classes))
#     macro_recall = float(recall_score(true_labels, predicted_labels, average='macro', labels=classes))
#     macro_f1 = float(f1_score(true_labels, predicted_labels, average='macro', labels=classes))
    
#     weighted_precision = float(precision_score(true_labels, predicted_labels, average='weighted', labels=classes))
#     weighted_recall = float(recall_score(true_labels, predicted_labels, average='weighted', labels=classes))
#     weighted_f1 = float(f1_score(true_labels, predicted_labels, average='weighted', labels=classes))
    
#     conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=classes)
#     report_dict = classification_report(true_labels, predicted_labels, labels=classes, output_dict=True)
#     report_df = pd.DataFrame(report_dict).transpose()
#     report_df.to_csv(os.path.join(output_dir, "classification_report.csv"))
    
#     # Probabilities for AUC
#     y_true_bin = label_binarize(true_labels, classes=classes)
#     probabilities = np.array(probabilities)
    
#     macro_auc = float(roc_auc_score(y_true_bin, probabilities, average='macro', multi_class='ovr'))
#     weighted_auc = float(roc_auc_score(y_true_bin, probabilities, average='weighted', multi_class='ovr'))
    
#     per_class_auc = {}
#     for i, cls in enumerate(classes):
#         per_class_auc[cls] = float(roc_auc_score(y_true_bin[:, i], probabilities[:, i]))
        
#     # Plot Confusion Matrix
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
#     plt.title('Confusion Matrix')
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
#     plt.close()
    
#     # Plot ROC curves
#     plt.figure(figsize=(10, 8))
#     for i, cls in enumerate(classes):
#         fpr, tpr, _ = roc_curve(y_true_bin[:, i], probabilities[:, i])
#         roc_auc_val = auc(fpr, tpr)
#         plt.plot(fpr, tpr, lw=2, label=f'ROC curve of {cls} (area = {roc_auc_val:0.4f})')
        
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic (One-vs-Rest)')
#     plt.legend(loc="lower right")
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, "roc_auc_curve.png"))
#     plt.close()
    
#     metrics = {
#         "accuracy": accuracy,
#         "macro_precision": macro_precision,
#         "macro_recall": macro_recall,
#         "macro_f1": macro_f1,
#         "weighted_precision": weighted_precision,
#         "weighted_recall": weighted_recall,
#         "weighted_f1": weighted_f1,
#         "macro_auc": macro_auc,
#         "weighted_auc": weighted_auc,
#         "per_class_auc": per_class_auc,
#         "confusion_matrix": conf_matrix.tolist()
#     }
    
#     with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
#         json.dump(metrics, f, indent=4)
        
#     return metrics

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--csv", type=str, help="Path to ground truth CSV")
#     parser.add_argument("--img_dir", type=str, help="Path to image directory")
#     args = parser.parse_args()
    
#     if args.csv and args.img_dir:
#         res = evaluate_model(ground_truth_csv=args.csv, image_directory=args.img_dir)
#         print("Evaluation complete.")
#         print(f"Accuracy: {res['accuracy']:.4f}")
#         print(f"Macro F1: {res['macro_f1']:.4f}")
#         print(f"Macro AUC: {res['macro_auc']:.4f}")
#     else:
#         print("Please provide --csv and --img_dir to run evaluation.")
import os
import json
from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

from prediction import load_prediction_model
from utils import load_class_mapping


# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 64

CLASS_NAMES = [
    "BCC",
    "Melanoma",
    "Nevus"
]

CLASS_TO_IDX = {
    "BCC": 0,
    "Melanoma": 1,
    "Nevus": 2
}

IDX_TO_CLASS = {
    0: "BCC",
    1: "Melanoma",
    2: "Nevus"
}


# ============================================================
# PATH CLEANING
# ============================================================

def clean_path(path):
    """
    Clean Windows paths entered through Streamlit.
    Removes surrounding quotes and whitespace.
    """
    if path is None:
        return None

    return str(path).strip().strip('"').strip("'")


# ============================================================
# IMAGE PATH RESOLUTION
# ============================================================

def build_image_path(image_id, image_directory):
    """
    Resolve the actual JPG path.

    Supports:
        image_directory/ISIC_xxxxxxx.jpg

    and nested dataset structures:
        image_directory/ISIC_2019_Training_Input/ISIC_xxxxxxx.jpg

    Returns:
        Path object or None
    """

    image_directory = Path(image_directory)
    image_id = str(image_id)

    # Remove extension if CSV already contains it
    if image_id.lower().endswith(".jpg"):
        filename = image_id
    elif image_id.lower().endswith(".jpeg"):
        filename = image_id
    else:
        filename = image_id + ".jpg"

    # --------------------------------------------------------
    # Direct path
    # --------------------------------------------------------

    direct_path = image_directory / filename

    if direct_path.exists():
        return direct_path

    # --------------------------------------------------------
    # Recursive search
    # --------------------------------------------------------

    matches = list(image_directory.rglob(filename))

    if len(matches) > 0:
        return matches[0]

    return None


# ============================================================
# RECREATE EXACT VALIDATION DATASET
# ============================================================

def prepare_validation_dataset(
    ground_truth_csv,
    image_directory
):
    """
    Recreates the exact validation split used during
    the verified ISIC 2019 evaluation.

    Selection:
        Melanoma = all MEL images
        Nevus    = first 4600 NV images
        BCC      = all BCC images

    Split:
        80% training
        20% validation

    Stratification:
        label

    Random state:
        42
    """

    ground_truth_csv = clean_path(ground_truth_csv)
    image_directory = clean_path(image_directory)

    csv_path = Path(ground_truth_csv).expanduser()
    img_dir = Path(image_directory).expanduser()

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Ground truth CSV not found:\n{csv_path}"
        )

    if not csv_path.is_file():
        raise ValueError(
            f"Ground truth path is not a file:\n{csv_path}"
        )

    if not img_dir.exists():
        raise FileNotFoundError(
            f"Image directory not found:\n{img_dir}"
        )

    if not img_dir.is_dir():
        raise ValueError(
            f"Image directory is not a directory:\n{img_dir}"
        )

    # --------------------------------------------------------
    # Load CSV
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING ISIC 2019 DATASET")
    print("=" * 70)

    print(f"Ground Truth CSV:")
    print(csv_path)

    print()
    print("Image Directory:")
    print(img_dir)

    df = pd.read_csv(csv_path)

    print()
    print(f"Original dataset shape: {df.shape}")

    required_columns = [
        "image",
        "MEL",
        "NV",
        "BCC"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # Select the exact three classes
    # --------------------------------------------------------

    mel_df = df[df["MEL"] == 1.0].copy()

    # IMPORTANT:
    # This matches the previously verified pipeline.
    nv_df = df[df["NV"] == 1.0].head(4600).copy()

    bcc_df = df[df["BCC"] == 1.0].copy()

    mel_df["class"] = "Melanoma"
    nv_df["class"] = "Nevus"
    bcc_df["class"] = "BCC"

    selected_df = pd.concat(
        [
            mel_df,
            nv_df,
            bcc_df
        ],
        ignore_index=True
    )

    print()
    print("Selected dataset:")
    print(f"Total selected images: {len(selected_df)}")

    print()
    print("Class distribution:")
    print(
        selected_df["class"].value_counts()
        .reindex(CLASS_NAMES)
    )

    # --------------------------------------------------------
    # Convert classes to labels
    # --------------------------------------------------------

    data = selected_df[
        ["image", "class"]
    ].copy()

    data["label"] = data["class"].map(CLASS_TO_IDX)

    # --------------------------------------------------------
    # Exact 80/20 stratified split
    # --------------------------------------------------------

    train_df, val_df = train_test_split(
        data,
        test_size=0.20,
        stratify=data["label"],
        random_state=42
    )

    # Reset index
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print()
    print("=" * 70)
    print("VALIDATION SPLIT")
    print("=" * 70)

    print()
    print(f"Training images:   {len(train_df)}")
    print(f"Validation images: {len(val_df)}")

    print()
    print("Training distribution:")
    print(
        train_df["class"]
        .value_counts()
        .reindex(CLASS_NAMES)
    )

    print()
    print("Validation distribution:")
    print(
        val_df["class"]
        .value_counts()
        .reindex(CLASS_NAMES)
    )

    # --------------------------------------------------------
    # Resolve actual image paths
    # --------------------------------------------------------

    print()
    print("Resolving validation image paths...")

    val_df["image_path"] = val_df["image"].apply(
        lambda x: build_image_path(
            x,
            img_dir
        )
    )

    missing_mask = val_df["image_path"].isna()

    missing_count = int(missing_mask.sum())

    print()
    print(f"Validation images: {len(val_df)}")
    print(f"Images found:      {len(val_df) - missing_count}")
    print(f"Images missing:    {missing_count}")

    if missing_count > 0:

        print()
        print("First missing images:")

        print(
            val_df.loc[
                missing_mask,
                "image"
            ].head(20).tolist()
        )

        raise FileNotFoundError(
            f"{missing_count} validation images could not be found. "
            "Evaluation stopped to prevent incorrect metrics."
        )

    # Convert Path objects to strings
    val_df["image_path"] = val_df[
        "image_path"
    ].apply(str)

    print()
    print("All validation images successfully resolved.")

    return train_df, val_df


# ============================================================
# IMAGE LOADING / PREPROCESSING
# ============================================================

def load_and_preprocess_image(
    image_path,
    label
):
    """
    Loads one image and applies the EXACT preprocessing
    used by the verified MobileNetV2 pipeline.

    IMPORTANT:
        resize -> float32 -> /255.0
    """

    image = tf.io.read_file(image_path)

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    # EXACT normalization used in the verified evaluation
    image = tf.cast(
        image,
        tf.float32
    ) / 255.0

    return image, label


# ============================================================
# CREATE VALIDATION DATASET
# ============================================================

def create_validation_tf_dataset(val_df):
    """
    Creates an optimized TensorFlow validation dataset.

    Uses:
        parallel image loading
        batching
        prefetching
    """

    image_paths = val_df[
        "image_path"
    ].values

    labels = val_df[
        "label"
    ].values.astype(np.int32)

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            image_paths,
            labels
        )
    )

    dataset = dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# RUN BATCHED INFERENCE
# ============================================================

def generate_predictions(
    model,
    val_df
):
    """
    Runs model inference on the complete validation set
    using batched TensorFlow inference.

    Returns:
        y_true
        y_prob
        y_pred
    """

    print()
    print("=" * 70)
    print("STARTING BATCHED VALIDATION INFERENCE")
    print("=" * 70)

    print()
    print(f"Total validation images: {len(val_df)}")
    print(f"Batch size:              {BATCH_SIZE}")

    validation_dataset = create_validation_tf_dataset(
        val_df
    )

    print()
    print("Running model.predict()...")

    # --------------------------------------------------------
    # IMPORTANT:
    # This returns the raw probability outputs.
    # These probabilities are used for ROC/AUC.
    # --------------------------------------------------------

    y_prob = model.predict(
        validation_dataset,
        verbose=1
    )

    y_prob = np.asarray(
        y_prob,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # True labels
    # --------------------------------------------------------

    y_true = val_df[
        "label"
    ].to_numpy(
        dtype=np.int32
    )

    # --------------------------------------------------------
    # Predicted class
    #
    # Argmax is ONLY used for class prediction.
    # It is NOT used for ROC/AUC.
    # --------------------------------------------------------

    y_pred = np.argmax(
        y_prob,
        axis=1
    ).astype(np.int32)

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    print()
    print("Inference complete.")

    print()
    print("Prediction shapes:")
    print(f"y_true: {y_true.shape}")
    print(f"y_prob: {y_prob.shape}")
    print(f"y_pred: {y_pred.shape}")

    if len(y_true) != len(val_df):
        raise ValueError(
            "y_true length does not match validation dataset."
        )

    if y_prob.shape[0] != len(val_df):
        raise ValueError(
            "Number of probability predictions does not "
            "match validation dataset."
        )

    if y_prob.shape[1] != 3:
        raise ValueError(
            f"Expected 3 model outputs, got {y_prob.shape[1]}"
        )

    # Probability sanity check
    probability_sums = np.sum(
        y_prob,
        axis=1
    )

    print()
    print("Probability sanity check:")
    print(
        "Minimum probability sum:",
        float(np.min(probability_sums))
    )

    print(
        "Maximum probability sum:",
        float(np.max(probability_sums))
    )

    return (
        y_true,
        y_prob,
        y_pred
    )


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_prob,
    y_pred
):
    """
    Calculates all required performance metrics.
    """

    print()
    print("=" * 70)
    print("CALCULATING PERFORMANCE METRICS")
    print("=" * 70)

    # --------------------------------------------------------
    # Classification metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0
    )

    weighted_precision = precision_score(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average="weighted",
        zero_division=0
    )

    weighted_recall = recall_score(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average="weighted",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average="weighted",
        zero_division=0
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2]
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    # --------------------------------------------------------
    # ROC / AUC
    # --------------------------------------------------------

    y_true_bin = label_binarize(
        y_true,
        classes=[0, 1, 2]
    )

    # Per-class AUC
    auc_values = {}

    fpr_values = {}
    tpr_values = {}

    for i, class_name in enumerate(CLASS_NAMES):

        fpr, tpr, _ = roc_curve(
            y_true_bin[:, i],
            y_prob[:, i]
        )

        class_auc = auc(
            fpr,
            tpr
        )

        fpr_values[class_name] = fpr
        tpr_values[class_name] = tpr

        auc_values[class_name] = float(
            class_auc
        )

    # Macro AUC
    macro_auc = roc_auc_score(
        y_true_bin,
        y_prob,
        average="macro",
        multi_class="ovr"
    )

    # Weighted AUC
    weighted_auc = roc_auc_score(
        y_true_bin,
        y_prob,
        average="weighted",
        multi_class="ovr"
    )

    return {
        "accuracy": float(accuracy),

        "macro_precision": float(
            macro_precision
        ),

        "macro_recall": float(
            macro_recall
        ),

        "macro_f1": float(
            macro_f1
        ),

        "weighted_precision": float(
            weighted_precision
        ),

        "weighted_recall": float(
            weighted_recall
        ),

        "weighted_f1": float(
            weighted_f1
        ),

        "auc_bcc": auc_values["BCC"],
        "auc_melanoma": auc_values["Melanoma"],
        "auc_nevus": auc_values["Nevus"],

        "macro_auc": float(
            macro_auc
        ),

        "weighted_auc": float(
            weighted_auc
        ),

        "confusion_matrix": cm.tolist(),

        "classification_report": report_dict,

        "roc": {
            "fpr": fpr_values,
            "tpr": tpr_values
        }
    }


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

def save_classification_report(
    metrics,
    output_dir
):

    report_df = pd.DataFrame(
        metrics["classification_report"]
    ).transpose()

    report_path = (
        Path(output_dir)
        / "classification_report.csv"
    )

    report_df.to_csv(
        report_path
    )

    print()
    print(
        f"Classification report saved: {report_path}"
    )


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    metrics,
    output_dir
):

    cm = np.array(
        metrics["confusion_matrix"]
    )

    plt.figure(
        figsize=(8, 6)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )

    plt.title(
        "Confusion Matrix — ISIC 2019 MobileNetV2"
    )

    plt.ylabel(
        "Actual"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.tight_layout()

    output_path = (
        Path(output_dir)
        / "confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Confusion matrix saved: {output_path}"
    )


# ============================================================
# SAVE ROC CURVE
# ============================================================

def save_roc_curve(
    metrics,
    output_dir
):

    plt.figure(
        figsize=(9, 7)
    )

    for class_name in CLASS_NAMES:

        fpr = np.array(
            metrics["roc"]["fpr"][class_name]
        )

        tpr = np.array(
            metrics["roc"]["tpr"][class_name]
        )

        auc_value = metrics[
            f"auc_{class_name.lower()}"
        ]

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=(
                f"{class_name} "
                f"(AUC = {auc_value:.4f})"
            )
        )

    # Random classifier
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=2,
        label="Random Classifier"
    )

    plt.xlim(
        [0.0, 1.0]
    )

    plt.ylim(
        [0.0, 1.05]
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curves — ISIC 2019 MobileNetV2"
    )

    plt.legend(
        loc="lower right"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    output_path = (
        Path(output_dir)
        / "roc_auc_curve.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"ROC/AUC curve saved: {output_path}"
    )


# ============================================================
# SAVE METRICS JSON
# ============================================================

def save_metrics_json(
    metrics,
    output_dir
):

    json_metrics = {
        "accuracy": metrics["accuracy"],

        "macro_precision":
            metrics["macro_precision"],

        "macro_recall":
            metrics["macro_recall"],

        "macro_f1":
            metrics["macro_f1"],

        "weighted_precision":
            metrics["weighted_precision"],

        "weighted_recall":
            metrics["weighted_recall"],

        "weighted_f1":
            metrics["weighted_f1"],

        "auc_bcc":
            metrics["auc_bcc"],

        "auc_melanoma":
            metrics["auc_melanoma"],

        "auc_nevus":
            metrics["auc_nevus"],

        "macro_auc":
            metrics["macro_auc"],

        "weighted_auc":
            metrics["weighted_auc"],

        "confusion_matrix":
            metrics["confusion_matrix"]
    }

    output_path = (
        Path(output_dir)
        / "metrics.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            json_metrics,
            f,
            indent=4
        )

    print(
        f"Metrics JSON saved: {output_path}"
    )


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

def print_final_results(
    metrics
):

    print()
    print()
    print("=" * 70)
    print("       ISIC 2019 — MobileNetV2 PERFORMANCE")
    print("=" * 70)

    print()

    print(
        f"Accuracy:              "
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    print()

    print("--- Macro Metrics ---")

    print(
        f"Precision:             "
        f"{metrics['macro_precision'] * 100:.2f}%"
    )

    print(
        f"Recall:                "
        f"{metrics['macro_recall'] * 100:.2f}%"
    )

    print(
        f"F1 Score:              "
        f"{metrics['macro_f1'] * 100:.2f}%"
    )

    print()

    print("--- Weighted Metrics ---")

    print(
        f"Precision:             "
        f"{metrics['weighted_precision'] * 100:.2f}%"
    )

    print(
        f"Recall:                "
        f"{metrics['weighted_recall'] * 100:.2f}%"
    )

    print(
        f"F1 Score:              "
        f"{metrics['weighted_f1'] * 100:.2f}%"
    )

    print()

    print("--- ROC / AUC ---")

    print(
        f"BCC          AUC: "
        f"{metrics['auc_bcc']:.4f}"
    )

    print(
        f"Melanoma     AUC: "
        f"{metrics['auc_melanoma']:.4f}"
    )

    print(
        f"Nevus        AUC: "
        f"{metrics['auc_nevus']:.4f}"
    )

    print(
        f"Macro AUC:           "
        f"{metrics['macro_auc']:.4f}"
    )

    print(
        f"Weighted AUC:        "
        f"{metrics['weighted_auc']:.4f}"
    )

    print()
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    report_df = pd.DataFrame(
        metrics["classification_report"]
    ).transpose()

    print(
        report_df.to_string()
    )

    print()
    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print(
        np.array(
            metrics["confusion_matrix"]
        )
    )

    print("=" * 70)


# ============================================================
# COMPLETE EVALUATION
# ============================================================

def evaluate_model(
    model=None,
    ground_truth_csv=None,
    image_directory=None,
    output_dir="metrics"
):
    """
    Complete batched performance evaluation.

    IMPORTANT:
    - Does NOT retrain the model.
    - Does NOT modify the model.
    - Uses the complete 2,489-image validation set.
    - Uses real model probability outputs.
    """

    if not ground_truth_csv:
        raise ValueError(
            "Ground truth CSV must be provided."
        )

    if not image_directory:
        raise ValueError(
            "Image directory must be provided."
        )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    if model is None:

        print()
        print("Loading saved model...")

        model = load_prediction_model()

    print()
    print("Model loaded successfully.")

    print(
        f"Input shape:  {model.input_shape}"
    )

    print(
        f"Output shape: {model.output_shape}"
    )

    # --------------------------------------------------------
    # Prepare exact validation dataset
    # --------------------------------------------------------

    train_df, val_df = prepare_validation_dataset(
        ground_truth_csv,
        image_directory
    )

    # --------------------------------------------------------
    # Generate batched predictions
    # --------------------------------------------------------

    y_true, y_prob, y_pred = generate_predictions(
        model,
        val_df
    )

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        y_true,
        y_prob,
        y_pred
    )

    # --------------------------------------------------------
    # Save artifacts
    # --------------------------------------------------------

    save_classification_report(
        metrics,
        output_dir
    )

    save_confusion_matrix(
        metrics,
        output_dir
    )

    save_roc_curve(
        metrics,
        output_dir
    )

    save_metrics_json(
        metrics,
        output_dir
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print_final_results(
        metrics
    )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print()
    print(f"Output directory: {output_dir}")

    print()
    print("Generated files:")

    print(
        f"* {output_dir / 'metrics.json'}"
    )

    print(
        f"* {output_dir / 'classification_report.csv'}"
    )

    print(
        f"* {output_dir / 'confusion_matrix.png'}"
    )

    print(
        f"* {output_dir / 'roc_auc_curve.png'}"
    )

    return metrics


# ============================================================
# COMMAND LINE EXECUTION
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "ISIC 2019 MobileNetV2 "
            "Performance Evaluation"
        )
    )

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to ISIC 2019 ground truth CSV"
    )

    parser.add_argument(
        "--img_dir",
        type=str,
        required=True,
        help="Path to ISIC 2019 image directory"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="metrics",
        help="Directory for evaluation artifacts"
    )

    args = parser.parse_args()

    results = evaluate_model(
        ground_truth_csv=args.csv,
        image_directory=args.img_dir,
        output_dir=args.output_dir
    )

    print()
    print("Final Accuracy:")
    print(
        f"{results['accuracy'] * 100:.2f}%"
    )

    print()
    print("Final Macro F1:")
    print(
        f"{results['macro_f1'] * 100:.2f}%"
    )

    print()
    print("Final Macro AUC:")
    print(
        f"{results['macro_auc']:.4f}"
    )