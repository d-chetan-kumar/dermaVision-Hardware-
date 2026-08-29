import os
import json
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 64

# Verification against class_names.json
with open('models/class_names.json', 'r') as f:
    class_mapping = json.load(f)
    CLASS_NAMES = [class_mapping["0"], class_mapping["1"], class_mapping["2"]]
    CLASS_TO_IDX = {v: int(k) for k, v in class_mapping.items()}

# ============================================================
# DATASET RECONSTRUCTION
# ============================================================

def clean_path(path):
    if path is None: return None
    return str(path).strip().strip('"').strip("'")

def build_image_path(image_id, image_directory):
    image_directory = Path(image_directory)
    image_id = str(image_id)
    if image_id.lower().endswith(".jpg"): filename = image_id
    elif image_id.lower().endswith(".jpeg"): filename = image_id
    else: filename = image_id + ".jpg"

    direct_path = image_directory / filename
    if direct_path.exists(): return direct_path

    matches = list(image_directory.rglob(filename))
    if len(matches) > 0: return matches[0]
    return None

print("Reconstructing 20,720 image dataset split...")
ground_truth_csv = "ISIC_2019_Training_GroundTruth.csv"
image_directory = "ISIC_2019_Training_Input"

csv_path = Path(ground_truth_csv)
img_dir = Path(image_directory)

df = pd.read_csv(csv_path)

# Extract 3 classes without caps
mel_df = df[df["MEL"] == 1.0].copy()
nv_df = df[df["NV"] == 1.0].copy()
bcc_df = df[df["BCC"] == 1.0].copy()

mel_df["class"] = "Melanoma"
nv_df["class"] = "Nevus"
bcc_df["class"] = "BCC"

selected_df = pd.concat([mel_df, nv_df, bcc_df], ignore_index=True)
data = selected_df[["image", "class"]].copy()
data["label"] = data["class"].map(CLASS_TO_IDX)

# Deterministic reproducible split
train_df, val_df = train_test_split(
    data, test_size=0.20, stratify=data["label"], random_state=42
)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

# Build paths
val_df["image_path"] = val_df["image"].apply(lambda x: build_image_path(x, img_dir))
val_df["image_path"] = val_df["image_path"].apply(str)

# Verify split counts
print(f"Total samples: {len(data)}")
print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")

assert len(data) == 20720, f"Expected 20,720 total samples, got {len(data)}"
assert len(train_df) == 16576, f"Expected 16,576 training samples, got {len(train_df)}"
assert len(val_df) == 4144, f"Expected 4,144 validation samples, got {len(val_df)}"

val_dist = val_df['class'].value_counts()
print("\nValidation class distribution:")
print(val_dist)

assert val_dist['BCC'] == 665, f"Expected BCC=665, got {val_dist['BCC']}"
assert val_dist['Melanoma'] == 904, f"Expected Melanoma=904, got {val_dist['Melanoma']}"
assert val_dist['Nevus'] == 2575, f"Expected Nevus=2575, got {val_dist['Nevus']}"

# Save image order
os.makedirs('results', exist_ok=True)
val_df[['image', 'class', 'label']].to_csv('results/validation_image_order.csv', index=False)
print("\nSaved validation image order to results/validation_image_order.csv")

# ============================================================
# MOBILENETV2 (TF) INFERENCE
# ============================================================

def load_and_preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def create_validation_tf_dataset(val_df):
    image_paths = val_df["image_path"].values
    labels = val_df["label"].values.astype(np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset

print("\nLoading MobileNetV2...")
mb_model = tf.keras.models.load_model('models/best_mobilenetv2.keras')
print("Running MobileNetV2 inference...")
mb_dataset = create_validation_tf_dataset(val_df)
mb_y_prob = mb_model.predict(mb_dataset, verbose=1)
mb_y_prob = np.asarray(mb_y_prob, dtype=np.float32)
mb_y_true = val_df["label"].to_numpy(dtype=np.int32)
mb_y_pred = np.argmax(mb_y_prob, axis=1).astype(np.int32)

mb_acc = accuracy_score(mb_y_true, mb_y_pred)
print(f"MobileNetV2 Actual Standalone Validation Accuracy: {mb_acc * 100:.2f}%")

# ============================================================
# DINOV2 (PYTORCH) INFERENCE
# ============================================================

class DINOv2Dataset(torch.utils.data.Dataset):
    def __init__(self, val_df):
        self.val_df = val_df
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.val_df)
        
    def __getitem__(self, idx):
        row = self.val_df.iloc[idx]
        img_path = row['image_path']
        label = row['label']
        image = Image.open(img_path).convert('RGB')
        tensor = self.transform(image)
        return tensor, label

class DINOv2LinearClassifier(nn.Module):
    def __init__(self, feature_dim=768, num_classes=3):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x):
        return self.classifier(x)

print('\nLoading DINOv2 backbone...')
backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', trust_repo=True)
backbone.eval()

print('Loading DINOv2 classifier (models/dinov2_classifier_best.pth)...')
ckpt = torch.load('models/dinov2_classifier_best.pth', map_location='cpu')
classifier = DINOv2LinearClassifier(
    feature_dim=ckpt.get("feature_dim", 768),
    num_classes=ckpt.get("num_classes", 3)
)
if "model_state_dict" in ckpt:
    classifier.load_state_dict(ckpt["model_state_dict"])
else:
    classifier.load_state_dict(ckpt)
classifier.eval()

dataset = DINOv2Dataset(val_df)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
backbone = backbone.to(device)
classifier = classifier.to(device)

dino_y_prob_list = []
dino_y_true_list = []

print('Running DINOv2 inference...')
with torch.no_grad():
    for images, labels in tqdm(loader, desc="DINOv2 Batch Progress"):
        images = images.to(device)
        features = backbone(images)
        logits = classifier(features)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        dino_y_prob_list.append(probs)
        dino_y_true_list.append(labels.numpy())

dino_y_prob = np.concatenate(dino_y_prob_list, axis=0)
dino_y_true = np.concatenate(dino_y_true_list, axis=0)
dino_y_pred = np.argmax(dino_y_prob, axis=1).astype(np.int32)

dino_acc = accuracy_score(dino_y_true, dino_y_pred)
print(f"DINOv2 Actual Standalone Validation Accuracy: {dino_acc * 100:.2f}%")

# ============================================================
# ALIGNMENT VERIFICATION
# ============================================================

assert np.array_equal(mb_y_true, dino_y_true), "Labels mismatch between models!"
assert mb_y_prob.shape == (4144, 3), "MobileNetV2 shape mismatch!"
assert dino_y_prob.shape == (4144, 3), "DINOv2 shape mismatch!"
print("\nValidation verification passed: Both arrays predict on the same 4,144 images in the exact same order.")

# ============================================================
# ENSEMBLE EVALUATION
# ============================================================

weights = [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2), (0.9, 0.1)]
ensemble_results = []

for mb_w, dino_w in weights:
    ens_prob = (mb_w * mb_y_prob) + (dino_w * dino_y_prob)
    ens_pred = np.argmax(ens_prob, axis=1).astype(np.int32)
    acc = accuracy_score(mb_y_true, ens_pred)
    
    ensemble_results.append({
        'MobileNet_W': mb_w,
        'DINOv2_W': dino_w,
        'Accuracy': acc * 100,
        'Predictions': ens_pred
    })

# Convert to DataFrame to display and find best
results_df = pd.DataFrame(ensemble_results).drop(columns=['Predictions'])
print("\n============================================================")
print("ENSEMBLE WEIGHT EVALUATION")
print("============================================================")
print(results_df.to_string(index=False, float_format="%.2f"))

best_ens = sorted(ensemble_results, key=lambda x: x['Accuracy'], reverse=True)[0]
best_mb_w = best_ens['MobileNet_W']
best_dino_w = best_ens['DINOv2_W']
best_pred = best_ens['Predictions']
best_acc = best_ens['Accuracy']

print("\n============================================================")
print(f"BEST ENSEMBLE: MobileNetV2={best_mb_w} / DINOv2={best_dino_w}")
print(f"BEST ACCURACY: {best_acc:.2f}%")
print("============================================================")

# Generate Classification Report
print("\nClassification Report (Best Ensemble):")
report = classification_report(mb_y_true, best_pred, target_names=CLASS_NAMES, zero_division=0)
print(report)

# Output CSV Report
report_dict = classification_report(mb_y_true, best_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
pd.DataFrame(report_dict).transpose().to_csv('results/best_ensemble_classification_report.csv')

# Generate Confusion Matrix
cm = confusion_matrix(mb_y_true, best_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title(f'Confusion Matrix (Ensemble {best_mb_w}/{best_dino_w})')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('graphs/best_ensemble_confusion_matrix.png', dpi=200, bbox_inches='tight')
plt.close()

# Generate Accuracy Comparison Graph
models = ['MobileNetV2', 'DINOv2', 'Best Ensemble']
accuracies = [mb_acc * 100, dino_acc * 100, best_acc]

plt.figure(figsize=(8, 6))
bars = plt.bar(models, accuracies, color=['#3b82f6', '#10b981', '#f59e0b'], width=0.5)
plt.title('Final Model Accuracy Comparison')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('graphs/final_accuracy_comparison.png', dpi=200, bbox_inches='tight')
plt.close()

print("\nAll artifacts saved successfully.")
print("- results/best_ensemble_classification_report.csv")
print("- graphs/best_ensemble_confusion_matrix.png")
print("- graphs/final_accuracy_comparison.png")
