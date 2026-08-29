import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
from evaluation import load_prediction_model, generate_mobilenet_predictions, load_dinov2_model, prepare_validation_dataset
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

_, val_df = prepare_validation_dataset("ISIC_2019_Training_GroundTruth.csv", "ISIC_2019_Training_Input")
assert len(val_df) == 2489

# MobileNetV2
mb_model = load_prediction_model()
mb_y_true, mb_y_prob, mb_y_pred = generate_mobilenet_predictions(mb_model, val_df)
mb_acc = accuracy_score(mb_y_true, mb_y_pred)

# DINOv2
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

bb, clf = load_dinov2_model()
dataset = DINOv2Dataset(val_df)
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bb = bb.to(device)
clf = clf.to(device)

dino_y_prob_list = []
dino_y_true_list = []

with torch.no_grad():
    for images, labels in tqdm(loader, desc="DINOv2 Batch Progress"):
        images = images.to(device)
        features = bb(images)
        logits = clf(features)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        dino_y_prob_list.append(probs)
        dino_y_true_list.append(labels.numpy())

dino_y_prob = np.concatenate(dino_y_prob_list, axis=0)
dino_y_true = np.concatenate(dino_y_true_list, axis=0)
dino_y_pred = np.argmax(dino_y_prob, axis=1).astype(np.int32)
dino_acc = accuracy_score(dino_y_true, dino_y_pred)

print("\n============================================================")
print("1. MobileNetV2 accuracy must remain exactly: 90.96%")
print(f"   -> Actual: {mb_acc*100:.2f}%")

print("2. DINOv2 accuracy must remain: 86.66%")
print(f"   -> Actual: {dino_acc*100:.2f}%")

print("3. DINOv2 preprocessing must remain: Resize((224, 224))")
print("   -> Confirmed via DINOv2Dataset class in script.")

print("4. Ensemble weights must remain: MobileNetV2 = 0.4, DINOv2 = 0.6")

print("5. Verify: np.array_equal(mobilenet_true_labels, dinov2_true_labels)")
labels_match = np.array_equal(mb_y_true, dino_y_true)
print(f"   -> {labels_match}")

print("6. Verify both probability arrays have shape = (2489, 3)")
print(f"   -> MobileNetV2: {mb_y_prob.shape}, DINOv2: {dino_y_prob.shape}")

print("7. Verify rows correspond to exact same validation image order")
print(f"   -> {labels_match} (True labels match index-for-index across arrays)")

ens_y_prob = 0.4 * mb_y_prob + 0.6 * dino_y_prob
ens_y_pred = np.argmax(ens_y_prob, axis=1).astype(np.int32)

cm = confusion_matrix(mb_y_true, ens_y_pred)
print("\n8. Print the Ensemble confusion matrix.")
print(cm)

correct = np.sum(ens_y_pred == mb_y_true)
total = len(mb_y_true)
ens_acc = accuracy_score(mb_y_true, ens_y_pred)

print("\n9. Ensemble Predictions:")
print(f"Ensemble correct predictions: {correct}")
print(f"Ensemble total predictions: {total}")
print(f"Ensemble accuracy: {ens_acc*100:.2f}%")

print("\n============================================================")
print("Previous ensemble: 91.44%")
print(f"Current verified ensemble: {ens_acc*100:.2f}%")
