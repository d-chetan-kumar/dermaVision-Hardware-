import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from evaluation import load_dinov2_model, prepare_validation_dataset
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

print("CURRENT TRANSFORM:")
print("transforms.Compose([")
print("    transforms.Resize(256),")
print("    transforms.CenterCrop(224),")
print("    transforms.ToTensor(),")
print("    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])")
print("])\n")

print("CORRECTED TRANSFORM:")
print("transforms.Compose([")
print("    transforms.Resize((224, 224)),")
print("    transforms.ToTensor(),")
print("    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])")
print("])\n")

_, val_df = prepare_validation_dataset("ISIC_2019_Training_GroundTruth.csv", "ISIC_2019_Training_Input")
assert len(val_df) == 2489

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

y_prob_list = []
y_true_list = []

print("\nRunning ONLY DINOv2 validation inference...")
with torch.no_grad():
    for images, labels in tqdm(loader, desc="DINOv2 Batch Progress"):
        images = images.to(device)
        features = bb(images)
        logits = clf(features)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        y_prob_list.append(probs)
        y_true_list.append(labels.numpy())

y_prob = np.concatenate(y_prob_list, axis=0)
y_true = np.concatenate(y_true_list, axis=0)
y_pred = np.argmax(y_prob, axis=1).astype(np.int32)

print("\nDINOv2 probability shape:", y_prob.shape)
print("DINOv2 prediction shape:", y_pred.shape)
print("DINOv2 true-label shape:", y_true.shape)

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
cm = confusion_matrix(y_true, y_pred)

print(f"\nAccuracy: {acc:.4f}")
print(f"Macro Precision: {prec:.4f}")
print(f"Macro Recall: {rec:.4f}")
print(f"Macro F1: {f1:.4f}")
print("\nConfusion Matrix:")
print(cm)
