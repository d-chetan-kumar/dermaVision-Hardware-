import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from evaluation import load_prediction_model, generate_mobilenet_predictions, load_dinov2_model, prepare_validation_dataset
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

print("Generating probabilities (since they were not cached to disk)...")
_, val_df = prepare_validation_dataset("ISIC_2019_Training_GroundTruth.csv", "ISIC_2019_Training_Input")

# MobileNet
mb_model = load_prediction_model()
mb_y_true, mb_y_prob, _ = generate_mobilenet_predictions(mb_model, val_df)

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
        return self.transform(Image.open(row['image_path']).convert('RGB')), row['label']

bb, clf = load_dinov2_model()
dataset = DINOv2Dataset(val_df)
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bb = bb.to(device)
clf = clf.to(device)

dino_y_prob_list = []

with torch.no_grad():
    for images, _ in tqdm(loader, desc="DINOv2 Batch Progress"):
        probs = torch.softmax(clf(bb(images.to(device))), dim=1).cpu().numpy()
        dino_y_prob_list.append(probs)

dino_y_prob = np.concatenate(dino_y_prob_list, axis=0)

print("\nStarting probability sweep...")

results = []
weights = [(100, 0), (90, 10), (80, 20), (70, 30), (60, 40), (50, 50), (40, 60), (30, 70), (20, 80), (10, 90), (0, 100)]

for mb_w, dino_w in weights:
    ens_y_prob = (mb_w / 100.0) * mb_y_prob + (dino_w / 100.0) * dino_y_prob
    ens_y_pred = np.argmax(ens_y_prob, axis=1).astype(np.int32)
    
    acc = accuracy_score(mb_y_true, ens_y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(mb_y_true, ens_y_pred, average='macro', zero_division=0)
    
    results.append({
        'MobileNetV2_W': mb_w,
        'DINOv2_W': dino_w,
        'Accuracy': acc * 100,
        'Macro_Precision': prec * 100,
        'Macro_Recall': rec * 100,
        'Macro_F1': f1 * 100
    })

df = pd.DataFrame(results)
df = df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)

print("\n============================================================")
print("WEIGHT SWEEP RESULTS (Sorted by Accuracy)")
print("============================================================")
print(df.to_string(index=False, float_format="%.2f"))

best = df.iloc[0]
print("\n============================================================")
print("BEST WEIGHT CONFIGURATION")
print("============================================================")
print(f"BEST WEIGHT: MobileNetV2={int(best['MobileNetV2_W'])} / DINOv2={int(best['DINOv2_W'])}")
print(f"BEST ACCURACY: {best['Accuracy']:.2f}%")
print(f"BEST MACRO F1: {best['Macro_F1']:.2f}%")
