import os
import torch
import torch.nn as nn

checkpoint_path = "models/dinov2_100_epoch_best.pth"
abs_path = os.path.abspath(checkpoint_path)
exists = os.path.exists(abs_path)
file_size = os.path.getsize(abs_path) if exists else 0

print("1. Exact resolved checkpoint path:", checkpoint_path)
print("2. Whether the file exists:", exists)
print("3. Absolute path:", abs_path)
print("4. File size:", file_size, "bytes")

if not exists:
    print("File does not exist. STOPPING.")
    exit(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(checkpoint_path, map_location=device)

print("\n--- CHECKPOINT METADATA ---")
print("feature_dim =", checkpoint.get("feature_dim"))
print("num_classes =", checkpoint.get("num_classes"))
print("class_names =\n", checkpoint.get("class_names"))
print("class_to_idx =\n", checkpoint.get("class_to_idx"))
print("best_val_accuracy =", checkpoint.get("best_val_accuracy"))
print("best_epoch =", checkpoint.get("best_epoch"))
print("epochs =", checkpoint.get("epochs"))
print("learning_rate =", checkpoint.get("learning_rate"))

print("\n--- STATE DICT KEYS ---")
if "model_state_dict" in checkpoint:
    keys = list(checkpoint["model_state_dict"].keys())
    print(keys)
else:
    keys = list(checkpoint.keys())
    print(keys)

class DINOv2LinearClassifier(nn.Module):
    def __init__(self, feature_dim=768, num_classes=3):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x):
        return self.classifier(x)

classifier = DINOv2LinearClassifier(
    feature_dim=checkpoint.get("feature_dim", 768),
    num_classes=checkpoint.get("num_classes", 3)
).to(device)

if "model_state_dict" in checkpoint:
    classifier.load_state_dict(checkpoint["model_state_dict"])
else:
    classifier.load_state_dict(checkpoint)

print("\n============================================================")
print("DINOV2 CHECKPOINT VERIFICATION")
print("============================================================")
print(f"Loaded file:\n{abs_path}")
print(f"File exists:\n{exists}")
print(f"Feature dimension:\n{checkpoint.get('feature_dim')}")
print(f"Classes:\n{checkpoint.get('num_classes')}")
if checkpoint.get('best_val_accuracy'):
    print(f"Best validation accuracy:\n{checkpoint.get('best_val_accuracy')*100:.2f}%")
else:
    print("Best validation accuracy:\nNone")
print(f"Best epoch:\n{checkpoint.get('best_epoch')}")
print(f"Training epochs:\n{checkpoint.get('epochs')}")

if checkpoint.get('best_epoch') == 50 and checkpoint.get('epochs') == 100:
    print("\n✓ CORRECT 100-EPOCH CHECKPOINT CONFIRMED")
else:
    print("\n✗ CHECKPOINT MISMATCH")
