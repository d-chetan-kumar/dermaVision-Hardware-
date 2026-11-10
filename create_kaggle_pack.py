import zipfile
from pathlib import Path

zip_path = Path("dermavision_kaggle_pack.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    z.write("DermaVision_Revision_V1_Training_Kaggle.ipynb", arcname="DermaVision_Revision_V1_Training_Kaggle.ipynb")
    z.write("data/splits/revision_v1/train.csv", arcname="train.csv")
    z.write("data/splits/revision_v1/val.csv", arcname="val.csv")
    z.write("data/splits/revision_v1/test.csv", arcname="test.csv")

print(f"Package created: {zip_path.name} ({zip_path.stat().st_size:,} bytes)")

