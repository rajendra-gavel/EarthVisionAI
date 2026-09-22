from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path("/mnt/d/pocs/EarthVisionAI")

DATA = (
    ROOT
    / "data/raw/india/iccd/sample/ICCD_Sample/Agra/labeled"
)

OUTPUT = ROOT / "results/india/agra"
OUTPUT.mkdir(parents=True, exist_ok=True)

# First labeled Agra pair
before_path = DATA / "im1/Agra_0_2022_r00_c01.png"
after_path = DATA / "im2/Agra_0_2023_r00_c01.png"
label_path = DATA / "label/Agra_0_2023_r00_c01.png"

before = np.array(Image.open(before_path).convert("RGB"))
after = np.array(Image.open(after_path).convert("RGB"))
label = np.array(Image.open(label_path))

print("Before :", before.shape, before.dtype)
print("After  :", after.shape, after.dtype)
print("Label  :", label.shape, label.dtype)

print("\nLabel values:")
values, counts = np.unique(label, return_counts=True)

for value, count in zip(values, counts):
    percentage = count / label.size * 100
    print(f"  {value}: {count:,} pixels ({percentage:.2f}%)")

# Save copies for the POC
Image.fromarray(before).save(OUTPUT / "agra_before.png")
Image.fromarray(after).save(OUTPUT / "agra_after.png")
Image.fromarray(label).save(OUTPUT / "agra_ground_truth.png")

print("\nSaved:")
print(OUTPUT / "agra_before.png")
print(OUTPUT / "agra_after.png")
print(OUTPUT / "agra_ground_truth.png")