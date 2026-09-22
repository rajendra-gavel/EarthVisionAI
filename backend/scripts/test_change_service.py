from pathlib import Path

from app.services.change_service import ChangeDetectionService


ROOT = Path("/mnt/d/pocs/EarthVisionAI")

DATA = (
    ROOT
    / "data/raw/india/iccd/sample/ICCD_Sample/Agra/labeled"
)

before = DATA / "im1/Agra_0_2022_r00_c01.png"
after = DATA / "im2/Agra_0_2023_r00_c01.png"

service = ChangeDetectionService()

result = service.detect_change(
    str(before),
    str(after),
)

print("\nEarthVision Change Detection")
print("=" * 40)

for key, value in result.items():
    print(f"{key}: {value}")