from pathlib import Path

import cv2
import numpy as np


ROOT = Path("/mnt/d/pocs/EarthVisionAI")

DATA = (
    ROOT
    / "data/raw/india/iccd/sample/ICCD_Sample/Agra/labeled"
)


def metrics(pred, truth):
    pred = pred > 0
    truth = truth > 0

    tp = np.logical_and(pred, truth).sum()
    fp = np.logical_and(pred, ~truth).sum()
    fn = np.logical_and(~pred, truth).sum()

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0
    )

    union = np.logical_or(pred, truth).sum()
    iou = tp / union if union else 0

    return precision, recall, f1, iou


for before_path in sorted((DATA / "im1").glob("*.png")):

    filename = before_path.name

    # Convert 2022 filename → corresponding 2023/label filename
    after_name = filename.replace("_2022_", "_2023_")

    after_path = DATA / "im2" / after_name
    label_path = DATA / "label" / after_name

    before = cv2.imread(str(before_path))
    after = cv2.imread(str(after_path))
    label = cv2.imread(
        str(label_path),
        cv2.IMREAD_GRAYSCALE
    )

    if before is None or after is None or label is None:
        print("Skipping:", filename)
        continue

    diff = cv2.absdiff(before, after)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    _, prediction = cv2.threshold(
        gray,
        30,
        255,
        cv2.THRESH_BINARY
    )

    precision, recall, f1, iou = metrics(
        prediction,
        label
    )

    changed = np.count_nonzero(prediction) / prediction.size * 100
    actual = np.count_nonzero(label) / label.size * 100

    print("\n", filename)
    print("-" * 50)
    print(f"Baseline changed area : {changed:.2f}%")
    print(f"Ground truth area     : {actual:.2f}%")
    print(f"Precision             : {precision:.4f}")
    print(f"Recall                : {recall:.4f}")
    print(f"F1                    : {f1:.4f}")
    print(f"IoU                   : {iou:.4f}")