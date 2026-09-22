from pathlib import Path

import cv2
import numpy as np


class ChangeDetectionService:

    def __init__(self):
        self.project_root = Path("/mnt/d/pocs/EarthVisionAI")

        self.change_map_dir = (
            self.project_root
            / "data"
            / "results"
            / "change_maps"
        )

        self.change_map_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def _align_image(self, before, after):
        """
        Align AFTER image to BEFORE using ECC translation.
        Falls back to the original image if alignment fails.
        """

        before_gray = cv2.cvtColor(
            before,
            cv2.COLOR_BGR2GRAY
        )

        after_gray = cv2.cvtColor(
            after,
            cv2.COLOR_BGR2GRAY
        )

        before_gray = before_gray.astype(np.float32) / 255.0
        after_gray = after_gray.astype(np.float32) / 255.0

        warp_matrix = np.eye(
            2,
            3,
            dtype=np.float32
        )

        criteria = (
            cv2.TERM_CRITERIA_EPS
            | cv2.TERM_CRITERIA_COUNT,
            100,
            1e-6,
        )

        try:
            _, warp_matrix = cv2.findTransformECC(
                before_gray,
                after_gray,
                warp_matrix,
                cv2.MOTION_TRANSLATION,
                criteria,
            )

            aligned = cv2.warpAffine(
                after,
                warp_matrix,
                (
                    after.shape[1],
                    after.shape[0],
                ),
                flags=cv2.INTER_LINEAR
                | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT,
            )

            return aligned, True

        except cv2.error:
            return after, False

    def _normalize_intensity(self, before, after):
        """
        Normalize AFTER brightness to BEFORE using
        per-channel mean/std normalization.
        """

        before_float = before.astype(np.float32)
        after_float = after.astype(np.float32)

        before_mean = before_float.mean(
            axis=(0, 1),
            keepdims=True,
        )

        before_std = before_float.std(
            axis=(0, 1),
            keepdims=True,
        )

        after_mean = after_float.mean(
            axis=(0, 1),
            keepdims=True,
        )

        after_std = after_float.std(
            axis=(0, 1),
            keepdims=True,
        )

        after_std = np.maximum(
            after_std,
            1.0,
        )

        normalized = (
            (after_float - after_mean)
            / after_std
            * before_std
            + before_mean
        )

        return np.clip(
            normalized,
            0,
            255,
        ).astype(np.uint8)

    def detect_change(
        self,
        before_path: str,
        after_path: str,
    ):

        before = cv2.imread(before_path)
        after = cv2.imread(after_path)

        if before is None:
            raise ValueError(
                f"Unable to read before image: {before_path}"
            )

        if after is None:
            raise ValueError(
                f"Unable to read after image: {after_path}"
            )

        if before.shape != after.shape:
            raise ValueError(
                f"Image dimensions do not match: "
                f"{before.shape} vs {after.shape}"
            )

        height, width = before.shape[:2]

        # --------------------------------------------------
        # 1. Image registration
        # --------------------------------------------------

        aligned_after, alignment_success = (
            self._align_image(
                before,
                after,
            )
        )

        # --------------------------------------------------
        # 2. Intensity normalization
        # --------------------------------------------------

        normalized_after = (
            self._normalize_intensity(
                before,
                aligned_after,
            )
        )

        # --------------------------------------------------
        # 3. Blur to suppress pixel-level noise
        # --------------------------------------------------

        before_blur = cv2.GaussianBlur(
            before,
            (5, 5),
            0,
        )

        after_blur = cv2.GaussianBlur(
            normalized_after,
            (5, 5),
            0,
        )

        # --------------------------------------------------
        # 4. Difference
        # --------------------------------------------------

        difference = cv2.absdiff(
            before_blur,
            after_blur,
        )

        gray_difference = cv2.cvtColor(
            difference,
            cv2.COLOR_BGR2GRAY,
        )

        # --------------------------------------------------
        # 5. Otsu adaptive threshold
        # --------------------------------------------------

        threshold_value, change_mask = cv2.threshold(
            gray_difference,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        # --------------------------------------------------
        # 6. Morphological cleanup
        # --------------------------------------------------

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5),
        )

        change_mask = cv2.morphologyEx(
            change_mask,
            cv2.MORPH_OPEN,
            kernel,
        )

        change_mask = cv2.morphologyEx(
            change_mask,
            cv2.MORPH_CLOSE,
            kernel,
        )

        # --------------------------------------------------
        # 7. Remove very small regions
        # --------------------------------------------------

        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                change_mask,
                connectivity=8,
            )
        )

        cleaned_mask = np.zeros_like(
            change_mask
        )

        min_area = max(
            50,
            int(width * height * 0.00005),
        )

        for label in range(1, num_labels):

            area = stats[
                label,
                cv2.CC_STAT_AREA
            ]

            if area >= min_area:
                cleaned_mask[
                    labels == label
                ] = 255

        change_mask = cleaned_mask

        # --------------------------------------------------
        # 8. Statistics
        # --------------------------------------------------

        changed_pixels = int(
            cv2.countNonZero(
                change_mask
            )
        )

        total_pixels = width * height

        change_percentage = (
            changed_pixels
            / total_pixels
        ) * 100

        # --------------------------------------------------
        # 9. Save binary change map
        # --------------------------------------------------

        before_name = Path(
            before_path
        ).stem

        after_name = Path(
            after_path
        ).stem

        output_name = (
            f"{before_name}_to_"
            f"{after_name}_change.png"
        )

        output_path = (
            self.change_map_dir
            / output_name
        )

        if not cv2.imwrite(
            str(output_path),
            change_mask,
        ):
            raise RuntimeError(
                f"Failed to save change map: "
                f"{output_path}"
            )

        # --------------------------------------------------
        # 10. Create visual overlay
        # --------------------------------------------------

        overlay = aligned_after.copy()

        red_layer = np.zeros_like(
            overlay
        )

        red_layer[:, :, 2] = 255

        mask_bool = change_mask > 0

        overlay[mask_bool] = cv2.addWeighted(
            overlay[mask_bool],
            0.45,
            red_layer[mask_bool],
            0.55,
            0,
        )

        overlay_name = (
            f"{before_name}_to_"
            f"{after_name}_overlay.png"
        )

        overlay_path = (
            self.change_map_dir
            / overlay_name
        )

        if not cv2.imwrite(
            str(overlay_path),
            overlay,
        ):
            raise RuntimeError(
                f"Failed to save overlay: "
                f"{overlay_path}"
            )

        return {
            "before_image": before_path,
            "after_image": after_path,

            "width": width,
            "height": height,

            "changed_pixels": changed_pixels,
            "total_pixels": total_pixels,

            "change_percentage": round(
                change_percentage,
                2,
            ),

            "method": (
                "registered_normalized_difference"
            ),

            "threshold": round(
                float(threshold_value),
                2,
            ),

            "alignment": alignment_success,

            "change_map": (
                f"/data/results/change_maps/"
                f"{output_name}"
            ),

            "change_overlay": (
                f"/data/results/change_maps/"
                f"{overlay_name}"
            ),
        }