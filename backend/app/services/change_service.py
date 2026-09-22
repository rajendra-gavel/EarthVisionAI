from pathlib import Path

import cv2
import numpy as np


class ChangeDetectionService:

    def __init__(self):

        self.output_dir = (
            Path("/mnt/d/pocs/EarthVisionAI")
            / "data"
            / "results"
            / "change_maps"
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # =========================================================
    # IMAGE LOADING
    # =========================================================

    def _load_image(self, path: str):

        image = cv2.imread(
            path,
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise ValueError(
                f"Unable to read image: {path}"
            )

        return image

    # =========================================================
    # FEATURE REGISTRATION
    # =========================================================

    def _feature_register(
        self,
        before,
        after,
    ):

        before_gray = cv2.cvtColor(
            before,
            cv2.COLOR_BGR2GRAY,
        )

        after_gray = cv2.cvtColor(
            after,
            cv2.COLOR_BGR2GRAY,
        )

        orb = cv2.ORB_create(
            nfeatures=3000,
            scaleFactor=1.2,
            nlevels=8,
        )

        keypoints_before, descriptors_before = (
            orb.detectAndCompute(
                before_gray,
                None,
            )
        )

        keypoints_after, descriptors_after = (
            orb.detectAndCompute(
                after_gray,
                None,
            )
        )

        if (
            descriptors_before is None
            or descriptors_after is None
        ):
            return after.copy(), False, 0.0

        matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING,
            crossCheck=False,
        )

        matches = matcher.knnMatch(
            descriptors_after,
            descriptors_before,
            k=2,
        )

        good_matches = []

        for pair in matches:

            if len(pair) < 2:
                continue

            first, second = pair

            if first.distance < 0.75 * second.distance:
                good_matches.append(first)

        if len(good_matches) < 8:
            return after.copy(), False, 0.0

        source_points = np.float32(
            [
                keypoints_after[m.queryIdx].pt
                for m in good_matches
            ]
        ).reshape(-1, 1, 2)

        destination_points = np.float32(
            [
                keypoints_before[m.trainIdx].pt
                for m in good_matches
            ]
        ).reshape(-1, 1, 2)

        homography, mask = cv2.findHomography(
            source_points,
            destination_points,
            cv2.RANSAC,
            5.0,
        )

        if homography is None or mask is None:
            return after.copy(), False, 0.0

        inliers = int(
            np.count_nonzero(mask)
        )

        inlier_ratio = (
            inliers / len(good_matches)
        )

        if inliers < 8 or inlier_ratio < 0.20:
            return after.copy(), False, inlier_ratio

        aligned = cv2.warpPerspective(
            after,
            homography,
            (
                before.shape[1],
                before.shape[0],
            ),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        return (
            aligned,
            True,
            float(inlier_ratio),
        )

    # =========================================================
    # ECC REFINEMENT
    # =========================================================

    def _ecc_refine(
        self,
        before,
        after,
    ):

        before_gray = cv2.cvtColor(
            before,
            cv2.COLOR_BGR2GRAY,
        )

        after_gray = cv2.cvtColor(
            after,
            cv2.COLOR_BGR2GRAY,
        )

        before_gray = cv2.GaussianBlur(
            before_gray,
            (5, 5),
            0,
        )

        after_gray = cv2.GaussianBlur(
            after_gray,
            (5, 5),
            0,
        )

        before_float = (
            before_gray.astype(
                np.float32
            )
            / 255.0
        )

        after_float = (
            after_gray.astype(
                np.float32
            )
            / 255.0
        )

        warp_matrix = np.eye(
            2,
            3,
            dtype=np.float32,
        )

        criteria = (
            cv2.TERM_CRITERIA_EPS
            | cv2.TERM_CRITERIA_COUNT,
            100,
            1e-6,
        )

        try:

            correlation, warp_matrix = (
                cv2.findTransformECC(
                    before_float,
                    after_float,
                    warp_matrix,
                    cv2.MOTION_AFFINE,
                    criteria,
                    None,
                    1,
                )
            )

            refined = cv2.warpAffine(
                after,
                warp_matrix,
                (
                    before.shape[1],
                    before.shape[0],
                ),
                flags=(
                    cv2.INTER_LINEAR
                    | cv2.WARP_INVERSE_MAP
                ),
                borderMode=cv2.BORDER_REFLECT,
            )

            return (
                refined,
                True,
                float(correlation),
            )

        except cv2.error:

            return (
                after.copy(),
                False,
                0.0,
            )

    # =========================================================
    # RADIOMETRIC NORMALIZATION
    # =========================================================

    def _normalize_intensity(
        self,
        before,
        after,
    ):

        before_lab = cv2.cvtColor(
            before,
            cv2.COLOR_BGR2LAB,
        )

        after_lab = cv2.cvtColor(
            after,
            cv2.COLOR_BGR2LAB,
        )

        before_l = before_lab[:, :, 0]
        after_l = after_lab[:, :, 0]

        before_mean = float(
            np.mean(before_l)
        )

        before_std = float(
            np.std(before_l)
        )

        after_mean = float(
            np.mean(after_l)
        )

        after_std = float(
            np.std(after_l)
        )

        if after_std < 1e-6:
            return after.copy()

        normalized_l = (
            (
                after_l.astype(
                    np.float32
                )
                - after_mean
            )
            * (
                before_std
                / max(after_std, 1e-6)
            )
            + before_mean
        )

        normalized_l = np.clip(
            normalized_l,
            0,
            255,
        ).astype(
            np.uint8
        )

        normalized_lab = after_lab.copy()

        normalized_lab[:, :, 0] = (
            normalized_l
        )

        return cv2.cvtColor(
            normalized_lab,
            cv2.COLOR_LAB2BGR,
        )

    # =========================================================
    # STRUCTURAL DIFFERENCE
    # =========================================================

    def _structural_difference(
        self,
        before,
        after,
    ):

        before_gray = cv2.cvtColor(
            before,
            cv2.COLOR_BGR2GRAY,
        )

        after_gray = cv2.cvtColor(
            after,
            cv2.COLOR_BGR2GRAY,
        )

        before_gray = cv2.GaussianBlur(
            before_gray,
            (7, 7),
            0,
        )

        after_gray = cv2.GaussianBlur(
            after_gray,
            (7, 7),
            0,
        )

        before_float = (
            before_gray.astype(
                np.float32
            )
        )

        after_float = (
            after_gray.astype(
                np.float32
            )
        )

        mu_before = cv2.GaussianBlur(
            before_float,
            (11, 11),
            1.5,
        )

        mu_after = cv2.GaussianBlur(
            after_float,
            (11, 11),
            1.5,
        )

        sigma_before = (
            cv2.GaussianBlur(
                before_float ** 2,
                (11, 11),
                1.5,
            )
            - mu_before ** 2
        )

        sigma_after = (
            cv2.GaussianBlur(
                after_float ** 2,
                (11, 11),
                1.5,
            )
            - mu_after ** 2
        )

        sigma_cross = (
            cv2.GaussianBlur(
                before_float * after_float,
                (11, 11),
                1.5,
            )
            - mu_before * mu_after
        )

        c1 = 6.5025
        c2 = 58.5225

        numerator = (
            (2 * mu_before * mu_after + c1)
            * (2 * sigma_cross + c2)
        )

        denominator = (
            (
                mu_before ** 2
                + mu_after ** 2
                + c1
            )
            * (
                sigma_before
                + sigma_after
                + c2
            )
        )

        similarity = (
            numerator
            / (denominator + 1e-6)
        )

        similarity = np.clip(
            similarity,
            -1,
            1,
        )

        return (
            1.0 - similarity
        )

    # =========================================================
    # SPECTRAL DIFFERENCE
    # =========================================================

    def _spectral_difference(
        self,
        before,
        after,
    ):

        before_lab = cv2.cvtColor(
            before,
            cv2.COLOR_BGR2LAB,
        ).astype(
            np.float32
        )

        after_lab = cv2.cvtColor(
            after,
            cv2.COLOR_BGR2LAB,
        ).astype(
            np.float32
        )

        difference = np.linalg.norm(
            before_lab - after_lab,
            axis=2,
        )

        difference /= np.sqrt(
            255 ** 2 * 3
        )

        return np.clip(
            difference,
            0,
            1,
        )

    # =========================================================
    # NORMALIZATION
    # =========================================================

    def _robust_normalize(
        self,
        values,
    ):

        values = values.astype(
            np.float32
        )

        median = np.median(values)

        mad = np.median(
            np.abs(
                values - median
            )
        )

        if mad < 1e-6:

            minimum = values.min()
            maximum = values.max()

            if maximum - minimum < 1e-6:
                return np.zeros_like(
                    values
                )

            return (
                values - minimum
            ) / (
                maximum - minimum
            )

        robust_score = (
            values - median
        ) / (
            1.4826 * mad
            + 1e-6
        )

        score = 1.0 - np.exp(
            -np.maximum(
                robust_score,
                0,
            )
            / 3.0
        )

        return np.clip(
            score,
            0,
            1,
        )

    # =========================================================
    # REGION EXTRACTION
    # =========================================================

    def _extract_regions(
        self,
        mask,
        score,
    ):

        number_labels, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                mask,
                connectivity=8,
            )
        )

        height, width = mask.shape

        image_area = (
            height * width
        )

        # Ignore tiny noise.
        min_area = max(
            150,
            int(image_area * 0.00015),
        )

        regions = []

        clean_mask = np.zeros_like(
            mask
        )

        for label in range(
            1,
            number_labels,
        ):

            area = int(
                stats[
                    label,
                    cv2.CC_STAT_AREA,
                ]
            )

            if area < min_area:
                continue

            x = int(
                stats[
                    label,
                    cv2.CC_STAT_LEFT,
                ]
            )

            y = int(
                stats[
                    label,
                    cv2.CC_STAT_TOP,
                ]
            )

            region_width = int(
                stats[
                    label,
                    cv2.CC_STAT_WIDTH,
                ]
            )

            region_height = int(
                stats[
                    label,
                    cv2.CC_STAT_HEIGHT,
                ]
            )

            region_score = float(
                np.mean(
                    score[
                        labels == label
                    ]
                )
            )

            if region_score >= 0.75:
                severity = "high"

            elif region_score >= 0.50:
                severity = "medium"

            else:
                severity = "low"

            clean_mask[
                labels == label
            ] = 255

            regions.append(
                {
                    "id": len(regions) + 1,
                    "area_pixels": area,
                    "bbox": [
                        x,
                        y,
                        region_width,
                        region_height,
                    ],
                    "severity": severity,
                    "score": round(
                        region_score,
                        4,
                    ),
                }
            )

        regions.sort(
            key=lambda item: item["area_pixels"],
            reverse=True,
        )

        # Re-number after sorting.
        for index, region in enumerate(
            regions,
            start=1,
        ):
            region["id"] = index

        return clean_mask, regions

    # =========================================================
    # OVERLAY
    # =========================================================

    def _create_overlay(
        self,
        image,
        mask,
        regions,
    ):

        overlay = image.copy()

        red_layer = np.zeros_like(
            image
        )

        red_layer[:, :] = (
            0,
            0,
            255,
        )

        changed = (
            mask > 0
        )

        if np.any(changed):

            overlay[changed] = (
                cv2.addWeighted(
                    image[changed],
                    0.45,
                    red_layer[changed],
                    0.55,
                    0,
                )
            )

        # Draw region boundaries.
        for region in regions:

            x, y, w, h = (
                region["bbox"]
            )

            cv2.rectangle(
                overlay,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                2,
            )

            label = (
                f"#{region['id']} "
                f"{region['severity']}"
            )

            cv2.putText(
                overlay,
                label,
                (x, max(18, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        return overlay

    # =========================================================
    # MAIN CHANGE ANALYSIS
    # =========================================================

    def detect_change(
        self,
        before_path: str,
        after_path: str,
    ):

        before = self._load_image(
            before_path
        )

        after = self._load_image(
            after_path
        )

        # -----------------------------------------------------
        # Same dimensions
        # -----------------------------------------------------

        if (
            before.shape[:2]
            != after.shape[:2]
        ):

            after = cv2.resize(
                after,
                (
                    before.shape[1],
                    before.shape[0],
                ),
                interpolation=cv2.INTER_LINEAR,
            )

        # -----------------------------------------------------
        # Feature registration
        # -----------------------------------------------------

        feature_aligned, feature_ok, feature_score = (
            self._feature_register(
                before,
                after,
            )
        )

        # -----------------------------------------------------
        # ECC refinement
        # -----------------------------------------------------

        if feature_ok:

            aligned_after, ecc_ok, ecc_score = (
                self._ecc_refine(
                    before,
                    feature_aligned,
                )
            )

        else:

            aligned_after, ecc_ok, ecc_score = (
                self._ecc_refine(
                    before,
                    after,
                )
            )

        alignment_applied = (
            feature_ok or ecc_ok
        )

        # -----------------------------------------------------
        # Radiometric normalization
        # -----------------------------------------------------

        normalized_after = (
            self._normalize_intensity(
                before,
                aligned_after,
            )
        )

        # -----------------------------------------------------
        # Difference maps
        # -----------------------------------------------------

        structural = (
            self._structural_difference(
                before,
                normalized_after,
            )
        )

        spectral = (
            self._spectral_difference(
                before,
                normalized_after,
            )
        )

        structural_score = (
            self._robust_normalize(
                structural
            )
        )

        spectral_score = (
            self._robust_normalize(
                spectral
            )
        )

        # -----------------------------------------------------
        # Multi-scale smoothing
        # -----------------------------------------------------

        structural_small = cv2.GaussianBlur(
            structural_score,
            (5, 5),
            0,
        )

        structural_large = cv2.GaussianBlur(
            structural_score,
            (11, 11),
            0,
        )

        spectral_smooth = cv2.GaussianBlur(
            spectral_score,
            (7, 7),
            0,
        )

        combined_score = (
            0.55 * structural_small
            + 0.25 * structural_large
            + 0.20 * spectral_smooth
        )

        combined_score = np.clip(
            combined_score,
            0,
            1,
        )

        # -----------------------------------------------------
        # Candidate threshold
        # -----------------------------------------------------

        threshold = float(
            np.percentile(
                combined_score,
                98.0,
            )
        )

        raw_mask = (
            combined_score >= threshold
        ).astype(
            np.uint8
        ) * 255

        # -----------------------------------------------------
        # Morphological cleanup
        # -----------------------------------------------------

        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (11, 11),
        )

        raw_mask = cv2.morphologyEx(
            raw_mask,
            cv2.MORPH_CLOSE,
            close_kernel,
        )

        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5),
        )

        raw_mask = cv2.morphologyEx(
            raw_mask,
            cv2.MORPH_OPEN,
            open_kernel,
        )

        # -----------------------------------------------------
        # Region extraction
        # -----------------------------------------------------

        clean_mask, regions = (
            self._extract_regions(
                raw_mask,
                combined_score,
            )
        )

        # -----------------------------------------------------
        # Overlay
        # -----------------------------------------------------

        overlay = self._create_overlay(
            normalized_after,
            clean_mask,
            regions,
        )

        # -----------------------------------------------------
        # Statistics
        # -----------------------------------------------------

        height, width = (
            before.shape[:2]
        )

        total_pixels = (
            height * width
        )

        changed_pixels = int(
            np.count_nonzero(
                clean_mask
            )
        )

        change_percentage = round(
            changed_pixels
            / total_pixels
            * 100,
            2,
        )

        # -----------------------------------------------------
        # Output
        # -----------------------------------------------------

        base_name = (
            Path(after_path).stem
        )

        change_map_path = (
            self.output_dir
            / f"{base_name}_change_map.png"
        )

        overlay_path = (
            self.output_dir
            / f"{base_name}_change_overlay.png"
        )

        score_path = (
            self.output_dir
            / f"{base_name}_change_score.png"
        )

        aligned_path = (
            self.output_dir
            / f"{base_name}_aligned_after.png"
        )

        cv2.imwrite(
            str(change_map_path),
            clean_mask,
        )

        cv2.imwrite(
            str(overlay_path),
            overlay,
        )

        score_image = (
            np.clip(
                combined_score * 255,
                0,
                255,
            )
            .astype(
                np.uint8
            )
        )

        cv2.imwrite(
            str(score_path),
            score_image,
        )

        cv2.imwrite(
            str(aligned_path),
            normalized_after,
        )

        # -----------------------------------------------------
        # Response
        # -----------------------------------------------------

        return {

            "width": width,

            "height": height,

            "changed_pixels": changed_pixels,

            "change_percentage": change_percentage,

            "candidate_regions": len(
                regions
            ),

            "regions": regions,

            "method": (
                "feature_registered_structural_spectral"
            ),

            "method_label": (
                "Feature Registration + "
                "Structural + Spectral Analysis"
            ),

            "alignment": alignment_applied,

            "feature_registration": feature_ok,

            "feature_match_score": round(
                float(feature_score),
                4,
            ),

            "ecc_refinement": ecc_ok,

            "alignment_score": round(
                float(ecc_score),
                4,
            ),

            "threshold": round(
                threshold,
                4,
            ),

            "change_map": (
                "/data/results/change_maps/"
                + change_map_path.name
            ),

            "change_overlay": (
                "/data/results/change_maps/"
                + overlay_path.name
            ),

            "change_score": (
                "/data/results/change_maps/"
                + score_path.name
            ),

            "aligned_image": (
                "/data/results/change_maps/"
                + aligned_path.name
            ),

            "experimental": True,

            "note": (
                "Potential visual differences "
                "identified using feature registration, "
                "structural analysis and spectral "
                "analysis. Results may include "
                "illumination, seasonal, acquisition, "
                "or residual registration differences "
                "and should not be interpreted as "
                "confirmed geographic change."
            ),
        }