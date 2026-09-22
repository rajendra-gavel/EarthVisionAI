from collections import Counter

import cv2
import numpy as np


class ImageAnalysisService:

    COLOR_RANGES = {
        "red": [
            ((0, 70, 50), (10, 255, 255)),
            ((170, 70, 50), (180, 255, 255)),
        ],
        "orange": [
            ((10, 70, 50), (25, 255, 255)),
        ],
        "yellow": [
            ((25, 70, 50), (35, 255, 255)),
        ],
        "green": [
            ((35, 40, 40), (85, 255, 255)),
        ],
        "cyan": [
            ((85, 40, 40), (100, 255, 255)),
        ],
        "blue": [
            ((100, 40, 40), (135, 255, 255)),
        ],
        "purple": [
            ((135, 40, 40), (170, 255, 255)),
        ],
    }

    def analyze(self, image: np.ndarray) -> dict:

        if image is None:
            raise ValueError("Image is empty.")

        height, width = image.shape[:2]

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV,
        )

        total_pixels = height * width

        color_counts = Counter()

        for color_name, ranges in self.COLOR_RANGES.items():

            mask = np.zeros(
                hsv.shape[:2],
                dtype=np.uint8,
            )

            for lower, upper in ranges:

                lower_np = np.array(
                    lower,
                    dtype=np.uint8,
                )

                upper_np = np.array(
                    upper,
                    dtype=np.uint8,
                )

                mask |= cv2.inRange(
                    hsv,
                    lower_np,
                    upper_np,
                )

            color_counts[color_name] = int(
                np.count_nonzero(mask)
            )

        # Neutral colors
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        color_counts["white"] = int(
            np.count_nonzero(
                (saturation < 40) &
                (value > 180)
            )
        )

        color_counts["gray"] = int(
            np.count_nonzero(
                (saturation < 40) &
                (value >= 60) &
                (value <= 180)
            )
        )

        color_counts["black"] = int(
            np.count_nonzero(
                value < 60
            )
        )

        percentages = {
            name: round(
                count / total_pixels * 100,
                2,
            )
            for name, count in color_counts.items()
            if count > 0
        }

        dominant_colors = sorted(
            percentages.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:5]

        gray_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        brightness = float(
            np.mean(gray_image)
        )

        if brightness < 70:
            brightness_label = "dark"
        elif brightness < 160:
            brightness_label = "moderately lit"
        else:
            brightness_label = "bright"

        return {
            "width": width,
            "height": height,
            "dominant_colors": [
                {
                    "name": name,
                    "percentage": percentage,
                }
                for name, percentage in dominant_colors
            ],
            "brightness": round(
                brightness,
                2,
            ),
            "brightness_label": brightness_label,
        }

    def get_dominant_color(
        self,
        image: np.ndarray,
    ) -> str:

        if image is None or image.size == 0:
            return "unknown"

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV,
        )

        total_pixels = image.shape[0] * image.shape[1]

        if total_pixels == 0:
            return "unknown"

        color_scores = {}

        for color_name, ranges in self.COLOR_RANGES.items():

            mask = np.zeros(
                hsv.shape[:2],
                dtype=np.uint8,
            )

            for lower, upper in ranges:

                lower_np = np.array(
                    lower,
                    dtype=np.uint8,
                )

                upper_np = np.array(
                    upper,
                    dtype=np.uint8,
                )

                mask |= cv2.inRange(
                    hsv,
                    lower_np,
                    upper_np,
                )

            count = np.count_nonzero(mask)

            color_scores[color_name] = (
                count / total_pixels
            )

        # Neutral colors

        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        white_ratio = np.count_nonzero(
            (saturation < 40) &
            (value > 180)
        ) / total_pixels

        gray_ratio = np.count_nonzero(
            (saturation < 40) &
            (value >= 60) &
            (value <= 180)
        ) / total_pixels

        black_ratio = np.count_nonzero(
            value < 60
        ) / total_pixels

        color_scores["white"] = white_ratio
        color_scores["gray"] = gray_ratio
        color_scores["black"] = black_ratio

        dominant_color = max(
            color_scores,
            key=color_scores.get,
        )

        # Avoid calling tiny color regions dominant.
        if color_scores[dominant_color] < 0.08:
            return "mixed"

        return dominant_color

image_analysis_service = ImageAnalysisService()