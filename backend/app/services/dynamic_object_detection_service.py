from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from ultralytics import YOLO

from app.services.image_analysis_service import (
    image_analysis_service,
)


ROOT = Path(__file__).resolve().parents[3]


MODEL_PATHS = {
    "aerial": ROOT / "yolo26n-obb.pt",
    "general": ROOT / "yolo11n.pt",
}


MODEL_NAMES = {
    "aerial": "YOLO26n-OBB",
    "general": "YOLO11n",
}


OUTPUT_DIR = (
    ROOT
    / "data"
    / "results"
    / "object_detection"
    / "uploads"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


class DynamicObjectDetectionService:

    def __init__(self):
        self.models = {}


    # =========================================================
    # MODEL LOADING
    # =========================================================

    def _get_model(
        self,
        model_type: str,
    ):

        if model_type not in MODEL_PATHS:

            raise ValueError(
                f"Unsupported model '{model_type}'. "
                f"Use aerial or general."
            )


        if model_type not in self.models:

            model_path = MODEL_PATHS[
                model_type
            ]


            if not model_path.exists():

                raise FileNotFoundError(
                    f"Model not found: {model_path}"
                )


            self.models[model_type] = YOLO(
                str(model_path)
            )


        return self.models[model_type]


    # =========================================================
    # DOMINANT COLOR FOR DETECTED OBJECT
    # =========================================================

    def _get_bbox_color(
        self,
        image: np.ndarray,
        bbox: list,
    ) -> str:

        if not bbox or len(bbox) != 4:
            return "unknown"


        x1, y1, x2, y2 = [
            int(value)
            for value in bbox
        ]


        height, width = image.shape[:2]


        x1 = max(
            0,
            min(x1, width - 1),
        )

        x2 = max(
            0,
            min(x2, width),
        )

        y1 = max(
            0,
            min(y1, height - 1),
        )

        y2 = max(
            0,
            min(y2, height),
        )


        if x2 <= x1 or y2 <= y1:
            return "unknown"


        crop = image[
            y1:y2,
            x1:x2,
        ]


        if crop.size == 0:
            return "unknown"


        return (
            image_analysis_service
            .get_dominant_color(crop)
        )


    # =========================================================
    # DETECTION
    # =========================================================

    def detect_bytes(
        self,
        image_bytes: bytes,
        filename: str,
        model_type: str = "aerial",
    ):

        if not image_bytes:

            raise ValueError(
                "Uploaded image is empty."
            )


        # -----------------------------------------------------
        # Decode image
        # -----------------------------------------------------

        array = np.frombuffer(
            image_bytes,
            dtype=np.uint8,
        )


        image = cv2.imdecode(
            array,
            cv2.IMREAD_COLOR,
        )


        if image is None:

            raise ValueError(
                "Unable to decode image. "
                "Please upload a valid JPG or PNG."
            )


        height, width = image.shape[:2]


        # -----------------------------------------------------
        # Image-level analysis
        # -----------------------------------------------------

        image_analysis = (
            image_analysis_service.analyze(
                image
            )
        )


        # -----------------------------------------------------
        # Load model
        # -----------------------------------------------------

        model = self._get_model(
            model_type
        )


        # -----------------------------------------------------
        # Run inference
        # -----------------------------------------------------

        results = model.predict(
            source=image,
            conf=0.40,
            verbose=False,
        )


        result = results[0]


        detections = []

        counts = {}


        # =====================================================
        # OBB MODEL
        # =====================================================

        if getattr(
            result,
            "obb",
            None,
        ) is not None:

            obb = result.obb


            if obb.cls is not None:

                classes = (
                    obb.cls
                    .cpu()
                    .numpy()
                )


                confidences = (

                    obb.conf
                    .cpu()
                    .numpy()

                    if obb.conf is not None

                    else np.ones(
                        len(classes)
                    )
                )


                polygons = (

                    obb.xyxyxyxy
                    .cpu()
                    .numpy()

                    if obb.xyxyxyxy is not None

                    else None
                )


                for i, cls_id in enumerate(
                    classes
                ):

                    class_id = int(
                        cls_id
                    )


                    class_name = (
                        result.names[
                            class_id
                        ]
                    )


                    confidence = float(
                        confidences[i]
                    )


                    polygon = []


                    if polygons is not None:

                        polygon = (
                            polygons[i]
                            .reshape(-1, 2)
                            .tolist()
                        )


                    detections.append(
                        {
                            "class": class_name,
                            "confidence": confidence,
                            "polygon": polygon,
                        }
                    )


                    counts[class_name] = (
                        counts.get(
                            class_name,
                            0,
                        )
                        + 1
                    )


        # =====================================================
        # STANDARD YOLO BOUNDING BOX MODEL
        # =====================================================

        elif getattr(
            result,
            "boxes",
            None,
        ) is not None:

            boxes = result.boxes


            if boxes.cls is not None:

                classes = (
                    boxes.cls
                    .cpu()
                    .numpy()
                )


                confidences = (

                    boxes.conf
                    .cpu()
                    .numpy()

                    if boxes.conf is not None

                    else np.ones(
                        len(classes)
                    )
                )


                xyxy = (

                    boxes.xyxy
                    .cpu()
                    .numpy()

                    if boxes.xyxy is not None

                    else None
                )


                for i, cls_id in enumerate(
                    classes
                ):

                    class_id = int(
                        cls_id
                    )


                    class_name = (
                        result.names[
                            class_id
                        ]
                    )


                    confidence = float(
                        confidences[i]
                    )


                    bbox = []


                    if xyxy is not None:

                        bbox = (
                            xyxy[i]
                            .astype(float)
                            .tolist()
                        )


                    # -------------------------------------------------
                    # Analyze dominant color inside detected object
                    # -------------------------------------------------

                    dominant_color = (
                        self._get_bbox_color(
                            image,
                            bbox,
                        )
                    )


                    detections.append(
                        {
                            "class": class_name,
                            "confidence": confidence,
                            "bbox": bbox,
                            "dominant_color": dominant_color,
                        }
                    )


                    counts[class_name] = (
                        counts.get(
                            class_name,
                            0,
                        )
                        + 1
                    )


        # =====================================================
        # ANNOTATED OUTPUT
        # =====================================================

        annotated = result.plot()


        output_name = (
            f"{uuid4().hex}_"
            f"{Path(filename).stem}.jpg"
        )


        output_path = (
            OUTPUT_DIR / output_name
        )


        cv2.imwrite(
            str(output_path),
            annotated,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                92,
            ],
        )


        # =====================================================
        # RESPONSE
        # =====================================================

        return {
            "model": MODEL_NAMES[
                model_type
            ],

            "model_type": model_type,

            "filename": filename,

            "width": width,

            "height": height,

            "detections": detections,

            "counts": counts,

            "total_detections": len(
                detections
            ),

            "image_analysis": image_analysis,

            "annotated_image": (
                "/data/results/"
                "object_detection/uploads/"
                + output_name
            ),
        }


# =============================================================
# SINGLE SERVICE INSTANCE
# =============================================================

dynamic_object_detection_service = (
    DynamicObjectDetectionService()
)