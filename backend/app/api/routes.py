from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.services.assistant_service import AssistantService
from app.services.change_service import ChangeDetectionService
from app.services.dynamic_object_detection_service import (
    dynamic_object_detection_service,
)


router = APIRouter()

change_service = ChangeDetectionService()
assistant_service = AssistantService()


# =========================================================
# CHANGE DETECTION
# =========================================================

class ChangeDetectionRequest(BaseModel):
    before: str
    after: str


@router.post("/change-detection")
def change_detection(
    request: ChangeDetectionRequest,
):
    before = Path(request.before)
    after = Path(request.after)

    if not before.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Before image not found: {before}",
        )

    if not after.exists():
        raise HTTPException(
            status_code=404,
            detail=f"After image not found: {after}",
        )

    try:
        return change_service.detect_change(
            str(before),
            str(after),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# =========================================================
# OBJECT / FEATURE DETECTION
# =========================================================

@router.post("/object-detection/upload")
async def object_detection_upload(
    image: UploadFile = File(...),
    model: str = Form("aerial"),
):
    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/jpg",
    }

    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG or PNG image.",
        )

    try:
        image_bytes = await image.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty.",
            )

        if len(image_bytes) > 15 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="Image size must be 15 MB or smaller.",
            )

        return dynamic_object_detection_service.detect_bytes(
            image_bytes=image_bytes,
            filename=image.filename or "uploaded_image",
            model_type=model,
        )

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Object detection failed: {exc}",
        )


# =========================================================
# EARTHVISION AI ASSISTANT
# =========================================================

class AssistantRequest(BaseModel):
    question: str
    change_result: dict | None = None
    object_result: dict | None = None


@router.post("/assistant")
def assistant(
    request: AssistantRequest,
):
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        return assistant_service.answer(
            question=request.question,
            change_result=request.change_result,
            object_result=request.object_result,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )