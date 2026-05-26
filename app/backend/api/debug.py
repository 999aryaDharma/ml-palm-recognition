from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from services.image_service import upload_to_pil

router = APIRouter()


@router.post("/extract-roi")
async def extract_roi(
    request: Request,
    image: UploadFile = File(...),
):
    """
    Debug endpoint to test ROI extraction from palm image.
    
    Args:
        image: Palm image file
        
    Returns:
        Debug info about detected hand and ROI
    """
    try:
        # Load and validate image
        pil_image = await upload_to_pil(image, request.app.state.settings.max_upload_mb)
        
        # Try to detect hand
        detector = request.app.state.detector
        if detector is None:
            raise HTTPException(status_code=503, detail={"error": "detector_not_ready"})
        
        detection_result = detector.detect(pil_image)
        
        if detection_result is None:
            return {
                "status": "no_hand_detected",
                "image_size": pil_image.size,
                "detection": None,
            }
        
        # Try to extract ROI
        from ml.roi import extract_palm_roi
        palm_roi = extract_palm_roi(pil_image, detection_result)
        
        return {
            "status": "success" if palm_roi else "roi_extraction_failed",
            "image_size": pil_image.size,
            "detection": {
                "box": detection_result.get("box"),
                "landmarks_count": len(detection_result.get("landmarks", [])),
            },
            "roi": {
                "size": palm_roi.size if palm_roi else None,
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={"error": "debug_failed", "message": str(e)}
        )

