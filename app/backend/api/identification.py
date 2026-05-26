from fastapi import APIRouter, Request, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import User
from schemas.identification import IdentifyResponse, IdentifiedUser
from services.image_service import upload_to_pil
from services.identification_service import IdentificationService

router = APIRouter()


@router.post("/identify", response_model=IdentifyResponse)
async def identify_palm(
    request: Request,
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Identify user from palm image.
    
    Args:
        image: Palm image file upload
        
    Returns:
        IdentifyResponse with status, user info, and match score
    """
    try:
        # Validate and convert image
        pil_image = await upload_to_pil(image, request.app.state.settings.max_upload_mb)
        
        # Create identification service
        service = IdentificationService(request.app.state, db)
        
        # Perform identification
        result, latency_ms = service.identify_palm(pil_image)
        
        # Build response
        if result["status"] == "identified":
            user = IdentifiedUser(
                id=result["user_id"],
                name=result["user_name"]
            )
            return IdentifyResponse(
                status=result["status"],
                user=user,
                score=round(result["score"], 4),
                latency_ms=latency_ms
            )
        elif result["status"] == "unknown":
            return IdentifyResponse(
                status="unknown",
                user=None,
                score=round(result["score"], 4),
                latency_ms=latency_ms
            )
        else:
            # Error case
            error_code = result.get("error_code", "unknown_error")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": error_code,
                    "message": f"Failed to process palm image: {error_code}"
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_server_error",
                "message": f"Identification failed: {str(e)}"
            }
        )
