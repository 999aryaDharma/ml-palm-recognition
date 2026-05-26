"""
Image Service Module
Handles image upload validation and processing
"""

from PIL import Image
from fastapi import UploadFile, HTTPException
import io


ALLOWED_FORMATS = {"JPEG", "PNG"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


async def upload_to_pil(image: UploadFile, max_mb: int = 8) -> Image.Image:
    """
    Convert uploaded file to PIL Image with validation.
    
    Args:
        image: FastAPI UploadFile
        max_mb: Maximum file size in megabytes
        
    Returns:
        PIL Image object
        
    Raises:
        HTTPException: If file is invalid
    """
    # Check file size
    image_bytes = await image.read()
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > max_mb:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "file_too_large",
                "message": f"File size {size_mb:.2f}MB exceeds limit {max_mb}MB"
            }
        )
    
    # Check file format
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        if pil_image.format not in ALLOWED_FORMATS:
            raise ValueError(f"Format {pil_image.format} not supported")
        # Convert to RGB if needed
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        return pil_image
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_image",
                "message": f"Failed to process image: {str(e)}"
            }
        )
