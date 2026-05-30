"""
Palm Recognition Module
Extracts embeddings from palm images
"""

import numpy as np
import torch
from PIL import Image
from typing import Optional
import os


class PalmRecognizer:
    """Extracts embeddings from palm images using deep learning."""
    
    def __init__(self, model_path: str):
        """
        Initialize palm recognizer.
        
        Args:
            model_path: Path to palm_recognizer.pt model
        """
        self.model_path = model_path
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not os.path.exists(model_path):
            print(f"[PalmRecognizer] WARNING: Model file not found at {model_path}")
            return
            
        try:
            # We assume it's a TorchScript model exported for inference
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
            print(f"[PalmRecognizer] Loaded TorchScript model from: {model_path} (device: {self.device})")
        except Exception as e:
            print(f"[PalmRecognizer] Failed to load model {model_path}: {e}")
    
    @torch.no_grad()
    def extract_embedding(self, palm_roi: Image.Image) -> Optional[np.ndarray]:
        """
        Extract embedding vector from palm ROI.
        
        Args:
            palm_roi: PIL Image of palm region (assumed to be 112x112)
            
        Returns:
            Embedding vector (128-dim float32 L2-normalized), or None if extraction fails
        """
        if self.model is None:
            return None
            
        # Convert to numpy and normalize
        img_np = np.array(palm_roi.convert("RGB")).astype(np.float32)
        
        # HWC to CHW
        img_np = np.transpose(img_np, (2, 0, 1))
        
        # Normalize to [-1, 1] as defined in our training config: 
        # img / 255.0 then (img - 0.5) / 0.5
        img_np = (img_np / 255.0 - 0.5) / 0.5
        
        # Add batch dimension
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).float().to(self.device)
        
        # Forward pass
        embedding = self.model(img_tensor)
        
        # Some TorchScript exports already include L2 normalization, 
        # but doing it here guarantees we output cosine-ready vectors
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        
        # Return 1D numpy array
        return embedding.cpu().numpy()[0]
