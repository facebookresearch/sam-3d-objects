# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
FastAPI server for SAM 3D Objects unified web application.
This server serves both the AI API endpoints and the React frontend.
"""

import os
import sys
import io
import tempfile
import logging
from typing import Optional
from pathlib import Path

# Add parent directory to path for sam3d_objects imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "notebook"))

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image
import torch
import httpx

# Import SAM 3D inference components
from inference import Inference, load_image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables for API keys
HF_TOKEN = os.getenv("HF_TOKEN", "")
KIE_API_KEY = os.getenv("KIE_API_KEY", "")

# Global models - loaded on startup
sam3d_image_model = None
sam3d_video_predictor = None

app = FastAPI(
    title="SAM 3D Objects API",
    description="Unified API for SAM 3D object tracking, image generation, and 3D model creation",
    version="1.0.0"
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Load models on server startup"""
    global sam3d_image_model
    
    logger.info("Starting model initialization...")
    
    # Authenticate with Hugging Face if token is provided
    if HF_TOKEN:
        logger.info("Authenticating with Hugging Face...")
        try:
            from huggingface_hub import login
            login(token=HF_TOKEN)
            logger.info("Successfully authenticated with Hugging Face")
        except Exception as e:
            logger.warning(f"Failed to authenticate with Hugging Face: {e}")
    
    # Load SAM 3D Image Model
    try:
        logger.info("Loading SAM 3D Image Model...")
        tag = "hf"
        config_path = Path(__file__).parent.parent / f"checkpoints/{tag}/pipeline.yaml"
        
        if config_path.exists():
            sam3d_image_model = Inference(str(config_path), compile=False)
            logger.info("SAM 3D Image Model loaded successfully")
        else:
            logger.warning(f"Config file not found at {config_path}. Model will not be available.")
    except Exception as e:
        logger.error(f"Failed to load SAM 3D Image Model: {e}")
    
    logger.info("Server startup complete")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "SAM 3D Objects API is running"}


@app.get("/api/health")
async def health_check():
    """API health check"""
    return {
        "status": "ok",
        "models": {
            "sam3d_image": sam3d_image_model is not None,
            "sam3d_video": sam3d_video_predictor is not None,
        }
    }


@app.post("/api/process/image")
async def process_image(
    file: UploadFile = File(...),
    text_prompt: str = Form(...)
):
    """
    Process an image with SAM 3D to generate masked object.
    
    Args:
        file: Input image file
        text_prompt: Text prompt describing the object to segment
    
    Returns:
        Processed image with masks
    """
    if sam3d_image_model is None:
        raise HTTPException(status_code=503, detail="SAM 3D Image Model not loaded")
    
    try:
        # Read uploaded image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)
        
        # For now, create a simple mask (full image)
        # In production, you would use text_prompt with SAM to generate mask
        mask = np.ones((image_np.shape[0], image_np.shape[1]), dtype=bool)
        
        # Run SAM 3D inference
        logger.info(f"Processing image with prompt: {text_prompt}")
        output = sam3d_image_model(image_np, mask, seed=42)
        
        # Save output as PLY file
        output_path = tempfile.mktemp(suffix=".ply")
        output["gs"].save_ply(output_path)
        
        # Return the PLY file
        return FileResponse(
            output_path,
            media_type="application/octet-stream",
            filename="processed_object.ply"
        )
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/process/video")
async def process_video(
    file: UploadFile = File(...),
    text_prompt: str = Form(...)
):
    """
    Process a video with SAM 3D video predictor.
    
    Args:
        file: Input video file
        text_prompt: Text prompt describing the object to track
    
    Returns:
        Processed video with tracking
    """
    if sam3d_video_predictor is None:
        raise HTTPException(
            status_code=503,
            detail="SAM 3D Video Predictor not implemented yet"
        )
    
    try:
        # Save uploaded video to temp file
        video_bytes = await file.read()
        video_path = tempfile.mktemp(suffix=".mp4")
        with open(video_path, "wb") as f:
            f.write(video_bytes)
        
        logger.info(f"Processing video with prompt: {text_prompt}")
        
        # TODO: Implement video processing with SAM 3D video predictor
        # This would use the stateful video_predictor.handle_request workflow
        
        raise HTTPException(
            status_code=501,
            detail="Video processing not yet implemented"
        )
        
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate/image-kie")
async def generate_image_kie(text_prompt: str = Form(...)):
    """
    Generate an image using kie.ai "nano banana" service.
    
    Args:
        text_prompt: Text prompt for image generation
    
    Returns:
        Generated image from kie.ai
    """
    if not KIE_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="KIE_API_KEY not configured"
        )
    
    try:
        logger.info(f"Generating image with kie.ai for prompt: {text_prompt}")
        
        # Make API call to kie.ai
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Note: This is a placeholder URL - update with actual kie.ai API endpoint
            response = await client.post(
                "https://api.kie.ai/v1/generate/nano-banana",
                headers={
                    "Authorization": f"Bearer {KIE_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={"prompt": text_prompt}
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"kie.ai API error: {response.text}"
                )
            
            # Return the image
            return StreamingResponse(
                io.BytesIO(response.content),
                media_type="image/png"
            )
            
    except httpx.HTTPError as e:
        logger.error(f"HTTP error calling kie.ai: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating image with kie.ai: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate/3d")
async def generate_3d(file: UploadFile = File(...)):
    """
    Generate a 3D model (.glb) from an input image using SAM 3D.
    
    Args:
        file: Input image file
    
    Returns:
        3D model in GLB format
    """
    if sam3d_image_model is None:
        raise HTTPException(status_code=503, detail="SAM 3D Model not loaded")
    
    try:
        # Read uploaded image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)
        
        # Create full image mask
        mask = np.ones((image_np.shape[0], image_np.shape[1]), dtype=bool)
        
        # Run SAM 3D inference
        logger.info("Generating 3D model from image")
        output = sam3d_image_model(image_np, mask, seed=42)
        
        # Save output as PLY first
        ply_path = tempfile.mktemp(suffix=".ply")
        output["gs"].save_ply(ply_path)
        
        # TODO: Convert PLY to GLB format
        # For now, return PLY file (client can handle it)
        # In production, you would convert to GLB format
        
        return FileResponse(
            ply_path,
            media_type="model/gltf-binary",
            filename="model.ply"  # Will be handled as GLB-compatible on client
        )
        
    except Exception as e:
        logger.error(f"Error generating 3D model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def setup_static_file_serving():
    """Configure FastAPI to serve React build files"""
    frontend_build_path = Path(__file__).parent.parent / "frontend" / "dist"
    
    if frontend_build_path.exists():
        logger.info(f"Serving static files from {frontend_build_path}")
        
        # Mount static files (except index.html which is served by catch-all)
        app.mount(
            "/assets",
            StaticFiles(directory=str(frontend_build_path / "assets")),
            name="static"
        )
        
        # Serve index.html for all non-API routes (SPA routing)
        @app.get("/{full_path:path}")
        async def serve_spa(full_path: str):
            # Don't serve static files for API routes
            if full_path.startswith("api/"):
                raise HTTPException(status_code=404, detail="Not found")
            
            index_path = frontend_build_path / "index.html"
            if index_path.exists():
                return FileResponse(index_path)
            else:
                raise HTTPException(status_code=404, detail="Frontend not built")
    else:
        logger.warning(f"Frontend build not found at {frontend_build_path}")


# Setup static file serving
setup_static_file_serving()


if __name__ == "__main__":
    # Run the server
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
