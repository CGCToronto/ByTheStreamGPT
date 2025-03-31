import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import logging
from pathlib import Path
import json

from scripts.local_model_handler import LocalModelHandler
from scripts.volume_updater import VolumeUpdater
from config.config import MODEL_NAME, MODELS_DIR

# Initialize FastAPI app
app = FastAPI(
    title="溪水旁 GPT API",
    description="API for the By The Stream Christian Magazine GPT service",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize model handler
model_handler = None

class Query(BaseModel):
    text: str
    language: str = "simplified"
    max_length: Optional[int] = 200

class UpdateRequest(BaseModel):
    force_update: Optional[bool] = False

@app.on_event("startup")
async def startup_event():
    global model_handler
    try:
        model_path = MODELS_DIR / 'latest'
        if not model_path.exists():
            model_path = MODEL_NAME  # Use base model if no fine-tuned model exists
        
        model_handler = LocalModelHandler(str(model_path))
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail="Error loading model")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "model": "溪水旁 GPT"}

@app.post("/query")
async def process_query(query: Query):
    """Process a query and return the response."""
    try:
        response = model_handler.process_query(
            query.text,
            max_length=query.max_length,
            language=query.language
        )
        return {"response": response}
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail="Error processing query")

@app.post("/update")
async def update_model(request: UpdateRequest):
    """Update the model with new volumes."""
    try:
        updater = VolumeUpdater()
        if request.force_update or updater.find_new_volumes():
            success = updater.process_new_volumes()
            if success:
                # Reload model with updated weights
                global model_handler
                model_handler = LocalModelHandler(str(MODELS_DIR / 'latest'))
                info = updater.get_model_info()
                return {
                    "status": "success",
                    "message": "Model updated successfully",
                    "info": info
                }
            else:
                raise HTTPException(status_code=500, detail="Error updating model")
        else:
            return {
                "status": "success",
                "message": "No updates needed",
                "info": updater.get_model_info()
            }
    except Exception as e:
        logger.error(f"Error in update process: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/info")
async def get_model_info():
    """Get information about the current model."""
    try:
        updater = VolumeUpdater()
        info = updater.get_model_info()
        return info
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail="Error getting model info") 