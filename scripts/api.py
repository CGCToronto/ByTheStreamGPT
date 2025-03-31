from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from .local_model_handler import LocalModelHandler
import os

# Initialize FastAPI app
app = FastAPI(
    title="溪水旁 GPT API",
    description="API for the 溪水旁 (By The Stream) Christian magazine GPT service",
    version="1.0.0"
)

# Initialize model handler
model_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'models',
    'deepseek-coder-1.3b-base'
)
model_handler = LocalModelHandler(model_path)

# Request models
class QueryRequest(BaseModel):
    query: str
    language: Optional[str] = None

class ArticleRequest(BaseModel):
    article_id: str

# Endpoints
@app.post("/query")
async def process_query(request: QueryRequest):
    """Process a user query and return relevant articles and response."""
    result = model_handler.process_query(request.query, request.language)
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@app.post("/summarize")
async def summarize_article(request: ArticleRequest):
    """Get a summary of a specific article."""
    result = model_handler.get_article_summary(request.article_id)
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 