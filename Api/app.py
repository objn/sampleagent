"""Additional HTTP routes mounted into LangGraph Agent Server."""

import asyncio
from uuid import UUID

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile, status
from pydantic import BaseModel, Field

from Models.postgresql import get_upload
from RAG.knowledge import MAX_UPLOAD_BYTES, search_knowledge
from RAG.uploads import queue_upload


app = FastAPI()


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=4, ge=1, le=10)
    category: str | None = None


@app.post("/rag/upload", status_code=status.HTTP_201_CREATED)
async def upload_rag_file(response: Response, file: UploadFile = File(...), category: str = Form("general")) -> dict:
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    try:
        record = await asyncio.to_thread(
            queue_upload, file.filename or "", file.content_type, data, category
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    response.headers["Location"] = record["status_url"]
    return record


@app.get("/rag/uploads/{upload_id}")
async def get_rag_upload(upload_id: UUID) -> dict:
    record = await asyncio.to_thread(get_upload, upload_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Upload not found")
    return record


@app.post("/rag/search")
async def search_rag(request: SearchRequest) -> dict:
    try:
        return {"results": await asyncio.to_thread(
            search_knowledge, request.query, request.limit, request.category
        )}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
