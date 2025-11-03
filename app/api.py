# app/04_api.py
import os
from typing import Optional, List
from fastapi import FastAPI
from pydantic import BaseModel
from app.query import ask  # 同ディレクトリで import 可能にするか、関数をここにコピペ

app = FastAPI(title="Neko RAG API")

class QueryIn(BaseModel):
    q: str
    max_chapter_allowed: Optional[int] = None

class Citation(BaseModel):
    chapter: int
    start_pos: int
    end_pos: int

class QueryOut(BaseModel):
    answer: str
    citations: List[Citation]

@app.post("/query", response_model=QueryOut)
def query(payload: QueryIn):
    answer, cites = ask(payload.q, max_chapter_allowed=payload.max_chapter_allowed)
    return {
        "answer": answer,
        "citations": [{"chapter": c[0], "start_pos": c[1], "end_pos": c[2]} for c in cites]
    }

