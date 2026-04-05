"""
FastAPI Server
==============
Endpoints:
  GET  /             → health check
  GET  /status       → trạng thái index
  POST /upload       → upload tài liệu (PDF / TXT)
  POST /chat         → hỏi đáp
  GET  /documents    → danh sách tài liệu đã index

Chạy: uvicorn server:app --port 8000
"""

import io
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_engine import RAGEngine

# ── Khởi tạo engine ─────────────────────────────────────────────────────────
INDEX_DIR = os.getenv("INDEX_DIR", ".")
engine    = RAGEngine(index_dir=INDEX_DIR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load engine khi server khởi động"""
    index_path = Path(INDEX_DIR) / "vector.index"
    if index_path.exists():
        try:
            engine.load()
        except Exception as e:
            print(f"[WARN] Không load được engine: {e}")
    else:
        print("[INFO] Chưa có index. Upload tài liệu để bắt đầu.")
    yield


app = FastAPI(
    title="RAG Chatbot API",
    description="Hỏi đáp tài liệu với Transformer + FAISS",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — cho phép React frontend gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # production: thay bằng domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ─────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    top_k: int = 4


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    question: str


class StatusResponse(BaseModel):
    ready: bool
    num_chunks: int
    num_documents: int
    documents: List[str]
    model: Optional[str] = None


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "RAG Chatbot API đang chạy!", "docs": "/docs"}


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Kiểm tra trạng thái index"""
    if not engine.loaded:
        return StatusResponse(
            ready=False,
            num_chunks=0,
            num_documents=0,
            documents=[],
        )
    return StatusResponse(
        ready=True,
        num_chunks=len(engine.chunks),
        num_documents=len(engine.meta.get("documents", [])),
        documents=engine.meta.get("documents", []),
        model=engine.meta.get("model_name"),
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Nhận câu hỏi → RAG pipeline → trả lời

    Flow:
    1. Encode câu hỏi bằng Sentence-BERT
    2. FAISS tìm top_k chunks liên quan
    3. T5 sinh câu trả lời từ context
    """
    if not engine.loaded:
        raise HTTPException(
            status_code=503,
            detail="Model chưa sẵn sàng. Vui lòng upload tài liệu trước.",
        )

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")
    if len(question) > 500:
        raise HTTPException(status_code=400, detail="Câu hỏi quá dài (tối đa 500 ký tự).")

    try:
        result = engine.query(question, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        question=question,
    )


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload PDF hoặc TXT → tự động index vào FAISS
    Không cần chạy lại indexing.py
    """
    allowed_types = {".pdf", ".txt", ".json"}
    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Chỉ chấp nhận file .pdf, .txt, hoặc .json. Nhận được: {suffix}",
        )

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="File quá lớn (tối đa 10MB).")

    # Trích xuất text
    text = ""
    if suffix == ".txt":
        text = content.decode("utf-8", errors="ignore")
    elif suffix == ".pdf":
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for page in pdf.pages:
                    text += (page.extract_text() or "") + "\n"
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="pdfplumber chưa được cài. Chạy: pip install pdfplumber",
            )
    elif suffix == ".json":
        from json_loader import extract_text_from_vimqa
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".json", delete=False
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            text = extract_text_from_vimqa(tmp_path)
        finally:
            os.unlink(tmp_path)

    if not text.strip():
        raise HTTPException(status_code=400, detail="Không trích xuất được text từ file.")

    # Nếu engine chưa load → khởi tạo index mới
    if not engine.loaded:
        _bootstrap_engine(file.filename, text)
        num_chunks = len(engine.chunks)
    else:
        num_chunks = engine.add_document(file.filename, text)

    return {
        "message": "Upload thành công!",
        "filename": file.filename,
        "num_chunks": num_chunks,
        "total_chunks": len(engine.chunks),
    }


def _bootstrap_engine(filename: str, text: str):
    """
    Khởi tạo index từ đầu khi chưa có sẵn.
    FIX: không hardcode EMBED_DIM, tự detect từ embeddings thực tế.
    """
    from indexing import chunk_text
    import pickle
    import json
    import faiss as fs
    from sentence_transformers import SentenceTransformer

    model_name = os.getenv(
        "ENCODER_MODEL", "keepitreal/vietnamese-sbert"
    )
    engine.model  = SentenceTransformer(model_name)
    chunks        = chunk_text(text, filename)
    engine.chunks = chunks

    texts      = [c["text"] for c in chunks]
    embeddings = engine.model.encode(
        texts, normalize_embeddings=True
    ).astype("float32")

    # FIX: dim từ embeddings thực tế, không hardcode
    actual_dim   = embeddings.shape[1]
    engine.index = fs.IndexFlatIP(actual_dim)
    engine.index.add(embeddings)

    idx_dir = Path(INDEX_DIR)
    fs.write_index(engine.index, str(idx_dir / "vector.index"))
    with open(idx_dir / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    engine.meta = {
        "model_name": model_name,
        "documents": [filename],
        "num_chunks": len(chunks),
        "embed_dim": actual_dim,
    }
    with open(idx_dir / "index_meta.json", "w", encoding="utf-8") as f:
        json.dump(engine.meta, f, ensure_ascii=False, indent=2)

    engine._load_generator()
    engine.loaded = True
    print(f"[Bootstrap] Index tạo xong: {len(chunks)} chunks, dim={actual_dim}")


@app.get("/documents")
async def list_documents():
    """Danh sách tài liệu đã được index"""
    if not engine.loaded:
        return {"documents": []}

    doc_stats = {}
    for chunk in engine.chunks:
        src = chunk["source"]
        doc_stats[src] = doc_stats.get(src, 0) + 1

    return {
        "documents": [
            {"name": name, "chunks": count}
            for name, count in doc_stats.items()
        ]
    }