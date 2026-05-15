import os
import shutil
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.pipeline import RAGPipeline


load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

XAI_API_KEY = os.getenv("XAI_API_KEY")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-3")

app = FastAPI(title="RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag: RAGPipeline | None = None


class ChatRequest(BaseModel):
    question: str
    top_k: int = 4


@app.on_event("startup")
def startup():
    global rag

    try:
        rag = RAGPipeline(
            data_dir=DATA_DIR,
            api_key=XAI_API_KEY,
            model_name=GROK_MODEL,
        )
    except Exception as e:
        print(f"❌ Failed to load RAG pipeline: {e}")
        rag = None


@app.get("/")
def root():
    return {
        "message": "RAG Chatbot API is running"
    }


@app.get("/status")
def status():
    if rag is None:
        return {
            "ready": False,
            "num_chunks": 0,
            "documents": [],
        }

    return rag.status()


@app.post("/chat")
def chat(req: ChatRequest):
    if rag is None:
        raise HTTPException(
            status_code=500,
            detail="RAG pipeline chưa sẵn sàng. Kiểm tra data/faiss_index.bin, chunks.json, config.json hoặc API key.",
        )

    if not req.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question không được để trống.",
        )

    try:
        return rag.ask(
            question=req.question,
            top_k=req.top_k,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    Tạm thời chỉ upload file vào thư mục uploads.
    Chưa rebuild index động.
    Vì index chính đang lấy từ UIT-ViQuAD2.0 đã build sẵn.
    """

    if not file.filename:
        raise HTTPException(status_code=400, detail="File không hợp lệ.")

    save_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "message": "Upload thành công. Hiện tại backend chưa rebuild index động từ file upload.",
        "filename": file.filename,
        "num_chunks": 0,
    }