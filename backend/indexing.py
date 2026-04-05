"""
RAG Indexing Pipeline
=====================
Đọc tài liệu (PDF / TXT) → chunk → encode bằng Sentence-BERT → lưu FAISS index
Chạy: python indexing.py --docs_dir ../docs
"""

import os
import json
import argparse
import pickle
from pathlib import Path
from typing import List, Dict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Cấu hình ────────────────────────────────────────────────────────────────
MODEL_NAME   = "paraphrase-multilingual-MiniLM-L12-v2"  # ~120MB, hỗ trợ VI+EN
CHUNK_SIZE   = 300    # số từ mỗi đoạn
CHUNK_OVERLAP = 50    # số từ trùng lắp giữa 2 đoạn liền kề
INDEX_PATH   = "vector.index"
CHUNKS_PATH  = "chunks.pkl"
EMBED_DIM    = 384    # chiều vector của model trên


# ── 1. Đọc tài liệu ─────────────────────────────────────────────────────────
def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf(path: str) -> str:
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
        return text
    except ImportError:
        print("  [!] pdfplumber chưa cài. Chạy: pip install pdfplumber")
        return ""


def load_documents(docs_dir: str) -> List[Dict]:
    """Trả về list of {filename, text}"""
    docs = []
    for p in Path(docs_dir).rglob("*"):
        if p.suffix.lower() == ".txt":
            text = read_txt(str(p))
        elif p.suffix.lower() == ".pdf":
            text = read_pdf(str(p))
        elif p.suffix.lower() == ".json":      # ← THÊM DÒNG NÀY
            text = read_json_vimqa(str(p))
        else:
            continue
        if text.strip():
            docs.append({"filename": p.name, "text": text})
            print(f"  ✓ Đã đọc: {p.name} ({len(text.split())} từ)")
    return docs


# ── 2. Chia chunks ──────────────────────────────────────────────────────────
def chunk_text(text: str, filename: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """Chia văn bản thành các đoạn có overlap"""
    words = text.split()
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(words):
        end = min(start + size, len(words))
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)

        chunks.append({
            "id": f"{filename}_chunk_{chunk_id}",
            "source": filename,
            "text": chunk_text_str,
            "start_word": start,
            "end_word": end,
        })

        chunk_id += 1
        start += size - overlap  # trượt cửa sổ có overlap

    return chunks


def build_chunks(docs: List[Dict]) -> List[Dict]:
    all_chunks = []
    for doc in docs:
        chunks = chunk_text(doc["text"], doc["filename"])
        all_chunks.extend(chunks)
        print(f"  ✓ {doc['filename']}: {len(chunks)} chunks")
    return all_chunks


# ── 3. Encode bằng Transformer ──────────────────────────────────────────────
def encode_chunks(chunks: List[Dict], model: SentenceTransformer, batch_size: int = 32) -> np.ndarray:
    """
    Đây là bước ứng dụng Transformer:
    - Mỗi chunk text được tokenize và đưa qua Sentence-BERT
    - Output là vector 384 chiều (mean pooling của hidden states)
    - Cosine similarity giữa các vectors → tìm đoạn liên quan
    """
    texts = [c["text"] for c in chunks]
    print(f"\n  Encoding {len(texts)} chunks bằng {MODEL_NAME}...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # normalize → cosine sim = dot product
    )
    return embeddings.astype("float32")


# ── 4. Xây dựng FAISS index ─────────────────────────────────────────────────
def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    FAISS IndexFlatIP: Inner Product search
    Vì embeddings đã normalize → IP = cosine similarity
    """
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)
    print(f"  ✓ FAISS index: {index.ntotal} vectors")
    return index


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="RAG Indexing Pipeline")
    parser.add_argument("--docs_dir", default="../docs", help="Thư mục chứa tài liệu")
    parser.add_argument("--output_dir", default=".", help="Thư mục lưu index")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 50)
    print("RAG INDEXING PIPELINE")
    print("=" * 50)

    # Bước 1: Load tài liệu
    print("\n[1/4] Đọc tài liệu...")
    docs = load_documents(args.docs_dir)
    if not docs:
        print("  [!] Không tìm thấy tài liệu nào. Tạo file mẫu...")
        # _create_sample_docs(args.docs_dir)
        docs = load_documents(args.docs_dir)

    # Bước 2: Chia chunks
    print("\n[2/4] Chia chunks...")
    chunks = build_chunks(docs)
    print(f"  Tổng cộng: {len(chunks)} chunks")

    # Bước 3: Load model & encode
    print("\n[3/4] Load Transformer model & encode...")
    print(f"  Model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = encode_chunks(chunks, model)

    # Bước 4: Build & lưu FAISS index
    print("\n[4/4] Xây dựng FAISS index...")
    index = build_faiss_index(embeddings)

    index_path  = output_dir / INDEX_PATH
    chunks_path = output_dir / CHUNKS_PATH

    faiss.write_index(index, str(index_path))
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    # Lưu metadata
    meta = {
        "model_name": MODEL_NAME,
        "num_docs": len(docs),
        "num_chunks": len(chunks),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embed_dim": EMBED_DIM,
        "documents": [d["filename"] for d in docs],
    }
    with open(output_dir / "index_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n✅ Indexing hoàn thành!")
    print(f"   Index saved : {index_path}")
    print(f"   Chunks saved: {chunks_path}")
    print(f"   Total chunks: {len(chunks)}")


def read_json_vimqa(path: str) -> str:
    """Đọc VIMQA JSON → plain text để chunk bình thường"""
    from json_loader import extract_text_from_vimqa
    return extract_text_from_vimqa(path)

def _create_sample_docs(docs_dir: str):
    """Tạo tài liệu mẫu để test"""
    Path(docs_dir).mkdir(parents=True, exist_ok=True)
    sample = """Transformer là kiến trúc deep learning được giới thiệu năm 2017 trong bài báo "Attention Is All You Need".
Kiến trúc này dựa hoàn toàn vào cơ chế self-attention, loại bỏ hoàn toàn các lớp RNN và CNN truyền thống.

Cơ chế Self-Attention cho phép mô hình tập trung vào các từ liên quan trong câu khi xử lý mỗi từ.
Với mỗi từ, mô hình tính toán ba vector: Query (Q), Key (K) và Value (V).
Attention score được tính bằng: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V

BERT (Bidirectional Encoder Representations from Transformers) là mô hình Transformer encoder được Google giới thiệu năm 2018.
BERT được pre-train trên lượng văn bản khổng lồ bằng hai nhiệm vụ: Masked Language Model và Next Sentence Prediction.
Sau đó có thể fine-tune cho nhiều tác vụ NLP như phân loại văn bản, NER, hỏi đáp.

RAG (Retrieval-Augmented Generation) kết hợp hai kỹ thuật: truy xuất thông tin và sinh văn bản.
Đầu tiên, hệ thống tìm kiếm các đoạn văn bản liên quan từ knowledge base sử dụng semantic search.
Sau đó, LLM sử dụng các đoạn đó làm context để sinh câu trả lời chính xác hơn.

Sentence-BERT là biến thể của BERT được fine-tune để tạo ra sentence embeddings chất lượng cao.
Nó sử dụng Siamese network với mean pooling để tạo vector đại diện cho toàn bộ câu.
Các vector này có thể so sánh bằng cosine similarity để tìm câu tương đồng về ngữ nghĩa.
"""
    with open(f"{docs_dir}/sample_nlp.txt", "w", encoding="utf-8") as f:
        f.write(sample)
    print("  ✓ Đã tạo tài liệu mẫu: docs/sample_nlp.txt")


if __name__ == "__main__":
    main()
