"""
RAG Indexing Pipeline
=====================
Đọc tài liệu (PDF / TXT / VIMQA JSON) → chunk → encode → FAISS index
Chạy: python indexing.py --docs_dir ../docs
"""

import os
import json
import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ── Cấu hình ────────────────────────────────────────────────────────────────
MODEL_NAME    = "keepitreal/vietnamese-sbert"
CHUNK_SIZE    = 300
CHUNK_OVERLAP = 50
INDEX_PATH    = "vector.index"
CHUNKS_PATH   = "chunks.pkl"
# EMBED_DIM không hardcode — tự detect từ embeddings.shape[1]


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


def load_documents(docs_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Trả về:
      docs         — list {filename, text} cho TXT/PDF → sẽ chunk theo câu
      json_chunks  — list chunks sẵn từ VIMQA JSON (paragraph-level)
    """
    docs: List[Dict]        = []
    json_chunks: List[Dict] = []

    for p in Path(docs_dir).rglob("*"):
        if p.suffix.lower() == ".txt":
            text = read_txt(str(p))
            if text.strip():
                docs.append({"filename": p.name, "text": text})
                print(f"  ✓ TXT: {p.name} ({len(text.split())} từ)")

        elif p.suffix.lower() == ".pdf":
            text = read_pdf(str(p))
            if text.strip():
                docs.append({"filename": p.name, "text": text})
                print(f"  ✓ PDF: {p.name} ({len(text.split())} từ)")

        elif p.suffix.lower() == ".json":
            # VIMQA JSON → paragraph-level chunks, không chunk theo từ
            chunks = _read_json_vimqa_chunks(str(p))
            if chunks:
                json_chunks.extend(chunks)
                n_sup = sum(1 for c in chunks if c.get("is_supporting"))
                print(f"  ✓ JSON: {p.name} → {len(chunks)} chunks "
                      f"({n_sup} supporting)")

    return docs, json_chunks


def _read_json_vimqa_chunks(path: str) -> List[Dict]:
    """Gọi load_vimqa_as_chunks() — paragraph-level, không cắt giữa câu."""
    try:
        from json_loader import load_vimqa_as_chunks
        return load_vimqa_as_chunks(path)
    except Exception as e:
        print(f"  [!] Lỗi đọc JSON {path}: {e}")
        return []


# ── 2. Chia chunks (chỉ dùng cho TXT/PDF) ──────────────────────────────────
def chunk_text(text: str, filename: str,
               size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """
    Chia văn bản TXT/PDF tại ranh giới câu (không cắt giữa câu).
    VIMQA JSON dùng load_vimqa_as_chunks() thay thế.
    """
    import re

    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks    = []
    chunk_id  = 0
    current_words: List[str] = []
    current_sents: List[str] = []

    for sent in sentences:
        sent_words = sent.split()
        if not sent_words:
            continue

        if current_words and len(current_words) + len(sent_words) > size:
            chunks.append({
                "id":         f"{filename}_chunk_{chunk_id}",
                "source":     filename,
                "text":       " ".join(current_words),
                "start_word": 0,
                "end_word":   len(current_words),
            })
            chunk_id += 1
            # Overlap: giữ câu cuối
            overlap_sents = current_sents[-1:] if current_sents else []
            current_words = " ".join(overlap_sents).split()
            current_sents = overlap_sents

        current_words.extend(sent_words)
        current_sents.append(sent)

    if current_words:
        chunks.append({
            "id":         f"{filename}_chunk_{chunk_id}",
            "source":     filename,
            "text":       " ".join(current_words),
            "start_word": 0,
            "end_word":   len(current_words),
        })

    return chunks


def build_chunks(docs: List[Dict]) -> List[Dict]:
    all_chunks = []
    for doc in docs:
        chunks = chunk_text(doc["text"], doc["filename"])
        all_chunks.extend(chunks)
        print(f"  ✓ {doc['filename']}: {len(chunks)} chunks")
    return all_chunks


# ── 3. Encode bằng Transformer ──────────────────────────────────────────────
def encode_chunks(chunks: List[Dict], model: SentenceTransformer,
                  batch_size: int = 32) -> np.ndarray:
    texts = [c["text"] for c in chunks]
    print(f"\n  Encoding {len(texts)} chunks bằng {MODEL_NAME}...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embeddings.astype("float32")


# ── 4. Xây dựng FAISS index ─────────────────────────────────────────────────
def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]   # tự detect: 768 (vietnamese-sbert) hoặc 384 (MiniLM)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"  ✓ FAISS index: {index.ntotal} vectors (dim={dim})")
    return index


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="RAG Indexing Pipeline")
    parser.add_argument("--docs_dir",   default="../docs", help="Thư mục chứa tài liệu")
    parser.add_argument("--output_dir", default=".",       help="Thư mục lưu index")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 50)
    print("RAG INDEXING PIPELINE")
    print("=" * 50)

    # Bước 1: Load tài liệu
    print("\n[1/4] Đọc tài liệu...")
    docs, json_chunks = load_documents(args.docs_dir)
    if not docs and not json_chunks:
        print("  [!] Không tìm thấy tài liệu nào trong", args.docs_dir)
        return

    # Bước 2: Chia chunks
    print("\n[2/4] Chia chunks...")
    txt_chunks = build_chunks(docs)
    all_chunks = txt_chunks + json_chunks
    print(f"  TXT/PDF chunks : {len(txt_chunks)}")
    print(f"  JSON chunks    : {len(json_chunks)}")
    print(f"  Tổng cộng      : {len(all_chunks)} chunks")

    # Bước 3: Load model & encode
    print("\n[3/4] Load Transformer model & encode...")
    print(f"  Model: {MODEL_NAME}")
    model      = SentenceTransformer(MODEL_NAME)
    embeddings = encode_chunks(all_chunks, model)

    # Bước 4: Build & lưu FAISS index
    print("\n[4/4] Xây dựng FAISS index...")
    index = build_faiss_index(embeddings)

    faiss.write_index(index, str(output_dir / INDEX_PATH))
    with open(output_dir / CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    actual_dim = embeddings.shape[1]
    meta = {
        "model_name":    MODEL_NAME,
        "num_docs":      len(docs) + len({c["source"] for c in json_chunks}),
        "num_chunks":    len(all_chunks),
        "chunk_size":    CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embed_dim":     actual_dim,
        "documents":     [d["filename"] for d in docs] +
                         list({c["source"] for c in json_chunks}),
    }
    with open(output_dir / "index_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n✅ Indexing hoàn thành!")
    print(f"   Index saved : {output_dir / INDEX_PATH}")
    print(f"   Chunks saved: {output_dir / CHUNKS_PATH}")
    print(f"   Total chunks: {len(all_chunks)}")
    print(f"   Embed dim   : {actual_dim}")


if __name__ == "__main__":
    main()