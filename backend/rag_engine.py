"""
RAG Engine
==========
Retrieval + Generation pipeline được dùng bởi FastAPI server

CHANGELOG (bug fixes):
  FIX-1: use_fast=False  → tránh ký tự rác ỠẪẲẰÕẺ từ fast tokenizer
  FIX-2: prompt format   → dùng "question: ... context: ..." thay instruction dài
  FIX-3: max_length=512  → đúng với window vit5-base (không phải 1024)
  FIX-4: max_new_tokens  → thay max_length trong generate() để không count input
  FIX-5: MAX_CONTEXT_CHUNKS=2 → giới hạn chunks tránh vượt 512 tokens
  FIX-6: _clean_output() → cắt phần echo prompt còn sót, xóa ký tự rác
"""

import os
import json
import pickle
import re
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_PATH  = "vector.index"
CHUNKS_PATH = "chunks.pkl"
META_PATH   = "index_meta.json"

MAX_CONTEXT_CHARS  = 600   # ký tự tối đa mỗi chunk đưa vào prompt
MAX_CONTEXT_CHUNKS = 2     # FIX-5: chỉ dùng top-2, tránh vượt 512-token window


class RAGEngine:
    def __init__(self, index_dir: str = "."):
        self.index_dir   = Path(index_dir)
        self.model       = None
        self.generator   = None
        self.tokenizer   = None
        self.index       = None
        self.chunks: List[Dict] = []
        self.meta: Dict  = {}
        self.loaded      = False

    # ── Load ────────────────────────────────────────────────────────────────
    def load(self):
        index_path  = self.index_dir / INDEX_PATH
        chunks_path = self.index_dir / CHUNKS_PATH
        meta_path   = self.index_dir / META_PATH

        if not index_path.exists():
            raise FileNotFoundError(
                f"Chưa tìm thấy FAISS index tại {index_path}. "
                "Hãy chạy indexing.py trước!"
            )

        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)

        model_name = self.meta.get("model_name", "paraphrase-multilingual-MiniLM-L12-v2")
        print(f"[RAG] Loading encoder: {model_name}")
        self.model = SentenceTransformer(model_name)

        print("[RAG] Loading FAISS index...")
        self.index = faiss.read_index(str(index_path))

        print("[RAG] Loading chunks...")
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        self._load_generator()
        self.loaded = True
        print(f"[RAG] Ready! {len(self.chunks)} chunks indexed.")

    def _load_generator(self):
        gen_model = os.getenv("GENERATOR_MODEL", "VietAI/vit5-base")
        print(f"[RAG] Loading generator: {gen_model}")
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

            self.tokenizer = AutoTokenizer.from_pretrained(
                gen_model,
                legacy=False,
                use_fast=False,       # ← FIX-1
            )
            self.generator = AutoModelForSeq2SeqLM.from_pretrained(gen_model)
            print("[RAG] Generator loaded.")
        except Exception as e:
            print(f"[RAG] Không load được T5 ({e}). Fallback: chỉ dùng retrieval.")
            self.generator = None

    # ── Retrieve ────────────────────────────────────────────────────────────
    def retrieve(self, query: str, top_k: int = 4) -> List[Dict]:
        q_vec = self.model.encode(
            [query], normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(q_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            if score < 0.6:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)

        return sorted(results, key=lambda x: x["score"], reverse=True)

    # ── Generate ────────────────────────────────────────────────────────────
    def generate(self, query: str, context_chunks: List[Dict]) -> str:
        """
        Extractive QA: trích câu liên quan nhất từ chunk
        Không dùng generator → không bị hallucination
        """
        if not context_chunks:
            return "Không tìm thấy thông tin phù hợp trong tài liệu."

        query_words = set(query.lower().split())

        best_sent  = ""
        best_score = 0

        for chunk in context_chunks[:2]:      # chỉ xét top-2 chunks
            sentences = re.split(r'(?<=[.!?])\s+', chunk["text"])
            for sent in sentences:
                if len(sent) < 15:            # bỏ câu quá ngắn
                    continue
                sent_words = set(sent.lower().split())
                # đếm từ query xuất hiện trong câu
                overlap = len(query_words & sent_words)
                # kết hợp với chunk score
                score = overlap + chunk["score"] * 2
                if score > best_score:
                    best_score = score
                    best_sent  = sent.strip()

        if not best_sent:
            return context_chunks[0]["text"][:300].strip()

        return best_sent

    def _clean_output(self, answer: str) -> str:
        """
        FIX-6: Dọn dẹp output sau generate:
        - Cắt tại dấu hiệu echo prompt (model vẫn có thể lặp một phần)
        - Xóa ký tự Unicode rác (private-use range, replacement char)
        - Xóa artifact ]Ỡ]Ẫ kiểu Wikipedia dump còn sót trong chunks
        - Normalize whitespace
        """
        # Cắt nếu model bắt đầu echo cấu trúc prompt
        echo_markers = [
            "question:", "context:", "Câu hỏi:", "Context:",
            "Trả lời ngắn", "Nếu không có", "Chỉ được trả lời",
        ]
        for marker in echo_markers:
            idx = answer.find(marker)
            if idx > 0:          # idx > 0: chỉ cắt nếu marker KHÔNG ở đầu
                answer = answer[:idx]

        # Xóa ký tự Unicode private-use và replacement char
        answer = re.sub(r'[\ufffd\ue000-\uf8ff]', '', answer)

        # Xóa artifact ]X]Y]Z kiểu Wikipedia HTML dump
        answer = re.sub(r'\][A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠ-Ỹ\s]+', '', answer)

        # Normalize whitespace
        answer = re.sub(r'\s+', ' ', answer).strip()

        if not answer or len(answer) < 2:
            return "Không tìm thấy câu trả lời phù hợp."

        return answer

    def _extract_answer(self, query: str, chunks: List[Dict]) -> str:
        """Fallback khi không có generator"""
        if not chunks:
            return "Không tìm thấy thông tin phù hợp trong tài liệu."
        if all(c["score"] < 0.65 for c in chunks):
            return "Không đủ thông tin để trả lời."

        best = chunks[0]
        text = best["text"]
        query_words = set(query.lower().split())
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sent in sentences:
            if any(w in sent.lower() for w in query_words):
                return sent.strip()
        return text[:400].strip() + ("..." if len(text) > 400 else "")

    # ── Main API ────────────────────────────────────────────────────────────
    def query(self, question: str, top_k: int = 4) -> Dict:
        if not self.loaded:
            raise RuntimeError("RAGEngine chưa được load. Gọi .load() trước.")

        retrieved = self.retrieve(question, top_k=top_k)
        answer    = self.generate(question, retrieved)

        return {
            "answer": answer,
            "sources": [
                {
                    "source": c["source"],
                    "text": c["text"][:300] + "..." if len(c["text"]) > 300 else c["text"],
                    "score": round(c["score"], 4),
                }
                for c in retrieved
            ],
        }

    # ── Index new document (runtime) ────────────────────────────────────────
    def add_document(self, filename: str, text: str):
        from indexing import chunk_text

        new_chunks = chunk_text(text, filename)
        if not new_chunks:
            return 0

        texts = [c["text"] for c in new_chunks]
        embeddings = self.model.encode(
            texts, normalize_embeddings=True
        ).astype("float32")

        self.index.add(embeddings)
        self.chunks.extend(new_chunks)

        faiss.write_index(self.index, str(self.index_dir / INDEX_PATH))
        with open(self.index_dir / CHUNKS_PATH, "wb") as f:
            pickle.dump(self.chunks, f)

        return len(new_chunks)