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
  FIX-7: threshold 0.6 → 0.3, thêm keyword fallback
  FIX-8: generator thực sự được gọi trong generate()
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
SCORE_THRESHOLD    = 0.3   # FIX-7: hạ từ 0.6 → 0.3 cho tiếng Việt


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

        model_name = self.meta.get("model_name", "keepitreal/vietnamese-sbert")
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
        if not gen_model:
            print("[RAG] GENERATOR_MODEL rỗng. Chỉ dùng extractive mode.")
            return

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
            print(f"[RAG] Không load được generator ({e}). Fallback: extractive mode.")
            self.generator = None
            self.tokenizer = None

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
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)

        # FIX-7: threshold 0.3 thay vì 0.6
        filtered = [r for r in results if r["score"] >= SCORE_THRESHOLD]

        # Keyword fallback nếu không có chunk nào pass threshold
        if not filtered:
            filtered = self._keyword_fallback(query, results)

        return sorted(filtered, key=lambda x: x["score"], reverse=True)

    def _keyword_fallback(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """Tìm theo từ khóa khi semantic search thất bại"""
        stopwords = {"là", "gì", "nào", "năm", "của", "và", "trong",
                     "có", "không", "được", "với", "cho", "hay", "hoặc",
                     "bao", "nhiêu", "khi", "tại", "sao", "thế", "nào"}
        query_words = set(query.lower().split()) - stopwords

        if not query_words:
            return candidates[:2]  # trả về top-2 nếu không có keyword

        matched = []
        for chunk in self.chunks:
            text_lower = chunk["text"].lower()
            hits = sum(1 for kw in query_words if kw in text_lower)
            if hits > 0:
                c = dict(chunk)
                c["score"] = 0.4 + hits * 0.05  # score tượng trưng
                matched.append(c)

        return sorted(matched, key=lambda x: x["score"], reverse=True)[:top_k if hasattr(self, '_top_k') else 4]

    # ── Generate ────────────────────────────────────────────────────────────
    def generate(self, query: str, context_chunks: List[Dict]) -> str:
        if not context_chunks:
            return "Không tìm thấy thông tin phù hợp trong tài liệu."

        # Ghép context từ top-2 chunks (FIX-5: tránh vượt 512 token)
        context = "\n".join(
            c["text"][:MAX_CONTEXT_CHARS] for c in context_chunks[:MAX_CONTEXT_CHUNKS]
        )

        # FIX-8: Thực sự gọi generator nếu có
        if self.generator and self.tokenizer:
            # FIX-2: prompt format ngắn gọn phù hợp với ViT5/Flan-T5
            prompt = f"question: {query} context: {context}"

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,        # FIX-3: đúng window size
                truncation=True,
            )

            outputs = self.generator.generate(
                **inputs,
                max_new_tokens=150,    # FIX-4: không đếm input tokens
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self._clean_output(answer)

        # Fallback: extractive nếu không có generator
        return self._extract_answer(query, context_chunks)

    def _clean_output(self, answer: str) -> str:
        """
        FIX-6: Dọn dẹp output sau generate:
        - Cắt tại dấu hiệu echo prompt
        - Xóa ký tự Unicode rác
        - Normalize whitespace
        """
        echo_markers = [
            "question:", "context:", "Câu hỏi:", "Context:",
            "Trả lời ngắn", "Nếu không có", "Chỉ được trả lời",
        ]
        for marker in echo_markers:
            idx = answer.find(marker)
            if idx > 0:
                answer = answer[:idx]

        answer = re.sub(r'[\ufffd\ue000-\uf8ff]', '', answer)
        answer = re.sub(r'\][A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠ-Ỹ\s]+', '', answer)
        answer = re.sub(r'\s+', ' ', answer).strip()

        if not answer or len(answer) < 2:
            return "Không tìm thấy câu trả lời phù hợp."

        return answer

    def _extract_answer(self, query: str, chunks: List[Dict]) -> str:
        if not chunks:
            return "Không tìm thấy thông tin phù hợp trong tài liệu."

        query_words = set(query.lower().split())
        stopwords = {"là", "gì", "nào", "của", "và", "trong", "có",
                    "không", "được", "lần", "thứ", "giải"}
        keywords = query_words - stopwords

        best_chunk = chunks[0]
        best_score = -1

        # Chọn chunk có nhiều keyword nhất
        for chunk in chunks[:MAX_CONTEXT_CHUNKS]:
            text_lower = chunk["text"].lower()
            hits = sum(1 for kw in keywords if kw in text_lower)
            score = hits + chunk["score"] * 2
            if score > best_score:
                best_score = score
                best_chunk = chunk

        text = best_chunk["text"]

        # Thử tìm câu chứa keyword — nếu câu đủ dài thì trả nguyên câu đó
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sent in sentences:
            if len(sent) < 20:
                continue
            sent_lower = sent.lower()
            if any(kw in sent_lower for kw in keywords):
                # Nếu câu đủ đầy đủ (>80 ký tự), trả về câu đó
                if len(sent) >= 80:
                    return sent.strip()

        # Fallback: trả về từ đầu chunk, cắt tại dấu chấm gần nhất trước 500 ký tự
        return _trim_to_sentence(text, max_chars=500)


    def _trim_to_sentence(text: str, max_chars: int = 500) -> str:
        """Cắt text tại dấu chấm gần nhất, không cắt giữa câu"""
        if len(text) <= max_chars:
            return text.strip()

        # Tìm dấu chấm cuối cùng trong khoảng max_chars
        cutoff = text[:max_chars]
        last_period = max(
            cutoff.rfind('.'),
            cutoff.rfind('!'),
            cutoff.rfind('?'),
        )

        if last_period > max_chars // 2:   # có dấu chấm hợp lý
            return text[:last_period + 1].strip()

        return cutoff.strip() + "..."

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