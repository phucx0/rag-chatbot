import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

        config_path = os.path.join(data_dir, "config.json")
        chunks_path = os.path.join(data_dir, "chunks.json")
        index_path = os.path.join(data_dir, "faiss_index.bin")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing config.json: {config_path}")

        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f"Missing chunks.json: {chunks_path}")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Missing faiss_index.bin: {index_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self.index = faiss.read_index(index_path)

        self.model_name = self.config["embed_model"]
        self.query_prefix = self.config.get("query_prefix", "")
        self.normalize = self.config.get("normalize", True)

        self.model = SentenceTransformer(self.model_name)

        print(f"✅ Retriever ready")
        print(f"   Model  : {self.model_name}")
        print(f"   Chunks : {len(self.chunks):,}")

    def encode_query(self, question: str):
        text = f"{self.query_prefix}{question}"

        vec = self.model.encode(
            [text],
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")

        if self.normalize:
            faiss.normalize_L2(vec)

        return vec

    def search(self, question: str, top_k: int = 4):
        query_vec = self.encode_query(question)

        scores, indices = self.index.search(query_vec, top_k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue

            chunk = self.chunks[idx]

            results.append({
                "id": chunk.get("id", str(idx)),
                "title": chunk.get("title", "Không rõ tiêu đề"),
                "context": chunk.get("context") or chunk.get("text") or chunk.get("page_content", ""),
                "score": float(score),
                "metadata": chunk.get("metadata", {}),
            })

        return results

    def count_chunks(self):
        return len(self.chunks)