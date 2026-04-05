"""
VIMQA JSON Loader
=================
Convert VIMQA format → text sạch cho RAG pipeline

Format VIMQA mỗi entry:
{
  "_id": "abc123",
  "question": "Tổ chức nào trao giải Oscar?",
  "answer": "Viện Hàn lâm",
  "supporting_facts": [["Giải Oscar", 0], ["Giải Oscar", 1]],
  "context": [
    ["Giải Oscar", ["Câu 1...", "Câu 2...", "Câu 3..."]],
    ["Viện Hàn lâm", ["Câu A...", "Câu B..."]]
  ]
}
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Set


# ── Cleaning ────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Làm sạch text tiếng Việt từ VIMQA:
    - Xóa ký tự control (non-printable)
    - Xóa artifact dạng ]Ỡ]Ẫ]Ằ (encoding lỗi từ Wikipedia dump)
    - Normalize khoảng trắng
    """
    if not text:
        return ""

    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'\][A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠ-Ỹ\s]+(?=\]|$)', '', text)
    text = re.sub(r'\](?=[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯ])', ' ', text)
    text = re.sub(r'[\ufffd\ue000-\uf8ff]', '', text)
    text = re.sub(r'(?<!\[)\](?!\])', ' ', text)
    text = re.sub(r'\[(?!\[)', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# ── Helpers ──────────────────────────────────────────────────────────────────
def _get_supporting_map(entry: dict) -> Dict[str, Set[int]]:
    """
    Trả về dict: entity_name → set các câu index là supporting facts
    Ví dụ: {"Kelly Rowland": {0, 1}, "Destiny's Child": {0}}
    """
    supporting = {}
    for fact in entry.get("supporting_facts", []):
        if len(fact) < 2:
            continue
        entity, sent_idx = fact[0], fact[1]
        if entity not in supporting:
            supporting[entity] = set()
        supporting[entity].add(sent_idx)
    return supporting


# ── Loaders ─────────────────────────────────────────────────────────────────
def extract_text_from_vimqa(json_path: str) -> str:
    """
    Convert toàn bộ VIMQA → plain text sạch có cấu trúc.
    Dùng để đưa vào chunk_text() của indexing.py.

    Mỗi QA entry được format:
      Câu hỏi: ...
      Trả lời: ...

      EntityName [quan trọng]: [KEY] câu supporting... câu thường...
      EntityName: câu A. câu B.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lines = []
    seen_titles: Set[str] = set()

    for entry in data:
        question = clean_text(entry.get("question", ""))
        answer   = clean_text(entry.get("answer", ""))
        supporting = _get_supporting_map(entry)

        # Header: câu hỏi + đáp án giúp tăng cosine similarity khi query
        if question:
            lines.append(f"Câu hỏi: {question}")
        if answer:
            lines.append(f"Trả lời: {answer}")
        if question or answer:
            lines.append("")

        # Context: mỗi entity một đoạn, đánh dấu supporting facts
        for para_title, sentences in entry.get("context", []):
            clean_title = clean_text(para_title)
            if not clean_title:
                continue

            is_supporting = clean_title in supporting
            seen_key = f"{entry.get('_id', '')}::{clean_title}"

            # Dedup paragraph theo (entry_id, title) thay vì chỉ title
            # → tránh bỏ sót cùng entity xuất hiện ở entry khác với context khác
            if seen_key in seen_titles:
                continue
            seen_titles.add(seen_key)

            # Format câu, đánh dấu [KEY] cho supporting sentences
            formatted_sents = []
            for i, sent in enumerate(sentences):
                cleaned = clean_text(sent)
                if not cleaned or len(cleaned) < 10:
                    continue
                if is_supporting and i in supporting.get(clean_title, set()):
                    formatted_sents.append(f"[KEY] {cleaned}")
                else:
                    formatted_sents.append(cleaned)

            if not formatted_sents:
                continue

            body = " ".join(formatted_sents)
            label = f"{clean_title} [quan trọng]" if is_supporting else clean_title
            lines.append(f"{label}: {body}")

        lines.append("")  # dòng trống phân cách giữa các entry

    return "\n".join(lines)


def load_vimqa_as_chunks(json_path: str) -> List[Dict]:
    """
    Đọc VIMQA JSON → list chunks, mỗi paragraph = 1 chunk.

    Ưu điểm so với chunk_text() theo số từ:
    - Không bao giờ cắt giữa câu
    - Mỗi chunk giữ nguyên ngữ nghĩa của 1 entity
    - Supporting facts được đưa lên đầu chunk → tăng similarity khi retrieve

    Dùng trực tiếp trong rag_engine.add_document() hoặc indexing.py
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = []
    filename  = Path(json_path).name
    seen_keys: Set[str] = set()

    for entry in data:
        entry_id   = entry.get("_id", "unknown")
        question   = clean_text(entry.get("question", ""))
        answer     = clean_text(entry.get("answer", ""))
        supporting = _get_supporting_map(entry)

        for para_title, sentences in entry.get("context", []):
            clean_title = clean_text(para_title)
            if not clean_title:
                continue

            seen_key = f"{entry_id}::{clean_title}"
            if seen_key in seen_keys:
                continue
            seen_keys.add(seen_key)

            is_supporting = clean_title in supporting

            # Tách supporting sentences và normal sentences
            key_sents    = []
            normal_sents = []
            for i, sent in enumerate(sentences):
                cleaned = clean_text(sent)
                if not cleaned or len(cleaned) < 10:
                    continue
                if is_supporting and i in supporting.get(clean_title, set()):
                    key_sents.append(f"[KEY] {cleaned}")
                else:
                    normal_sents.append(cleaned)

            if not key_sents and not normal_sents:
                continue

            # Đưa supporting sentences lên đầu → tăng relevance
            all_sents = key_sents + normal_sents
            body = " ".join(all_sents)

            # Prefix câu hỏi vào chunk supporting → cosine similarity cao hơn khi query
            if is_supporting and question:
                text = f"Câu hỏi: {question} Trả lời: {answer}\n{clean_title}: {body}"
            else:
                text = f"{clean_title}: {body}"

            chunks.append({
                "id":             f"{entry_id}_{clean_title}",
                "source":         filename,
                "text":           text,
                "title":          clean_title,
                "qa_id":          entry_id,
                "is_supporting":  is_supporting,
                "question":       question,
                "answer":         answer,
            })

    return chunks


def load_vimqa_qa_pairs(json_path: str) -> List[Dict]:
    """
    Trả về list câu hỏi - đáp án.
    Dùng để test/đánh giá chất lượng retrieval sau khi index.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    qa_pairs = []
    for entry in data:
        question = clean_text(entry.get("question", ""))
        answer   = clean_text(entry.get("answer", ""))

        if question and answer:
            qa_pairs.append({
                "id":       entry.get("_id"),
                "question": question,
                "answer":   answer,
                "type":     entry.get("type", ""),
            })

    return qa_pairs


# ── Test nhanh ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python json_loader.py <path_to_vimqa.json>")
        sys.exit(1)

    path = sys.argv[1]

    print("=" * 50)
    print("KIỂM TRA JSON LOADER")
    print("=" * 50)

    # Test QA pairs
    pairs = load_vimqa_qa_pairs(path)
    print(f"\nTổng số câu hỏi: {len(pairs)}")
    print("\nVí dụ 3 câu đầu:")
    for p in pairs[:3]:
        print(f"  Q: {p['question']}")
        print(f"  A: {p['answer']}")
        print(f"  Type: {p['type']}")
        print()

    # Test chunks
    chunks = load_vimqa_as_chunks(path)
    print(f"Tổng số chunks (paragraph-level): {len(chunks)}")
    supporting_count = sum(1 for c in chunks if c["is_supporting"])
    print(f"  Trong đó supporting: {supporting_count}")
    print(f"  Non-supporting: {len(chunks) - supporting_count}")

    print("\nVí dụ 2 chunk đầu:")
    for c in chunks[:2]:
        print(f"  [{c['title']}] (supporting={c['is_supporting']})")
        print(f"  {c['text'][:200]}...")
        print()

    # Test extract text
    text = extract_text_from_vimqa(path)
    print(f"\nText sau khi extract: {len(text.split())} từ")
    print("\n500 ký tự đầu tiên:")
    print(text[:500])
    print("...")