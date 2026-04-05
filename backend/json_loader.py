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
from typing import List, Dict


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

    # Xóa ký tự control (ASCII 0-8, 11-12, 14-31, 127)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Xóa artifact ]X]Y]Z (dấu ] theo sau bởi chữ cái viết hoa/có dấu)
    # Đây là lỗi phổ biến khi parse Wikipedia HTML dump sang JSON
    text = re.sub(r'\][A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠ-Ỹ\s]+(?=\]|$)', '', text)
    text = re.sub(r'\](?=[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯ])', ' ', text)

    # Chỉ xóa các ký tự THỰC SỰ vô nghĩa / không in được (không phải Unicode hợp lệ)
    # Giữ lại tất cả ký tự Unicode có thể in được, bao gồm:
    # & – — [] chữ Ả Rập tiếng nước ngoài trong ngoặc, v.v.
    # Chỉ loại bỏ các ký tự thay thế (replacement char) và private-use Unicode
    text = re.sub(r'[\ufffd\ue000-\uf8ff]', '', text)  # replacement char + private use

    # Xóa dấu ] đứng một mình (artifact từ Wikipedia markup [[...]]) 
    # nhưng chỉ khi nó không nằm trong cặp ngoặc hợp lệ
    text = re.sub(r'(?<!\[)\](?!\])', ' ', text)
    text = re.sub(r'\[(?!\[)', ' ', text)

    # Normalize nhiều khoảng trắng → 1
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# ── Loaders ─────────────────────────────────────────────────────────────────
def extract_text_from_vimqa(json_path: str) -> str:
    """
    Convert toàn bộ context trong VIMQA → plain text sạch
    Dùng để đưa vào chunk_text() của indexing.py

    Mỗi paragraph → một đoạn text với tiêu đề
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lines = []
    seen_titles = set()  # tránh duplicate paragraph cùng tiêu đề

    for entry in data:
        for para_title, sentences in entry.get("context", []):

            # Bỏ qua paragraph đã xử lý (VIMQA có nhiều entry dùng chung context)
            if para_title in seen_titles:
                continue
            seen_titles.add(para_title)

            # Clean tiêu đề
            clean_title = clean_text(para_title)
            if not clean_title:
                continue

            lines.append(f"# {clean_title}")

            for sent in sentences:
                cleaned = clean_text(sent)
                # Bỏ câu quá ngắn (< 10 ký tự) — thường là artifact
                if cleaned and len(cleaned) >= 10:
                    lines.append(cleaned)

            lines.append("")  # dòng trống phân cách giữa các paragraph

    return "\n".join(lines)


def load_vimqa_as_chunks(json_path: str) -> List[Dict]:
    """
    Đọc VIMQA JSON → list chunks (mỗi paragraph = 1 chunk)
    Dùng khi muốn giữ nguyên cấu trúc paragraph thay vì chunk theo số từ
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = []
    filename = Path(json_path).name
    seen_titles = set()

    for entry in data:
        entry_id = entry.get("_id", "unknown")

        for para_title, sentences in entry.get("context", []):
            if para_title in seen_titles:
                continue
            seen_titles.add(para_title)

            # Ghép và clean các câu
            full_text = " ".join(
                clean_text(s) for s in sentences
                if clean_text(s) and len(clean_text(s)) >= 10
            )

            if not full_text.strip():
                continue

            chunks.append({
                "id":     f"{entry_id}_{clean_text(para_title)}",
                "source": filename,
                "text":   full_text,
                "title":  clean_text(para_title),
                "qa_id":  entry_id,
            })

    return chunks


def load_vimqa_qa_pairs(json_path: str) -> List[Dict]:
    """
    Trả về list câu hỏi - đáp án
    Dùng để test/đánh giá chất lượng retrieval sau khi index
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
        print()

    # Test extract text
    text = extract_text_from_vimqa(path)
    print(f"Text sau khi extract: {len(text.split())} từ")
    print("\n500 ký tự đầu tiên:")
    print(text[:500])
    print("...")