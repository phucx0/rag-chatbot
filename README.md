# RAG Chatbot — Hỏi đáp tài liệu với Transformer

Ứng dụng chatbot hỏi đáp tài liệu sử dụng kiến trúc RAG (Retrieval-Augmented Generation)
kết hợp Sentence-BERT (Transformer encoder) + FAISS + Flan-T5.

## Kiến trúc hệ thống

```
Tài liệu (PDF/TXT)
      │
      ▼
[Sentence-BERT]  ←── Transformer encoder (keepitreal/vietnamese-sbert)
      │ encode thành vector 384 chiều
      ▼
[FAISS Index]    ←── Vector database, lưu & tìm kiếm nhanh
      │
      │  Lúc query:
      │  Câu hỏi → encode → cosine similarity search → top-K chunks
      ▼
[Flan-T5]        ←── Transformer generator (VietAI/vit5-base)
      │ sinh câu trả lời từ context
      ▼
[FastAPI Server] ←── REST API: /upload, /chat, /status
      │
      ▼
[React Website]  ←── Giao diện chat + upload tài liệu
```

## Cấu trúc thư mục

```
rag-chatbot/
└── backend
    └── __pycache__
    └── .hf_cache
    └── .model_cache
    └── data
    └── notebook
    └── src
        ├── api.py
        ├── pipeline.py
        ├── reader.py
        ├── retriever.py
    ├── README.md
    └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── index.css
│   │   └── components/
│   │       ├── ChatWindow.jsx
│   │       └── Sidebar.jsx
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
```

---

## Hướng dẫn chạy

### Bước 1 — Cài đặt Python

```bash
cd backend
pip install -r requirements.txt
```

> Lần đầu sẽ tải model ~1GB. Cần kết nối internet.

### Bước 2 — Chuẩn bị tài liệu

Đặt file `.pdf` hoặc `.txt` vào thư mục `docs/`:

```bash
mkdir docs
# copy file tài liệu của bạn vào đây
cp your_document.pdf ../docs/
```

### Bước 3 — Chạy Server

```bash
cd backend
uvicorn src.api:app --reload --port 8000
```

### Bước 5 — Chạy React Website

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

Mở trình duyệt: http://localhost:3000

---

## API Endpoints

| Method | Endpoint | Mô tả                         |
| ------ | -------- | ----------------------------- |
| GET    | /status  | Trạng thái index              |
| POST   | /chat    | Gửi câu hỏi, nhận câu trả lời |

## Giải thích kỹ thuật

### 1. Transformer Encoder (Sentence-BERT)

- Model: `keepitreal/vietnamese-sbert`
- Kiến trúc: 12 lớp Transformer encoder, 384 chiều hidden
- Kỹ thuật: Mean pooling trên tất cả token hidden states → sentence vector
- Hỗ trợ 50+ ngôn ngữ bao gồm tiếng Việt

### 2. FAISS Vector Search

- Dùng `IndexFlatIP` (Inner Product)
- Embeddings đã normalize → IP = cosine similarity
- Tìm top-K chunks liên quan nhất với câu hỏi

### 3. Transformer Generator (Flan-T5)

- Model: `VietAI/vit5-base`
- Kiến trúc: Encoder-Decoder Transformer (seq2seq)
- Fine-tuned theo instruction-following
- Sinh câu trả lời từ context chunks

### 4. RAG Pipeline

```
query → encode(query) → FAISS.search(top_k) → T5.generate(query + context) → answer
```

---

## Biến môi trường

| Biến            | Mặc định                | Mô tả                    |
| --------------- | ----------------------- | ------------------------ |
| INDEX_DIR       | `.`                     | Thư mục chứa FAISS index |
| GENERATOR_MODEL | `VietAI/vit5-base`      | Model T5 để generate     |
| VITE_API_URL    | `http://localhost:8000` | URL của backend API      |

---

## Lưu ý khi chạy trên CPU

- Lần đầu load model mất ~30-60 giây
- Encode một câu hỏi: ~0.5-1 giây
- Generate với Flan-T5: ~3-8 giây/câu trả lời
- Nếu quá chậm: set `GENERATOR_MODEL=` rỗng để dùng extraction mode (nhanh hơn)
