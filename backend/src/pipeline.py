from openai import OpenAI
from src.retriever import Retriever


class Reader:
    def __init__(self, model_name: str = "grok-3", api_key: str | None = None):
        self.model_name = model_name

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
        )

        print(f"✅ Reader ready — Grok model: {model_name}")

    def answer(self, question: str, contexts: list) -> str:
        context_text = "\n\n".join([
            f"[{i + 1}] ({r['title']})\n{r['context']}"
            for i, r in enumerate(contexts)
        ])

        prompt = f"""
Bạn là trợ lý hỏi đáp tiếng Việt dựa trên tài liệu được cung cấp.

Yêu cầu:
- Chỉ trả lời dựa trên các đoạn văn bản bên dưới.
- Không tự bịa thông tin ngoài tài liệu.
- Nếu không tìm thấy thông tin liên quan, hãy trả lời:
"Tôi không tìm thấy thông tin liên quan trong tài liệu."

Các đoạn văn bản:
{context_text}

Câu hỏi:
{question}

Trả lời:
""".strip()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1024,
        )

        return response.choices[0].message.content


class RAGPipeline:
    def __init__(
        self,
        data_dir: str,
        api_key: str | None = None,
        model_name: str = "grok-3",
    ):
        self.retriever = Retriever(data_dir=data_dir)
        self.reader = Reader(model_name=model_name, api_key=api_key)

        print("✅ RAG pipeline ready")

    def ask(self, question: str, top_k: int = 4):
        contexts = self.retriever.search(question, top_k=top_k)

        answer = self.reader.answer(
            question=question,
            contexts=contexts,
        )

        sources = [
            {
                "source": r["title"],
                "text": r["context"][:500],
                "score": r["score"],
            }
            for r in contexts
        ]

        return {
            "answer": answer,
            "sources": sources,
        }

    def status(self):
        return {
            "ready": True,
            "num_chunks": self.retriever.count_chunks(),
            "documents": []
        }