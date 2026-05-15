from openai import OpenAI


class Reader:
    def __init__(self, model_name: str = 'grok-3', api_key: str = None):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key,
            base_url='https://api.x.ai/v1',  # endpoint của xAI
        )
        print(f'✅ Reader ready — Grok model: {model_name}')

    def answer(self, question: str, contexts: list) -> str:
        context_text = '\n\n'.join([
            f"[{i+1}] ({r['title']})\n{r['context']}"
            for i, r in enumerate(contexts)
        ])

        prompt = f"""Bạn là trợ lý trả lời câu hỏi dựa trên các đoạn văn bản được cung cấp.
Chỉ trả lời dựa trên thông tin có trong các đoạn văn bản dưới đây.
Nếu không tìm thấy câu trả lời, hãy nói "Tôi không tìm thấy thông tin liên quan."

Các đoạn văn bản:
{context_text}

Câu hỏi: {question}

Trả lời:"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.1,
            max_tokens=1024,
        )
        return response.choices[0].message.content