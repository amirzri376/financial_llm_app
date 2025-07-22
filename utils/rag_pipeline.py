import os
import openai
from dotenv import load_dotenv

load_dotenv()


class RAGPipeline:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        openai.api_key = api_key

        print("[Model Load] ChatGPT API (GPT-4 Turbo) initialized successfully")

    def generate_answer(self, context_chunks, question, metadata_list=None):
        context_with_meta = []
        references = []
        for idx, chunk in enumerate(context_chunks[:10]):
            meta = (
                metadata_list[idx] if metadata_list and idx < len(metadata_list) else {}
            )
            meta_info = f"[{meta.get('company', 'Unknown')} - {meta.get('filename', 'Unknown')} - Page {meta.get('page', 'Unknown')} - Year {meta.get('year', 'Unknown')}]"
            context_with_meta.append(f"{meta_info}\n{chunk}")
            references.append(meta_info)

        context = "\n\n".join(context_with_meta)
        limited_context = "\n\n".join(context_with_meta[:10])

        messages = [
            {
                "role": "system",
                "content": (
                    "When the question is about the Job tittels (e.g., where is the CEO of Crayon comes from), extract names and use it to find the answer to the question.\n\n"
                    "You are a financial assistant answering questions based on document excerpts. "
                    "Always extract relevant facts directly from the context. Be accurate, specific, and do not speculate.\n\n"
                    "You must cite the exact source tag (e.g., [Crayon - Crayon_annual-report_2023.pdf"
                    "When the question is about a number (e.g., revenue, profit, growth), extract the exact value and unit from the text. "
                    "Prioritize recent, explicitly dated values. Avoid paraphrasing.\n\n"
                    "When the question is about people (e.g., board members, CEO background), extract names and roles exactly as stated.\n\n"
                    "When the question is about risks or qualitative information, summarize the most relevant sentence or phrase from the context.\n\n"
                    "If the answer is not clearly stated in the context, say it is not available."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{limited_context}\n\nQuestion:\n{question}\n\nAnswer:",
            },
        ]

        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=200,
            temperature=0.7,
            top_p=0.95,
        )

        answer = response.choices[0].message.content.strip()
        return {"answer": answer}
