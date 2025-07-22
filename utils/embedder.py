import os
import pickle
import re
import faiss
import numpy as np
import openai
import tiktoken
from dotenv import load_dotenv

# Load OpenAI API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
openai.api_key = api_key


# ... [unchanged imports and OpenAI setup] ...


class EmbeddingStore:
    def __init__(self, model_name="text-embedding-3-small"):
        self.model_name = model_name
        self.index = None
        self.documents = []

    @staticmethod
    def make_question_temporally_explicit(query):
        present_tense_keywords = (
            r"\b(is|are|does|do|has|have|includes|include|currently|today)\b"
        )
        contains_present_tense = re.search(present_tense_keywords, query.lower())
        contains_explicit_year = re.search(r"\b20\d{2}\b", query)

        if contains_present_tense and not contains_explicit_year:
            return f"{query.strip()} (Assume the question refers to the most recent available year.)"
        return query

    def _embed(self, texts, max_tokens_per_batch=230000, max_items_per_batch=2048):
        if isinstance(texts, str):
            texts = [texts]
        before_count = len(texts)
        texts = [t for t in texts if isinstance(t, str) and t.strip()]
        after_count = len(texts)

        # print(
        #   f"[Filter] Dropped {before_count - after_count} invalid or empty chunks, {after_count} remaining."
        # )
        client = openai.OpenAI()
        enc = tiktoken.encoding_for_model(self.model_name)

        all_embeddings = []
        current_batch = []
        current_token_count = 0

        for idx, text in enumerate(texts):
            token_count = len(enc.encode(text))

            if token_count > max_tokens_per_batch:
                # print(
                #    f"Warning: Single chunk #{idx} exceeds {max_tokens_per_batch} tokens. Skipping."
                # )
                continue

            if (current_token_count + token_count > max_tokens_per_batch) or (
                len(current_batch) >= max_items_per_batch
            ):
                # print(
                #    f"Submitting batch with {len(current_batch)} chunks, {current_token_count} tokens."
                # )
                # print(f"First chunk preview: {current_batch[0][:100]}...")
                response = client.embeddings.create(
                    model=self.model_name, input=current_batch
                )
                batch_embeddings = [e.embedding for e in response.data]
                all_embeddings.extend(batch_embeddings)
                current_batch = []
                current_token_count = 0

            current_batch.append(text)
            current_token_count += token_count

        if current_batch:
            #  print(
            #     f"Submitting final batch with {len(current_batch)} chunks, {current_token_count} tokens."
            # )
            # print(f"First chunk preview: {current_batch[0][:100]}...")
            response = client.embeddings.create(
                model=self.model_name, input=current_batch
            )
            batch_embeddings = [e.embedding for e in response.data]
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings).astype("float32")

    def build_index(self, chunks, metadata):
        if not chunks:
            raise ValueError("No chunks provided to build the index.")
        # print("Generating embeddings via OpenAI API...")
        embeddings = self._embed(chunks)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        self.documents = list(zip(chunks, metadata))

    def search(self, query, top_k=10, company_filter=None, year_filter=None):
        if self.index is None:
            raise ValueError("Index not built or loaded.")
        if isinstance(company_filter, str):
            company_filter = [company_filter]
        if isinstance(year_filter, str):
            year_filter = [year_filter]

        query = EmbeddingStore.make_question_temporally_explicit(query)
        query_embedding = self._embed(query)[0].reshape(1, -1)
        distances, indices = self.index.search(query_embedding, top_k * 10)

        filtered = []
        for idx in indices[0]:
            if idx >= len(self.documents):
                continue
            chunk, meta = self.documents[idx]
            if not isinstance(meta, dict):
                continue

            company = meta.get("company", "").lower()
            if company_filter and company not in [c.lower() for c in company_filter]:
                continue

            if year_filter and str(meta.get("year")) not in set(map(str, year_filter)):
                continue

            score = self._score_chunk(chunk, query, meta)
            filtered.append((chunk, meta, score))

        filtered.sort(key=lambda x: x[2], reverse=True)
        return [(chunk, meta) for chunk, meta, _ in filtered[:top_k]]

    def _score_chunk(self, chunk, query, meta):
        score = 0
        chunk_lower = chunk.lower()
        query_lower = query.lower()

        if re.search(
            r"\d[\d,\.]*( million| billion| percent|%)?", chunk, re.IGNORECASE
        ):
            score += 1

        for term in query_lower.split():
            if term in chunk_lower:
                score += 1

        def get_dynamic_keywords(q):
            if "ceo" in q:
                return [
                    "ceo",
                    "chief executive",
                    "leadership",
                    "background",
                    "experience",
                ]
            elif "growth" in q or "grew" in q:
                return ["growth", "increase", "expansion", "year over year"]
            elif "solid" in q or "strong" in q or "stable" in q:
                return [
                    "stability",
                    "solvency",
                    "liquidity",
                    "balance sheet",
                    "resilience",
                ]
            elif "board" in q:
                return ["board", "director", "governance"]
            elif "risk" in q:
                return ["risk", "uncertainty", "exposure", "challenge"]
            elif "profit" in q:
                return ["profit", "net income", "ebitda", "margin"]
            elif "revenue" in q:
                return ["revenue", "sales", "top line"]
            else:
                return []

        for keyword in get_dynamic_keywords(query_lower):
            if keyword in chunk_lower:
                score += 1

        if "ceo" in query_lower:
            if re.search(r"\b[A-Z][a-z]+\s[A-Z][a-z]+\b", chunk):
                score += 1

        year_bonus = self._query_year_match_bonus(query_lower, meta)
        score += year_bonus

        return score

    def _query_year_match_bonus(self, query_lower, meta):
        """Returns +3 if one year matches, +5 if two or more years match a range."""
        all_years = re.findall(r"\b(20\d{2})\b", query_lower)
        match_count = 0

        range_match = re.search(
            r"(?:from\s+)?(20\d{2})\s*(?:to|-|-)\s*(20\d{2})", query_lower
        )
        if range_match:
            try:
                start, end = int(range_match.group(1)), int(range_match.group(2))
                if start <= end:
                    year_range = {str(y) for y in range(start, end + 1)}
                    if str(meta.get("year")) in year_range:
                        match_count += 1
            except ValueError:
                pass

        if str(meta.get("year")) in all_years:
            match_count += 1

        if match_count >= 2:
            return 5
        elif match_count == 1:
            return 3
        else:
            return 0

    def _query_year_in_meta(self, query, meta):
        text = query.lower()
        all_years = re.findall(r"\b(20\d{2})\b", text)

        range_match = re.search(r"(?:from\s+)?(20\d{2})\s*(?:to|-|-)\s*(20\d{2})", text)
        if range_match:
            try:
                start, end = int(range_match.group(1)), int(range_match.group(2))
                if start <= end:
                    year_range = {str(y) for y in range(start, end + 1)}
                    return str(meta.get("year")) in year_range
            except ValueError:
                pass

        return str(meta.get("year")) in all_years if all_years else False

    def save(self, index_path="faiss_index", doc_path="documents.pkl"):
        faiss.write_index(self.index, index_path)
        with open(doc_path, "wb") as f:
            pickle.dump(self.documents, f)

    def load(self, index_path="faiss_index", doc_path="documents.pkl"):
        if not os.path.exists(index_path) or not os.path.exists(doc_path):
            raise FileNotFoundError("Index or document store not found.")
        self.index = faiss.read_index(index_path)
        with open(doc_path, "rb") as f:
            self.documents = pickle.load(f)
