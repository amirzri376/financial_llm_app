from typing import List
import tiktoken
import nltk
from nltk.tokenize import sent_tokenize

# Ensure punkt is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


def chunk_text_by_tokens(
    text: str,
    max_tokens: int = 8192,
    overlap: int = 100,
    model_name: str = "text-embedding-3-small",
) -> List[str]:
    """
    Token-limited, sentence-aware chunking that splits long sentences if needed.
    """
    enc = tiktoken.encoding_for_model(model_name)
    sentences = sent_tokenize(text)

    chunks = []
    current_chunk_tokens = []
    current_len = 0

    for sent in sentences:
        sent_tokens = enc.encode(sent)

        # If the sentence itself exceeds max_tokens, split it safely
        if len(sent_tokens) > max_tokens:
            start = 0
            while start < len(sent_tokens):
                end = min(start + max_tokens, len(sent_tokens))
                chunk_tokens = sent_tokens[start:end]
                chunks.append(enc.decode(chunk_tokens))
                start += max_tokens - overlap  # Apply overlap to split parts
            continue  # Move to next sentence after splitting this one

        # If adding this sentence exceeds limit, finalize current chunk first
        if current_len + len(sent_tokens) > max_tokens:
            if current_chunk_tokens:
                chunks.append(enc.decode(current_chunk_tokens))
            # Apply overlap
            overlap_tokens = (
                current_chunk_tokens[-overlap:]
                if overlap < len(current_chunk_tokens)
                else current_chunk_tokens
            )
            current_chunk_tokens = overlap_tokens + sent_tokens
            current_len = len(current_chunk_tokens)
        else:
            # Safe to add sentence
            current_chunk_tokens.extend(sent_tokens)
            current_len += len(sent_tokens)

    # Finalize last chunk if any
    if current_chunk_tokens:
        chunks.append(enc.decode(current_chunk_tokens))

    return chunks
