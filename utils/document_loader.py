import os
import sys
import pickle
import logging
import re
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import nltk

nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

sys.path.append(os.path.dirname(__file__))

import chunker
import embedder
import rag_pipeline

logging.basicConfig(level=logging.INFO)

CHUNKS_PATH = "chunks.pkl"
INDEX_PATH = "faiss_index"
DOCS_PATH = "documents.pkl"


def safe_text(text):
    try:
        return text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    except Exception:
        return ""


def extract_year_from_filename(filename):
    for year in ["2017", "2018", "2019", "2020", "2021", "2022", "2023"]:
        if year in filename:
            return year
    return None


def extract_facts_from_html_file(file_path):
    with open(file_path, "rb") as f:
        content = f.read().decode(errors="ignore")
    soup = BeautifulSoup(content, "html.parser")

    text_blocks = []
    for tag in soup.find_all(["p", "div", "span"]):
        text = tag.get_text(separator=" ", strip=True)
        if text and len(text.split()) > 3:
            text_blocks.append(text)

    if not text_blocks:
        text_blocks = [soup.get_text(separator=" ", strip=True)]
    return text_blocks


def load_and_chunk_all(folder_path, max_tokens=8192, overlap=50):
    all_chunks, all_metadata = [], []
    for company in os.listdir(folder_path):
        company_folder = os.path.join(folder_path, company)
        if os.path.isdir(company_folder):
            for filename in os.listdir(company_folder):
                file_path = os.path.join(company_folder, filename)
                year = extract_year_from_filename(filename)

                if filename.endswith(".pdf"):
                    reader = PdfReader(file_path)
                    for page_num, page in enumerate(reader.pages):
                        text = safe_text(page.extract_text() or "")
                        if not text.strip():
                            continue
                        chunks = chunker.chunk_text_by_tokens(
                            text, max_tokens=max_tokens, overlap=overlap
                        )
                        metadata = {
                            "company": company,
                            "filename": filename,
                            "page": page_num + 1,
                            "year": year,
                        }
                        all_chunks.extend(chunks)
                        all_metadata.extend([metadata] * len(chunks))

                elif filename.endswith(".html"):
                    facts = extract_facts_from_html_file(file_path)
                    for idx, fact in enumerate(facts):
                        chunks = chunker.chunk_text_by_tokens(
                            fact, max_tokens=max_tokens, overlap=overlap
                        )
                        metadata = {
                            "company": company,
                            "filename": filename,
                            "page": idx + 1,
                            "year": year,
                        }
                        all_chunks.extend(chunks)
                        all_metadata.extend([metadata] * len(chunks))

    return all_chunks, all_metadata


def save_chunks(chunks, metadata):
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump((chunks, metadata), f)


def load_chunks():
    with open(CHUNKS_PATH, "rb") as f:
        return pickle.load(f)


def infer_filters_from_question(question):
    company_keywords = {
        "crayon": "Crayon",
        "softwareone": "SoftwareOne",
        "uber": "UBER",
    }

    # Normalize case
    question_lower = question.lower()

    company_filter = [
        name for key, name in company_keywords.items() if key in question_lower
    ]

    # Match both individual years and ranges like "2017–2023" or "from 2017 to 2023"
    year_matches = re.findall(r"\b(20\d{2})\b", question)
    range_match = re.search(r"from (\d{4}) to (\d{4})", question.lower())

    year_filter = set(year_matches)

    if range_match:
        start_year, end_year = map(int, range_match.groups())
        year_filter.update(str(y) for y in range(start_year, end_year + 1))

    return company_filter if company_filter else None, (
        sorted(year_filter) if year_filter else None
    )


def log_formatted_answer(question: str, result: dict):
    logging.info("\n" + "=" * 100)
    logging.info(f"❓ Question: {question}")
    logging.info("🧠 Answer:\n" + result["answer"].strip())
    logging.info("=" * 100 + "\n")


if __name__ == "__main__":
    data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

    if os.path.exists(CHUNKS_PATH):
        logging.info("Loading pre-processed chunks from disk...")
        chunks, metadata = load_chunks()
    else:
        logging.info("Processing PDFs and HTML/XBRL and generating chunks...")
        chunks, metadata = load_and_chunk_all(data_folder)
        save_chunks(chunks, metadata)

    logging.info(f"Total chunks loaded: {len(chunks)}")

    store = embedder.EmbeddingStore()
    if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
        logging.info("Loading existing FAISS index and metadata...")
        store.load(INDEX_PATH, DOCS_PATH)
    else:
        logging.info("Building new FAISS index...")
        store.build_index(chunks, metadata)
        store.save(INDEX_PATH, DOCS_PATH)

    benchmark_questions = [
        "What was the total revenue for Crayon in 2023?",
        "What was the total revenue for SoftwareOne in 2022?",
        "Which company Crayon or SoftwareOne was most profitable in 2022 and 2023?",
        "Who was on the board in Crayon in 2021?",
        "What are the most significant risk factors for SoftwareOne in 2021?",
        "What are the most significant risk factors for UBER in 2021?",
        "What did the CEO of Crayon do before joining Crayon?",
        "Where is the CEO of Crayon from?",
        "How much did Crayon grow from 2017 through 2023?",
        "Which of the three companies, Uber, SoftwareOne and Crayon are the most solid?",
        "Compare the profitability for Uber, SoftwareOne and Crayon, year over year from 2019 through 2023?",
    ]

    for question in benchmark_questions:
        logging.info(f"Processing Question: {question}")
        company_filter, year_filter = infer_filters_from_question(question)
        results = store.search(
            question, top_k=10, company_filter=company_filter, year_filter=year_filter
        )

        if results:
            chunks_only = [chunk for chunk, _ in results]
            metas_only = [meta for _, meta in results]

            rag = rag_pipeline.RAGPipeline()

            # Print the 10 relevant chunks with their tags
            # logging.info(f"--- Retrieved Chunks for: '{question}' ---")
            # for idx, (chunk, meta) in enumerate(zip(chunks_only, metas_only)):
            #   tag = f"[{meta.get('company', 'Unknown')} - Page {meta.get('page', 'Unknown')}, {meta.get('year', 'Unknown')}]"
            #  logging.info(f"Chunk {idx + 1}:\n{tag}\n{chunk}\n")
            # logging.info(
            #    "-------------------------------------------------------------"
            # )

            answer = rag.generate_answer(chunks_only, question, metas_only)
            log_formatted_answer(question, answer)
        else:
            logging.info(f"No relevant results found for '{question}'.\n")
