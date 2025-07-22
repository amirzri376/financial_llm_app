# Financial LLM App

This project is a Retrieval-Augmented Generation (RAG) system for answering financial questions from company annual reports (PDF and HTML). It uses OpenAI's GPT-4o for generation and FAISS for semantic search over document chunks.

---

## What It Does

Given a financial question like:

- “What was the total revenue for Crayon in 2023?”
- “Who was on the board of Crayon in 2021?”
- “What did the CEO of Crayon do before joining the company?”

The app:

1. Loads financial reports from the `data/` folder (PDF or HTML).
2. Extracts text, splits into token-limited chunks (8192 tokens, with overlap).
3. Embeds the chunks using `text-embedding-3-small` via OpenAI API.
4. Stores embeddings in a FAISS index.
5. Retrieves relevant chunks for a given question.
6. Sends them to GPT-4o with a structured system prompt to extract accurate, cited answers.

---

## Project Structure

```
financial_llm_app/
├── utils/
│   ├── document_loader.py     # Loads reports, extracts content, chunks text, builds/loads index
│   ├── chunker.py             # Token-limited, sentence-aware chunking
│   ├── embedder.py            # Embeds chunks, builds & queries FAISS index
│   ├── rag_pipeline.py        # GPT-4o prompt + answer generation with citation
│
├── data/                      # PDF and HTML annual reports
│   └── Crayon_annual-report_2023.pdf, etc.
│
├── index/                     # Generated FAISS index + chunk metadata
│   ├── faiss_index/
│   ├── documents.pkl
│   ├── chunks.pkl
│
├── .env                       # Contains your OpenAI API key (not in Git)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## How to Run

1. **Install dependencies**

```
pip install -r requirements.txt
```

2. **Add your OpenAI key**

Create a file named `.env` in the root folder with the following content:

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

3. **Add annual reports**

Place PDFs/HTMLs inside the `data/` folder. Example:

```
data/
├── Crayon_annual-report_2023.pdf
├── SoftwareOne_annual-report_2022.pdf
└── ...
```

4. **Run the full pipeline**

```
python utils/document_loader.py
```

This script:
- Loads or rebuilds the FAISS index
- Applies metadata filtering for company/year
- Runs a predefined list of benchmark financial questions
- Prints answers with source citations

---

## Example Benchmark Questions

- What was the total revenue for Crayon in 2023?
- Who was on the board in Crayon in 2021?
- What did the CEO of Crayon do before joining Crayon?
- How much did Crayon grow from 2017 through 2023?
- Compare the profitability for Uber, SoftwareOne, and Crayon from 2019 through 2023.
- What are the most significant risk factors for SoftwareOne in 2021?

---

## Notes

- `.env`, FAISS index files, and `.pkl` files are ignored by Git (`.gitignore`).
- Chunk size is optimized for long-form documents (8192 token max).
- GPT-4o prompt is designed for high-precision answers with strict source citation.
- The app is fully modular: you can replace the model, embedding, or retriever logic independently.

---

## 📄 License

This project is provided for educational and research use. Do not upload proprietary or confidential documents.
