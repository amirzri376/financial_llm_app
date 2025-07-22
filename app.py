import streamlit as st
from utils.embedder import EmbeddingStore
from utils.rag_pipeline import RAGPipeline

st.set_page_config(page_title="Financial Document Q&A", layout="wide")
st.title("Financial Document Q&A")


# Load index and RAG pipeline only once
@st.cache_resource
def load_resources():
    store = EmbeddingStore()
    store.load()
    rag = RAGPipeline()
    return store, rag


store, rag = load_resources()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "accumulated_context" not in st.session_state:
    st.session_state.accumulated_context = []

# User input
query = st.text_input("Ask your financial question:", key="user_query")

if st.button("Submit"):
    if query.strip():
        # Search for new context
        results = store.search(query, top_k=5)
        if results:
            # Format results with citations
            formatted_contexts = []
            for chunk, meta in results:
                meta_info = (
                    f"{meta.get('company', 'Unknown')} - {meta.get('filename', 'Unknown')} - Page {meta.get('page', 'Unknown')}"
                    if isinstance(meta, dict)
                    else str(meta)
                )
                formatted_contexts.append(f"[{meta_info}] {chunk}")

            # Update accumulated context with new chunks
            new_chunks = [chunk for chunk, _ in results]
            st.session_state.accumulated_context.extend(new_chunks)

            # Generate answer using all accumulated context and history
            answer = rag.generate_answer(
                st.session_state.accumulated_context,
                query,
                history=st.session_state.chat_history,
            )

            # Store the interaction
            st.session_state.chat_history.append(
                {"question": query, "answer": answer, "citations": formatted_contexts}
            )
        else:
            st.session_state.chat_history.append(
                {
                    "question": query,
                    "answer": "No relevant results found.",
                    "citations": [],
                }
            )

# Display chat history
st.write("---")
for chat in st.session_state.chat_history:
    st.markdown(f"**User:** {chat['question']}")
    st.markdown(f"**Answer:** {chat['answer']}")
    if chat["citations"]:
        with st.expander("Show Citations"):
            for citation in chat["citations"]:
                st.markdown(citation)
    st.write("---")
