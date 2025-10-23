import streamlit as st
from io import BytesIO
import PyPDF2
import re
from typing import List, Dict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# -------------------------
# Gemini API Setup
# -------------------------
if "GEMINI_API_KEY" not in st.secrets:
    st.error("⚠️ GEMINI_API_KEY not found in Streamlit secrets!")
else:
    st.info("✅ GEMINI_API_KEY found, configuring Gemini...")
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        st.success("Gemini API configured successfully.")
    except Exception as e:
        st.error(f"Gemini API configuration failed: {e}")

# -------------------------
# Function to call Gemini LLM
# -------------------------
def call_gemini(prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
        )
        st.info("✅ Gemini call successful")
        return response.text.strip()
    except Exception as e:
        st.error(f"⚠️ Gemini API Error: {e}")
        return f"Gemini API Error: {e}"

# -------------------------
# PDF → Text
# -------------------------
def pdf_to_pages(file_bytes: bytes) -> List[str]:
    try:
        reader = PyPDF2.PdfReader(BytesIO(file_bytes))
        pages = []
        for i, page in enumerate(reader.pages):
            txt = page.extract_text() or ""
            pages.append(txt.strip())
            st.write(f"Page {i+1} length: {len(txt)} chars")
        return pages
    except Exception as e:
        st.error(f"PDF read error: {e}")
        return []

# -------------------------
# Clean + Chunk Text
# -------------------------
def clean_text(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()

def chunk_text_with_meta(pages: List[str], chunk_size: int = 1200, overlap: int = 300) -> List[Dict]:
    chunks = []
    for p_idx, page_text in enumerate(pages):
        text = clean_text(page_text)
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(text_len, start + chunk_size)
            chunk = text[start:end]
            chunks.append({
                "id": f"p{p_idx}_s{start}",
                "text": chunk,
                "page": p_idx + 1
            })
            if end == text_len:
                break  # stop if we've reached the end
            start = start + chunk_size - overlap  # advance start properly
    return chunks


# -------------------------
# Embeddings + FAISS Index
# -------------------------
@st.cache_resource
def get_embedding_model():
    st.info("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    st.success("Embedding model loaded")
    return model

def build_faiss_index(embeddings: np.ndarray):
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    st.info(f"FAISS index built with {index.ntotal} vectors")
    return index

# -------------------------
# RAG Answering
# -------------------------
def answer_question(question: str, chunks: List[Dict], index, embeddings_model, chunk_embeddings, top_k: int = 5):
    st.info(f"Encoding question: {question}")
    try:
        q_emb = embeddings_model.encode([question])
        q_emb = np.array(q_emb).astype("float32")
        faiss.normalize_L2(q_emb)
    except Exception as e:
        st.error(f"Question embedding failed: {e}")
        return {"answer": f"Error embedding question: {e}", "retrieved": []}

    try:
        D, I = index.search(q_emb, top_k)
        st.info(f"Top {top_k} chunks retrieved with scores: {D[0]}")
    except Exception as e:
        st.error(f"FAISS search failed: {e}")
        return {"answer": f"FAISS search error: {e}", "retrieved": []}

    retrieved = []
    for score, idx in zip(D[0], I[0]):
        retrieved.append({
            "score": float(score),
            "chunk": chunks[idx],
            "text": chunks[idx]["text"]
        })

    context_text = "\n\n---\n\n".join(
        [f"[Page {r['chunk']['page']}] {r['text'][:1000]}" for r in retrieved]
    )

    system_prompt = (
        "You are a helpful assistant. Use only the context below to answer the user's question. "
        "If the answer is not found in the context, say 'I don't know'. "
        "Cite page numbers where relevant."
    )

    full_prompt = f"{system_prompt}\n\nContext:\n{context_text}\n\nQuestion: {question}\nAnswer:"

    try:
        model_answer = call_gemini(full_prompt)
    except Exception as e:
        st.error(f"Gemini call failed: {e}")
        model_answer = f"Error calling Gemini: {e}"

    return {
        "answer": model_answer,
        "retrieved": retrieved
    }

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="AI PDF Agent (Gemini)", layout="wide")
st.title("🤖 AI PDF Agent — Debug Mode")

uploaded = st.file_uploader("📄 Upload your PDF", type=["pdf"])

if uploaded is not None:
    file_bytes = uploaded.read()
    st.info("🔍 Extracting text from PDF...")
    pages = pdf_to_pages(file_bytes)
    if not pages:
        st.error("No text extracted from PDF. It may be scanned or image-based.")

    st.info("📚 Chunking text for retrieval...")
    chunks = chunk_text_with_meta(pages)
    if not chunks:
        st.error("No chunks created. PDF might be empty or unreadable.")

    st.info("⚙️ Embedding chunks (first time may take some time)...")
    try:
        emb_model = get_embedding_model()
        texts = [c["text"] for c in chunks]
        chunk_embeddings = emb_model.encode(texts, show_progress_bar=True)
        chunk_embeddings = np.array(chunk_embeddings).astype("float32")
        faiss.normalize_L2(chunk_embeddings)
        index = build_faiss_index(chunk_embeddings)
        st.success("🚀 PDF is ready for Q&A!")
    except Exception as e:
        st.error(f"Embedding / FAISS build failed: {e}")

    question = st.text_input("💬 Ask a question about your PDF:")
    if question:
        with st.spinner("🧠 Thinking..."):
            result = answer_question(question, chunks, index, emb_model, chunk_embeddings)
        st.subheader("🟩 Answer")
        st.write(result["answer"])

        with st.expander("📑 Retrieved Sources"):
            for r in result["retrieved"]:
                st.markdown(f"- **Page {r['chunk']['page']}** (score: {r['score']:.3f})")
                st.caption(r["chunk"]["text"][:300] + "...")
