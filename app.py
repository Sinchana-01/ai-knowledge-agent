import os
import re
import streamlit as st
from dotenv import load_dotenv

import google.generativeai as genai

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from pypdf import PdfReader


# -----------------------------------------------------------
# 1. CONFIGURE GEMINI
# -----------------------------------------------------------

load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")

if not gemini_key:
    raise ValueError("⚠️ GEMINI_API_KEY missing! Add it to your .env file.")

genai.configure(api_key=gemini_key)

# Use a model that your key supports (from your list_models output)
GEMINI_MODEL_NAME = "models/gemini-flash-latest"
model = genai.GenerativeModel(GEMINI_MODEL_NAME)


# -----------------------------------------------------------
# 2. FILE & DOCUMENT HELPERS (KNOWLEDGE BASE)
# -----------------------------------------------------------

DATA_DIR = "data"


def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)


def get_unique_filename(filename: str) -> str:
    """If filename exists, add _1, _2, etc. to make it unique."""
    base, ext = os.path.splitext(filename)
    counter = 1
    new_name = filename

    while os.path.exists(os.path.join(DATA_DIR, new_name)):
        new_name = f"{base}_{counter}{ext}"
        counter += 1

    return new_name


def save_uploaded_files(uploaded_files):
    """Save uploaded TXT/PDF files into the data/ folder, avoiding duplicates."""
    ensure_data_dir()
    saved_files = []

    for file in uploaded_files:
        original_name = file.name

        # If we've already saved a file with this original name in this session, skip it
        if original_name in st.session_state.saved_filenames:
            continue

        # Make sure the actual filename on disk is unique
        filename = get_unique_filename(original_name)
        filepath = os.path.join(DATA_DIR, filename)

        try:
            with open(filepath, "wb") as f:
                f.write(file.getbuffer())
            saved_files.append(filename)
            # Remember this original name so we don't save it again on rerun
            st.session_state.saved_filenames.add(original_name)
        except PermissionError:
            st.warning(
                f"⚠️ Could not save {filename} due to file permission issues. "
                "Please close any open copies of this file and try again."
            )
        except Exception as e:
            st.warning(f"⚠️ Error saving {filename}: {e}")

    return saved_files


def load_txt_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_pdf_file(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"
    return text


def load_documents(data_dir=DATA_DIR):
    """Read all TXT/PDF files from data/ and return (documents, sources)."""
    docs = []
    sources = []

    if not os.path.exists(data_dir):
        return [], []

    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)

        if os.path.isfile(filepath):
            try:
                if filename.lower().endswith(".txt"):
                    content = load_txt_file(filepath)
                elif filename.lower().endswith(".pdf"):
                    content = load_pdf_file(filepath)
                else:
                    continue  # ignore other file types

                if content.strip():
                    docs.append(content)
                    sources.append(filename)
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    return docs, sources


# -----------------------------------------------------------
# 3. BUILD TF-IDF INDEX (KNOWLEDGE BASE)
# -----------------------------------------------------------

def build_index():
    

    documents, sources = load_documents(DATA_DIR)

    if not documents:
        st.warning("No documents found in the data/ folder. Please upload some files.")
        return None, None, None, None

    

    st.info("🔍 Building TF-IDF index...")
    vectorizer = TfidfVectorizer(stop_words="english")
    doc_vectors = vectorizer.fit_transform(documents)

    st.success("Ask questions now!")
    return vectorizer, doc_vectors, documents, sources


# -----------------------------------------------------------
# 4. STREAMLIT PAGE SETUP & STATE
# -----------------------------------------------------------

st.set_page_config(page_title="Multi-Agent AI (KB + Resume Screening)", page_icon="🤖", layout="wide")

st.title("🤖 Multi-Agent AI")
st.caption("📚 Knowledge Base Agent + 📄 Resume Screening Agent (Gemini-powered)")

# Session state init
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "saved_filenames" not in st.session_state:
    st.session_state.saved_filenames = set()

# Predefine vars used later
resume_files = None
jd_file = None


def add_message(role, content):
    st.session_state.chat_history.append({"role": role, "content": content})


# -----------------------------------------------------------
# 5. SIDEBAR: AGENT SELECTOR + PER-AGENT UI
# -----------------------------------------------------------

with st.sidebar:
    st.header("🧠 Select Agent")
    agent_type = st.selectbox(
        "Choose agent mode",
        ["Knowledge Base Agent", "Resume Screening"],
    )

    st.markdown("---")

    if agent_type == "Knowledge Base Agent":
        st.subheader("📁 Upload Documents (for KB Agent)")
        uploaded_files = st.file_uploader(
            "Upload TXT or PDF files",
            type=["txt", "pdf"],
            accept_multiple_files=True,
            key="kb_files",
        )

        if uploaded_files:
            saved = save_uploaded_files(uploaded_files)
            if saved:
                st.success(f"✅ Saved {len(saved)} file(s) to the data/ folder:")
                for name in saved:
                    st.write(f"- {name}")
                # Re-run app so new docs are indexed
                st.rerun()

        if st.button("🗑 Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

        st.markdown("---")
        st.header("ℹ️ About KB Agent")
        st.write(
            """
            - Uses **TF-IDF + cosine similarity** over uploaded TXT/PDF documents  
            - Uses **Gemini** to generate answers based on those documents  
            - Can also answer general questions when docs don't have the info  
            """
        )

    elif agent_type == "Resume Screening":
        st.subheader("📄 Upload for Resume Screening")
        jd_file = st.file_uploader(
            "Upload Job Description (PDF/TXT)",
            type=["pdf", "txt"],
            key="jd_file",
        )
        resume_files = st.file_uploader(
            "Upload one or more Resumes (PDF/TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key="resume_files",
        )

        st.markdown("---")
        st.header("ℹ️ About Resume Agent")
        st.write(
            """
            - Uses **Gemini** to score each resume against the job description  
            - Outputs a **score out of 10** and a short explanation  
            - Ranks candidates from best to worst match  
            """
        )


# -----------------------------------------------------------
# 6. BUILD KB INDEX (SHARED, EVEN IF NOT USING)
# -----------------------------------------------------------

vectorizer, doc_vectors, documents, sources = build_index()


# -----------------------------------------------------------
# 7. GEMINI HELPERS (TEXT EXTRACTION + MODES)
# -----------------------------------------------------------

def safe_extract_text(response):
    """
    Safely extract text from a Gemini response.
    Avoids ValueError when response.text is not available.
    """
    try:
        if hasattr(response, "text") and response.text:
            return response.text.strip()
    except Exception:
        pass

    try:
        parts = []
        if hasattr(response, "candidates") and response.candidates:
            for cand in response.candidates:
                content = getattr(cand, "content", None)
                if content and getattr(content, "parts", None):
                    for p in content.parts:
                        t = getattr(p, "text", None)
                        if t:
                            parts.append(t)
        if parts:
            return "\n".join(parts).strip()
    except Exception:
        pass

    return "I couldn't generate a response for this question."


def build_context_from_docs(query, top_k=3, max_chars=1200):
    """Always return the top_k most similar docs as context (no hard threshold)."""
    if vectorizer is None or doc_vectors is None or documents is None or sources is None:
        return "", []

    query_vec = vectorizer.transform([query])
    cosine_similarities = linear_kernel(query_vec, doc_vectors).flatten()

    related_doc_indices = cosine_similarities.argsort()[::-1][:top_k]

    context_parts = []
    used_sources = []

    for idx in related_doc_indices:
        source_name = sources[idx]
        full_text = documents[idx]
        snippet = full_text[:max_chars]

        context_parts.append(f"Source: {source_name}\n\n{snippet}")
        used_sources.append(source_name)

    if not context_parts:
        return "", []

    context = "\n\n---\n\n".join(context_parts)
    return context, used_sources


def ask_gemini(context, question):
    """Doc-based answer: only use provided context."""
    prompt = f"""
You are a helpful assistant answering questions based ONLY on the given context.
If the answer is not in the context, say exactly:
"I cannot find this information in the documents."

CONTEXT:
{context}

QUESTION:
{question}

Provide a clear, concise answer.
"""
    try:
        response = model.generate_content(prompt)
        return safe_extract_text(response)
    except Exception as e:
        st.error("❌ Error while calling Gemini API (context mode).")
        st.exception(e)
        return "I ran into an error while generating the answer. Please check the error above."


def ask_gemini_general(question):
    """General chat mode: let Gemini answer from its own knowledge."""
    prompt = f"""
You are a helpful and friendly assistant.

The user may ask general questions or small talk that are NOT related to any documents.
In that case, answer normally using your own knowledge.

QUESTION:
{question}

Provide a clear, concise answer.
"""
    try:
        response = model.generate_content(prompt)
        return safe_extract_text(response)
    except Exception as e:
        st.error("❌ Error while calling Gemini API (general mode).")
        st.exception(e)
        return "I ran into an error while answering your question. Please check the error above."


# -----------------------------------------------------------
# 8. KNOWLEDGE BASE AGENT LOGIC
# -----------------------------------------------------------

def answer_kb_question(query):
    try:
        # Normalize query
        q = query.strip().lower()

        # 1) Handle very basic small-talk without hitting the API
        greetings = {"hi", "hello", "hey", "yo", "hiya", "hai"}
        if q in greetings or "how are you" in q:
            return (
                "Hi! 👋 I'm your Knowledge Base Agent.\n\n"
                "You can:\n"
                "- Upload PDFs/TXTs in the sidebar (KB Agent mode)\n"
                "- Ask me about those documents (policies, manuals, FAQs)\n"
                "- Or ask general questions, and I'll answer using Gemini."
            )

        # 2) If we don't have an index yet, go straight to general mode
        if vectorizer is None or doc_vectors is None:
            return ask_gemini_general(query)

        # 3) Always try documents first (RAG)
        context, used_sources = build_context_from_docs(query)

        if context:
            answer = ask_gemini(context, query)

            # If Gemini says it can't find it in docs -> fall back to general answer
            if "i cannot find this information in the documents" in answer.lower():
                return ask_gemini_general(query)

            # Otherwise it's a doc-based answer; show sources
            if used_sources:
                answer += "\n\n---\n\nSources used: " + ", ".join(set(used_sources))
            return answer

        # 4) No context built at all -> general answer
        return ask_gemini_general(query)

    except Exception as e:
        st.error("❌ Unexpected error while answering the question.")
        st.exception(e)
        return "Something went wrong while processing your question. Please check the error above."


# -----------------------------------------------------------
# 9. RESUME SCREENING AGENT LOGIC
# -----------------------------------------------------------

def read_uploaded_file(file):
    """Read text from an in-memory UploadedFile (TXT/PDF)."""
    if file is None:
        return ""
    name = file.name.lower()
    try:
        if name.endswith(".txt"):
            file.seek(0)
            return file.read().decode("utf-8", errors="ignore")
        elif name.endswith(".pdf"):
            file.seek(0)
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += (page.extract_text() or "") + "\n"
            return text
    except Exception as e:
        st.warning(f"⚠️ Error reading file {file.name}: {e}")
    return ""


def score_single_resume(job_desc_text, resume_text, resume_name):
    """Use Gemini to rate a single resume against the job description."""
    prompt = f"""
You are a resume screening assistant.

JOB DESCRIPTION:
{job_desc_text}

CANDIDATE RESUME ({resume_name}):
{resume_text}

Rate how well this candidate matches the job on a scale of 0 to 10.
Reply in this exact format:

Score: <number>/10
Summary: <one short paragraph explaining why>

Make sure 'Score:' is on its own line and the score is a number between 0 and 10.
"""

    try:
        response = model.generate_content(prompt)
        text = safe_extract_text(response)

        score = 0.0
        summary = text.strip()

        # Try to parse "Score: X/10"
        for line in text.splitlines():
            if "score" in line.lower():
                nums = re.findall(r"(\d+(\.\d+)?)", line)
                if nums:
                    score = float(nums[0][0])
                    break

        return {
            "name": resume_name,
            "score": score,
            "raw_response": text,
            "summary": summary,
        }
    except Exception as e:
        st.error(f"❌ Error while scoring resume {resume_name}.")
        st.exception(e)
        return {
            "name": resume_name,
            "score": 0.0,
            "raw_response": "",
            "summary": "Error while scoring this resume.",
        }


def run_resume_screening(jd_file, resume_files):
    st.subheader("📄 Resume Screening Results")

    if jd_file is None:
        st.info("Please upload a **Job Description** in the sidebar.")
        return

    if not resume_files:
        st.info("Please upload at least one **Resume** in the sidebar.")
        return

    # Read JD text
    jd_text = read_uploaded_file(jd_file)
    if not jd_text.strip():
        st.error("Could not read text from the Job Description file.")
        return

    st.write("✅ Job Description loaded. Now evaluating resumes...")

    results = []
    for rf in resume_files:
        resume_text = read_uploaded_file(rf)
        if not resume_text.strip():
            st.warning(f"Could not read text from resume: {rf.name}")
            continue

        with st.spinner(f"Scoring {rf.name}..."):
            res = score_single_resume(jd_text, resume_text, rf.name)
            results.append(res)

    if not results:
        st.error("No valid resumes could be scored.")
        return

    # Sort by score (descending)
    results.sort(key=lambda x: x["score"], reverse=True)

    st.success("✅ Screening complete! Ranked candidates below:")

    # Show ranked results
    for i, r in enumerate(results, start=1):
        st.markdown(
            f"""
**Rank #{i}: {r['name']}**  
Score: **{r['score']:.1f} / 10**

{r['summary']}
---
"""
        )


# -----------------------------------------------------------
# 10. MAIN UI PER AGENT
# -----------------------------------------------------------

if agent_type == "Knowledge Base Agent":
    # Show existing chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if vectorizer is None:
        st.info(
            "📥 Please upload at least one TXT or PDF file in the sidebar (KB Agent mode) "
            "to start asking document-based questions.\n\n"
            "You can still ask general questions, though!"
        )

    user_input = st.chat_input(
        "Ask your question (general or about the uploaded documents)..."
    )

    if user_input:
        add_message("user", user_input)
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking with Gemini..."):
                answer = answer_kb_question(user_input)
                st.markdown(answer)
                add_message("assistant", answer)

elif agent_type == "Resume Screening":
    st.subheader("📄 Resume Screening Agent")
    st.write(
        "Upload a **Job Description** and one or more **Resumes** in the sidebar, "
        "then click the button below to rank candidates."
    )

    if st.button("🚀 Run Resume Screening"):
        run_resume_screening(jd_file, resume_files)
