import streamlit as st
import requests
import json
import PyPDF2
from typing import List

# -----------------------------
# CONFIG
# -----------------------------
OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3.2:1b"

st.set_page_config(page_title="EMSI-Chatbot", page_icon="🎓")
st.title("🎓 EMSI-Chatbot (Ollama + Llama 3.2:1B)")

st.markdown(
    "Chatbot basé sur **Ollama** + **llama3.2:1b**, enrichi avec la plaquette officielle EMSI.\n\n"
    "⚠️ Le chatbot répond **uniquement** à partir du PDF chargé."
)

# -----------------------------
# RAG UTILS
# -----------------------------
def chunk_text(text: str, chunk_size=500, overlap=100) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def retrieve_relevant_chunks(question: str, chunks: List[str], top_k=3) -> List[str]:
    question_words = question.lower().split()
    scored = []

    for chunk in chunks:
        score = sum(1 for w in question_words if w in chunk.lower())
        if score > 0:
            scored.append((score, chunk))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [chunk for _, chunk in scored[:top_k]]


# -----------------------------
# SIDEBAR — LOAD PDF
# -----------------------------
st.sidebar.subheader("📄 Charger la plaquette EMSI")

uploaded_pdf = st.sidebar.file_uploader("Sélectionner le PDF EMSI", type=["pdf"])

if uploaded_pdf:
    reader = PyPDF2.PdfReader(uploaded_pdf)
    full_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    if full_text.strip():
        st.session_state["pdf_chunks"] = chunk_text(full_text)
        st.sidebar.success(f"PDF chargé ✔ ({len(st.session_state['pdf_chunks'])} segments)")
    else:
        st.sidebar.error("PDF vide ou illisible.")

# -----------------------------
# SIDEBAR — MODEL PARAMETERS
# -----------------------------
st.sidebar.header("⚙️ Paramètres du modèle")

model_name = st.sidebar.text_input("Modèle Ollama :", value=DEFAULT_MODEL)

temperature = st.sidebar.slider(
    "Température (créativité)", 0.0, 1.5, 0.2, 0.05
)

max_tokens = st.sidebar.slider(
    "Max tokens (num_predict)", 64, 1024, 256, 32
)

st.sidebar.caption("⚠️ Ollama doit être lancé avec : `ollama serve`")

# -----------------------------
# STREAMING FUNCTION
# -----------------------------
def stream_ollama(messages, model, options):
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": options,
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            stream=True,
            timeout=600,
        )
    except requests.exceptions.ConnectionError:
        yield "⚠️ Impossible de se connecter à Ollama."
        return

    if response.status_code != 200:
        yield f"⚠️ Erreur Ollama: {response.status_code}"
        return

    for line in response.iter_lines():
        if not line:
            continue

        try:
            data = json.loads(line.decode("utf-8"))
        except:
            continue

        if "message" in data and "content" in data["message"]:
            yield data["message"]["content"]

        if data.get("done"):
            break


# -----------------------------
# CHAT HISTORY INIT
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# -----------------------------
# DISPLAY CHAT HISTORY
# -----------------------------
for msg in st.session_state.chat_history:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])


# -----------------------------
# CHAT INPUT
# -----------------------------
user_input = st.chat_input("Pose ta question sur l’EMSI…")

if user_input:

    # USER MESSAGE
    st.session_state.chat_history.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # CHECK PDF
    if "pdf_chunks" not in st.session_state:
        with st.chat_message("assistant"):
            st.markdown("⚠️ Veuillez d’abord charger la plaquette EMSI.")
        st.stop()

    # 🔍 RETRIEVE RELEVANT CHUNKS
    relevant_chunks = retrieve_relevant_chunks(
        user_input,
        st.session_state["pdf_chunks"]
    )

    if not relevant_chunks:
        with st.chat_message("assistant"):
            st.markdown(
                "Je ne trouve pas cette information dans la plaquette EMSI."
            )
        st.stop()

    context = "\n\n".join(relevant_chunks)

    # 🧠 STRONG SYSTEM PROMPT
    system_prompt = (
        "Tu es EMSI-Chatbot.\n"
        "Tu dois répondre STRICTEMENT et UNIQUEMENT avec les informations ci-dessous.\n"
        "Si la réponse n’existe pas dans ces informations, dis exactement :\n"
        "'Je ne trouve pas cette information dans la plaquette EMSI.'\n\n"
        "===== CONTEXTE EMSI =====\n"
        f"{context}\n"
        "===== FIN CONTEXTE ====="
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    options = {
        "temperature": float(temperature),
        "num_predict": int(max_tokens),
    }

    # STREAM RESPONSE
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        for chunk in stream_ollama(messages, model_name, options):
            full_response += chunk
            placeholder.markdown(full_response)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": full_response}
    )
