from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import requests
import PyPDF2
from typing import List

# -----------------------------
# CONFIG
# -----------------------------
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.2:1b"

# -----------------------------
# APP INIT
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# GLOBAL MEMORY (simple RAG)
# -----------------------------
pdf_chunks: List[str] = []


# -----------------------------
# UTILS
# -----------------------------
def chunk_text(text: str, chunk_size=500, overlap=100) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def retrieve_relevant_chunks(question: str, chunks: List[str], top_k=3) -> List[str]:
    """Simple keyword-based retrieval."""
    question_words = question.lower().split()
    scored = []

    for chunk in chunks:
        score = sum(1 for w in question_words if w in chunk.lower())
        if score > 0:
            scored.append((score, chunk))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [chunk for _, chunk in scored[:top_k]]


# -----------------------------
# UPLOAD PDF
# -----------------------------
@app.post("/upload_pdf")
async def upload_pdf(pdf: UploadFile = File(...)):
    global pdf_chunks

    reader = PyPDF2.PdfReader(pdf.file)
    full_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    if not full_text.strip():
        return {"message": "PDF vide ou illisible."}

    pdf_chunks = chunk_text(full_text)

    return {
        "message": "PDF EMSI traité avec succès.",
        "chunks": len(pdf_chunks),
    }


# -----------------------------
# CHAT (RAG)
# -----------------------------
@app.post("/chat")
async def chat(data: dict):
    global pdf_chunks

    user_message = data.get("message")

    if not user_message:
        return {"response": "Message vide."}

    if not pdf_chunks:
        return {
            "response": (
                "⚠️ Aucun PDF EMSI n’a été chargé. "
                "Veuillez uploader la plaquette officielle."
            )
        }

    # 🔍 Retrieve relevant PDF chunks
    relevant_chunks = retrieve_relevant_chunks(user_message, pdf_chunks)

    if not relevant_chunks:
        return {
            "response": (
                "Je ne trouve pas cette information dans la plaquette EMSI."
            )
        }

    context = "\n\n".join(relevant_chunks)

    # 🧠 STRONG SYSTEM PROMPT
    system_prompt = (
        "Tu es EMSI-Chatbot.\n"
        "Tu dois répondre STRICTEMENT et UNIQUEMENT à partir des informations ci-dessous.\n"
        "Si la réponse n’existe pas dans ces informations, dis exactement :\n"
        "'Je ne trouve pas cette information dans la plaquette EMSI.'\n\n"
        "===== CONTEXTE EMSI =====\n"
        f"{context}\n"
        "===== FIN CONTEXTE ====="
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "options": {
            "temperature": 0.2,     # 🔴 VERY IMPORTANT
            "num_predict": 256,
        },
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        data = response.json()
        answer = data["message"]["content"]

        return {"response": answer}

    except Exception as e:
        return {"response": f"Erreur serveur: {str(e)}"}


# -----------------------------
# CLEAR MEMORY
# -----------------------------
@app.post("/clear_chat")
async def clear_chat():
    global pdf_chunks
    pdf_chunks = []
    return {"message": "Mémoire PDF réinitialisée."}
