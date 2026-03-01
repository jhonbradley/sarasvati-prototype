from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import shutil
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")


# ---------- PAGINA PRINCIPAL ----------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ---------- SUBIR ARCHIVOS ----------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    filepath = f"docs/{file.filename}"
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"mensaje": "Archivo subido correctamente"}


# ---------- INDEXAR DOCUMENTOS ----------
import os
import shutil

@app.get("/index")
def index_documents():

    # 🔥 BORRAR ÍNDICE ANTERIOR SI EXISTE
    if os.path.exists("storage"):
        shutil.rmtree("storage")

    documents = SimpleDirectoryReader("docs").load_data()

    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=OpenAIEmbedding()
    )

    index.storage_context.persist(persist_dir="storage")

    return {"mensaje": "Documentos RE-indexados correctamente"}


# ---------- CHAT (MEJORADO) ----------
@app.post("/chat", response_class=HTMLResponse)
async def chat_with_docs(question: str = Form(...)):

    # Cargar índice
    storage_context = StorageContext.from_defaults(persist_dir="storage")
    index = load_index_from_storage(storage_context)

    # Crear query engine MEJORADO
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        response_mode="tree_summarize"
    )

    # Prompt del sistema (MUY IMPORTANTE)
    system_prompt = """
Eres Sarasvati, un asistente de IA empresarial.
Tu trabajo es responder usando únicamente la información de los documentos internos.

Reglas:
- Responde de forma clara y completa.
- Si piden resumen, entrega un resumen real del documento.
- Explica con detalle.
- Nunca respondas con frases vacías como "puedo hacerlo".
- Si la información no existe en los documentos, dilo claramente.
"""

    # Consulta al motor RAG
    response = query_engine.query(system_prompt + question)

    # Mostrar respuesta en HTML
    return f"""
    <h2>Respuesta de Sarasvati</h2>
    <p><b>Pregunta:</b> {question}</p>
    <p><b>Respuesta:</b> {response}</p>
    <br><br>
    <a href="/">⬅ Volver</a>
    """