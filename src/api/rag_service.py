"""
Servico de RAG (Retrieval-Augmented Generation) usando ChromaDB e Ollama.

Indexa documentos do projeto e responde perguntas usando contexto recuperado.
"""

import os
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from src.api.llm_service import generate_text

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8100"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
COLLECTION_NAME = "project_docs"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_chroma_client() -> chromadb.HttpClient:
    """Retorna cliente HTTP do ChromaDB."""
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


def _get_embedding_function() -> OllamaEmbeddingFunction:
    return OllamaEmbeddingFunction(
        url=f"{OLLAMA_BASE_URL}/api/embeddings",
        model_name=EMBEDDING_MODEL,
    )


def get_or_create_collection(client: chromadb.HttpClient):
    """Retorna a collection do projeto, criando se nao existir."""
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_get_embedding_function(),
    )


def _chunk_text(text: str, max_chunk_size: int = 500) -> list[str]:
    """Divide texto em chunks por paragrafos."""
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) > max_chunk_size and current:
            chunks.append(current.strip())
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current.strip():
        chunks.append(current.strip())
    return chunks


def index_documents() -> int:
    """Indexa README.md e params.yaml no ChromaDB. Idempotente."""
    try:
        client = get_chroma_client()
        collection = get_or_create_collection(client)

        if collection.count() > 0:
            return collection.count()

        files_to_index = ["README.md", "params.yaml"]
        documents = []
        metadatas = []
        ids = []

        for filename in files_to_index:
            filepath = PROJECT_ROOT / filename
            if not filepath.exists():
                continue
            text = filepath.read_text(encoding="utf-8")
            chunks = _chunk_text(text)
            for i, chunk in enumerate(chunks):
                documents.append(chunk)
                metadatas.append({"source": filename, "chunk_index": i})
                ids.append(f"{filename}_{i}")

        if documents:
            collection.add(documents=documents, metadatas=metadatas, ids=ids)

        return len(documents)
    except Exception:
        return 0


def query_context(question: str, n_results: int = 3) -> list[str]:
    """Busca contexto relevante no ChromaDB."""
    try:
        client = get_chroma_client()
        collection = get_or_create_collection(client)
        results = collection.query(query_texts=[question], n_results=n_results)
        return results["documents"][0] if results["documents"] else []
    except Exception:
        return []


def chat_about_model(question: str) -> str:
    """Responde pergunta usando RAG: busca contexto + gera resposta via LLM."""
    context_chunks = query_context(question)

    if context_chunks:
        context = "\n\n".join(context_chunks)
        prompt = (
            "Use o seguinte contexto sobre o projeto de previsao de acoes da Nike "
            "para responder a pergunta.\n\n"
            f"Contexto:\n{context}\n\n"
            f"Pergunta: {question}\n\n"
            "Responda de forma concisa em portugues."
        )
    else:
        prompt = (
            "Voce e um assistente especializado em um projeto de previsao de acoes "
            "da Nike usando LSTM e MLflow. "
            f"Responda a seguinte pergunta de forma concisa em portugues: {question}"
        )

    return generate_text(prompt)
