"""
Servico de integracao com LLM via Ollama.

Expoe funcoes para gerar texto e explicar previsoes do modelo LSTM.
"""

import os

import httpx

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "30"))

_FALLBACK_MESSAGE = "LLM service unavailable"


def generate_text(prompt: str) -> str:
    """Envia prompt ao Ollama e retorna o texto gerado."""
    try:
        with httpx.Client(timeout=OLLAMA_TIMEOUT) as client:
            response = client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            )
            response.raise_for_status()
            return response.json()["response"]
    except (httpx.HTTPError, KeyError):
        return _FALLBACK_MESSAGE


def explain_prediction(
    close: float,
    predicted_price: float,
    predicted_return: float,
) -> str:
    """Gera explicacao em linguagem natural para uma previsao do modelo."""
    direction = "alta" if predicted_return > 0 else "queda"
    prompt = (
        "Voce e um analista financeiro. O modelo LSTM previu que a acao da Nike (NKE), "
        f"com ultimo fechamento de ${close:.2f}, tera um retorno de "
        f"{predicted_return:.4%} no proximo dia, resultando em um preco previsto de "
        f"${predicted_price:.2f} ({direction}). "
        "Explique essa previsao de forma simples para um investidor, em 2-3 frases em portugues."
    )
    return generate_text(prompt)
