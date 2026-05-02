"""
MCP Server para o projeto Nike Forecast.

Expoe tools para previsao, explicacao, informacoes do modelo e chat RAG.
Todas as tools delegam para a FastAPI via HTTP.
"""

import json
import os

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://localhost:8000")
HTTP_TIMEOUT = float(os.getenv("MCP_HTTP_TIMEOUT", "60"))

server = Server("nike-forecast")


def _call_api(method: str, path: str, payload: dict | None = None) -> str:
    """Chama a FastAPI e retorna o JSON como string."""
    url = f"{FASTAPI_BASE_URL}{path}"
    with httpx.Client(timeout=HTTP_TIMEOUT) as client:
        if method == "GET":
            response = client.get(url)
        else:
            response = client.post(url, json=payload)
        response.raise_for_status()
        return json.dumps(response.json(), indent=2, ensure_ascii=False)


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="predict",
            description="Preve o proximo preco de fechamento da Nike dado o ultimo close",
            inputSchema={
                "type": "object",
                "properties": {
                    "close": {
                        "type": "number",
                        "description": "Ultimo preco de fechamento conhecido",
                    }
                },
                "required": ["close"],
            },
        ),
        Tool(
            name="model_info",
            description="Retorna informacoes do modelo LSTM registrado no MLflow",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="explain_prediction",
            description="Gera explicacao em linguagem natural da previsao do modelo",
            inputSchema={
                "type": "object",
                "properties": {
                    "close": {
                        "type": "number",
                        "description": "Ultimo preco de fechamento conhecido",
                    }
                },
                "required": ["close"],
            },
        ),
        Tool(
            name="ask_about_model",
            description="Pergunta sobre o projeto ou modelo usando RAG",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Pergunta sobre o modelo ou projeto",
                    }
                },
                "required": ["question"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "predict":
            result = _call_api("POST", "/predict", {"close": arguments["close"]})
        elif name == "model_info":
            result = _call_api("GET", "/model-info")
        elif name == "explain_prediction":
            result = _call_api("POST", "/explain", {"close": arguments["close"]})
        elif name == "ask_about_model":
            result = _call_api("POST", "/chat", {"question": arguments["question"]})
        else:
            result = json.dumps({"error": f"Tool desconhecida: {name}"})
    except httpx.HTTPError as exc:
        result = json.dumps({"error": f"Erro ao chamar API: {str(exc)}"})

    return [TextContent(type="text", text=result)]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
