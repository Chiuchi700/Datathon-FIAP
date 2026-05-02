"""
Testes unitarios para o servico RAG (src/api/rag_service.py).

Todas as dependencias externas (ChromaDB, Ollama) sao mockadas.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.api.rag_service import (
    _chunk_text,
    chat_about_model,
    index_documents,
    query_context,
)


# ---------------------------------------------------------------------------
# _chunk_text
# ---------------------------------------------------------------------------


class TestChunkText:
    def test_splits_on_double_newline(self):
        text = "Paragrafo 1\n\nParagrafo 2\n\nParagrafo 3"
        chunks = _chunk_text(text, max_chunk_size=500)
        assert len(chunks) >= 1
        assert "Paragrafo 1" in chunks[0]

    def test_respects_max_chunk_size(self):
        text = "A" * 100 + "\n\n" + "B" * 100 + "\n\n" + "C" * 100
        chunks = _chunk_text(text, max_chunk_size=150)
        for chunk in chunks:
            assert len(chunk) <= 200  # margem por causa de paragrafo unico

    def test_returns_empty_for_empty_text(self):
        assert _chunk_text("") == []

    def test_single_paragraph(self):
        chunks = _chunk_text("Apenas um paragrafo")
        assert chunks == ["Apenas um paragrafo"]


# ---------------------------------------------------------------------------
# index_documents
# ---------------------------------------------------------------------------


class TestIndexDocuments:
    @patch("src.api.rag_service.get_or_create_collection")
    @patch("src.api.rag_service.get_chroma_client")
    def test_skips_when_already_indexed(self, mock_client, mock_collection):
        mock_col = MagicMock()
        mock_col.count.return_value = 10
        mock_collection.return_value = mock_col

        result = index_documents()

        assert result == 10
        mock_col.add.assert_not_called()

    @patch("src.api.rag_service.get_or_create_collection")
    @patch("src.api.rag_service.get_chroma_client")
    def test_indexes_files_when_empty(self, mock_client, mock_collection, tmp_path):
        mock_col = MagicMock()
        mock_col.count.return_value = 0
        mock_collection.return_value = mock_col

        # Cria arquivos temporarios
        (tmp_path / "README.md").write_text("# Titulo\n\nConteudo do README")
        (tmp_path / "params.yaml").write_text("data:\n  ticker: NKE")

        with patch("src.api.rag_service.PROJECT_ROOT", tmp_path):
            result = index_documents()

        assert result > 0
        mock_col.add.assert_called_once()

    @patch("src.api.rag_service.get_chroma_client")
    def test_returns_zero_on_connection_error(self, mock_client):
        mock_client.side_effect = Exception("Connection refused")
        result = index_documents()
        assert result == 0


# ---------------------------------------------------------------------------
# query_context
# ---------------------------------------------------------------------------


class TestQueryContext:
    @patch("src.api.rag_service.get_or_create_collection")
    @patch("src.api.rag_service.get_chroma_client")
    def test_returns_list_of_strings(self, mock_client, mock_collection):
        mock_col = MagicMock()
        mock_col.query.return_value = {
            "documents": [["chunk 1", "chunk 2"]],
            "metadatas": [[{"source": "README.md"}, {"source": "params.yaml"}]],
        }
        mock_collection.return_value = mock_col

        result = query_context("Qual o ticker?")
        assert result == ["chunk 1", "chunk 2"]

    @patch("src.api.rag_service.get_chroma_client")
    def test_returns_empty_on_connection_error(self, mock_client):
        mock_client.side_effect = Exception("Connection refused")
        result = query_context("Qualquer pergunta")
        assert result == []

    @patch("src.api.rag_service.get_or_create_collection")
    @patch("src.api.rag_service.get_chroma_client")
    def test_returns_empty_when_no_results(self, mock_client, mock_collection):
        mock_col = MagicMock()
        mock_col.query.return_value = {"documents": [[]]}
        mock_collection.return_value = mock_col

        result = query_context("Pergunta sem match")
        assert result == []


# ---------------------------------------------------------------------------
# chat_about_model
# ---------------------------------------------------------------------------


class TestChatAboutModel:
    @patch("src.api.rag_service.generate_text")
    @patch("src.api.rag_service.query_context")
    def test_returns_llm_response_with_context(self, mock_query, mock_gen):
        mock_query.return_value = ["O modelo usa LSTM bidirecional."]
        mock_gen.return_value = "O modelo e LSTM."

        result = chat_about_model("Qual modelo?")
        assert result == "O modelo e LSTM."

    @patch("src.api.rag_service.generate_text")
    @patch("src.api.rag_service.query_context")
    def test_prompt_includes_context(self, mock_query, mock_gen):
        mock_query.return_value = ["chunk relevante"]
        mock_gen.return_value = "resposta"

        chat_about_model("Pergunta?")

        prompt = mock_gen.call_args[0][0]
        assert "chunk relevante" in prompt
        assert "Pergunta?" in prompt

    @patch("src.api.rag_service.generate_text")
    @patch("src.api.rag_service.query_context")
    def test_falls_back_without_context(self, mock_query, mock_gen):
        mock_query.return_value = []
        mock_gen.return_value = "resposta sem contexto"

        result = chat_about_model("Pergunta?")
        assert result == "resposta sem contexto"

        prompt = mock_gen.call_args[0][0]
        assert "Pergunta?" in prompt
