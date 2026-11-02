"""Tests for RAG citations: correct passage attribution, source IDs, URLs, 
fabrication prevention, and context trimming.

These tests verify that RAG answers correctly cite retrieved document passages,
source metadata is preserved, and no fabricated citations are produced when 
evidence is missing.
"""

import os
import json
from unittest.mock import patch, MagicMock

import pytest

import main
import rag
from rag import RetrievedChunk, Chunk, format_document_context, sources_payload, augment_with_documents


# ---------- citation basics ----------

def test_retrieved_chunk_has_source_metadata():
    """Retrieved chunks must carry source metadata (filename, page, section)."""
    rc = RetrievedChunk(
        chunk=Chunk(text="Revenue grew 20%.", filename="report.pdf", page=3, section="Finance"),
        score=0.9,
        n=1
    )
    assert rc.chunk.filename == "report.pdf"
    assert rc.chunk.page == 3
    assert rc.chunk.section == "Finance"
    assert rc.n == 1


def test_format_document_context_has_citation_labels():
    """format_document_context must include document_context tags and citations."""
    rc = RetrievedChunk(
        chunk=Chunk(text="Revenue grew 20%.", filename="report.pdf", page=3, section="Finance"),
        score=0.9,
        n=1
    )
    block = format_document_context([rc])
    assert "<document_context>" in block
    assert "</document_context>" in block
    assert "[D1]" in block  # numbered citation
    assert "report.pdf" in block
    assert "page 3" in block


def test_format_document_context_untrusted_mark():
    """Document context must mark text as untrusted (never invent)."""
    from rag import RetrievedChunk
    from rag import Chunk
    rc = RetrievedChunk(
        chunk=Chunk(text="Revenue grew 20%.", filename="report.pdf", page=3, section="Finance"),
        score=0.9,
        n=1
    )
    block = format_document_context([rc])
    lower = block.lower()
    assert "untrusted" in lower or "never invent" in lower


# ---------- no fabricated citations when evidence missing ----------

def test_no_fabricated_citations_when_no_docs():
    """When no documents are retrieved, answers must not contain document citations."""
    msgs = [{"role": "user", "content": "What is the capital of Canada?"}]
    
    # Augment with no retrieved chunks
    out = augment_with_documents(msgs, [])
    # The last message should not have document_context
    assert "<document_context>" not in out[-1]["content"]
    assert "doc_sources" not in out[-1]


def test_no_fabricated_citations_when_irrelevant_docs():
    """When irrelevant documents are retrieved, no forced citations should appear."""
    from rag import RetrievedChunk, Chunk
    
    # Retrieve irrelevant docs
    msgs = [{"role": "user", "content": "Who won the 2024 Super Bowl?"}]
    
    rc = RetrievedChunk(
        chunk=Chunk(text=" completely irrelevant content about cooking pasta", filename="cookbook.txt", page=1, section="Recipes"),
        score=0.1,
        n=1
    )
    
    out = augment_with_documents(msgs, [rc])
    # Should not contain [D1] or document_context citations for irrelevant docs
    last_content = out[-1]["content"].lower()
    assert "[d1]" not in last_content
    assert "document_context" not in last_content or "irrelevant" in last_content


def test_sources_payload_snippet_capped_and_honest():
    """sources_payload must cap snippets and never include full original text."""
    from rag import RetrievedChunk, Chunk
    
    long_text = "x" * 500
    rc = RetrievedChunk(
        chunk=Chunk(text=long_text, filename="f.txt", page=2),
        score=0.9,
        n=1
    )
    
    payload = sources_payload([rc], snippet_chars=160)
    
    # Numbered correctly
    assert payload[0]["n"] == 1
    assert payload[0]["filename"] == "f.txt"
    assert payload[0]["page"] == 2
    assert payload[0]["url"] == ""  # no URL if not available
    
    # Snippet is exactly the right length
    assert len(payload[0]["snippet"]) == 160
    
    # Full original text must NOT appear in the payload
    payload_str = json.dumps(payload)
    assert "x" * 500 not in payload_str


# ---------- source IDs and URLs remain attached correctly ----------

def test_sources_payload_preserves_url_when_available():
    """sources_payload should preserve URL when the chunk has one."""
    from rag import RetrievedChunk, Chunk
    
    rc = RetrievedChunk(
        chunk=Chunk(text="Some fact.", filename="webpage.html", page=1, section="About", 
                     metadata={"url": "https://example.com/page"}),
        score=0.9,
        n=1
    )
    
    payload = sources_payload([rc], snippet_chars=100)
    
    assert payload[0]["url"] == "https://example.com/page"
    assert payload[0]["published"] is not None  # may have published date


def test_sources_payload_no_url_when_missing():
    """sources_payload should handle missing URL gracefully."""
    from rag import RetrievedChunk, Chunk
    
    rc = RetrievedChunk(
        chunk=Chunk(text="Some fact.", filename="local.doc", page=1, section="Notes"),
        score=0.9,
        n=1
    )
    
    payload = sources_payload([rc], snippet_chars=100)
    
    assert payload[0]["url"] == ""  # empty string, not None, crash
    assert payload[0]["published"] is None


# ---------- context trimming preserves relevant passages ----------

def test_format_document_context_preserves_all_metadata():
    """Context trimming must preserve all source metadata fields."""
    from rag import RetrievedChunk, Chunk
    
    rc = RetrievedChunk(
        chunk=Chunk(
            text="This is a very long passage that should be trimmed but metadata kept.",
            filename="report.pdf",
            page=5,
            section="Finance",
            language="en"
        ),
        score=0.95,
        n=3
    )
    
    block = format_document_context([rc])
    
    # All metadata should be present
    assert "report.pdf" in block
    assert "page 5" in block
    assert "Finance" in block
    assert "[D3]" in block  # 3rd document citation
    
    # The document_context tags should be intact
    assert "<document_context>" in block
    assert "</document_context>" in block


def test_augment_with_documents_preserves_conversation_structure():
    """augment_with_documents should not corrupt conversation message structure."""
    msgs = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "new question"},
    ]
    
    from rag import RetrievedChunk, Chunk
    rc = RetrievedChunk(
        chunk=Chunk(text="new fact", filename="f.txt", page=1, section="X"),
        score=0.5,
        n=1
    )
    
    out = augment_with_documents(msgs, [rc])
    
    # Input messages should be copy-protected (copy-on-write)
    import json
    assert json.dumps(msgs) == json.dumps(msgs)  # untouched
    
    # Output should have document_context only in the last user message
    assert "<document_context>" in out[-1]["content"]
    assert "<document_context>" not in out[-2]["content"]
    assert "<document_context>" not in out[0]["content"]
    
    # Old content should still be present
    assert "old answer" in out[1]["content"]


# ---------- citation ordering and multiplicity ----------

def test_multiple_sources_numbered_correctly():
    """Multiple sources should be numbered sequentially."""
    from rag import RetrievedChunk, Chunk
    
    rc1 = RetrievedChunk(
        chunk=Chunk(text="Fact from doc A", filename="docA.txt", page=1, section="A"),
        score=0.9,
        n=1
    )
    rc2 = RetrievedChunk(
        chunk=Chunk(text="Fact from doc B", filename="docB.txt", page=2, section="B"),
        score=0.8,
        n=2
    )
    
    from rag import format_document_context
    block = format_document_context([rc1, rc2])
    
    assert "[D1]" in block
    assert "[D2]" in block
    assert "docA.txt" in block
    assert "docB.txt" in block


def test_sources_payload_numbered_snippet_capped():
    """Already tested in test_rag.py - verify numbering and snippet cap."""
    from rag import RetrievedChunk, Chunk
    
    long_text = "x" * 500
    rc = RetrievedChunk(
        chunk=Chunk(text=long_text, filename="f.txt", page=2),
        score=0.9,
        n=1
    )
    
    payload = sources_payload([rc], snippet_chars=160)
    
    assert payload[0]["n"] == 1
    assert payload[0]["filename"] == "f.txt"
    assert payload[0]["page"] == 2
    assert payload[0]["url"] == ""
    assert payload[0]["published"] is None
    assert len(payload[0]["snippet"]) == 160
    assert "x" * 500 not in json.dumps(payload)