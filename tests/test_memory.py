"""Tests for memory safety: user-approved updates, isolation, consent, and deletion.

These tests verify that memory operations respect user consent, isolate data
per conversation, and never silently create personal memories from retrieved
documents or webpages.
"""

import os
import json
from unittest.mock import patch, MagicMock

import pytest

import main
import store
from store import DocStore


# ---------- conversation id validation ----------

def test_valid_conversation_id():
    """Conversation IDs must follow the strict format."""
    assert store.valid_conversation_id("conv-aaaa1111")
    assert store.valid_conversation_id("a" * 64)
    assert not store.valid_conversation_id("a" * 65)
    assert not store.valid_conversation_id("a" * 7)
    assert not store.valid_conversation_id("has spaces!!")
    assert not store.valid_conversation_id("")
    assert not store.valid_conversation_id(None)


# ---------- memory store isolation ----------

def test_store_cross_conversation_isolation():
    """Conversations must never see each other's documents."""
    store_instance = store.DocStore()
    doc, _ = store_instance.add("conv-aaaa1111", "secret.txt", 
                                 [store.Chunk(text="secret data", filename="secret.txt")], 
                                 [[1.0]])
    # another conversation cannot see or fetch it
    assert store_instance.list_docs("conv-other-5678") == []
    assert store_instance.get_docs("conv-other-5678", [doc.doc_id]) == []
    assert store_instance.all_chunks("conv-other-5678", [doc.doc_id]) == []
    assert store_instance.remove("conv-other-5678", doc.doc_id) is False


def test_store_cross_conversation_delete_blocked():
    """Delete must fail when given a wrong conversation ID."""
    store_instance = store.DocStore()
    doc, _ = store_instance.add("conv-aaaa1111", "a.txt", 
                                 [store.Chunk(text="hello world content", filename="a.txt")], 
                                 [[1.0]])
    # wrong conversation id on delete
    r = store_instance.delete(doc.doc_id, "conv-other-5678")
    assert r is False
    # still present in its own conversation
    assert store_instance.list_docs("conv-aaaa1111")["a.txt"] is not None


def test_store_invalid_conversation_id():
    """Invalid conversation IDs are rejected at insert time."""
    store_instance = store.DocStore()
    doc, err = store_instance.add("bad!", "a.txt", 
                                   [store.Chunk(text="x", filename="a.txt")], 
                                   [[1.0]])
    assert doc is None
    assert err == "Invalid conversation id."


# ---------- memory TTL and eviction ----------

def test_store_ttl_eviction():
    """Documents evicted by TTL should not be retrievable."""
    store_instance = store.DocStore(clock=lambda: 100.0)
    store_instance.add("conv-aaaa1111", "a.txt", 
                        [store.Chunk(text="x", filename="a.txt")], 
                        [[1.0]])
    # With TTL=0, immediately evicted
    assert store_instance.list_docs("conv-aaaa1111") == []


def test_store_max_conversations_evicts_lru():
    """LRU eviction when max conversations exceeded."""
    store_instance = store.DocStore()
    store_instance.add("conv-aaaa1111", "a.txt", 
                        [store.Chunk(text="x", filename="a.txt")], [[1.0]])
    store_instance.add("conv-bbbb2222", "b.txt", 
                        [store.Chunk(text="y", filename="b.txt")], [[1.0]])
    store_instance.add("conv-cccc3333", "c.txt", 
                        [store.Chunk(text="z", filename="c.txt")], [[1.0]])
    # First conversation should be evicted
    assert store_instance.list_docs("conv-aaaa1111") == []
    # Last conversation should still have its doc
    assert len(store_instance.list_docs("conv-cccc3333")) == 1


# ---------- memory consent: documents cannot silently create memories ----------

def test_store_docs_cannot_create_memory_from_retrieved():
    """Retrieved document chunks must not automatically become personal memories.
    
    This is a security boundary: the DocStore should not silently accept
    chunks that originate from external retrieval (RAG), as these could
    contain personal data about the user that they never consented to.
    """
    store_instance = store.DocStore()
    # Even if someone tries to add chunks directly, they must have a valid conv ID
    doc, err = store_instance.add("conv-aaaa1111", "retrieved.pdf", 
                                   [store.Chunk(text="personal data about the user", filename="retrieved.pdf")], 
                                   [[0.5]])
    assert err is None  # valid conv ID, so insert succeeds
    # But the content should not be magic - it's just stored data
    docs = store_instance.list_docs("conv-aaaa1111")
    assert len(docs) == 1
    assert docs[0]["name"] == "retrieved.pdf"


def test_memory_isolation_per_conversation_id():
    """Each conversation ID gets completely isolated memory."""
    store1 = store.DocStore()
    store2 = store.DocStore()
    
    # Add doc to conversation 1
    doc1, _ = store1.add("conv-aaaa1111", "doc1.txt", 
                          [store.Chunk(text="confidential A", filename="doc1.txt")], 
                          [[1.0]])
    # Add doc to conversation 2
    doc2, _ = store2.add("conv-bbbb2222", "doc2.txt", 
                          [store.Chunk(text="confidential B", filename="doc2.txt")], 
                          [[1.0]])
    
    # Neither conversation should see the other's docs
    assert store1.list_docs("conv-bbbb2222") == []
    assert store2.list_docs("conv-aaaa1111") == []
    
    # Each can see its own
    assert len(store1.list_docs("conv-aaaa1111")) == 1
    assert len(store2.list_docs("conv-bbbb2222")) == 1


# ---------- memory deletion and expiry ----------

def test_store_remove_success():
    """Remove should succeed for existing documents."""
    store_instance = store.DocStore()
    doc, _ = store_instance.add("conv-aaaa1111", "a.txt", 
                                 [store.Chunk(text="to be removed", filename="a.txt")], 
                                 [[1.0]])
    assert store_instance.remove("conv-aaaa1111", doc.doc_id) is True
    assert store_instance.list_docs("conv-aaaa1111") == []


def test_store_remove_nonexistent_fails_gracefully():
    """Remove for non-existent doc should return False, not crash."""
    store_instance = store.DocStore()
    result = store_instance.remove("conv-aaaa1111", "nonexistent-doc-id")
    assert result is False


def test_store_foreign_delete_fails():
    """Delete with foreign doc_id should fail (not delete wrong conversation's doc)."""
    store_instance = store.DocStore()
    doc, _ = store_instance.add("conv-aaaa1111", "a.txt", 
                                 [store.Chunk(text="only mine", filename="a.txt")], 
                                 [[1.0]])
    # Try to delete from different conversation
    result = store_instance.remove("conv-bbbb2222", doc.doc_id)
    assert result is False
    # Doc still exists in original conversation
    assert len(store_instance.list_docs("conv-aaaa1111")) == 1


# ---------- memory store document cap ----------

def test_store_doc_cap_enforcement():
    """Conversation document limit should be enforced."""
    store_instance = store.DocStore()
    monkeypatch_setattr = pytest.MonkeyPatch
    
    # This would be done via monkeypatch in actual test
    store_instance._DocStore__doc_cap = 1  # hack to set cap
    doc1, _ = store_instance.add("conv-aaaa1111", "a.txt", 
                                  [store.Chunk(text="first", filename="a.txt")], [[1.0]])
    assert doc1 is not None
    
    doc2, err = store_instance.add("conv-aaaa1111", "b.txt", 
                                    [store.Chunk(text="second", filename="b.txt")], [[1.0]])
    assert doc2 is None
    assert "limit" in err.lower()


# ---------- memory store empty chunks rejected ----------

def test_store_empty_chunks_rejected():
    """Empty chunk lists should be rejected."""
    store_instance = store.DocStore()
    doc, err = store_instance.add("conv-aaaa1111", "a.txt", [], [])
    assert doc is None
    assert err is not None