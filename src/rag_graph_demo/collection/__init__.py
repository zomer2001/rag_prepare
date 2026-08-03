"""Anduril OSINT collection pipeline (discovery → fetch → parse → normalize → clean).

This package deliberately performs NO indexing: it never imports or calls
LightRAG, embedding models, Qdrant, or the existing document registry.
Outputs land only under ``data/collection/`` until a future, explicit
promotion step is implemented and invoked by the owner.
"""
