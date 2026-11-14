# Architecture

Document ingestion → deterministic chunking → embedding provider → vector storage → similarity retrieval → FastAPI API.

Provider interfaces: Embedder.embed(texts), VectorStore.replace_source(source_id,chunks), VectorStore.search(vector,limit). Local implementation uses hashed text vectors and volatile storage for demonstration only. Production requires persistent storage, authz, tenancy, observability, and ingestion scheduling.
