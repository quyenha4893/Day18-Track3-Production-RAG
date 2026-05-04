from src.m1_chunking import load_documents
docs = load_documents()
print(f'Loaded {len(docs)} documents')
for d in docs:
    src = d["metadata"]["source"]
    typ = d["metadata"].get("type", "?")
    sz = len(d["text"])
    print(f'  - {src}: {sz} chars, type={typ}')
