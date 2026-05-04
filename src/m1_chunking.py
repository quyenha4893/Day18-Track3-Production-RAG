"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load all markdown/text/PDF files from data/. (Đã implement sẵn)"""
    docs = []
    
    # Load markdown files
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp), "type": "markdown"}})
    
    # Load PDF files
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        try:
            import pdfplumber
            with pdfplumber.open(fp) as pdf:
                text = ""
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text() or ""
                    # Try to extract tables as well
                    tables = page.extract_tables() or []
                    for table in tables:
                        for row in table:
                            page_text += " | ".join([str(cell) if cell else "" for cell in row]) + "\n"
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                if text.strip():
                    docs.append({"text": text, "metadata": {"source": os.path.basename(fp), "type": "pdf", "pages": len(pdf.pages)}})
        except ImportError:
            # pdfplumber not available — try PyPDF2 as fallback
            try:
                from PyPDF2 import PdfReader
                with open(fp, "rb") as f:
                    reader = PdfReader(f)
                    text = ""
                    for page_num, page in enumerate(reader.pages):
                        text += f"\n--- Page {page_num + 1} ---\n{page.extract_text() or ''}"
                    if text.strip():
                        docs.append({"text": text, "metadata": {"source": os.path.basename(fp), "type": "pdf", "pages": len(reader.pages)}})
            except ImportError:
                    # As a last resort, try a dependency-free binary-text extraction.
                    try:
                        with open(fp, "rb") as f:
                            data = f.read()
                        # Extract long runs of printable ASCII characters as a heuristic
                        import string
                        printable = set(bytes(string.printable, "ascii"))
                        sequences = []
                        cur = bytearray()
                        for b in data:
                            if b in printable:
                                cur.append(b)
                            else:
                                if len(cur) > 100:
                                    sequences.append(cur.decode("ascii", errors="ignore"))
                                cur = bytearray()
                        if len(cur) > 100:
                            sequences.append(cur.decode("ascii", errors="ignore"))
                        text = "\n\n".join(sequences)
                        if text.strip():
                            docs.append({"text": text, "metadata": {"source": os.path.basename(fp), "type": "pdf", "pages": None, "note": "binary-extract"}})
                        else:
                            print(f"⚠️  Cannot read {os.path.basename(fp)} — pdfplumber and PyPDF2 not installed and binary extraction failed")
                    except Exception:
                        print(f"⚠️  Cannot read {os.path.basename(fp)} — pdfplumber and PyPDF2 not installed and binary extraction failed")
        except Exception as e:
            print(f"⚠️  Error reading {os.path.basename(fp)}: {e}")
    
    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.

    Args:
        text: Input text.
        threshold: Cosine similarity threshold. Dưới threshold → tách chunk mới.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects grouped by semantic similarity.
    """
    metadata = metadata or {}
    # Simple, dependency-free semantic grouping using lexical overlap heuristic.
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text) if s.strip()]
    if not sentences:
        return []
    def tok_set(s):
        return {w.lower() for w in re.findall(r"\w+", s)}

    chunks: list[Chunk] = []
    current = [sentences[0]]
    prev_set = tok_set(sentences[0])
    for s in sentences[1:]:
        sset = tok_set(s)
        # similarity = intersection / min(size) to be stricter
        denom = min(max(len(prev_set), 1), max(len(sset), 1))
        sim = len(prev_set & sset) / denom if denom else 0.0
        if sim < threshold:
            chunks.append(Chunk(text=" ".join(current), metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"}))
            current = [s]
        else:
            current.append(s)
        prev_set = sset
    if current:
        chunks.append(Chunk(text=" ".join(current), metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"}))
    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Args:
        text: Input text.
        parent_size: Chars per parent chunk.
        child_size: Chars per child chunk.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    parents: list[Chunk] = []
    children: list[Chunk] = []
    current = []
    cur_len = 0
    p_index = 0
    for para in paragraphs:
        if cur_len + len(para) > parent_size and current:
            parent_text = "\n\n".join(current).strip()
            pid = f"parent_{p_index}"
            parents.append(Chunk(text=parent_text, metadata={**metadata, "chunk_type": "parent", "parent_id": pid}))
            p_index += 1
            current = [para]
            cur_len = len(para)
        else:
            current.append(para)
            cur_len += len(para)
    if current:
        parent_text = "\n\n".join(current).strip()
        pid = f"parent_{p_index}"
        parents.append(Chunk(text=parent_text, metadata={**metadata, "chunk_type": "parent", "parent_id": pid}))

    # Create children by slicing parents into child_size windows (non-overlapping)
    for p in parents:
        pid = p.metadata.get("parent_id")
        text_len = len(p.text)
        if text_len <= child_size:
            children.append(Chunk(text=p.text, metadata={**metadata, "chunk_type": "child"}, parent_id=pid))
        else:
            start = 0
            while start < text_len:
                end = min(start + child_size, text_len)
                chunk_text = p.text[start:end].strip()
                if chunk_text:
                    children.append(Chunk(text=chunk_text, metadata={**metadata, "chunk_type": "child"}, parent_id=pid))
                start = end

    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.

    Args:
        text: Markdown text.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects, mỗi chunk = 1 section (header + content).
    """
    metadata = metadata or {}
    # Split by headers (keep header lines)
    pattern = re.compile(r'(^#{1,6}\s+.+$)', flags=re.MULTILINE)
    parts = pattern.split(text)
    chunks: list[Chunk] = []
    current_header = ""
    current_content = ""
    if parts:
        # parts alternates between content and header when split starts with content
        i = 0
        while i < len(parts):
            part = parts[i]
            if pattern.match(part):
                # header
                if current_header or current_content.strip():
                    chunks.append(Chunk(text=(current_header + "\n" + current_content).strip(), metadata={**metadata, "section": current_header.strip(), "strategy": "structure"}))
                current_header = part.strip()
                current_content = ""
                i += 1
                # next part (if exists) is content
                if i < len(parts):
                    current_content = parts[i]
                    i += 1
            else:
                # no header (leading content)
                if not current_header:
                    current_content += part
                else:
                    current_content += part
                i += 1
        # append last
        if current_header or current_content.strip():
            chunks.append(Chunk(text=(current_header + "\n" + current_content).strip(), metadata={**metadata, "section": current_header.strip() if current_header else "", "strategy": "structure"}))
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.

    Returns:
        {"basic": {...}, "semantic": {...}, "hierarchical": {...}, "structure": {...}}
    """
    results = {}
    for doc in documents:
        text = doc.get("text", "")
        basic = chunk_basic(text)
        semantic = chunk_semantic(text)
        parents, children = chunk_hierarchical(text)
        structure = chunk_structure_aware(text)

        def stats(chunks):
            if not chunks:
                return {"num": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
            lengths = [len(c.text) for c in chunks]
            return {"num": len(chunks), "avg_len": sum(lengths) / len(lengths), "min_len": min(lengths), "max_len": max(lengths)}

        results = {
            "basic": stats(basic),
            "semantic": stats(semantic),
            "hierarchical": {"parents": stats(parents), "children": stats(children)},
            "structure": stats(structure),
        }
        # Only run for first doc (intended usage)
        break
    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
