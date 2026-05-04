"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.

    Args:
        text: Raw chunk text.

    Returns:
        Summary string (2-3 câu).
    """
    # Simple extractive summary: take first 2 sentences
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    if not sentences:
        return text[:100]
    summary = ". ".join(sentences[:min(2, len(sentences))]) + "."
    return summary


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).

    Args:
        text: Raw chunk text.
        n_questions: Số câu hỏi cần generate.

    Returns:
        List of question strings.
    """
    # Simple pattern-based question generation
    import re
    questions = []
    
    # Extract key phrases (usually capitalized or after "là", "được", "có")
    text_lower = text.lower()
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    
    # Pattern 1: "X là Y" → "Y là gì?"
    matches = re.findall(r"(\w+[\w\s]*?)\s+là\s+([^,\.]+)", text)
    for subject, definition in matches[:1]:
        questions.append(f"Cái gì là {definition.strip()}?")
    
    # Pattern 2: Extract numbers with context → "Bao nhiêu?"
    numbers = re.findall(r"(\d+)\s+(\w+)", text)
    for num, unit in numbers[:1]:
        questions.append(f"Con số này là bao nhiêu? ({num} {unit})")
    
    # Pattern 3: First sentence as implicit question
    if sentences:
        questions.append(f"Điều gì được nói trong: {sentences[0][:60]}?")
    
    # Pad to n_questions if needed
    while len(questions) < n_questions:
        questions.append(f"Nội dung bổ sung từ {text[:40]}?")
    
    return questions[:n_questions]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).

    Args:
        text: Raw chunk text.
        document_title: Tên document gốc.

    Returns:
        Text với context prepended.
    """
    # Simple rule-based context: extract first heading or infer from content
    if not text.strip():
        return text
    
    lines = text.split("\n")
    first_heading = ""
    for line in lines:
        if line.startswith("#"):
            first_heading = line.replace("#", "").strip()
            break
    
    context = ""
    if document_title:
        context = f"Từ tài liệu: {document_title}. "
    if first_heading:
        context += f"Chủ đề: {first_heading}. "
    else:
        # Infer topic from first 20 words
        words = text.split()[:20]
        context += f"Nội dung về: {' '.join(words[:5])}... "
    
    return f"{context}\n\n{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.

    Args:
        text: Raw chunk text.

    Returns:
        Dict with extracted metadata fields.
    """
    # Simple keyword-based extraction (no LLM)
    text_lower = text.lower()
    
    # Categorization by keywords
    category = "other"
    if any(w in text_lower for w in ["nghỉ", "phép", "thử việc", "khen thưởng", "nhân sự", "thâm niên"]):
        category = "hr"
    elif any(w in text_lower for w in ["mật khẩu", "vpn", "bảo mật", "security", "wireguard"]):
        category = "security"
    elif any(w in text_lower for w in ["lương", "bảo hiểm", "thưởng", "finance"]):
        category = "finance"
    elif any(w in text_lower for w in ["công nghệ", "it", "server", "database"]):
        category = "it"
    elif any(w in text_lower for w in ["policy", "chính sách", "quy định"]):
        category = "policy"
    
    # Topic extraction: first non-empty line or heading
    import re
    topic = "General"
    headings = re.findall(r"^#+\s+(.+)$", text, flags=re.MULTILINE)
    if headings:
        topic = headings[0]
    else:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if lines:
            topic = lines[0][:50]
    
    # Language detection (simple)
    lang = "vi" if any(w in text_lower for w in ["không", "được", "các", "ngày", "người"]) else "en"
    
    # Extract entities (numbers, time periods)
    numbers = re.findall(r"\d+", text)
    entities = list(set(numbers[:5])) if numbers else []
    
    return {
        "topic": topic,
        "category": category,
        "language": lang,
        "entities": entities,
        "length": len(text),
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks.

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: List of methods to apply. Default: ["contextual", "hyqa", "metadata"]
                 Options: "summary", "hyqa", "contextual", "metadata", "full"

    Returns:
        List of EnrichedChunk objects.
    """
    if methods is None:
        methods = ["contextual", "hyqa", "metadata"]

    enriched = []

    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        metadata = chunk.get("metadata", {})
        
        # Apply requested enrichment methods
        summary = summarize_chunk(text) if "summary" in methods or "full" in methods else ""
        questions = generate_hypothesis_questions(text) if "hyqa" in methods or "full" in methods else []
        enriched_text = contextual_prepend(text, metadata.get("source", "")) if "contextual" in methods or "full" in methods else text
        auto_meta = extract_metadata(text) if "metadata" in methods or "full" in methods else {}
        
        # Create EnrichedChunk
        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**metadata, **auto_meta},
            method="+".join(methods),
        ))

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
