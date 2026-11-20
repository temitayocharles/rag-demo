"""Deterministic, source-backed FAQ responses for the public no-LLM demo.

Only answers explicitly published in the bundled knowledge file are eligible.
No model is called and no unrelated vector-search chunks are promoted to answers.
"""

import re
from difflib import SequenceMatcher


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _faq_entries(document: str) -> dict[str, str]:
    matches = re.findall(r"(?m)^\*\*Q:\s*(.+?)\*\*\s*\nA:\s*(.+?)\s*$", document)
    return {_normalize(question): answer.strip() for question, answer in matches}


def _format_answer(answer: str) -> str:
    return f"{answer}\n\nSource: https://tayoca.com/ (public knowledge)"


def resolve_public_demo_answer(question: str, document: str) -> str | None:
    """Answer only high-confidence FAQ questions; decline any uncertain retrieval."""
    entries = _faq_entries(document)
    query = _normalize(question)
    if not query or not entries:
        return None

    # Broad "What does Tayoca do?" is answered by combining TWO published
    # FAQ entries rather than selecting an arbitrary 800-character chunk.
    if query in {
        "what does tayoca do",
        "what does tayoca actually do",
        "what do you do at tayoca",
        "tell me what tayoca does",
        "what is tayoca about",
    }:
        overview = entries.get("what is tayoca")
        services = entries.get("what services does tayoca provide")
        if overview and services:
            return _format_answer(f"Tayoca: {overview}\n\nServices: {services}")
        return None

    exact = entries.get(query)
    if exact:
        return _format_answer(exact)

    # Conservative typo/rephrasing support: never guess for one-word queries
    # or unrelated questions, and require a UNIQUE close match.
    if len(query.split()) < 4:
        return None
    candidates = sorted(
        ((SequenceMatcher(None, query, key).ratio(), key) for key in entries),
        reverse=True,
    )
    if (
        candidates
        and candidates[0][0] >= 0.89
        and (len(candidates) == 1 or candidates[0][0] - candidates[1][0] >= 0.08)
    ):
        return _format_answer(entries[candidates[0][1]])
    return None
