from typing import Dict, Any, List, Optional


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def _format_options(options: Dict[str, str]) -> str:
    """Format option dict as lettered lines: 'A. ...\nB. ...'"""
    return '\n'.join(f"{k}. {v}" for k, v in sorted(options.items()))


# -------------------------------------------------------
# Embedding messages  (used to build FAISS indices)
# -------------------------------------------------------

def build_mcq_embedding_message(
    task: Dict[str, Any],
    include_gold: bool = True,
) -> Dict[str, Any]:
    """
    Compact embedding message for medical MCQ tasks (medqa / medmcqa / etc.).
    Includes the gold answer letter when include_gold=True so the embedding
    captures answer semantics.
    """
    opts_str = _format_options(task['options'])
    parts = [
        f"Medical QA: {task['question']}",
        opts_str,
    ]
    if include_gold:
        parts.append(f"Answer: {task['gold_answer']}")
    return {"role": "user", "content": "\n".join(parts)}


def build_pubmedqa_embedding_message(
    task: Dict[str, Any],
    include_gold: bool = True,
) -> Dict[str, Any]:
    """
    Embedding message for PubMedQA yes/no/maybe questions.

    include_gold=True  → question + long_answer (abstract body). The answer
                         label (yes/no/maybe) is NEVER included to avoid leakage.
    include_gold=False → question only.
    """
    parts = [f"Biomedical QA: {task['question']}"]
    if include_gold and task.get('long_answer'):
        parts.append(task['long_answer'])
    return {"role": "user", "content": "\n".join(parts)}


def build_pubmed_corpus_embedding_message(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Embedding message for a PubMed retrieval corpus entry (pubmed_corpus format).

    Uses the question field when available (populated from PubMedQA train set)
    so the index lives in the same semantic space as test queries.
    Falls back to title + abstract when no question is present.
    """
    if task.get('question'):
        return {"role": "user", "content": f"Biomedical QA: {task['question']}"}
    content = task.get('contents') or f"{task.get('title', '')} {task.get('content', '')}".strip()
    return {"role": "user", "content": content}


def build_pubmed_corpus_content_message(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Content embedding message for MLP injection: encodes passage title + abstract.
    Used to build the content embedding index (separate from the question
    retrieval index) for pubmed_corpus format entries.
    """
    content = task.get('contents') or f"{task.get('title', '')} {task.get('content', '')}".strip()
    if not content:
        content = task.get('question', '.')
    return {"role": "user", "content": content}


def build_combined_pool_embedding_message(
    task: Dict[str, Any],
    include_content: bool = False,
) -> Dict[str, Any]:
    """
    Embedding message for the combined pool (pqa_labeled + pqa_unlabeled).

    include_content=False (default) → question only (for query-index alignment).
    include_content=True            → question + corpus body (title + content).
                                      Answer label is NEVER added.

    Fallback when question is absent: title → content → ".".
    """
    question = task.get("question", "").strip()
    title    = task.get("title",   "").strip()
    content  = task.get("content", "").strip()
    body     = f"{title}\n{content}".strip() if title else content

    if question:
        text = f"Biomedical QA: {question}"
        if include_content and body:
            text = f"{text}\n{body}"
    else:
        text = body or "."

    return {"role": "user", "content": text}


# -------------------------------------------------------
# In-context RAG messages  (retrieved items + test question)
# -------------------------------------------------------

def build_mcq_vqa_message(
    task: Dict[str, Any],
    include_gold: bool = False,
) -> Dict[str, Any]:
    """
    Single-question VQA message for medical MCQ tasks (no retrieved context).

    include_gold=True appends the correct answer letter (used for teacher
    prompts during MLP training).
    """
    options_str = _format_options(task['options'])
    lines = [
        "You are an expert medical question-answering assistant.",
        "Answer the following multiple-choice question by selecting the single correct option.",
        "",
        f"Question: {task['question']}",
        "",
        "Options:",
        options_str,
        "",
        "Reply with ONLY the option letter (A, B, C, or D). Do not explain.",
    ]
    if include_gold:
        lines.append(f"Answer: {task['gold_answer']}")
    return {"role": "user", "content": "\n".join(lines)}


def build_mcq_incontext_message(
    test_task: Dict[str, Any],
    retrieved_tasks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    In-context RAG prompt for medical MCQ tasks: k solved examples + test question.
    """
    lines = [
        "You are an expert medical question-answering assistant.",
        "Use the following solved examples to answer the final question.",
        "Reply with ONLY the option letter (A, B, C, or D). Do not explain.",
        "",
        "--- Solved Examples ---",
    ]

    for i, ex in enumerate(retrieved_tasks, start=1):
        opts_str = _format_options(ex['options'])
        lines += [
            f"[Example {i}]",
            f"Question: {ex['question']}",
            "Options:",
            opts_str,
            f"Answer: {ex['gold_answer']}",
            "",
        ]

    lines += [
        "--- Your Question ---",
        f"Question: {test_task['question']}",
        "Options:",
        _format_options(test_task['options']),
        "",
        "Answer:",
    ]

    return {"role": "user", "content": "\n".join(lines)}


def build_pubmedqa_incontext_message(
    test_task: Dict[str, Any],
    retrieved_tasks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    In-context RAG prompt for PubMedQA using retrieved labeled train examples.
    Shows each example as Question + Answer (yes/no/maybe).
    """
    lines = [
        "You are an expert biomedical researcher.",
        "Use the following solved examples to answer the final question.",
        "The answer must be one of: yes / no / maybe.",
        "Reply with ONLY the option letter (A, B, or C). Do not explain.",
        "",
        "--- Solved Examples ---",
    ]

    for i, ex in enumerate(retrieved_tasks, start=1):
        lines += [
            f"[Example {i}]",
            f"Question: {ex['question']}",
            f"Answer: {ex['gold_answer']}",
            "",
        ]

    lines += [
        "--- Your Question ---",
        f"Question: {test_task['question']}",
        "Options:",
        _format_options(test_task['options']),
        "",
        "Answer:",
    ]

    return {"role": "user", "content": "\n".join(lines)}


def build_corpus_incontext_message(
    test_task: Dict[str, Any],
    retrieved_passages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    In-context RAG prompt using retrieved PubMed corpus passages
    (pubmed_corpus format: title + content fields).
    """
    lines = [
        "You are an expert biomedical researcher.",
        "Use the following reference passages to answer the question.",
        "Reply with ONLY the option letter (A, B, or C). Do not explain.",
        "",
        "--- Reference Passages ---",
    ]

    for i, p in enumerate(retrieved_passages, start=1):
        title = p.get('title', '').strip()
        content = p.get('content', '').strip()
        lines += [
            f"[Passage {i}]{(' ' + title) if title else ''}",
            content,
            "",
        ]

    lines += [
        "--- Your Question ---",
        f"Question: {test_task['question']}",
        "Options:",
        _format_options(test_task['options']),
        "",
        "Answer:",
    ]

    return {"role": "user", "content": "\n".join(lines)}


def build_combined_incontext_message(
    test_task: Dict[str, Any],
    retrieved_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    In-context RAG prompt for the combined pool (pqa_labeled + pqa_unlabeled).

    ALL items — labeled or not — are shown identically as reference passages
    using their corpus text (long_answer / abstract). The final_decision answer
    is never shown regardless of source.

      Every item → [Reference N] Question / Abstract
    """
    lines = [
        "You are an expert biomedical researcher.",
        "Use the following reference passages to answer the final question.",
        "The answer must be one of: yes / no / maybe.",
        "Reply with ONLY the option letter (A, B, or C). Do not explain.",
        "",
        "--- Reference Passages ---",
        "",
    ]

    for i, item in enumerate(retrieved_items, start=1):
        title   = item.get("title",   "").strip()
        content = item.get("content", "").strip()
        lines.append(f"[Reference {i}]{(' ' + title) if title else ''}")
        if content:
            lines.append(content)
        lines.append("")

    lines += [
        "--- Your Question ---",
        f"Question: {test_task['question']}",
        "Options:",
        _format_options(test_task.get("options", {"A": "yes", "B": "no", "C": "maybe"})),
        "",
        "Answer:",
    ]

    return {"role": "user", "content": "\n".join(lines)}
