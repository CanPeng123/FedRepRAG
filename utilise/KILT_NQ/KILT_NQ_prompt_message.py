def build_kilt_nq_embedding_message(record, include_gold=False):
    """
    Message used to embed a corpus item into the FAISS index.
    With include_gold=False: question only (for retrieval alignment).
    With include_gold=True:  question + first gold answer (for richer index).
    """
    q = record['question']
    if include_gold and record.get('gold_answers'):
        content = f"Question: {q}\nAnswer: {record['gold_answers'][0]}"
    else:
        content = f"Question: {q}"
    return {"role": "user", "content": content}


def build_kilt_nq_content_message(record):
    """
    Message used to embed the answer/content side (MLP injection input).
    Uses all gold answers joined together.
    """
    q = record['question']
    answers = ' | '.join(record.get('gold_answers', [])) or 'N/A'
    content = f"Question: {q}\nAnswer: {answers}"
    return {"role": "user", "content": content}


def build_kilt_nq_incontext_message(record, retrieved):
    """
    Build an in-context RAG prompt using retrieved train QA pairs as few-shot examples.
    retrieved: list of train records with 'question' and 'gold_answers'.
    """
    question = record['question']
    examples = ""
    for r in retrieved:
        ans = r['gold_answers'][0] if r.get('gold_answers') else 'N/A'
        examples += f"Question: {r['question']}\nAnswer: {ans}\n\n"
    content = (
        "Answer the following question with a short, concise phrase or entity. "
        "Do not explain — just give the answer.\n\n"
        "Here are some examples:\n\n"
        f"{examples}"
        f"Now answer this question:\n"
        f"Question: {question}\nAnswer:"
    )
    return {"role": "user", "content": content}


def build_kilt_nq_message(record):
    """
    Build a single-turn chat message for open-domain QA.
    Returns a dict suitable for tokenizer.apply_chat_template.
    """
    question = record['question']
    content = (
        "Answer the following question with a short, concise phrase or entity. "
        "Do not explain — just give the answer.\n\n"
        f"Question: {question}\nAnswer:"
    )
    return {"role": "user", "content": content}