from typing import List, Dict, Any, Optional, Tuple
from PIL.Image import Image as PILImage


# =========================================================
#  SLAKE message builders
# =========================================================

def build_slake_vqa_message(question: str, image: Optional[PILImage], answer_type: Optional[str] = None, 
base_type: Optional[str] = None, triple: Optional[Any] = None, gold_answer: Optional[str] = None, include_gold: bool = False) -> Dict[str, Any]:
    lines = [
        "You are a careful medical VQA assistant.",
        "Answer the question based on the radiology image.",
        "Reply with a short, direct answer (a single word or short phrase).",
        "If it is a yes/no question, answer exactly: Yes or No.",
        f"Question: {question}",
    ]

    if answer_type:
        lines.append(f"Answer type: {answer_type}")  # OPEN / CLOSED

    # optional kvqa hint (won't leak the answer if you only include relation/tail)
    if base_type and base_type.lower() == "kvqa" and isinstance(triple, (list, tuple)) and len(triple) >= 3:
        rel = str(triple[1])
        tail = str(triple[2])
        lines.append(f"Knowledge query (may help): relation={rel}, tail={tail}")

    # supervision hint (like your ScienceQA script)
    if include_gold and gold_answer is not None:
        lines.append(f"Correct answer: {gold_answer}")
    
    lines.append("Rules:\n- Do not output explanations.\n- Output ONLY the answer text.")

    content = [{"type": "text", "text": "\n".join(lines)}]
    if image is not None:
        content.append({"type": "image", "image": image})
    return {"role": "user", "content": content}


def build_slake_rag_incontext_message(test_task: Dict[str, Any], test_img: Optional[PILImage], retrieved_tasks: List[Dict[str, Any]], retrieved_imgs: List[Optional[PILImage]], 
retrieved_caps: Optional[List[str]] = None, include_example_images: bool = False, include_example_captions: bool = False) -> Dict[str, Any]:
    """
    In-context RAG prompt for SLAKE.

    include_example_images   : attach retrieved example images
    include_example_captions : include retrieved example captions in text
    """
    lines = [
        "You are a careful medical VQA assistant.",
        "Use the following solved examples to answer the final question.",
        "Reply with a short, direct answer (a single word or short phrase).",
        "If it is a yes/no question, answer exactly: Yes or No.",
        "",
        "Solved examples:",
    ]

    content: List[Dict[str, Any]] = [{"type": "text", "text": "\n".join(lines)}]

    if retrieved_caps is None:
        retrieved_caps = [""] * len(retrieved_tasks)

    for k, (ex, ex_img, ex_cap) in enumerate(zip(retrieved_tasks, retrieved_imgs, retrieved_caps), start=1):
        ex_q = ex.get("question", "")
        ex_a = ex.get("gold_answer", ex.get("answer", ""))

        block_lines = [f"[Example {k}]", f"Q: {ex_q}"]
        if include_example_captions and ex_cap:
            block_lines.append(f"Image caption: {ex_cap}")
        block_lines.append(f"A: {ex_a}")
        block = "\n".join(block_lines) + "\n"

        content.append({"type": "text", "text": block})

        if include_example_images and ex_img is not None:
            content.append({"type": "image", "image": ex_img})

    # final question
    content.append({"type": "text", "text": f"\n[Final]\nQ: {test_task.get('question','')}\nA:"})
    if test_img is not None:
        content.append({"type": "image", "image": test_img})

    return {"role": "user", "content": content}


# =========================================================
"""
def build_solved_example_message_slake(ex: Dict[str, Any]) -> Dict[str, Any]:
    
    # Minimal solved-example text for retrieval embedding.
    
    q = (ex.get("question") or "").strip()
    a = (ex.get("gold_answer") or ex.get("answer") or "").strip()

    # short single-paragraph format
    if not a:
        a = "."
    if q:
        text = f"Solved example: Q: {q} A: {a}"
    else:
        text = f"Solved example: A: {a}"

    return {"role": "user", "content": [{"type": "text", "text": text}]}
"""



def build_solved_example_message_slake(ex: Dict[str, Any], include_answer_type: bool = True, include_base_type: bool = True, include_kvqa_hint: bool = True, 
include_gold: bool = True) -> Dict[str, Any]:
    """
    Minimal solved-example text for SLAKE retrieval embedding.

    Produces ONE short paragraph, optionally including:
      - answer_type (OPEN/CLOSED)
      - base_type
      - kvqa hint: relation/tail (from triple)
      - gold answer (for solved examples)
    """
    q = (ex.get("question") or "").strip()
    a = (ex.get("gold_answer") or ex.get("answer") or "").strip()
    if not a and include_gold:
        a = "."

    parts = ["Solved example:"]
    if q:
        parts.append(f"Question: {q}")

    if include_answer_type:
        at = (ex.get("answer_type") or "").strip()
        if at:
            parts.append(f"Answer type: {at}.")

    bt = (ex.get("base_type") or "").strip()
    if include_base_type and bt:
        parts.append(f"Base type: {bt}.")

    # kvqa hint (relation/tail) if available
    if include_kvqa_hint and bt.lower() == "kvqa":
        triple = ex.get("triple", None)
        if isinstance(triple, (list, tuple)) and len(triple) >= 3:
            rel = str(triple[1]).strip()
            tail = str(triple[2]).strip()
            # only add if non-empty
            if rel or tail:
                parts.append(f"Knowledge query: relation={rel}, tail={tail}")

    if include_gold:
        parts.append(f"Answer: {a}.")

    text = " ".join(parts)
    return {"role": "user", "content": [{"type": "text", "text": text}]}
