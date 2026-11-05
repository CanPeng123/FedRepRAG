import torch
import numpy as np
from PIL.Image import Image as PILImage
from typing import List, Dict, Union, Literal, Optional
from qwen_vl_utils import process_vision_info
import torch.nn.functional as F

from VQA.utilise.data_process import load_image, load_pil_or_none
from VQA.utilise.ScienceQA.ScienceQA_data_process import discover_q_choice_by_convention, load_captions

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ----------------------------------------------------------------
# ---- Build message ([question, context, choices, image]) ----
# ----------------------------------------------------------------
def build_multi_image_vqa_message(question: str, choices: list[str], question_img: PILImage | None, choice_imgs: list[PILImage | None], 
context: str | None = None):

    lines = [
        "You are a helpful science VQA assistant.",
        "You will be given one question image and images for some answer choices.",
        "Look at all the images and the question text, then select exactly ONE letter for the correct answer.",
        f"Question: {question}"
    ]
    if context:
        lines.append(f"Context: {context}")
    lines.append("Options:")
    for i, c in enumerate(choices):
        lines.append(f"{LETTERS[i]}. {c if c else '(see corresponding image)'}")
    
    lines.append("\nRules:" \
    "\n- The first image is the question image (if provided)." \
    "\n- The next images correspond to choices in order A, B, C, ..." \
    "\n- Answer with ONLY the option letter (e.g., A).")

    content = [{"type": "text", "text": "\n".join(lines)}]

    if question_img is not None:
        content.append({"type": "text", "text": "Question Image:"})
        content.append({"type": "image", "image": question_img})

    for i, im in enumerate(choice_imgs):
        if im is None:
            continue
        content.append({"type": "text", "text": f"Choice {LETTERS[i]} Image:"})
        content.append({"type": "image", "image": im})

    return {"role": "user", "content": content}


def build_retrieval_augmented_message(test_task: dict, retrieved_tasks: list[dict], test_q_img: PILImage | None, 
test_choice_imgs: list[PILImage | None], include_example_images: bool = False, example_captions: List[str] | None = None,
retrieved_visual_w_choice_img: bool = False, use_qa: bool = False, use_ctx: bool = False, use_lec: bool = False, use_sol: bool = True) -> dict:
    """
    Build a message that includes solved training examples + the new test question.

    retrieved_tasks: each should have keys 'question', 'choices', 'gold',
    and optionally 'context'/'lecture'/'solution'.

    - include_example_images:
        If True, attach question/choice images for each retrieved example.
    - example_captions:
        If provided, should be a list of dicts, same length as retrieved_tasks.
        Each dict maps caption keys (e.g. "image", "A", "B", ...) to text,
        produced by load_captions(dir_path, image_only=...).
        We will:
          * use key "image" (if present) as the question image caption
          * use keys "A", "B", ... (if present) as choice image captions.
    """
    lines: list[str] = []

    print("build_retrieval_augmented_message | include_example_images: {0}".format(include_example_images))
    print("build_retrieval_augmented_message | example_captions: {0}".format(example_captions))

    # --- Shared header with build_multi_image_vqa_message ---
    lines.append("You are a helpful science VQA assistant.")
    lines.append("You will be given one question image and images for some answer choices.")
    lines.append(
        "Look at all the images and the question text, then select exactly ONE letter for the correct answer."
    )
    lines.append(
        "First, you will see some solved example questions. Study them carefully, then answer the final question."
    )
    lines.append("")  # blank line

    # --- Solved examples from retrieval ---
    for ex_i, ex in enumerate(retrieved_tasks):
        q = ex["question"]
        choices = ex["choices"]
        gold_idx = int(ex["gold"])  # answer
        gold_letter = LETTERS[gold_idx]
        
        ex_ctx = ex.get("context", "")  # hint 
        ex_lect = ex.get("lecture", "")
        ex_solu = ex.get("solution", "")

        # captions dict for this example (may be None / empty)
        caps_for_ex: Dict[str, str] = {}
        if example_captions is not None and ex_i < len(example_captions):
            caps_for_ex = example_captions[ex_i] or {}

        lines.append(f"Solved example {ex_i + 1}:")
        if use_qa:
            lines.append(f"Question: {q}")
        if ex_ctx and use_ctx:
            lines.append(f"Context: {ex_ctx}")
        if use_qa:
            lines.append("Options:")
            for j, c in enumerate(choices):
                # keep it consistent with build_multi_image_vqa_message
                lines.append(f"{LETTERS[j]}. {c if c else '(see corresponding image)'}")

        # --- OPTIONAL: caption for example images ---
        # Example question image caption
        q_cap = caps_for_ex.get("image", "").strip()
        if q_cap:
            lines.append(f"Question image caption: {q_cap}")
        
        if retrieved_visual_w_choice_img:
            # Example choice images caption
            for j, _ in enumerate(choices):
                letter = LETTERS[j]
                ch_cap = caps_for_ex.get("choice_{0}".format(j), "").strip()
                if ch_cap:
                    lines.append(f"Choice {letter} image caption: {ch_cap}")
        if use_qa:
            lines.append(f"Correct answer: {gold_letter}")
        if ex_lect and use_lec:  
            lines.append(f"Lecture: {ex_lect}")
        if ex_solu and use_sol:
            lines.append(f"Solution: {ex_solu}")
        lines.append("")  # blank line between examples

    # --- New test question (no answer given) ---
    lines.append("Now answer this new question:\n")
    lines.append(f"Question: {test_task['question']}")
    if test_task.get("context"):
        lines.append(f"Context: {test_task['context']}")
    lines.append("Options:")
    for j, c in enumerate(test_task["choices"]):
        lines.append(f"{LETTERS[j]}. {c if c else '(see corresponding image)'}")

    # --- Rules block (updated to reflect example + test images) ---
    rules_lines = [
        "Rules:",
        "- First, read the solved example questions (if any) in the text above.",
        "- The NEW question you must answer is under 'Now answer this new question:'.",
        "- Images for the new question are labeled 'Question Image:' and 'Choice X Image:'.",
        "- If present, images for solved examples are labeled "
        "like 'Solved example k – Question Image:' or 'Solved example k – Choice X Image:'.",
        "- Use all images and text as guidance, but ONLY answer the new question.",
        "- Answer with ONLY the option letter for the new question (e.g., A).",
    ]
    lines.append("\n" + "\n".join(rules_lines))

    # Build content list: text + test images (same pattern as build_multi_image_vqa_message)
    content: list[dict] = [{"type": "text", "text": "\n".join(lines)}]

    if test_q_img is not None:
        content.append({"type": "text", "text": "Question Image:"})
        content.append({"type": "image", "image": test_q_img})

    for i, im in enumerate(test_choice_imgs):
        if im is None:
            continue
        content.append({"type": "text", "text": f"Choice {LETTERS[i]} Image:"})
        content.append({"type": "image", "image": im})

    # -------- OPTION 1: include retrieved example IMAGES ----------
    if include_example_images:
        for ex_i, ex in enumerate(retrieved_tasks):
            image_dir = ex.get("image_dir") or (os.path.dirname(ex["image_path"]) if ex.get("image_path") else None)
            q_path, choice_paths = discover_q_choice_by_convention(image_dir, len(ex["choices"]))
            ex_q_img = load_pil_or_none(q_path)
            ex_choice_imgs = [load_pil_or_none(p) for p in choice_paths]

            # Example question image
            if ex_q_img is not None:
                print("*** Solved example {ex_i + 1} – Question Image added")
                content.append({"type": "text", "text": f"Solved example {ex_i + 1} – Question Image:"})
                content.append({"type": "image", "image": ex_q_img})
            
            # Example choice images
            if retrieved_visual_w_choice_img:
                for j, im in enumerate(ex_choice_imgs):
                    if im is None:
                        continue
                    content.append({
                        "type": "text",
                        "text": f"Solved example {ex_i + 1} – Choice {LETTERS[j]} Image:",
                    })
                    content.append({"type": "image", "image": im})
            
    return {"role": "user", "content": content}


def build_solved_example_message(ex: dict) -> dict:
    """
    Build a message that describes ONE solved example:
    question + context + options + gold + lecture/solution.
    """
    lines = []

    q = ex["question"]
    choices = ex["choices"]
    gold_idx = int(ex["gold"])
    gold_letter = LETTERS[gold_idx]

    ex_ctx = ex.get("context", "")
    ex_lect = ex.get("lecture", "")
    ex_solu = ex.get("solution", "")

    lines.append("Solved example:")
    lines.append(f"Question: {q}")
    if ex_ctx:
        lines.append(f"Context: {ex_ctx}")
    lines.append("Options:")
    for j, c in enumerate(choices):
        lines.append(f"{LETTERS[j]}. {c if c else '(see corresponding image)'}")
    lines.append(f"Correct answer: {gold_letter}")
    if ex_lect:
        lines.append(f"Lecture: {ex_lect}")
    if ex_solu:
        lines.append(f"Solution: {ex_solu}")

    content = [{"type": "text", "text": "\n".join(lines)}]
    return {"role": "user", "content": content}
