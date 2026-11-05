import torch
import os
from qwen_vl_utils import process_vision_info


def estimate_gflops(model, total_tokens_count: int):
    """
    Estimates GFLOPs for the forward pass.
    Formula: (2 * Number of Parameters * Total Tokens) / 10^9
    """
    # Get total parameters (requires_grad or total, usually total is used for capacity)
    params = sum(p.numel() for p in model.parameters())
    # 2 ops per param per token (multiply-add)
    flops = 2 * params * total_tokens_count
    return flops / 1e9


def get_batch_token_count(processor, messages_list: list):
    """
    Calculates the total number of tokens in a batch of messages.
    messages_list: list of single chat dicts, same format as used in batch_vqa_answer.
    """
    texts = [
        processor.apply_chat_template([m], tokenize=False, add_generation_prompt=True)
        for m in messages_list
    ]

    image_inputs, video_inputs = process_vision_info(messages_list)

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    return int(inputs["attention_mask"].sum().item())


def record_efficiency_to_txt(out_dir, metrics, args):
    """Records the efficiency results into a text file."""
    report_path = os.path.join(out_dir, "efficiency_report.txt")
    with open(report_path, "w") as f:
        f.write("===== Efficiency Evaluation Report =====\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Retrieval Mode: {getattr(args, 'retrieval_mode', 'none')}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Avg Retrieval Overhead (Tokens/Query): {metrics['avg_overhead']:.2f}\n")
        f.write(f"Avg Computational Cost (GFLOPs/Query):  {metrics['avg_gflops']:.2f}\n")
        f.write(f"Total Queries Processed:                {metrics['total_queries']}\n")
        f.write("========================================\n")
    print(f"Efficiency report saved to: {report_path}")