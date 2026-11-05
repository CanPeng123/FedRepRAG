import argparse

# -----------------------------
# Args
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser("Extract SLAKE train embeddings for retrieval (FAISS).")

    parser.add_argument("--model", type=str, required=True, choices=[
        "qwen2_5-vl-instruct:3b",
        "qwen2_5-vl-instruct:7b",
        "qwen2_5-vl-instruct:72b",
        "gemma3:4b",
        "gemma3:12b",
    ])
    parser.add_argument("--use_quantization", action="store_true")
    parser.add_argument("--use_deepspeed", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)

    parser.add_argument("--problems_json", type=str, required=True, help="SLAKE json, e.g. train.json")
    parser.add_argument("--images_root", type=str, default="", help="Root folder to prepend to relative image paths.")
    parser.add_argument("--split", type=str, default="train", help="train/val/test if your loader supports it.")
    parser.add_argument("--lang", type=str, default="en", help="en/zh if applicable.")
    parser.add_argument("--out_dir", type=str, default="SLAKE_embeddings")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--idx", type=int, default=0, help="which part (0-based)")
    parser.add_argument("--num_parts", type=int, default=1, help="total parts")

    # whether to append gold answer into prompt (like ScienceQA)
    parser.add_argument("--store_embeddings", action="store_true", help="Actually compute and save embeddings.")
    parser.add_argument("--embeddings_w_gold", action="store_true",
                        help="Append gold answer text to prompt when storing embeddings (supervision hint).")

    # generation debug (optional)
    parser.add_argument("--debug_generate", action="store_true", help="Also run generate() for a quick sanity check.")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)

    return parser.parse_args()