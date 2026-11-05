import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Generate visual descriptions for images using Qwen2.5-VL-Instruct.")
    parser.add_argument('--model', type=str, required=True, choices=[
        'qwen2_5-vl-instruct:2b', 
        'qwen2_5-vl-instruct:7b', 
        'qwen2_5-vl-instruct:72b'
    ], help="Model size: '2b', '7b', or '72b' of Qwen2.5-VL-Instruct.")
    parser.add_argument('--use_quantization', action='store_true', help="Use 4-bit quantization.")
    parser.add_argument('--use_deepspeed', action='store_true', help="Use DeepSpeed for distributed training.")
    parser.add_argument('--local_rank', type=int, default=-1, help="DeepSpeed internal use (auto-injected).")

    parser.add_argument('--idx', type=int, default=0, help='Which part to run (0-based)')
    parser.add_argument('--num_parts', type=int, default=1, help='Total number of parallel parts')
    parser.add_argument('--batch_size', type=int, default=8)

    parser.add_argument('--do_sample', action='store_true', help="During VLM inference, do samle.")
    parser.add_argument('--temperature', type=float, default=0.7, help='During VLM inference, temperature value for control output')

    parser.add_argument('--problems_json', type=str, help="Path to ScienceQA problems.json")
    parser.add_argument('--images_root', type=str, default="", help="Optional root to prepend to relative image paths in problems.json")
    parser.add_argument('--split', type=str, default=None, help="Filter split: train/val/test")
    parser.add_argument('--out_dir', type=str, default="VQA_preds")

    return parser.parse_args()