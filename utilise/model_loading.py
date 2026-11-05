import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig, AddedToken

REPRAG_TOKEN = "<RepRAG>"


def load_model_with_quantization(model_name, use_quantization):
    if use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
    else:
        quantization_config = None

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"  # *** REQUIRED FOR QWEN2.5
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        trust_remote_code=True
    )
    
    return processor, model

"""
def load_model_with_deepspeed(model_name, use_flash_attn=False):
    # 1) Load processor  
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"  # *** REQUIRED FOR QWEN2.5
    
    # 2) Build model kwargs
    model_kwargs = {
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
    }
    if use_flash_attn:
        # only set these if you have flash_attn installed
        model_kwargs.update({
            "attn_implementation": "flash_attention_2",
            "device_map": "auto",
        })

    # 3) Load the model
    model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)

    # 4) Wrap with DeepSpeed
    model_engine = deepspeed.init_inference(
        model=model,
        mp_size=torch.cuda.device_count(),
        dtype=torch.float16,
        replace_method='auto',  # Avoids replacing internal layers
        replace_with_kernel_inject=True,
        max_tokens=65536,
    )
    return processor, model_engine
"""


def load_model_with_deepspeed(model_name, use_flash_attn=False):
    import deepspeed

    # 1) Load processor
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tok = processor.tokenizer
    tok.padding_side = "left"  # REQUIRED for Qwen2.5 batching

    # 2) Load model (NOT wrapped yet)
    model_kwargs = {"torch_dtype": torch.float16, "trust_remote_code": True}
    if use_flash_attn:
        model_kwargs.update({
            "attn_implementation": "flash_attention_2",
            "device_map": "auto",
        })
    model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)

    # 3) Add <RepRAG> BEFORE DeepSpeed wrapping + resize
    reprag_id = tok.convert_tokens_to_ids(REPRAG_TOKEN)
    if reprag_id == tok.unk_token_id:
        tok.add_tokens([AddedToken(REPRAG_TOKEN, lstrip=False, rstrip=False)])
        model.resize_token_embeddings(len(tok))

    # sanity: must be single token
    reprag_token_id = tok.convert_tokens_to_ids(REPRAG_TOKEN)
    enc = tok(REPRAG_TOKEN, add_special_tokens=False)["input_ids"]
    assert len(enc) == 1 and enc[0] == reprag_token_id, f"'<RepRAG>' must be single token, got {enc}"

    # 4) Wrap with DeepSpeed
    model_engine = deepspeed.init_inference(
        model=model,
        mp_size=torch.cuda.device_count(),
        dtype=torch.float16,
        replace_method="auto",
        replace_with_kernel_inject=True,
        max_tokens=65536,
    )

    return processor, model_engine, reprag_token_id





def load_model_standard(model_name):
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"  # *** REQUIRED FOR QWEN2.5
    model = AutoModelForImageTextToText.from_pretrained(model_name)
    return processor, model


def load_gemma3_standard(model_name):
    processor = AutoProcessor.from_pretrained(model_name)
    processor.tokenizer.padding_side = "left"
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        # multi-GPU: let Accelerate distribute layers
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        # single GPU: direct placement avoids Accelerate hook overhead
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    return processor, model
