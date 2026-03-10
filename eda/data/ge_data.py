"""
Feature Extraction Script for EDA (Efficient Draft Adaptation).

This script extracts hidden states from a target LLM (e.g., Qwen2.5-Math-7B)
on a given dataset, producing .ckpt files used for draft model training.

Usage (single GPU, for testing):
    python ge_data.py --start 0 --end 100 --index 0 --gpu_index 0 \
        --outdir /path/to/output \
        --datafile /path/to/data.json \
        --modelname /path/to/TargetLLM \
        --data_type sharegpt

Usage (via allocation.py for multi-GPU parallel extraction):
    python allocation.py --outdir /path/to/output \
        --datafile /path/to/data.json \
        --modelname /path/to/TargetLLM

======================  CUSTOMIZATION GUIDE  =======================
--data_type:
    "sharegpt"  — ShareGPT-format JSON: list of {"id", "conversations":[
                   {"from":"human","value":"..."}, {"from":"gpt","value":"..."}]}
    "custom"    — Any other format: implement preprocess_custom() below.

--datafile :
    Path to the JSON file. Could be ShareGPT, DeepMath-generated, etc.

--modelname :
    HuggingFace-compatible path to the TARGET model (the large model you want
    to accelerate, e.g. Qwen2.5-Math-7B, Qwen2.5-Coder-7B-Instruct, etc.)

System prompt:
    Modify SYSTEM_PROMPT below to match what you used when generating answers.
    For ShareGPT (general):    "You are a helpful assistant."
    For Math domain:           "You are a helpful and thorough mathematics assistant."
    For Code domain:           "You are an expert programmer."
    For Medical domain:        "You are a helpful medical assistant."

Chat template separators (sep / sep2):
    These must match the target model's chat template.
    Qwen2.5 / Qwen2.5-Math / Qwen2.5-Coder / Meditron3-Qwen2.5 all use:
        sep  = "<|im_end|>\\n<|im_start|>assistant\\n"
        sep2 = "<|im_end|>\\n<|im_start|>user\\n"
    If you use a different model family, update them accordingly.
====================================================================
"""

import argparse

parser = argparse.ArgumentParser(description="Feature extraction for EDA training data")
parser.add_argument("--start", type=int, default=0, help="Start index in the dataset")
parser.add_argument("--end", type=int, default=100, help="End index (exclusive) in the dataset")
parser.add_argument("--index", type=int, default=0, help="Worker index (used for output subdirectory naming)")
parser.add_argument("--gpu_index", type=int, nargs="+", default=[0], help="GPU(s) to use")
parser.add_argument("--outdir", type=str, required=True, help="Root output directory for .ckpt files")
parser.add_argument("--datafile", type=str, required=True, help="Path to the input JSON dataset")
parser.add_argument("--modelname", type=str, required=True, help="Path to the target LLM (HuggingFace format)")
parser.add_argument("--data_type", type=str, default="sharegpt",
                    choices=["sharegpt", "custom"],
                    help="Dataset format type")
args = parser.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# ============================================================
#  MODIFY: system prompt — must match what was used during
#  answer generation (for domain-specific generated datasets)
# ============================================================
SYSTEM_PROMPT = "You are a helpful assistant."

# ============================================================
#  Chat template separators — Qwen2.5 family default
#  Change these if you use a different model (e.g. LLaMA-3)
# ============================================================
SEP = "<|im_end|>\n<|im_start|>assistant\n"
SEP2 = "<|im_end|>\n<|im_start|>user\n"


def preprocess_sharegpt(examples, tokenizer):
    """Process ShareGPT-format conversations."""
    new_examples = {
        "conversation": [],
        "input_ids": [],
        "loss_mask": [],
    }
    roles = {"human": "user", "gpt": "assistant"}
    convroles = ["user", "assistant"]

    for i in range(len(examples["id"])):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        source = examples["conversations"][i]
        if roles[source[0]["from"]] != "user":
            source = source[1:]
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == convroles[j % 2], f"Turn {j} role mismatch"
            messages.append({"role": role, "content": sentence["value"]})

        conversation = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        _append_sample(new_examples, conversation, tokenizer)
    return new_examples


def preprocess_custom(examples, tokenizer):
    """
    MODIFY THIS for non-ShareGPT datasets.

    Your dataset must expose at minimum:
        examples["id"]            — unique identifier per sample
        examples["conversations"] — list of turns, same format as ShareGPT
                                    OR implement your own formatting below.
    """
    raise NotImplementedError(
        "Implement preprocess_custom() for your dataset format. "
        "See the CUSTOMIZATION GUIDE at the top of this file."
    )


def _append_sample(new_examples, conversation, tokenizer):
    """Tokenize conversation and compute loss mask."""
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.unk_token_id

    input_ids = tokenizer(
        conversation,
        return_tensors="pt",
        max_length=2048,
        add_special_tokens=False,
    ).input_ids[0]
    loss_mask = torch.ones_like(input_ids)

    turns = conversation.split(SEP2)
    turns[1] = turns[0] + SEP2 + turns[1]
    turns = turns[1:]

    cur_len = 1
    loss_mask[:cur_len] = 0
    for k, turn in enumerate(turns):
        if turn == "":
            break
        turn_len = len(tokenizer(turn).input_ids)
        parts = turn.split(SEP)
        if len(parts) != 2:
            break
        parts[0] += SEP
        instruction_len = len(tokenizer(parts[0]).input_ids)
        if k == 0:
            loss_mask[0: cur_len + instruction_len - 2] = 0
        else:
            loss_mask[cur_len - 6: cur_len + instruction_len - 2] = 0
        cur_len += turn_len
        cur_len += 5
    loss_mask[cur_len:] = 0

    new_examples["conversation"].append(conversation)
    new_examples["input_ids"].append(input_ids[None, :])
    new_examples["loss_mask"].append(loss_mask[None, :])


def build_dataset(tokenizer):
    ds = load_dataset("json", data_files=args.datafile)
    ds = ds["train"]
    ds = ds.shuffle(seed=42)
    ds1 = ds.select(range(args.start, args.end))
    original_columns = ds1.column_names

    if args.data_type == "sharegpt":
        preprocess_fn = lambda examples: preprocess_sharegpt(examples, tokenizer)
    else:
        preprocess_fn = lambda examples: preprocess_custom(examples, tokenizer)

    ds1 = ds1.map(
        preprocess_fn,
        batched=True,
        remove_columns=original_columns,
        load_from_cache_file=False,
    )
    ds1.set_format(type="torch")
    return ds1


# ── Load model & tokenizer ───────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(args.modelname, use_fast=False)
dataset = build_dataset(tokenizer)

model = AutoModelForCausalLM.from_pretrained(
    args.modelname, device_map="auto", torch_dtype=torch.bfloat16
)
model.eval()


@torch.no_grad()
def extract_features(data):
    input_ids = data["input_ids"]
    outs = model(input_ids.cuda(), output_hidden_states=True)
    hidden_state = outs.hidden_states[-1]
    return {
        "input_ids": input_ids.cpu()[0],
        "hidden_state": hidden_state.cpu()[0],
        "loss_mask": data["loss_mask"].cpu()[0],
    }


outdir = os.path.join(args.outdir, str(args.index))
os.makedirs(outdir, exist_ok=True)


def save_sample(name, data_point):
    os.makedirs(name, exist_ok=True)
    idx = len(os.listdir(name))
    torch.save(data_point, os.path.join(name, f"data_{idx}.ckpt"))


for data in tqdm(dataset, total=len(dataset), desc=f"Worker {args.index}", unit="sample"):
    save_sample(outdir, extract_features(data))
