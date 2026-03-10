"""Evaluate original models without EAGLE acceleration on benchmarks.

This script evaluates the base model directly (no speculative decoding)
to get baseline generation speed for comparison with EAGLE.

Usage:
    # Evaluate single benchmark
    python eval_original.py --bench-name gsm8k --base-model-path /path/to/model --model-id test

    # With specific temperature
    python eval_original.py --bench-name gsm8k --base-model-path /path/to/model --temperature 1.0
"""
import argparse
import json
import os
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)

from accelerate.utils import set_seed
set_seed(0)

import time
import shortuuid
from fastchat.llm_judge.common import load_questions
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def run_eval(
        base_model_path,
        model_id,
        question_file,
        question_begin,
        question_end,
        answer_file,
        max_new_token,
        num_choices,
        temperature,
):
    questions = load_questions(question_file, question_begin, question_end)
    
    get_model_answers(
        base_model_path,
        model_id,
        questions,
        answer_file,
        max_new_token,
        num_choices,
        temperature,
    )


@torch.inference_mode()
def get_model_answers(
        base_model_path,
        model_id,
        questions,
        answer_file,
        max_new_token,
        num_choices,
        temperature,
):
    # Load model and tokenizer
    print(f"Loading model from {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    # Qwen2.5 chat template (same as gen_ea_answer_qwen2.py)
    def build_qwen_prompt(messages):
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

    question = questions[0]

    # warmup (same as gen_ea_answer_qwen2.py)
    for _ in range(3):
        torch.manual_seed(0)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]
        turns = []
        new_tokens = []
        wall_time = []
        for j in range(len(question["turns"])):
            qs = question["turns"][j]
            messages.append({"role": "user", "content": qs})
            prompt = build_qwen_prompt(messages)
            input_ids = tokenizer([prompt], add_special_tokens=False).input_ids

            torch.cuda.synchronize()
            start_time = time.time()

            # Standard generation (no EAGLE)
            if temperature > 1e-5:
                output_ids = model.generate(
                    torch.as_tensor(input_ids).cuda(),
                    max_new_tokens=max_new_token,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            else:
                output_ids = model.generate(
                    torch.as_tensor(input_ids).cuda(),
                    max_new_tokens=max_new_token,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            torch.cuda.synchronize()
            total_time = time.time() - start_time
            output_ids = output_ids[0][len(input_ids[0]):]

            # Qwen2.5 stop tokens
            stop_token_ids = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|im_end|>")
            ]

            if stop_token_ids:
                stop_token_ids_index = [
                    i for i, id in enumerate(output_ids)
                    if id in stop_token_ids
                ]
                if len(stop_token_ids_index) > 0:
                    output_ids = output_ids[: stop_token_ids_index[0]]

            output = tokenizer.decode(output_ids, spaces_between_special_tokens=False)
            for special_token in tokenizer.special_tokens_map.values():
                if isinstance(special_token, list):
                    for special_tok in special_token:
                        output = output.replace(special_tok, "")
                else:
                    output = output.replace(special_token, "")
            output = output.strip()

            turns.append(output)
            new_tokens.append(len(output_ids))
            wall_time.append(total_time)
            messages.append({"role": "assistant", "content": output})
    print('Warmup done')

    # Collect statistics
    total_new_tokens = 0
    total_time = 0.0

    answers = {}
    for question in tqdm(questions):
        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
            ]
            turns = []
            new_tokens = []
            wall_time = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                messages.append({"role": "user", "content": qs})
                prompt = build_qwen_prompt(messages)
                input_ids = tokenizer([prompt], add_special_tokens=False).input_ids

                torch.cuda.synchronize()
                start_time = time.time()

                # Standard generation (no EAGLE)
                if temperature > 1e-5:
                    output_ids = model.generate(
                        torch.as_tensor(input_ids).cuda(),
                        max_new_tokens=max_new_token,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                else:
                    output_ids = model.generate(
                        torch.as_tensor(input_ids).cuda(),
                        max_new_tokens=max_new_token,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                torch.cuda.synchronize()
                total_time_single = time.time() - start_time
                new_token = len(output_ids[0]) - len(input_ids[0])
                output_ids = output_ids[0][len(input_ids[0]):]

                # Qwen2.5 stop tokens
                stop_token_ids = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|im_end|>")
                ]

                if stop_token_ids:
                    stop_token_ids_index = [
                        i for i, id in enumerate(output_ids)
                        if id in stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]
                        new_token = len(output_ids)

                output = tokenizer.decode(output_ids, spaces_between_special_tokens=False)
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                turns.append(output)
                new_tokens.append(int(new_token))
                wall_time.append(total_time_single)
                messages.append({"role": "assistant", "content": output})
                
                # Accumulate stats
                total_new_tokens += new_token
                total_time += total_time_single

            choices.append({"index": i, "turns": turns, "new_tokens": new_tokens, "wall_time": wall_time})

        ans = {
            "question_id": question["question_id"],
            "answer_id": shortuuid.uuid(),
            "model_id": model_id,
            "choices": choices,
            "tstamp": time.time(),
        }
        answers[question["question_id"]] = json.dumps(ans, ensure_ascii=False) + "\n"

        # Print per-question stats
        total_tokens_q = sum(choices[0]["new_tokens"])
        total_time_q = sum(choices[0]["wall_time"])
        tokens_per_sec = total_tokens_q / total_time_q if total_time_q > 0 else 0
        print(f"Question {question['question_id']}: {total_tokens_q} tokens, {total_time_q:.2f}s, {tokens_per_sec:.2f} tok/s")

    # Print overall statistics
    print("\n" + "="*60)
    print("Overall Statistics (No EAGLE):")
    print(f"Total questions: {len(questions)}")
    print(f"Total tokens generated: {total_new_tokens}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average tokens/second: {total_new_tokens / total_time:.2f}" if total_time > 0 else "N/A")
    print("="*60)

    # Save answers
    os.makedirs(os.path.dirname(answer_file) if os.path.dirname(answer_file) else ".", exist_ok=True)
    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])
    
    print(f"Answers saved to {answer_file}")

    # Clean up
    del model
    del tokenizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model-path", 
        type=str, 
        required=True,
        help="The path to the base model."
    )
    parser.add_argument("--model-id", type=str, default="original_baseline")
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument("--question-begin", type=int, help="The begin index of questions.")
    parser.add_argument("--question-end", type=int, help="The end index of questions.")
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument("--max-new-token", type=int, default=1024)
    parser.add_argument("--num-choices", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)

    args = parser.parse_args()

    args.model_id = args.model_id + "-temperature-" + str(args.temperature) + "-original"

    question_file = f"{parent_dir}/data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"{args.bench_name}/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")
    print(f"Base model: {args.base_model_path}")
    print(f"Temperature: {args.temperature}")

    run_eval(
        args.base_model_path,
        args.model_id,
        question_file,
        args.question_begin,
        args.question_end,
        answer_file,
        args.max_new_token,
        args.num_choices,
        args.temperature,
    )
