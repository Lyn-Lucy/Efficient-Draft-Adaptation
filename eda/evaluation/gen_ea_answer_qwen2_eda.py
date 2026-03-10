"""Generate answers with EDA EAGLE for Qwen2.5 models on MT-Bench/GSM8K/HumanEval.

Usage:
python gen_ea_answer_qwen2_eda.py --base-model-path /path/to/qwen2.5 --ea-model-path /path/to/eda_eagle
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

from eda.model.ea_model_eda import EaModel
from eda.model.kv_cache import initialize_past_key_values
from eda.model.utils import *


def run_eval(
        base_model_path,
        ea_model_path,
        model_id,
        question_file,
        question_begin,
        question_end,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        num_gpus_total,
        max_gpu_memory,
        temperature,
        args
):
    questions = load_questions(question_file, question_begin, question_end)
    shuffled_ids = [q["question_id"] for q in questions]

    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                base_model_path,
                ea_model_path,
                model_id,
                questions[i: i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                temperature,
                args
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
        base_model_path,
        ea_model_path,
        model_id,
        questions,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        max_gpu_memory,
        temperature,
        args
):
    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        private_intermediate_size=args.private_intermediate_size,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    tokenizer = model.get_tokenizer()

    if temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=temperature)
    else:
        logits_processor = None

    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    # Qwen2.5 chat template
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

    # warmup
    for _ in range(3):
        torch.manual_seed(0)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]
        turns = []
        idxs = []
        new_tokens = []
        wall_time = []
        for j in range(len(question["turns"])):
            qs = question["turns"][j]
            messages.append({"role": "user", "content": qs})
            prompt = build_qwen_prompt(messages)
            input_ids = tokenizer([prompt], add_special_tokens=False).input_ids

            torch.cuda.synchronize()
            start_time = time.time()

            output_ids, new_token, idx = model.eagenerate(
                torch.as_tensor(input_ids).cuda(),
                temperature=temperature,
                log=True,
                is_llama3=False,
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
            idxs.append(int(idx))
            new_tokens.append(int(new_token))
            wall_time.append(total_time)
            messages.append({"role": "assistant", "content": output})
    print('Warmup done')

    # Collect statistics
    all_accept_lengths = []

    for question in tqdm(questions):
        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
            ]
            turns = []
            idxs = []
            new_tokens = []
            wall_time = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                messages.append({"role": "user", "content": qs})
                prompt = build_qwen_prompt(messages)
                input_ids = tokenizer([prompt], add_special_tokens=False).input_ids

                torch.cuda.synchronize()
                start_time = time.time()

                output_ids, new_token, idx = model.eagenerate(
                    torch.as_tensor(input_ids).cuda(),
                    temperature=temperature,
                    log=True,
                    is_llama3=False,
                )
                torch.cuda.synchronize()
                total_time = time.time() - start_time
                output_ids = output_ids[0][len(input_ids[0]):]

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

                # Calculate average accept length for this turn
                if new_token > 0 and idx > 0:
                    accept_length = new_token / idx
                    all_accept_lengths.append(accept_length)
                    print(f"Question {question['question_id']}, Turn {j+1}: "
                          f"new_tokens={new_token}, forward_passes={idx}, "
                          f"accept_length={accept_length:.2f}, time={total_time:.2f}s")

                turns.append(output)
                idxs.append(int(idx))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                messages.append({"role": "assistant", "content": output})

            choices.append({
                "index": i, 
                "turns": turns, 
                "idxs": idxs, 
                "new_tokens": new_tokens, 
                "wall_time": wall_time
            })

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")

    # Print summary statistics
    if all_accept_lengths:
        avg_accept = sum(all_accept_lengths) / len(all_accept_lengths)
        print("\n" + "="*60)
        print("EDA EAGLE Accept Length Statistics")
        print("="*60)
        print(f"Total turns evaluated: {len(all_accept_lengths)}")
        print(f"Average accept length: {avg_accept:.4f}")
        print(f"Min accept length: {min(all_accept_lengths):.4f}")
        print(f"Max accept length: {max(all_accept_lengths):.4f}")
        print("="*60)


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ea-model-path",
        type=str,
        default="/gemini/platform/public/linlx/moe_eagle/checkpoints/qwen25_7b_base_eda/state_19",
        help="The path to the EDA EAGLE model weights.",
    )
    parser.add_argument(
        "--base-model-path", 
        type=str, 
        default="/gemini/platform/public/linlx/moe_eagle/models/Qwen2.5-7B",
        help="The path to the base Qwen2.5 model."
    )
    parser.add_argument("--model-id", type=str, default="qwen25_7b_eda_eagle")
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
    parser.add_argument("--total-token", type=int, default=60)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--private-intermediate-size", type=int, default=None,
        help="Private expert intermediate size. If not set, will use value from checkpoint config.")
    parser.add_argument("--num-choices", type=int, default=1)
    parser.add_argument("--num-gpus-per-model", type=int, default=1)
    parser.add_argument("--num-gpus-total", type=int, default=1)
    parser.add_argument("--max-gpu-memory", type=str)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tree-choices", type=str, default="mc_sim_7b_63")

    args = parser.parse_args()

    args.model_id = args.model_id + "-temperature-" + str(args.temperature) + "-eda"
    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray
        ray.init()

    question_file = f"{parent_dir}/data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"{args.bench_name}/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")
    print(f"Base model: {args.base_model_path}")
    print(f"EDA EAGLE model: {args.ea_model_path}")

    run_eval(
        args.base_model_path,
        args.ea_model_path,
        args.model_id,
        question_file,
        args.question_begin,
        args.question_end,
        answer_file,
        args.max_new_token,
        args.num_choices,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.max_gpu_memory,
        args.temperature,
        args
    )

    reorg_answer_file(answer_file)
