# ⚡ Efficient Draft Adaptation (EDA)

**Efficient Domain Adaptation for Speculative Decoding via Shared-Private Mixture-of-Experts**

> We propose **EDA**, a transferable speculative decoding framework that adapts a single draft model to multiple domain-specific LLMs (Math, Code, Medical) with only ~43% trainable parameters, achieving 2.87×–3.45× speedup across domains.

---

## 🔍 Overview

**EDA** uses a **Shared-Private MoE** architecture for efficient speculative decoding adaptation:

- **Stage 1**: Train shared + private experts on general data (base model)
- **Stage 2**: Freeze shared experts, train only private experts on domain data (~43% params)

```
Hidden state
    ├─► Shared Router ──► Shared Expert(s)  [frozen in Stage 2]  ─┐
    │                                                              ├─► Output
    └─► Private Router ─► Private Expert(s) [trained in Stage 2] ─┘
```

**📊 Results** on Qwen2.5-7B family:

| Domain | Target Model | Speedup |
|--------|-------------|---------|
| 🔢 Math   | Qwen2.5-Math-7B | **3.45×** |
| 💻 Code   | Qwen2.5-Coder-7B-Instruct | **2.96×** |
| 🏥 Medical | Meditron3-Qwen2.5-7B | **2.87×** |

---

## 📁 Repository Structure

```
Efficient-Draft-Adaptation/
├── eda/
│   ├── model/          # EDA and baseline EAGLE model architectures
│   ├── train/          # DeepSpeed training scripts and configs
│   ├── data/           # Feature extraction (ge_data.py, allocation.py)
│   └── evaluation/     # Inference and speedup measurement scripts
├── scripts/
│   ├── train_stage1_base.sh
│   ├── train_stage2_transfer.sh
│   ├── eval_eda.sh
│   ├── eval_base_transfer.sh
│   └── eval_original.sh
├── requirements.txt
└── setup.py
```

---

## 🚀 Installation

```bash
git clone https://github.com/YOUR_USERNAME/Efficient-Draft-Adaptation.git
cd Efficient-Draft-Adaptation
pip install -r requirements.txt
pip install -e .
```

---

## 🤖 Models

```bash
huggingface-cli download Qwen/Qwen2.5-7B --local-dir models/Qwen2.5-7B
huggingface-cli download Qwen/Qwen2.5-Math-7B --local-dir models/Qwen2.5-Math-7B
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct --local-dir models/Qwen2.5-Coder-7B-Instruct
huggingface-cli download OpenMeditron/Meditron3-Qwen2.5-7B --local-dir models/Meditron3-Qwen2.5-7B
```

---

## 📦 Data Preparation

Feature extraction precomputes last-layer hidden states of the target model, saving them as `.ckpt` files for training.

### Stage 1: General Data (ShareGPT)

```bash
huggingface-cli download Aeala/ShareGPT_Vicuna_unfiltered --local-dir data/ShareGPT

python eda/data/allocation.py \
    --datafile data/ShareGPT/ShareGPT_V4.3_unfiltered_cleaned_split.json \
    --modelname models/Qwen2.5-7B \
    --outdir eagle_datas/sharegpt_base_mufp16 \
    --total 68000 \
    --gpus 0 1 2 3 4 5 6 7
```

### Stage 2: Domain Data

**Option A — Self-Generated (recommended)**: Generate answers with the target model on domain questions (e.g., [DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K)), save as ShareGPT-format JSON, then extract:

```bash
python eda/data/allocation.py \
    --datafile data/DeepMath_generated.json \
    --modelname models/Qwen2.5-Math-7B \
    --outdir eagle_datas/deepmath_generated_math_mufp16 \
    --total 66720 \
    --gpus 0 1 2 3 4 5 6 7
```

**Option B — ShareGPT with target model**:

```bash
python eda/data/allocation.py \
    --datafile data/ShareGPT/ShareGPT_V4.3_unfiltered_cleaned_split.json \
    --modelname models/Qwen2.5-Math-7B \
    --outdir eagle_datas/sharegpt_math_mufp16 \
    --total 68000 \
    --gpus 0 1 2 3 4 5 6 7
```

For custom datasets, implement `preprocess_custom()` in `eda/data/ge_data.py` and pass `--data_type custom`.

---

## 🏋️ Training

### Stage 1

```bash
bash scripts/train_stage1_base.sh
```

### Stage 2 (Domain Transfer)

```bash
bash scripts/train_stage2_transfer.sh
```

Key arguments (`main_deepspeed_eda.py`):

| Argument | Description |
|----------|-------------|
| `--basepath` | Path to target LLM |
| `--tmpdir` | Feature data directory |
| `--cpdir` | Output checkpoint directory |
| `--transfer` | Enable Stage 2 transfer mode |
| `--pretrained_eda` | Stage 1 checkpoint path |
| `--freeze_attention` | Freeze attention in Stage 2 |

Checkpoints saved as `state_0/` … `state_19/`.

---

## 📈 Evaluation

```bash
bash scripts/eval_eda.sh        # EDA (ours)
bash scripts/eval_base_transfer.sh # Base transfer baseline
bash scripts/eval_original.sh      # Autoregressive baseline
```

Supported benchmarks (set `TASK` in scripts):

| Domain | Tasks |
|--------|-------|
| Math | `gsm8k`, `aime_2024`, `svamp`, `hendrycks_math`, `math_qa` |
| Code | `humaneval`, `humaneval_plus`, `apps`, `bigcodebench`, `mbpp` |
| Medical | `medmcqa`, `medqa_usmle`, `pubmedqa`, `mmlu_clinical` |

```bash
python eda/evaluation/extract_results.py results/eda_math_gsm8k/
```

---

## 🏛️ Affiliations

1. Shanghai Innovation Institute
2. Xiamen University
3. TeleAI

---

## 🙏Acknowledgements

We are very grateful to the [EAGLE](https://github.com/SafeAILab/EAGLE) teams for creating awesome repo.