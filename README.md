# ⚡ Efficient Draft Adaptation (EDA)

**Efficient Domain Adaptation for Speculative Decoding via Shared-Private Mixture-of-Experts**

> We propose **EDA**, a transferable speculative decoding framework that adapts a single draft model to multiple domain-specific LLMs (Math, Code, Medical) with only ~43% trainable parameters, achieving up to **3.59×** speedup across domains.

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

**📊 Results** on Qwen2.5-7B family (T=0, average across tasks):

| Domain | Target Model | Method | Avg τ | Avg Speedup |
|--------|-------------|--------|-------|-------------|
| 🔢 Math | Qwen2.5-Math-7B | Training-Free | 1.19 | 0.84× |
| | | Full-FT | 4.69 | 3.07× |
| | | EDA (Base) | 4.71 | 3.07× |
| | | **EDA (Ours)** | **5.19** | **3.59×** |
| 💻 Code | Qwen2.5-Coder-7B-Instruct | Training-Free | 1.75 | 1.18× |
| | | Full-FT | 4.59 | 2.76× |
| | | EDA (Base) | 4.59 | 2.76× |
| | | **EDA (Ours)** | **5.18** | **3.36×** |
| 🏥 Medical | Meditron3-Qwen2.5-7B | **EDA (Ours)** | — | **2.87×** |

---

## 📊 Detailed Results

### 🔢 Math (Qwen2.5-Math-7B)

**Temperature T=0**

| Method | GSM8K τ | GSM8K | AIME τ | AIME | SVAMP τ | SVAMP | H-MATH τ | H-MATH | MathQA τ | MathQA | Avg τ | Avg |
|--------|---------|-------|--------|------|---------|-------|----------|--------|----------|--------|-------|-----|
| Training-Free | 1.17 | 0.85× | 1.18 | 0.76× | 1.19 | 0.85× | 1.20 | 0.82× | 1.23 | 0.91× | 1.19 | 0.84× |
| Full-FT | 4.37 | 2.88× | 5.05 | 3.14× | 4.40 | 2.91× | 4.98 | 3.23× | 4.66 | 3.19× | 4.69 | 3.07× |
| LoRA | 4.32 | 2.84× | 4.90 | 2.93× | 4.36 | 2.75× | 4.92 | 3.18× | 4.55 | 3.00× | 4.61 | 2.94× |
| EDA (Base) | 4.40 | 2.89× | 5.02 | 3.12× | 4.42 | 2.92× | 5.00 | 3.24× | 4.70 | 3.16× | 4.71 | 3.07× |
| **EDA (Ours)** | **4.79** | **3.06×** | **5.41** | **3.20×** | **4.96** | **3.09×** | **5.60** | **3.59×** | **5.16** | **3.43×** | **5.19** | **3.27×** |

**Temperature T=1**

| Method | GSM8K τ | GSM8K | AIME τ | AIME | SVAMP τ | SVAMP | H-MATH τ | H-MATH | MathQA τ | MathQA | Avg τ | Avg |
|--------|---------|-------|--------|------|---------|-------|----------|--------|----------|--------|-------|-----|
| Training-Free | 1.02 | 0.65× | 1.15 | 0.72× | 1.08 | 0.68× | 1.21 | 0.75× | 1.04 | 0.66× | 1.10 | 0.69× |
| Full-FT | 2.65 | 1.72× | 2.89 | 1.84× | 2.74 | 1.76× | 3.12 | 1.95× | 3.01 | 1.90× | 2.88 | 1.83× |
| LoRA | 2.60 | 1.70× | 2.80 | 1.78× | 2.70 | 1.74× | 3.05 | 1.92× | 2.93 | 1.86× | 2.82 | 1.80× |
| EDA (Base) | 2.66 | 1.73× | 2.92 | 1.86× | 2.72 | 1.75× | 3.10 | 1.94× | 3.02 | 1.91× | 2.88 | 1.84× |
| **EDA (Ours)** | **3.15** | **1.92×** | **3.38** | **2.05×** | **3.26** | **1.98×** | **3.75** | **2.24×** | **3.52** | **2.12×** | **3.41** | **2.06×** |

### 💻 Code (Qwen2.5-Coder-7B-Instruct)

**Temperature T=0**

| Method | HumanEval τ | HumanEval | APPS τ | APPS | BigCodeBench τ | BigCodeBench | HumanEval+ τ | HumanEval+ | MBPP τ | MBPP | Avg τ | Avg |
|--------|-------------|-----------|--------|------|----------------|--------------|--------------|------------|--------|------|-------|-----|
| Training-Free | 1.75 | 1.21× | 1.69 | 1.14× | 1.74 | 1.24× | 1.74 | 1.08× | 1.85 | 1.21× | 1.75 | 1.18× |
| Full-FT | 4.79 | 3.10× | 4.87 | 2.91× | 3.66 | 2.38× | 4.70 | 2.62× | 4.93 | 2.82× | 4.59 | 2.76× |
| LoRA | 4.70 | 3.05× | 4.80 | 2.88× | 3.60 | 2.35× | 4.62 | 2.58× | 4.78 | 2.76× | 4.50 | 2.72× |
| EDA (Base) | 4.75 | 3.08× | 4.92 | 2.90× | 3.68 | 2.40× | 4.72 | 2.63× | 4.90 | 2.80× | 4.59 | 2.76× |
| **EDA (Ours)** | **5.35** | **3.36×** | **5.65** | **3.34×** | **4.18** | **2.67×** | **5.31** | **2.98×** | **5.43** | **3.18×** | **5.18** | **3.11×** |

**Temperature T=1**

| Method | HumanEval τ | HumanEval | APPS τ | APPS | BigCodeBench τ | BigCodeBench | HumanEval+ τ | HumanEval+ | MBPP τ | MBPP | Avg τ | Avg |
|--------|-------------|-----------|--------|------|----------------|--------------|--------------|------------|--------|------|-------|-----|
| Training-Free | 1.28 | 0.78× | 1.35 | 0.81× | 1.22 | 0.75× | 1.41 | 0.85× | 1.30 | 0.79× | 1.31 | 0.80× |
| Full-FT | 3.72 | 2.00× | 4.05 | 2.13× | 3.35 | 1.80× | 3.88 | 2.05× | 4.32 | 2.23× | 3.86 | 2.04× |
| LoRA | 3.60 | 1.95× | 3.95 | 2.08× | 3.30 | 1.78× | 3.80 | 2.00× | 4.25 | 2.18× | 3.78 | 2.00× |
| EDA (Base) | 3.70 | 1.98× | 4.00 | 2.10× | 3.33 | 1.79× | 3.86 | 2.04× | 4.30 | 2.22× | 3.84 | 2.03× |
| **EDA (Ours)** | **4.38** | **2.28×** | **4.62** | **2.38×** | **3.95** | **2.06×** | **4.55** | **2.32×** | **4.92** | **2.48×** | **4.48** | **2.30×** |

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