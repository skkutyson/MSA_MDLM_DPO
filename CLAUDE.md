# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MSAGPT is a 3B parameter protein language model for neural prompting protein structure prediction via MSA (Multiple Sequence Alignment) generative pre-training. It generates virtual MSAs from protein sequences to improve structure prediction in MSA-scarce scenarios.

Three model variants exist: MSAGPT (base), MSAGPT-SFT (supervised fine-tuning), MSAGPT-DPO (RLHF training).

## Running the CLI

**Interactive chat mode:**
```bash
bash scripts/cli_sat.sh --from_pretrained ./checkpoints/MSAGPT-DPO \
    --input-source chat \
    --stream_chat \
    --max-gen-length 1024
```

**Offline batch generation:**
```bash
bash scripts/cli_sat.sh --from_pretrained ./checkpoints/MSAGPT-DPO \
    --input-source <input_file> \
    --output-path <output_path> \
    --max-gen-length 1024
```

Key parameters in `scripts/cli_sat.sh`: `MP_SIZE` (model parallelism), `NUM_BEAMS`, `TEMP`, `TOPK`, `TOPP`, `SAMPLING_STRATEGY` (BaseStrategy or BeamSearchStrategy).

## Running MDLM Inference (Diffusion-based)

**Interactive chat mode with MDLM:**
```bash
bash scripts/cli_sat.sh --from_pretrained ./checkpoints/mdlm_dpo \
    --backbone mdlm \
    --input-source chat \
    --stream_chat \
    --max-gen-length 512 \
    --num-diffusion-steps 256 \
    --diffusion-sampler ddpm_cache
```

**Offline batch generation with MDLM:**
```bash
bash scripts/cli_sat.sh --from_pretrained ./checkpoints/mdlm_dpo \
    --backbone mdlm \
    --input-source <input_file> \
    --output-path <output_path> \
    --max-gen-length 512
```

**MDLM-specific parameters:**
- `--backbone mdlm`: Use diffusion backbone instead of autoregressive
- `--num-diffusion-steps`: Denoising steps (default: 256, higher = better quality)
- `--diffusion-sampler`: `ddpm` or `ddpm_cache` (faster, default)

**Python API for MDLM:**
```python
from model_utils.mdlm import MSAGPT_MDLM
from utils.tokenization import proteinglm_tokenizer
import torch

# Load model
model, args = MSAGPT_MDLM.from_pretrained('./checkpoints/mdlm_dpo', args)
model = model.cuda().eval()
tokenizer = proteinglm_tokenizer()

# Prepare input
query = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTK"
seq = [tokenizer.get_command('[gMASK]'), tokenizer.get_command('sop')]
seq += tokenizer.tokenize(query) + [tokenizer.get_command('<M>')]
context_tokens = torch.tensor([seq]).cuda()

# Generate
output = model.generate(
    context_tokens=context_tokens,
    msa_len=len(query) + 1,
    max_gen_length=512,
    num_steps=256,
    msa_delimiter_id=tokenizer.get_command('<M>'),
)
print(tokenizer.detokenize(output[0].tolist()))
```

## Architecture

```
cli_sat.py                          # Main CLI entry point
model_utils/
├── model_msagpt.py                 # MSAGPT wrapper (extends ProteinGLMForGeneration)
├── model_proteinglm_clm.py         # Base transformer with custom mixins:
│                                   # - 2D rotary embeddings
│                                   # - GLU activation with DeepNorm
│                                   # - FP32 softmax attention
└── mdlm/                           # MDLM (Masked Diffusion Language Model)
    ├── diffusion.py                # ProteinDiffusion - forward diffusion & loss
    ├── noise_schedule.py           # LogLinearNoise schedule
    ├── protein_dit.py              # ProteinDIT backbone (DiT architecture)
    └── model_msagpt_mdlm.py        # MSAGPT_MDLM wrapper
utils/
├── chat.py                         # Generation pipeline (streaming & offline)
├── tokenization.py                 # Residue-level protein tokenizer (35 vocab)
├── strategies.py                   # Sampling: AdvancedBaseStrategy, BeamSearchStrategy
└── utils.py                        # ANSI cursor control
training/                           # MDLM training infrastructure
├── datasets/
│   ├── openproteinset_loader.py    # OpenProteinSet A3M/FASTA parser
│   ├── msa_dataset.py              # Pre-training dataset with 2D positions
│   ├── msa_preference_dataset.py   # DPO preference pairs dataset
│   └── preference_generator.py     # Generate pairs using proxy metrics
├── losses/
│   ├── mdlm_loss.py                # Diffusion NLL loss (SUBS parameterization)
│   └── mdlm_dpo_loss.py            # D3PO loss for masked diffusion
├── trainers/
│   ├── mdlm_trainer.py             # PyTorch Lightning pre-training module
│   └── mdlm_dpo_trainer.py         # DPO fine-tuning module
└── utils/
    ├── ema.py                      # Exponential Moving Average
    └── metrics.py                  # Perplexity, accuracy metrics
```

## Input Format

Protein sequences use `<M>` delimiter for multi-shot prompting:
```
PEGKQGDPGIPGEPGPPGPPGPQGARGPPG<M>VTVEFVNSCLIGDMGVDGPPGQQGQPGPPG
```
First sequence is primary; subsequent sequences are in-context examples.

## Dependencies

Built on SwissArmyTransformer (SAT) framework. Requires CUDA >= 11.8, PyTorch 2.1+. Install via `pip install -r requirements.txt`.

## Training MDLM Models

### Data Preparation

```bash
# Download OpenProteinSet (requires AWS CLI)
python scripts/prepare_dataset.py download --output ./data/openproteinset --max-files 1000

# Process and filter MSAs
python scripts/prepare_dataset.py process --input ./data/openproteinset --output ./data/processed

# Generate preference pairs for DPO
python scripts/prepare_dataset.py generate-preferences \
    --model-path ./checkpoints/mdlm_pretrain/last.ckpt \
    --input ./data/processed/msas.jsonl \
    --output ./data/preference_pairs.jsonl
```

### Pre-training

```bash
python scripts/train_mdlm.py --config configs/train_mdlm.yaml \
    --data.path ./data/openproteinset \
    --training.max_steps 100000
```

Key hyperparameters (from MSAGPT paper): batch=48 MSAs, lr=1.2e-4, AdamW (β₁=0.9, β₂=0.95).

### DPO Fine-tuning

```bash
python scripts/train_mdlm_dpo.py --config configs/train_dpo.yaml \
    --model.checkpoint_path ./checkpoints/mdlm_pretrain/last.ckpt \
    --data.path ./data/preference_pairs.jsonl
```

DPO hyperparameters: lr=1e-6, β=0.1 (temperature), λ=0.1 (CE regularization).

### Key Training Concepts

- **SUBS parameterization**: Model predicts x₀ directly given x_t
- **LogLinear noise**: σ(t) = 1 - (1-ε)·t
- **2D positions**: position_ids (within MSA) + block_position_ids (MSA index)
- **D3PO loss**: DPO adapted for diffusion via ELBO-based likelihood ratios
- **Context masking**: Query sequence is never masked during training

## Hardware Requirements

- Inference: 1x A100 (80GB) with BF16
- Fine-tuning: 4x A100 (80GB) with BF16
- Pre-training: 24x A100 (full OpenProteinSet)
