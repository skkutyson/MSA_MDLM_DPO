# MDLM Integration into MSAGPT: Technical Documentation

## 1. Architecture Overview

The original MSAGPT uses **autoregressive generation** (token-by-token) via SwissArmyTransformer (SAT). The MDLM integration adds a **parallel diffusion-based generation** alternative that generates all MSA positions simultaneously through iterative denoising.

```
Original MSAGPT Flow:
  Input → [gMASK][sop]query<M> → AR Model → token₁ → token₂ → ... → tokenₙ

MDLM Flow:
  Input → [gMASK][sop]query<M>[MASK][MASK]...<M>[MASK]...
        → Diffusion (t=1→0) → all tokens revealed simultaneously
```

---

## 2. Core Model Components (`model_utils/mdlm/`)

### 2.1 ProteinDIT (`protein_dit.py`)
A Diffusion Transformer backbone adapted for proteins:

```python
class ProteinDIT(nn.Module):
    def __init__(self, vocab_size=128, hidden_size=1024, num_heads=16, num_layers=24, ...):
        self.token_embed = nn.Embedding(vocab_size, hidden_size)
        self.time_embed = TimestepEmbedder(cond_dim)  # Diffusion timestep conditioning
        self.rotary_2d = ProteinRotary2D(head_dim)    # 2D rotary for MSA structure
        self.blocks = nn.ModuleList([DiTBlock(...) for _ in range(num_layers)])
        self.output_proj = nn.Linear(hidden_size, vocab_size)
```

**Key design**: Uses 2D rotary embeddings where:
- First half of head dimension: position within MSA sequence (0 to L-1)
- Second half: MSA block index (which MSA in the alignment)

### 2.2 ProteinDiffusion (`diffusion.py`)
Wraps the backbone with diffusion-specific logic:

```python
class ProteinDiffusion(nn.Module):
    def __init__(self, backbone, noise_schedule, mask_token_id=36, parameterization='subs'):
        # SUBS parameterization: model directly predicts x₀ given x_t

    def forward(self, x0, sigma, position_ids, block_position_ids, context_mask=None):
        # 1. Apply noise: x_t = mask tokens where random < sigma
        # 2. Preserve context (query) using context_mask
        # 3. Forward through backbone
        # 4. Return log probabilities
```

**Masking process** (forward diffusion):
```python
def q_xt(self, x0, move_chance):
    # move_chance = sigma (noise level)
    move_indices = torch.rand(*x0.shape) < move_chance
    x_t = torch.where(move_indices, self.mask_token_id, x0)
    return x_t
```

### 2.3 Noise Schedule (`noise_schedule.py`)
LogLinear schedule from MDLM paper:

```python
class LogLinearNoise:
    def __call__(self, t):
        # σ(t) = 1 - (1-ε)·t
        # At t=1: σ=ε (almost no masking)
        # At t=0: σ≈1 (fully masked)
        return 1 - (1 - self.eps) * t
```

### 2.4 MSAGPT_MDLM Wrapper (`model_msagpt_mdlm.py`)
SAT-compatible interface for CLI integration:

```python
class MSAGPT_MDLM(nn.Module):
    def generate(self, context_tokens, msa_len, max_gen_length, num_steps=256, ...):
        # 1. Initialize with context + masked generation region
        full_tokens[:, :context_len] = context_tokens
        full_tokens[:, context_len:] = mask_token_id

        # 2. Pre-place <M> delimiters at block boundaries (key fix!)
        for msa_idx in range(num_gen_msa):
            delimiter_pos = gen_start + (msa_idx + 1) * msa_len - 1
            full_tokens[:, delimiter_pos] = msa_delimiter_id  # Token 35
            context_mask[:, delimiter_pos] = 1  # Don't overwrite

        # 3. Build 2D position IDs
        position_ids, block_position_ids = self._build_2d_positions(...)

        # 4. Run diffusion sampling
        return self.diffusion.sample_ddpm_caching(...)
```

---

## 3. Sampling Methods (`diffusion.py`)

### 3.1 Standard DDPM Sampling
```python
def sample(self, batch_size, seq_len, num_steps, ...):
    x = torch.full(..., mask_token_id)  # Start fully masked

    for step in range(num_steps):
        t = 1.0 - step / num_steps
        sigma = self.noise_schedule(t)

        # Get predictions
        logits = self.get_logits(x, sigma, position_ids, block_position_ids)

        # Apply invalid token masking
        for token_id in invalid_token_ids:
            logits[..., token_id] = float('-inf')

        # Sample
        probs = F.softmax(logits / temperature, dim=-1)
        x_pred = torch.multinomial(probs.view(-1, vocab_size), 1)

        # Re-mask some predictions (not final step)
        if step < num_steps - 1:
            remask = torch.rand_like(x.float()) < sigma_next
            x = torch.where(remask, mask_token_id, x_pred)
        else:
            x = x_pred

        # Restore context
        x = torch.where(context_mask.bool(), context_tokens, x)
```

### 3.2 DDPM with Caching (faster)
Tracks which positions are "revealed" and only re-masks unrevealed positions:
```python
def sample_ddpm_caching(self, ...):
    revealed = torch.zeros_like(x, dtype=torch.bool)

    for step in range(num_steps):
        # Only update unrevealed positions
        # Progressively reveal based on confidence scores
        ...
```

---

## 4. CLI Integration (`cli_sat.py`)

### 4.1 Argument Parsing
```python
py_parser.add_argument("--backbone", type=str, default="gpt", choices=["gpt", "mdlm"])
py_parser.add_argument("--num-diffusion-steps", type=int, default=256)
py_parser.add_argument("--diffusion-sampler", type=str, default="ddpm_cache")
```

### 4.2 Model Loading
```python
if args.backbone == 'mdlm':
    model, args = MSAGPT_MDLM.from_pretrained(args.from_pretrained, args, ...)
else:
    model, args = MSAGPT.from_pretrained(args.from_pretrained, args, ...)
```

### 4.3 Strategy Selection
```python
if args.backbone == 'mdlm':
    # Exclude invalid tokens (SAT padding 37-127, special tokens)
    invalid_slices.extend([33, 34, 36])  # eop, eos, DIFFUSION_MASK
    invalid_slices.extend(list(range(37, 128)))

    strategy = get_diffusion_strategy(
        strategy_name=args.diffusion_sampler,
        num_steps=args.num_diffusion_steps,
        temperature=args.temperature,
        invalid_slices=invalid_slices,
    )
else:
    strategy = AdvancedBaseStrategy(...)  # or BeamSearchStrategy
```

---

## 5. Generation Pipeline (`utils/chat.py`)

### 5.1 Diffusion Sampling Function
```python
def diffusion_sampling(args, raw_text, model, tokenizer, strategy):
    # Build context: [gMASK] + [sop] + query + [<M>] + prompts...
    seq = [tokenizer.get_command("[gMASK]"), tokenizer.get_command("sop")]
    for each in raw_text:
        seq += tokenizer.tokenize(each) + [tokenizer.get_command('<M>')]

    context_tokens = torch.tensor([seq], device=args.device)
    msa_len = len(raw_text[0]) + 1  # +1 for <M> delimiter

    # Generate
    output = strategy.generate(
        model=model,
        context_tokens=context_tokens,
        msa_len=msa_len,
        max_gen_length=max_gen_length,
        msa_delimiter_id=tokenizer.get_command('<M>'),
    )

    # Filter invalid tokens before detokenization
    output = [t if (1 <= t <= 25 or t == 27 or t in {33,34,35,36}) else 27 for t in output]

    return tokenizer.detokenize(output)
```

---

## 6. Training Infrastructure (`training/`)

### 6.1 Dataset Classes
```python
class MSADataset:
    # Pre-training: returns tokenized MSAs with 2D positions
    def __getitem__(self, idx):
        return {
            'input_ids': ...,          # [gMASK, sop, query, <M>, msa1, <M>, ...]
            'position_ids': ...,        # Within-MSA position (0 to L-1)
            'block_position_ids': ...,  # MSA index (0, 1, 2, ...)
            'context_mask': ...,        # 1 for query, 0 for generation targets
        }

class MSAPreferenceDataset:
    # DPO: returns winner/loser pairs
    def __getitem__(self, idx):
        return {
            'winner_input_ids': ..., 'loser_input_ids': ...,
            'winner_position_ids': ..., 'loser_position_ids': ...,
            ...
        }
```

### 6.2 Loss Functions
```python
# Pre-training: Diffusion NLL
def compute_diffusion_loss(model, x0, position_ids, block_position_ids, noise_schedule, ...):
    t = torch.rand(batch_size)
    sigma = noise_schedule(t)
    log_probs = model.forward(x0, sigma, position_ids, block_position_ids, context_mask)
    loss = F.nll_loss(log_probs.view(-1, vocab_size), x0.view(-1))
    return loss

# DPO: D3PO for diffusion
def compute_dpo_loss(winner_batch, loser_batch, policy_model, ref_model, beta=0.1):
    # Sample shared timestep
    t = torch.rand(batch_size)

    # Compute log-likelihoods
    winner_log_prob = compute_log_prob(winner_batch, policy_model)
    loser_log_prob = compute_log_prob(loser_batch, policy_model)
    winner_log_prob_ref = compute_log_prob(winner_batch, ref_model)  # frozen
    loser_log_prob_ref = compute_log_prob(loser_batch, ref_model)

    # Bradley-Terry loss
    winner_ratio = winner_log_prob - winner_log_prob_ref
    loser_ratio = loser_log_prob - loser_log_prob_ref
    loss = -F.logsigmoid(beta * (winner_ratio - loser_ratio))
    return loss
```

---

## 7. Key Bug Fixes

### 7.1 Vocab Size Mismatch
**Problem**: SAT sets `args.vocab_size=100` (unpadded), but model needs 128 (padded).
```python
# Before (bug):
vocab_size = getattr(args, 'vocab_size', 128)  # Gets 100 from SAT

# After (fix):
vocab_size = 128  # Always use padded size
```

### 7.2 MSA Structure Preservation
**Problem**: Diffusion generates random tokens without MSA structure.
```python
# Fix: Pre-place <M> delimiters at block boundaries
for msa_idx in range(num_gen_msa):
    delimiter_pos = gen_start + (msa_idx + 1) * msa_len - 1
    full_tokens[:, delimiter_pos] = msa_delimiter_id
    context_mask[:, delimiter_pos] = 1  # Don't overwrite during sampling
```

### 7.3 Invalid Token Filtering
**Problem**: SAT padding tokens (37-127) cause detokenization errors.
```python
# Filter before detokenization
valid_special = {33, 34, 35, 36}
output = [t if (1 <= t <= 25 or t == 27 or t in valid_special) else 27 for t in output]
```

---

## 8. File Summary

| File | Purpose |
|------|---------|
| `model_utils/mdlm/protein_dit.py` | DiT backbone with 2D rotary embeddings |
| `model_utils/mdlm/diffusion.py` | Forward diffusion, loss, DDPM sampling |
| `model_utils/mdlm/noise_schedule.py` | LogLinear noise schedule |
| `model_utils/mdlm/model_msagpt_mdlm.py` | SAT-compatible wrapper, generate() |
| `utils/diffusion_sampling.py` | Strategy classes for CLI compatibility |
| `utils/chat.py` | `diffusion_sampling()` function |
| `cli_sat.py` | `--backbone mdlm` argument handling |
| `training/` | Datasets, losses, trainers for MDLM |

---

## 9. Usage

### CLI Inference
```bash
# Interactive chat with MDLM
bash scripts/cli_sat.sh \
    --from_pretrained ./checkpoints/mdlm_dpo \
    --backbone mdlm \
    --input-source chat \
    --stream_chat \
    --max-gen-length 512 \
    --num-diffusion-steps 256 \
    --diffusion-sampler ddpm_cache
```

### Python API
```python
from model_utils.mdlm import MSAGPT_MDLM
from utils.tokenization import proteinglm_tokenizer
import torch

# Load model
model, args = MSAGPT_MDLM.from_pretrained('./checkpoints/mdlm_dpo', args)
model = model.cuda().eval()
tokenizer = proteinglm_tokenizer()

# Generate
query = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTK"
seq = [tokenizer.get_command('[gMASK]'), tokenizer.get_command('sop')]
seq += tokenizer.tokenize(query) + [tokenizer.get_command('<M>')]
context_tokens = torch.tensor([seq]).cuda()

output = model.generate(
    context_tokens=context_tokens,
    msa_len=len(query) + 1,
    max_gen_length=512,
    num_steps=256,
    msa_delimiter_id=tokenizer.get_command('<M>'),
)
print(tokenizer.detokenize(output[0].tolist()))
```

### Training
```bash
# Pre-training
python scripts/train_mdlm.py --config configs/train_mdlm.yaml \
    --data.path ./data/openproteinset

# DPO fine-tuning
python scripts/train_mdlm_dpo.py --config configs/train_dpo.yaml \
    --model.checkpoint_path ./checkpoints/mdlm_pretrain/last.ckpt \
    --data.path ./data/preference_pairs.jsonl
```

---

## 10. References

- [MSAGPT Paper](https://arxiv.org/abs/2406.05347) - Original MSAGPT methodology
- [MDLM Paper](https://arxiv.org/abs/2406.07524) - Masked Diffusion Language Models
- [D3PO Paper](https://arxiv.org/abs/2503.08295) - DPO for discrete diffusion
- [OpenProteinSet](https://registry.opendata.aws/openfold/) - Training data source
