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

**CRITICAL**: MDLM uses a **BACKWARDS noise schedule** where small t = high σ = maximum masking. This is the opposite of standard diffusion. See [MDLM_CONVENTION_CLARIFICATION.md](MDLM_CONVENTION_CLARIFICATION.md) before reading further.

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
    def forward(self, t):
        # σ(t) = 1 - (1-ε)·t
        #
        # IMPORTANT: This is a REVERSED convention!
        # t ~ U(ε, 1) during training (sampled randomly)
        # 
        # Small t (t≈ε):     σ ≈ 1   (MAXIMUM masking)
        # Large t (t≈1):     σ ≈ 0   (MINIMUM masking)
        #
        # This is BACKWARDS from standard diffusion where
        # larger t = more noise. See MDLM_CONVENTION_CLARIFICATION.md
        return 1 - (1 - self.eps) * t
```

**Key Understanding**: 
- σ represents the **amount of masking** applied to the input
- σ = 0: ~0% of tokens masked (input is mostly original)
- σ = 1: ~100% of tokens masked (input is all MASK tokens)

During **training**: t sampled from U(ε, 1), giving varied σ values
During **sampling**: t goes from 1 → 0, which INCREASES σ (more re-masking)

See [MDLM_CONVENTION_CLARIFICATION.md](MDLM_CONVENTION_CLARIFICATION.md) for complete explanation of why this backwards convention works.

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

### 3.1 DDPM Sampling with Denoising and Re-masking

The actual MDLM sampling process follows this pattern:

```python
def sample(self, batch_size, seq_len, num_steps=256, ...):
    # Step 1: Initialize with all mask tokens (t=0, fully masked)
    x = torch.full((batch_size, seq_len), mask_token_id)
    
    # Step 2: Iterate from t=1 (nearly clean) to t=0 (fully masked)
    #         This is DENOISING - gradually revealing the original tokens
    dt = 1.0 / num_steps  # e.g., dt = 1/256
    
    for step in range(num_steps):
        t = 1.0 - step * dt  # t: 1.0 → 0.9961 → ... → 0.0
        sigma = noise_schedule(t)  # σ(t) = 1-(1-ε)·t
        
        # Step 2a: Get predictions from model
        # Model sees partially masked input x with noise level sigma
        logits = model(x, sigma=sigma, position_ids=..., block_position_ids=...)
        
        # Step 2b: Sample predictions
        probs = softmax(logits / temperature)
        x_pred = multinomial_sample(probs)  # Predicted tokens at this step
        
        # Step 2c: DDPM UPDATE - This is the KEY part
        # Decide whether to keep the prediction or re-mask it
        if step < num_steps - 1:  # Not the last step
            t_next = 1.0 - (step + 1) * dt  # Next timestep (even closer to t=0)
            sigma_next = noise_schedule(t_next)  # Even lower noise
            
            # move_chance = σ_next: probability of keeping mask token
            # σ_next is the noise level for the NEXT step
            # As t decreases, σ INCREASES (not decreases!)
            # σ(t) = 1 - (1-ε)·t, so at t=0: σ=1, at t=1: σ≈0
            move_chance = sigma_next
            
            # Re-mask some predictions with probability σ_next
            # Early steps (high t): σ_next ≈ 0.001, very few re-masks
            # Late steps (low t): σ_next ≈ 1.0, most get re-masked
            # This is CONFIDENCE-BASED filtering!
            remask = rand() < move_chance
            x = where(remask, mask_token_id, x_pred)
        else:
            # Final step: use predictions directly without re-masking
            x = x_pred
        
        # Step 2d: Always restore context tokens
        # Query and few-shot examples never get masked
        x = where(context_mask, context_tokens, x)
    
    return x
```

**Visualization of the re-masking pattern**:
```
Early steps (t close to 1.0):
  σ_next ≈ 0.001 (low)
  move_chance ≈ 0.001
  Re-mask ~0.1% of predictions
  Keep ~99.9% of model outputs
  
Middle steps (t ≈ 0.5):
  σ_next ≈ 0.5
  move_chance ≈ 0.5
  Re-mask ~50% of predictions
  Keep ~50% of model outputs
  
Late steps (t close to 0):
  σ_next ≈ 1.0 (high)
  move_chance ≈ 1.0
  Re-mask ~100% of unselected tokens
  Only keep final confident predictions
```

**The core MDLM mechanism (as you correctly described)**:
1. **Model predicts original tokens from noisy input**: x_pred = sample from p(x_0|x_t)
2. **Stochastic re-masking based on sigma_next**: Some predictions are masked again with probability σ_next
3. **Progressive acceptance**: Early predictions are tentatively kept, late predictions must be confident
4. **Final step**: No re-masking, final predictions are kept

See [MDLM_NOISE_AND_SAMPLING_DETAILED.md](MDLM_NOISE_AND_SAMPLING_DETAILED.md) for complete analysis with numerical examples.

---

## 3.3 Batch Processing and Loss Calculation

**Important**: Despite using random t sampling, training covers all noise levels adequately.

```python
# In each training step:
batch_size = 32

# Each sample gets different t:
t = torch.rand(batch_size) * (1 - eps) + eps
# Example: [0.1, 0.8, 0.3, 0.6, 0.2, 0.9, ...]

# Different σ for each sample:
sigma = noise_schedule(t)
# Example: [0.9, 0.2, 0.7, 0.4, 0.8, 0.1, ...]

# Each sample gets different masking amount
mask = torch.rand_like(x0.float()) < sigma.unsqueeze(-1)

# Loss is AVERAGED over batch
loss = (nll * loss_mask).sum() / (loss_mask.sum())  # Scalar
```

**Over many batches**:
```
Batch 1: 32 samples with random t ∈ [ε, 1]
Batch 2: 32 samples with random t ∈ [ε, 1]
...
After 100K steps: 3.2M samples with uniform σ distribution
```

All noise levels are covered. Training is sufficient!

See [TRAINING_BATCH_AND_ACCUMULATION.md](TRAINING_BATCH_AND_ACCUMULATION.md) for detailed batch, loss, and gradient accumulation explanation.

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

**Key difference**: MDLM training is fundamentally different from autoregressive training.

```python
# Training: Sample timestep, apply noise, predict original
def compute_diffusion_loss(model, x0, position_ids, block_position_ids, noise_schedule, ...):
    batch_size, seq_len = x0.shape
    device = x0.device
    
    # Step 1: Sample random timesteps for this batch
    # t ~ U(eps, 1): higher t = less noise, lower t = more noise
    t = torch.rand(batch_size, device=device) * (1 - eps) + eps
    sigma = noise_schedule(t)  # σ(t) = 1-(1-ε)·t
    
    # Step 2: Apply noise (masking) to create x_t from x_0
    # move_chance = sigma: probability of masking each token
    move_chance = sigma.unsqueeze(-1)
    
    # Don't mask context tokens (query, few-shot examples)
    if context_mask is not None:
        move_chance = move_chance * (1 - context_mask.float())
    
    # Step 3: Create noisy input x_t
    # Mask tokens where random < move_chance
    mask = torch.rand_like(x0.float()) < move_chance
    x_t = torch.where(mask, mask_token_id, x0)
    
    # Restore context tokens (never masked during training)
    if context_mask is not None:
        x_t = torch.where(context_mask.bool(), x0, x_t)
    
    # Step 4: Forward pass
    # Model sees noisy input x_t and must predict original tokens x_0
    logits = model(
        input_ids=x_t,
        sigma=sigma,
        position_ids=position_ids,
        block_position_ids=block_position_ids,
    )
    
    # Step 5: Compute loss
    # Cross-entropy between predicted and true original tokens
    log_probs = F.log_softmax(logits, dim=-1)
    nll = F.nll_loss(log_probs.view(-1, vocab_size), x0.view(-1), reduction='none')
    
    # Only compute loss on non-context tokens
    loss_mask = 1 - context_mask.float() if context_mask is not None else 1.0
    loss = (nll * loss_mask).sum() / (loss_mask.sum() + 1e-8)
    
    return {'loss': loss, 'accuracy': accuracy, ...}
```

**Key insight about training vs sampling**:

| Phase | Input | Model Task | What happens after |
|-------|-------|-----------|-------------------|
| **Training** | x_t (noisy) | Predict x_0 | Compute cross-entropy loss |
| **Sampling** | x_t (noisy) | Predict x_0 | Sample predictions, re-mask with probability σ_next, iterate |

The model is trained to predict the original tokens from noisy input. During sampling, we use this prediction ability iteratively, with progressive re-masking.

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
