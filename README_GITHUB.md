# MSAGPT: Masked Diffusion Language Model for Proteins

A diffusion-based language model trained on protein Multiple Sequence Alignments (MSAs) for generative protein design.

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/MSAGPT.git
cd MSAGPT
pip install -r requirements.txt
```

### Training

```bash
# Prepare data (convert raw MSAs to processed JSONL)
python scripts/prepare_dataset.py process \
    --input ./data/openproteinset \
    --output ./data/processed

# Train model
python scripts/train_mdlm.py \
    --config configs/train_mdlm.yaml \
    --data.path ./data/processed/msas.jsonl \
    --training.max_steps 100000
```

### Inference

```bash
python scripts/generate.py \
    --model-path ./checkpoints/mdlm_pretrain/last.ckpt \
    --query MVLKKVDPL \
    --num-samples 5 \
    --num-steps 50
```

## Architecture

### Model Components

- **ProteinDIT**: Diffusion Transformer with 1024 hidden size, 16 attention heads, 24 layers
- **Noise Schedule**: LogLinear diffusion schedule
- **Conditioning**: Timestep-conditioned via AdaLN (Adaptive Layer Normalization)
- **Position Embeddings**: 2D Rotary embeddings for (sequence position, MSA block position)

### Input Format

```
[gMASK, sop, query_tokens..., <M>, msa1_tokens..., <M>, msa2_tokens...]
  ↑     ↑     ↑                ↑    ↑
  start  seq-op  context       delimiter  generation targets
  (frozen during denoising)
```

## Key Features

- **Masked Diffusion**: Learns to denoise masked MSA sequences given query context
- **Few-Shot Learning**: Condition on 2-3 example MSAs for improved generation
- **DPO Fine-tuning**: Direct Preference Optimization for preference-aligned generation
- **Distributed Training**: DDP support for multi-GPU training
- **BF16 Mixed Precision**: Efficient training with automatic mixed precision

## Configuration

Main config: `configs/train_mdlm.yaml`

```yaml
model:
  hidden_size: 1024
  num_attention_heads: 16
  num_layers: 24

training:
  batch_size: 2
  learning_rate: 1.2e-4
  max_steps: 100000
  precision: bf16-mixed

data:
  max_seq_length: 2048
  max_msa_depth: 64
  num_msa_sequences: 8
```

## Project Structure

```
MSAGPT/
├── configs/              # Configuration files
├── scripts/              # Training, inference, data preparation
├── model_utils/          # Model implementations (ProteinDIT, DIT blocks)
├── training/
│   ├── datasets/         # Data loading (MSADataset, OpenProteinSet)
│   ├── trainers/         # Training logic (MDLMTrainer)
│   └── losses/           # Loss functions
├── checkpoints/          # Model checkpoints
└── data/                 # Data directory
```

## Data Format

Input: MSA files (`.a3m`, `.fasta`)

Processed: JSONL format (recommended for faster loading)
```json
{
  "id": "1abc_A",
  "query": "MVLKKVDPL...",
  "sequences": ["MVLKKVDPM...", "MVLKKIDPL...", ...],
  "depth": 156,
  "length": 245
}
```

## Performance

- **Throughput**: ~1-2 seconds/batch (batch_size=2, max_seq_length=2048)
- **Memory**: ~8GB VRAM (single GPU)
- **Training Time**: ~100 hours for 100K steps on H100

## Citation

```bibtex
@article{msagpt2024,
  title={MSAGPT: Protein Generation via Masked Diffusion Language Modeling},
  author={...},
  year={2024}
}
```

## License

MIT

## Contact

For questions or issues, please open a GitHub issue or contact us at [contact info].

---

**For detailed technical documentation, see [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md)**
