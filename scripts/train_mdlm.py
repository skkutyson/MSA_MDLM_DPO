#!/usr/bin/env python3
"""
MDLM Pre-training Script.

Train masked diffusion language models on protein MSA data.

Usage:
    python scripts/train_mdlm.py --config configs/train_mdlm.yaml
    python scripts/train_mdlm.py --config configs/train_mdlm.yaml --data.path ./data/small_subset

Example with overrides:
    python scripts/train_mdlm.py \
        --config configs/train_mdlm.yaml \
        --training.max_steps 1000 \
        --training.batch_size 16 \
        --hardware.devices 1
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model_utils.mdlm.protein_dit import ProteinDIT
from model_utils.mdlm.diffusion import ProteinDiffusion
from model_utils.mdlm.noise_schedule import get_noise_schedule
from training.datasets import MSADataset, MSACollator, create_dataloader
from training.trainers import MDLMTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load config from YAML file with optional overrides."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if overrides:
        # Apply nested overrides (e.g., "training.batch_size" -> config["training"]["batch_size"])
        for key, value in overrides.items():
            parts = key.split('.')
            d = config
            for part in parts[:-1]:
                if part not in d:
                    d[part] = {}
                d = d[part]
            d[parts[-1]] = value

    return config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train MDLM on protein MSA data')

    parser.add_argument('--config', type=str, default='configs/train_mdlm.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    # Allow overriding any config value via --key.subkey value
    # These are parsed dynamically

    args, unknown = parser.parse_known_args()

    # Parse unknown args as config overrides
    overrides = {}
    i = 0
    while i < len(unknown):
        if unknown[i].startswith('--'):
            key = unknown[i][2:]
            if i + 1 < len(unknown) and not unknown[i + 1].startswith('--'):
                value = unknown[i + 1]
                # Try to parse as number/bool
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        if value.lower() == 'true':
                            value = True
                        elif value.lower() == 'false':
                            value = False
                overrides[key] = value
                i += 2
            else:
                overrides[key] = True
                i += 1
        else:
            i += 1

    return args, overrides


def create_model(config: Dict[str, Any]) -> tuple:
    """Create MDLM model and noise schedule."""
    model_config = config['model']

    # Create backbone
    backbone = ProteinDIT(
        vocab_size=int(model_config.get('vocab_size', 128)),
        hidden_size=int(model_config.get('hidden_size', 1024)),
        num_heads=int(model_config.get('num_attention_heads', 16)),
        num_layers=int(model_config.get('num_layers', 24)),
        cond_dim=int(model_config.get('cond_dim', 256)),
        mlp_ratio=float(model_config.get('mlp_ratio', 4.0)),
        dropout=float(model_config.get('dropout', 0.0)),
    )

    # Create noise schedule
    diffusion_config = config.get('diffusion', {})
    noise_schedule = get_noise_schedule(
        model_config.get('noise_type', 'loglinear'),
        eps=float(diffusion_config.get('eps', 1e-3)),
    )

    # Create diffusion model
    diffusion = ProteinDiffusion(
        backbone=backbone,
        noise_schedule=noise_schedule,
        mask_token_id=int(model_config.get('mask_token_id', 36)),
        parameterization='subs',
    )

    logger.info(f"Created model with {sum(p.numel() for p in diffusion.parameters()):,} parameters")

    return diffusion, noise_schedule


def main():
    """Main training function."""
    args, overrides = parse_args()

    # Load config
    config = load_config(args.config, overrides)
    logger.info(f"Loaded config from {args.config}")

    # Create model
    model, noise_schedule = create_model(config)

    # Create datasets
    data_config = config['data']
    train_dataset = MSADataset(
        data_source=data_config['path'],
        max_seq_length=data_config.get('max_seq_length', 2048),
        max_msa_depth=data_config.get('max_msa_depth', 64),
        num_msa_sequences=data_config.get('num_msa_sequences', 8),
    )

    # Check for empty dataset
    if len(train_dataset) == 0:
        logger.error(f"No MSA files found in {data_config['path']}")
        logger.error("Please ensure the data directory contains .a3m or .fasta files.")
        logger.error("You can create test data with: python scripts/prepare_dataset.py process --input <src> --output <dst>")
        logger.error("Or use the test dataset: --data.path ./data/test_msa")
        sys.exit(1)

    # Create dataloaders
    train_config = config['training']
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=min(train_config.get('batch_size', 48), len(train_dataset)),
        shuffle=True,
        num_workers=config.get('hardware', {}).get('num_workers', 4),
        collate_fn=MSACollator(),
        pin_memory=True,
    )

    # Validation loader (optional)
    val_loader = None

    logger.info(f"Created dataloader with {len(train_dataset)} samples")

    # Check for Lightning
    try:
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
        from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
        HAS_LIGHTNING = True
    except ImportError:
        HAS_LIGHTNING = False
        logger.warning("PyTorch Lightning not found, using simple training loop")

    if HAS_LIGHTNING:
        # Create Lightning trainer
        ema_config = config.get('ema', {})
        trainer_module = MDLMTrainer(
            model=model.backbone,
            noise_schedule=noise_schedule,
            mask_token_id=config['model'].get('mask_token_id', 36),
            learning_rate=train_config.get('learning_rate', 1.2e-4),
            weight_decay=train_config.get('weight_decay', 0.01),
            betas=tuple(train_config.get('betas', [0.9, 0.95])),
            warmup_steps=int(train_config.get('max_steps', 100000) * train_config.get('warmup_ratio', 0.025)),
            max_steps=train_config.get('max_steps', 100000),
            ema_decay=ema_config.get('decay', 0.9999),
            use_ema=ema_config.get('enabled', True),
            eps=float(config.get('diffusion', {}).get('eps', 1e-3)),
        )

        # Callbacks
        ckpt_config = config.get('checkpoint', {})
        callbacks = [
            ModelCheckpoint(
                dirpath=ckpt_config.get('save_dir', './checkpoints/mdlm_pretrain'),
                filename='mdlm-{step:06d}-{val_loss:.4f}',
                every_n_train_steps=ckpt_config.get('save_every_n_steps', 1000),
                save_top_k=ckpt_config.get('save_top_k', 3),
                monitor=ckpt_config.get('monitor', 'train/loss'),
                mode='min',
                save_last=True,
            ),
            LearningRateMonitor(logging_interval='step'),
        ]

        # Logger
        log_config = config.get('logging', {})
        try:
            logger_instance = WandbLogger(
                project=log_config.get('project', 'mdlm-pretrain'),
                save_dir=ckpt_config.get('save_dir', './checkpoints'),
            )
        except Exception:
            logger_instance = TensorBoardLogger(
                save_dir=ckpt_config.get('save_dir', './checkpoints'),
                name='mdlm-pretrain',
            )

        # Hardware config
        hw_config = config.get('hardware', {})

        # Create Lightning trainer
        lightning_trainer = pl.Trainer(
            default_root_dir=ckpt_config.get('save_dir', './checkpoints/mdlm_pretrain'),
            max_steps=train_config.get('max_steps', 100000),
            max_epochs=train_config.get('max_epochs', -1),
            gradient_clip_val=train_config.get('gradient_clip_val', 1.0),
            accumulate_grad_batches=train_config.get('accumulate_grad_batches', 1),
            precision=train_config.get('precision', 'bf16-mixed'),
            devices=hw_config.get('devices', 1),
            strategy=hw_config.get('strategy', 'auto'),
            callbacks=callbacks,
            logger=logger_instance,
            log_every_n_steps=log_config.get('log_every_n_steps', 10),
        )

        # Train
        logger.info("Starting training...")
        lightning_trainer.fit(
            trainer_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=args.resume,
        )

    else:
        # Simple training loop without Lightning
        logger.info("Running simple training loop...")
        simple_train(model, noise_schedule, train_loader, config)


def simple_train(model, noise_schedule, train_loader, config):
    """Simple training loop without Lightning."""
    from training.losses.mdlm_loss import compute_diffusion_loss

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    noise_schedule = noise_schedule.to(device)

    train_config = config['training']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.get('learning_rate', 1.2e-4),
        weight_decay=train_config.get('weight_decay', 0.01),
        betas=tuple(train_config.get('betas', [0.9, 0.95])),
    )

    max_steps = train_config.get('max_steps', 100000)
    mask_token_id = config['model'].get('mask_token_id', 36)

    step = 0
    for epoch in range(1000):
        for batch in train_loader:
            if step >= max_steps:
                break

            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            result = compute_diffusion_loss(
                model=model.backbone,
                x0=batch['input_ids'],
                position_ids=batch['position_ids'],
                block_position_ids=batch['block_position_ids'],
                noise_schedule=noise_schedule,
                mask_token_id=mask_token_id,
                attention_mask=batch.get('attention_mask'),
                context_mask=batch.get('context_mask'),
            )

            result['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % 10 == 0:
                logger.info(f"Step {step}: loss={result['loss'].item():.4f}, acc={result['accuracy'].item():.4f}")

            step += 1

        if step >= max_steps:
            break

    # Save final checkpoint
    save_dir = config.get('checkpoint', {}).get('save_dir', './checkpoints/mdlm_pretrain')
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pt'))
    logger.info(f"Saved final model to {save_dir}")


if __name__ == '__main__':
    main()
