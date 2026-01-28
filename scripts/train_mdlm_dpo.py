#!/usr/bin/env python3
"""
MDLM DPO Fine-tuning Script.

Fine-tune MDLM using Direct Preference Optimization on preference pairs.

Usage:
    python scripts/train_mdlm_dpo.py --config configs/train_dpo.yaml
    python scripts/train_mdlm_dpo.py \
        --config configs/train_dpo.yaml \
        --model.checkpoint_path ./checkpoints/mdlm_pretrain/last.ckpt \
        --data.path ./data/preference_pairs.jsonl

Example with overrides:
    python scripts/train_mdlm_dpo.py \
        --config configs/train_dpo.yaml \
        --dpo.beta 0.05 \
        --training.learning_rate 5e-7
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
from training.datasets import MSAPreferenceDataset, PreferenceCollator, create_preference_dataloader
from training.trainers import MDLMDPOTrainer

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
    parser = argparse.ArgumentParser(description='DPO fine-tune MDLM on preference pairs')

    parser.add_argument('--config', type=str, default='configs/train_dpo.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    args, unknown = parser.parse_known_args()

    # Parse unknown args as config overrides
    overrides = {}
    i = 0
    while i < len(unknown):
        if unknown[i].startswith('--'):
            key = unknown[i][2:]
            if i + 1 < len(unknown) and not unknown[i + 1].startswith('--'):
                value = unknown[i + 1]
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
    """Create or load MDLM model."""
    model_config = config['model']

    # Create backbone
    backbone = ProteinDIT(
        vocab_size=int(model_config.get('vocab_size', 128)),
        hidden_size=int(model_config.get('hidden_size', 1024)),
        num_heads=int(model_config.get('num_attention_heads', 16)),
        num_layers=int(model_config.get('num_layers', 24)),
        cond_dim=int(model_config.get('cond_dim', 256)),
        mlp_ratio=float(model_config.get('mlp_ratio', 4.0)),
        dropout=0.0,  # No dropout during fine-tuning
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

    # Load pretrained weights if specified
    checkpoint_path = model_config.get('checkpoint_path')
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading pretrained weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Remove 'model.' prefix if present (from Lightning)
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        else:
            state_dict = checkpoint

        diffusion.load_state_dict(state_dict, strict=False)
        logger.info("Loaded pretrained weights successfully")

    logger.info(f"Model has {sum(p.numel() for p in diffusion.parameters()):,} parameters")

    return diffusion, noise_schedule


def main():
    """Main DPO training function."""
    args, overrides = parse_args()

    # Load config
    config = load_config(args.config, overrides)
    logger.info(f"Loaded config from {args.config}")

    # Create model
    model, noise_schedule = create_model(config)

    # Create dataset
    data_config = config['data']
    train_dataset = MSAPreferenceDataset(
        data_path=data_config['path'],
        max_seq_length=data_config.get('max_seq_length', 2048),
        max_msa_sequences=data_config.get('max_msa_sequences', 8),
    )

    # Config references
    train_config = config['training']
    hw_config = config.get('hardware', {})

    logger.info(f"Created dataloader with {len(train_dataset)} preference pairs")

    # Check for Lightning
    try:
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
        from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
        HAS_LIGHTNING = True
    except ImportError:
        HAS_LIGHTNING = False
        logger.warning("PyTorch Lightning not found, using simple training loop")

    dpo_config = config.get('dpo', {})

    if HAS_LIGHTNING:
        # Create DPO trainer module
        trainer_module = MDLMDPOTrainer(
            model=model.backbone,
            noise_schedule=noise_schedule,
            mask_token_id=int(config['model'].get('mask_token_id', 36)),
            beta=float(dpo_config.get('beta', 0.1)),
            lambda_ce=float(dpo_config.get('lambda_ce', 0.1)),
            learning_rate=float(train_config.get('learning_rate', 1e-6)),
            weight_decay=float(train_config.get('weight_decay', 0.01)),
            betas=tuple(float(b) for b in train_config.get('betas', [0.9, 0.95])),
            warmup_steps=int(train_config.get('warmup_steps', 100)),
            max_steps=int(train_config.get('max_steps', -1)),
            eps=float(config.get('diffusion', {}).get('eps', 1e-3)),
        )

        # Create validation dataloader (use 10% of data)
        val_size = max(1, len(train_dataset) // 10)
        train_size = len(train_dataset) - val_size
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )

        train_loader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=int(train_config.get('batch_size', 1)),
            shuffle=True,
            num_workers=int(hw_config.get('num_workers', 4)),
            collate_fn=PreferenceCollator(),
            pin_memory=True,
        )

        val_loader = torch.utils.data.DataLoader(
            val_subset,
            batch_size=int(train_config.get('batch_size', 1)),
            shuffle=False,
            num_workers=int(hw_config.get('num_workers', 4)),
            collate_fn=PreferenceCollator(),
            pin_memory=True,
        )

        logger.info(f"Split data: {train_size} train, {val_size} validation")

        # Callbacks
        ckpt_config = config.get('checkpoint', {})
        callbacks = [
            ModelCheckpoint(
                dirpath=ckpt_config.get('save_dir', './checkpoints/mdlm_dpo'),
                filename='mdlm-dpo-{step:06d}-{train_accuracy:.4f}',
                every_n_train_steps=int(ckpt_config.get('save_every_n_steps', 100)),
                save_top_k=int(ckpt_config.get('save_top_k', 3)),
                monitor=ckpt_config.get('monitor', 'train/accuracy'),
                mode='max',
                save_last=True,
            ),
            LearningRateMonitor(logging_interval='step'),
        ]

        # Logger
        log_config = config.get('logging', {})
        try:
            logger_instance = WandbLogger(
                project=log_config.get('project', 'mdlm-dpo'),
                save_dir=ckpt_config.get('save_dir', './checkpoints'),
            )
        except Exception:
            logger_instance = TensorBoardLogger(
                save_dir=ckpt_config.get('save_dir', './checkpoints'),
                name='mdlm-dpo',
            )

        # Create Lightning trainer
        lightning_trainer = pl.Trainer(
            default_root_dir=ckpt_config.get('save_dir', './checkpoints/mdlm_dpo'),
            max_epochs=train_config.get('max_epochs', 1),
            max_steps=train_config.get('max_steps', -1),
            gradient_clip_val=train_config.get('gradient_clip_val', 1.0),
            accumulate_grad_batches=train_config.get('accumulate_grad_batches', 8),
            precision=train_config.get('precision', 'bf16-mixed'),
            devices=hw_config.get('devices', 1),
            strategy=hw_config.get('strategy', 'auto'),
            callbacks=callbacks,
            logger=logger_instance,
            log_every_n_steps=log_config.get('log_every_n_steps', 1),
        )

        # Train
        logger.info("Starting DPO training...")
        lightning_trainer.fit(
            trainer_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=args.resume,
        )

        # Save final model
        save_dir = ckpt_config.get('save_dir', './checkpoints/mdlm_dpo')
        final_path = os.path.join(save_dir, 'mdlm_dpo_final.pt')
        torch.save(trainer_module.model.state_dict(), final_path)
        logger.info(f"Saved final model to {final_path}")

    else:
        # Simple training loop - create dataloader here
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=int(train_config.get('batch_size', 1)),
            shuffle=True,
            num_workers=int(hw_config.get('num_workers', 4)),
            collate_fn=PreferenceCollator(),
            pin_memory=True,
        )
        logger.info("Running simple DPO training loop...")
        simple_dpo_train(model, noise_schedule, train_loader, config)


def simple_dpo_train(model, noise_schedule, train_loader, config):
    """Simple DPO training loop without Lightning."""
    from training.trainers.mdlm_dpo_trainer import SimpleDPOTrainer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dpo_config = config.get('dpo', {})
    train_config = config['training']

    trainer = SimpleDPOTrainer(
        model=model.backbone,
        noise_schedule=noise_schedule,
        mask_token_id=int(config['model'].get('mask_token_id', 36)),
        beta=float(dpo_config.get('beta', 0.1)),
        lambda_ce=float(dpo_config.get('lambda_ce', 0.1)),
        learning_rate=float(train_config.get('learning_rate', 1e-6)),
        weight_decay=float(train_config.get('weight_decay', 0.01)),
        device=str(device),
    )

    max_epochs = train_config.get('max_epochs', 1)
    step = 0

    for epoch in range(max_epochs):
        for batch in train_loader:
            result = trainer.train_step(batch)

            if step % 1 == 0:
                logger.info(
                    f"Step {step}: loss={result['loss']:.4f}, "
                    f"dpo={result['dpo_loss']:.4f}, "
                    f"ce={result['ce_loss']:.4f}, "
                    f"acc={result['accuracy']:.4f}"
                )

            step += 1

    # Save final model
    save_dir = config.get('checkpoint', {}).get('save_dir', './checkpoints/mdlm_dpo')
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_checkpoint(os.path.join(save_dir, 'mdlm_dpo_final.pt'))
    logger.info(f"Saved final model to {save_dir}")


if __name__ == '__main__':
    main()
