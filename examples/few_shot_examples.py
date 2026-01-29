#!/usr/bin/env python3
"""
Few-Shot Learning Examples and Tests for MDLM

Demonstrates how to use the few-shot learning feature for different scenarios.
"""

import torch
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.datasets.msa_dataset import MSADataset, MSATokenizer


def example_1_standard_mode():
    """표준 모드: Query만 사용"""
    print("=" * 60)
    print("Example 1: Standard Mode (Query only)")
    print("=" * 60)
    
    dataset = MSADataset(
        data_source='./data/openproteinset',
        max_seq_length=2048,
        max_msa_depth=64,
        num_msa_sequences=8,
        use_few_shot=False,  # ← 표준 모드
    )
    
    if len(dataset) > 0:
        sample = dataset[0]
        
        # Context mask 분석
        context_mask = sample['context_mask']
        input_ids = sample['input_ids']
        
        context_count = (context_mask == 1).sum().item()
        generation_count = (context_mask == 0).sum().item()
        
        print(f"Sample length: {len(input_ids)}")
        print(f"Context tokens (query + delimiters): {context_count}")
        print(f"Generation tokens (MSAs): {generation_count}")
        print(f"Context ratio: {context_count / len(input_ids) * 100:.1f}%")
        print(f"\nContext mask: {context_mask[:50]}...")
        print("Pattern: [1, 1, 1, ..., 0, 0, 0, ...]")
        print("         └─ Query     └─ MSAs to generate")
    else:
        print("No data found. Please download data first.")
    
    print()


def example_2_fewshot_2():
    """Few-shot 모드: 2개 예제"""
    print("=" * 60)
    print("Example 2: Few-Shot Mode (2 examples)")
    print("=" * 60)
    
    dataset = MSADataset(
        data_source='./data/openproteinset',
        max_seq_length=2048,
        max_msa_depth=64,
        num_msa_sequences=6,  # ← 감소 (메모리)
        use_few_shot=True,     # ← Few-shot 활성화!
        few_shot_examples=2,   # ← 2개 예제
    )
    
    if len(dataset) > 0:
        sample = dataset[0]
        
        context_mask = sample['context_mask']
        input_ids = sample['input_ids']
        block_position_ids = sample['block_position_ids']
        
        context_count = (context_mask == 1).sum().item()
        generation_count = (context_mask == 0).sum().item()
        
        # Block ID로 어떤 것이 예제인지 확인
        unique_blocks = torch.unique(block_position_ids)
        
        print(f"Sample length: {len(input_ids)}")
        print(f"Context tokens (query + examples): {context_count}")
        print(f"Generation tokens (target MSAs): {generation_count}")
        print(f"Context ratio: {context_count / len(input_ids) * 100:.1f}%")
        print(f"\nNumber of blocks (query + examples + targets): {len(unique_blocks)}")
        print(f"Block IDs: {unique_blocks.tolist()}")
        print(f"\nContext mask pattern:")
        print(f"  Block 0 (Query): context_mask=1")
        print(f"  Block 1 (Example 1): context_mask=1  ← Few-shot!")
        print(f"  Block 2 (Example 2): context_mask=1  ← Few-shot!")
        print(f"  Block 3+ (Targets): context_mask=0")
        print(f"\nContext mask: {context_mask[:100]}...")
        print("Pattern: [1, 1, 1, ..., 1, 1, 1, ..., 0, 0, 0, ...]")
        print("         └─ Query    └─ Examples  └─ Targets")
    else:
        print("No data found. Please download data first.")
    
    print()


def example_3_fewshot_3():
    """Few-shot 모드: 3개 예제"""
    print("=" * 60)
    print("Example 3: Few-Shot Mode (3 examples)")
    print("=" * 60)
    
    dataset = MSADataset(
        data_source='./data/openproteinset',
        max_seq_length=2048,
        max_msa_depth=64,
        num_msa_sequences=5,   # ← 더 감소
        use_few_shot=True,
        few_shot_examples=3,   # ← 3개 예제 (더 많은 패턴)
    )
    
    if len(dataset) > 0:
        sample = dataset[0]
        
        context_mask = sample['context_mask']
        generation_count = (context_mask == 0).sum().item()
        context_count = (context_mask == 1).sum().item()
        
        print(f"Context tokens (query + 3 examples): {context_count}")
        print(f"Generation tokens (target MSAs): {generation_count}")
        print(f"Context ratio: {context_count / (context_count + generation_count) * 100:.1f}%")
        print(f"\nBenefits of 3 examples:")
        print(f"  ✓ 더 많은 패턴 학습")
        print(f"  ✓ 더 안정적인 생성")
        print(f"  ✗ 더 많은 메모리 사용")
        print(f"  ✗ 생성 대상 감소 (6→5)")
    else:
        print("No data found. Please download data first.")
    
    print()


def example_4_comparison():
    """표준 vs Few-shot 비교"""
    print("=" * 60)
    print("Example 4: Standard vs Few-Shot Comparison")
    print("=" * 60)
    
    modes = [
        ("Standard", False, 0),
        ("Few-Shot (2)", True, 2),
        ("Few-Shot (3)", True, 3),
    ]
    
    print(f"{'Mode':<20} {'Context %':<15} {'Tokens/Sample':<15} {'Memory':<15}")
    print("-" * 65)
    
    for name, use_few_shot, few_shot_examples in modes:
        try:
            dataset = MSADataset(
                data_source='./data/openproteinset',
                max_seq_length=2048,
                max_msa_depth=64,
                num_msa_sequences=6,
                use_few_shot=use_few_shot,
                few_shot_examples=few_shot_examples if use_few_shot else 0,
            )
            
            if len(dataset) > 0:
                sample = dataset[0]
                context_mask = sample['context_mask']
                input_ids = sample['input_ids']
                
                context_count = (context_mask == 1).sum().item()
                total = len(input_ids)
                context_pct = context_count / total * 100
                
                # 메모리는 대략적으로 추정
                memory_estimate = f"~{total * 2 / 1024:.1f}KB"
                
                print(f"{name:<20} {context_pct:<14.1f}% {total:<14} {memory_estimate:<15}")
        except Exception as e:
            print(f"{name:<20} {'Error':<15}")
    
    print()
    print("Key Insights:")
    print("  • Standard: 최소 메모리, 학습 느림")
    print("  • Few-Shot (2): 균형잡힌 설정 (추천)")
    print("  • Few-Shot (3): 최고 성능, 메모리 높음")
    print()


def example_5_training_command():
    """학습 명령어 예제"""
    print("=" * 60)
    print("Example 5: Training Commands")
    print("=" * 60)
    
    commands = [
        ("표준 모드", "python scripts/train_mdlm.py --config configs/train_mdlm.yaml"),
        ("Few-shot (2개)", "python scripts/train_mdlm.py --config configs/train_mdlm_fewshot.yaml"),
        ("Few-shot 커스텀", "python scripts/train_mdlm.py --config configs/train_mdlm.yaml --data.use_few_shot true --data.few_shot_examples 3"),
        ("혼합 학습 (Stage 1)", "python scripts/train_mdlm.py --config configs/train_mdlm.yaml --training.max_steps 50000 --data.use_few_shot false"),
        ("혼합 학습 (Stage 2)", "python scripts/train_mdlm.py --config configs/train_mdlm_fewshot.yaml --training.max_steps 50000 --training.warmup_ratio 0.01"),
    ]
    
    for i, (desc, cmd) in enumerate(commands, 1):
        print(f"\n{i}. {desc}")
        print(f"   {cmd}")
    
    print()


def example_6_loss_behavior():
    """Loss 계산 방식"""
    print("=" * 60)
    print("Example 6: How Loss is Calculated")
    print("=" * 60)
    
    print("표준 모드:")
    print("  logits: (batch, seq_len, vocab)")
    print("  target: (batch, seq_len)")
    print("  context_mask: [1, 1, ..., 0, 0, 0]")
    print("  ↓")
    print("  gen_mask = (1 - context_mask) = [0, 0, ..., 1, 1, 1]")
    print("  loss = cross_entropy(logits, target, reduction='none')")
    print("  loss = (loss * gen_mask).sum() / gen_mask.sum()")
    print("  ↓")
    print("  MSA 토큰에만 loss 적용 (Query는 제외)")
    print()
    
    print("Few-shot 모드:")
    print("  logits: (batch, seq_len, vocab)")
    print("  target: (batch, seq_len)")
    print("  context_mask: [1, 1, ..., 1, 1, ..., 0, 0, 0]")
    print("                 └ Query  └ Examples └ Targets")
    print("  ↓")
    print("  gen_mask = (1 - context_mask) = [0, 0, ..., 0, 0, ..., 1, 1, 1]")
    print("  loss = cross_entropy(logits, target, reduction='none')")
    print("  loss = (loss * gen_mask).sum() / gen_mask.sum()")
    print("  ↓")
    print("  Target MSA 토큰에만 loss 적용")
    print("  (Query + Examples는 모두 제외)")
    print()
    
    print("Key: loss는 0이 붙은 위치(generation targets)에만 계산됨")
    print()


def main():
    """Run all examples"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Few-Shot Learning Examples for MDLM".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    try:
        example_1_standard_mode()
        example_2_fewshot_2()
        example_3_fewshot_3()
        example_4_comparison()
        example_5_training_command()
        example_6_loss_behavior()
        
        print("=" * 60)
        print("요약 (Summary)")
        print("=" * 60)
        print("""
Few-shot 학습의 핵심:
  1. Query: 항상 context (context_mask=1)
  2. Few-shot 예제: context (context_mask=1) ← 새로운!
  3. Target MSAs: generation (context_mask=0)
  
Loss는 generation targets(context_mask=0)에만 적용됨

선택 가이드:
  • 표준 모드: 데이터 많음, 빠른 학습, 낮은 메모리
  • Few-shot (2): 균형잡힘, 좋은 성능 (추천)
  • Few-shot (3): 최고 성능, 높은 메모리
        """)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
