# SparseCascade: Cascaded Compression for Large Language Models

Anonymous code repository for ESWA submission.

## Structure

```
├── step1_pruning/           # Stage 1: Adaptive iterative pruning
│   ├── prune_channel/       #   Core: policy gradient + importance scoring
│   │   ├── main.py          #   Entry point for pruning
│   │   ├── pgpruning.py     #   REINFORCE policy gradient optimization
│   │   ├── eval.py          #   Evaluation utilities
│   │   └── ...
│   └── generate_random_mask.py
│
├── step2_quantization/      # Stage 2: Mask-guided mixed-precision quantization
│   │                        # Stage 3: Low-rank compensation (shared framework)
│   ├── main.py              #   Entry point for quantization + evaluation
│   ├── train_lora.py        #   Low-rank compensation training (Stage 3)
│   ├── quantize/            #   Quantization core (LWC, LET, AugLoss)
│   ├── preprocessing/       #   Activation scales
│   ├── models/              #   Quantized model definitions
│   └── lm_eval/             #   Zero-shot evaluation harness
│
├── figures/                 # Figure generation scripts (8 files)
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Requirements

- Ubuntu 22.04
- Python 3.10
- CUDA 11.8
- Single GPU with 24GB+ VRAM (e.g., RTX 3090)

## Installation

```bash
conda create -n sparsecascade python=3.10 -y
conda activate sparsecascade
pip install -r requirements.txt
```

## Pre-trained Weights

Download LLaMA-2-7B from Meta (requires approval):
https://llama.meta.com/

Place the model under a directory (e.g., `/root/autodl-tmp/models/Llama-2-7b-hf`) and update
the `--model` / `--model-name` arguments accordingly.

## Usage: Full Pipeline (30% Pruning)

All experiments were conducted with a global pruning rate of 30%.
Paths use `/root/autodl-tmp` as the working directory — adjust to your environment.

### Stage 1 — Adaptive Iterative Pruning

```bash
cd step1_pruning
python prune_channel/main.py \
    --model-name "/root/autodl-tmp/models/Llama-2-7b-hf" \
    --init-data-name c4 \
    --train-data-name c4 \
    --test-data-name wikitext2 \
    --dataset-size 12000 \
    --train-batch-size 4 \
    --test-batch-size 4 \
    --max-seq-length 128 \
    --init-seqlen 1024 \
    --nsamples 128 \
    --n-workers 1 \
    --init-type wanda-sp \
    --score-from-metric sigmap \
    --prune-rate-start 0.7 \
    --prune-rate-target 0.7 \
    --prune-start-iter-percentage 0.0 \
    --prune-end-iter-percentage 0.1 \
    --init-rate 0.7 \
    --attn-score-lr 0.002 \
    --mlp-score-lr 0.002 \
    --score-init-constant 1.0 \
    --ma-window-size 5 \
    --eval-per-steps 1000 \
    --save-folder "/root/autodl-tmp/exp_v36_full_pipeline/sparsity_30_corrected" \
    --penalty-lamda-init -1 \
    --penalty-lamda-final 0.0 \
    --K 2 \
    --n-epoches 10
```

Key designs:
- **Importance prior**: `--init-type wanda-sp` with SigmaP mapping (`--score-from-metric sigmap`)
- **Probabilistic mask sampling**: Bernoulli distribution over output dimensions
- **Policy gradient**: REINFORCE with smoothed loss (`--ma-window-size 5`)
- **Global budget**: 30% pruning (`--prune-rate-target 0.7` retention)

### Stage 2 — Mask-Guided Mixed-Precision Quantization

```bash
cd step2_quantization
python main.py \
    --model "/root/autodl-tmp/models/Llama-2-7b-hf" \
    --output_dir "/root/autodl-tmp/exp_v36_full_pipeline/quantized_sparsity_30" \
    --wbits 4 --abits 16 \
    --quant_type mix \
    --prune_info_path "/root/autodl-tmp/exp_v36_full_pipeline/sparsity_30_corrected/test/<timestamp>/prune_info.pt" \
    --act-scales "./act_scales_pruned/Llama-2-7b-hf_sparsity_30.pt" \
    --act-shifts "./act_shifts_pruned/Llama-2-7b-hf_sparsity_30.pt" \
    --eval_ppl \
    --epochs 20 \
    --lwc \
    --let \
    --aug_loss \
    --tasks piqa,hellaswag,winogrande,arc_easy,arc_challenge
```

Key designs:
- **Mask-guided calibration**: `--prune_info_path` anchors quantization to valid weights only
- **Mixed precision**: 1% FP16 outlier + 99% 4-bit (`--quant_type mix`)
- **Calibration optimization**: LWC + LET + AugLoss over 20 epochs
- **Effective bit-width**: (0.01 × 16 + 0.99 × 4) × (1 − 0.3) ≈ 2.9 bits

### Stage 3 — Dimension-Aligned Low-Rank Compensation

Stage 3 reuses the `step2_quantization` framework since the low-rank compensation path
shares the quantization backbone.

```bash
cd step2_quantization

# Train the low-rank compensation path
python train_lora.py \
    --model_id "/root/autodl-tmp/models/Llama-2-7b-hf" \
    --prune_info_path "/root/autodl-tmp/exp_v36_full_pipeline/sparsity_30_corrected/test/<timestamp>/prune_info.pt" \
    --output_dir "/root/autodl-tmp/exp_v36_full_pipeline/quantized_sparsity_30" \
    --wbits 4 --abits 16 \
    --quant_type mix \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lr 2e-4 \
    --train_steps 200 \
    --dataset c4 \
    --batch_size 1 \
    --gradient_accumulation_steps 4

# Evaluate with low-rank compensation
python main.py \
    --model "/root/autodl-tmp/models/Llama-2-7b-hf" \
    --output_dir "/root/autodl-tmp/exp_v36_full_pipeline/quantized_sparsity_30" \
    --tasks piqa,hellaswag,winogrande,arc_easy,arc_challenge \
    --wbits 4 --abits 16 \
    --quant_type mix \
    --eval_ppl \
    --prune_info_path "/root/autodl-tmp/exp_v36_full_pipeline/sparsity_30_corrected/test/<timestamp>/prune_info.pt" \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_path "/root/autodl-tmp/exp_v36_full_pipeline/quantized_sparsity_30/lora_50_merged.pth" \
    --epochs 0 \
    2>&1 | tee eval_lora_50_sparsity30.txt
```

Key designs:
- **Rank expansion**: `--lora_r 64 --lora_alpha 128` for enhanced compensation capacity
- **Frozen backbone**: Only the low-rank compensation parameters are trained
- **Dimension alignment**: Compensation paths adapt to pruned valid dimensions via `--prune_info_path`

### Verify Mixed-Precision Ratio

```bash
python step2_quantization/check_ratio.py \
    --mask_dir "/root/autodl-tmp/exp_v36_full_pipeline/quantized_sparsity_30_outlier001/mask/"
```

## Generating Figures

```bash
cd figures
python pruning_method_3_1_problem.py
python pruning_method_3_1_prior_effect.py
python pruning_method_3_1_outcome.py
python quantization_method_3_2_problem.py
python quantization_method_3_2_mechanism_try.py
python recovery_method_3_3_problem.py
python recovery_method_3_3_curve_evolution_try.py
python overall-pipeline-v3.py
```

## Citation

If you use this code, please cite our paper (to be updated upon acceptance).
