"""JustRL training with DeepSeek-R1-Distill-Qwen-1.5B + LoRA + GRPO.

Modes:
  --mode trial : 2 steps, 50 rows, 512-token completions
  --mode full  : paper hyperparams scaled for 24GB
"""
from __future__ import annotations

import argparse
import os

import torch
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer

from JustRL.data import load_dapo_math_17k
from JustRL.reward import dapo_binary_reward

PatchFastRL("GRPO")

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
LORA_RANK = 64
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def build_model_and_tokenizer(max_seq_length: int):
    """Load model and apply LoRA."""
    dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        dtype=dtype,
        max_lora_rank=LORA_RANK,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_RANK,
        lora_dropout=0.0,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    return model, tokenizer


def build_dataset(n_rows: int | None):
    """Load DAPO-Math-17k with boxed suffix."""
    ds = load_dapo_math_17k(append_boxed_suffix=True, dedup=False)
    if n_rows is not None:
        ds = ds.select(range(min(n_rows, len(ds))))
    return ds


def build_grpo_config(mode: str) -> GRPOConfig:
    """Build GRPOConfig for trial or full mode."""
    bf16 = is_bfloat16_supported()

    common = dict(
        learning_rate=1e-6,
        lr_scheduler_type="constant",
        temperature=1.0,
        beta=0.0,
        epsilon=0.2,
        epsilon_high=0.28,
        loss_type="dapo",
        max_grad_norm=1.0,
        weight_decay=0.1,
        warmup_steps=10,
        optim="adamw_8bit",
        bf16=bf16,
        fp16=not bf16,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=50,
        output_dir="outputs/justrl-deepseek-1.5b",
        report_to="none",
        remove_unused_columns=False,
        seed=3407,
    )

    if mode == "trial":
        return GRPOConfig(
            **common,
            max_steps=2,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_generations=4,
            max_prompt_length=512,
            max_completion_length=512,
        )
    else:
        return GRPOConfig(
            **common,
            max_steps=4380,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            num_generations=8,
            max_prompt_length=1024,
            max_completion_length=8192,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["trial", "full"], default="trial")
    parser.add_argument("--n_rows", type=int, default=None,
                        help="Subset dataset to N rows (trial default: 50)")
    args = parser.parse_args()

    if args.mode == "trial" and args.n_rows is None:
        args.n_rows = 50

    max_seq = 1024 + 8192 if args.mode == "full" else 512 + 512
    model, tokenizer = build_model_and_tokenizer(max_seq_length=max_seq)
    dataset = build_dataset(args.n_rows)
    config = build_grpo_config(args.mode)

    print(f"JustRL-DeepSeek-1.5B | mode={args.mode} | rows={len(dataset)} | "
          f"max_seq={max_seq} | bf16={is_bfloat16_supported()}")

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=dapo_binary_reward,
        args=config,
        train_dataset=dataset,
    )

    trainer.train()

    save_dir = os.path.join(config.output_dir, "final")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Saved to {save_dir}")


if __name__ == "__main__":
    main()
