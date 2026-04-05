"""KTO fine-tuning with QLoRA."""

import argparse
from pathlib import Path

import torch
from peft import LoraConfig as PeftLoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import KTOConfig, KTOTrainer

from safety_finetune.config import load_config
from safety_finetune.data_utils import build_kto_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Model ---
    model_kwargs = {"trust_remote_code": True}

    if cfg.model.quantization == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif cfg.model.quantization == "8bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    # Use MPS on Apple Silicon when no CUDA, otherwise CPU
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
    elif torch.backends.mps.is_available() and cfg.model.quantization is None:
        # bitsandbytes doesn't support MPS; only use MPS for non-quantized
        model_kwargs["device_map"] = "mps"

    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, **model_kwargs)

    if cfg.model.quantization:
        model = prepare_model_for_kbit_training(model)

    # --- LoRA ---
    peft_config = PeftLoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # --- Data ---
    data = build_kto_dataset(
        max_samples=cfg.data.max_samples,
        eval_split=cfg.data.eval_split,
    )

    # --- Training ---
    training_args = KTOConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        warmup_ratio=cfg.training.warmup_ratio,
        max_length=cfg.training.max_length,
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps if cfg.training.save_steps > 0 else None,
        bf16=cfg.training.bf16,
        remove_unused_columns=False,
        report_to="wandb" if cfg.output.push_to_hub else "none",
    )

    trainer = KTOTrainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["eval"],
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    print(f"Model saved to {output_dir / 'final'}")


if __name__ == "__main__":
    main()
