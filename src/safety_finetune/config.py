"""Load YAML configs and provide typed access."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    quantization: str | None = None


@dataclass
class LoraConfig:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class TrainingConfig:
    method: str = "kto"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 2
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    max_length: int = 512
    logging_steps: int = 10
    save_steps: int = 0
    bf16: bool = False


@dataclass
class DataConfig:
    max_samples: int | None = None
    eval_split: float = 0.2


@dataclass
class OutputConfig:
    dir: str = "outputs/smoke"
    push_to_hub: bool = False


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)

    return Config(
        model=ModelConfig(**raw.get("model", {})),
        lora=LoraConfig(**raw.get("lora", {})),
        training=TrainingConfig(**raw.get("training", {})),
        data=DataConfig(**raw.get("data", {})),
        output=OutputConfig(**raw.get("output", {})),
    )
