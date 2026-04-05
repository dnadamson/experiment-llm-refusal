"""Test config loading."""

from safety_finetune.config import load_config


def test_load_smoke_config():
    cfg = load_config("configs/smoke.yaml")
    assert cfg.model.name == "Qwen/Qwen2.5-0.5B-Instruct"
    assert cfg.model.quantization is None
    assert cfg.training.method == "kto"
    assert cfg.data.max_samples == 100


def test_load_full_config():
    cfg = load_config("configs/full.yaml")
    assert cfg.model.name == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.model.quantization == "4bit"
    assert cfg.training.bf16 is True
    assert cfg.data.max_samples is None
