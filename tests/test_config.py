"""Test config loading."""

from safety_finetune.config import load_config


def test_load_smoke_config():
    cfg = load_config("configs/smoke.yaml")
    assert cfg.model.name == "Qwen/Qwen2.5-0.5B-Instruct"
    assert cfg.model.quantization is None
    assert cfg.training.method == "kto"
    assert cfg.training.desirable_weight == 0.31
    assert cfg.training.beta == 0.1
    assert cfg.data.max_samples == 100
    assert cfg.output.report_to == "tensorboard"


def test_load_full_config():
    cfg = load_config("configs/full.yaml")
    assert cfg.model.name == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.model.quantization == "4bit"
    assert cfg.training.bf16 is True
    assert cfg.training.desirable_weight == 0.31
    assert cfg.data.max_samples is None
    assert cfg.output.report_to == "tensorboard"
