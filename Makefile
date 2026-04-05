.PHONY: build smoke train eval test lint notebook clean

IMAGE := safety-finetune
RUN := docker compose run --rm train

build:
	docker compose build

smoke:
	$(RUN) safety_finetune.train --config configs/smoke.yaml

train:
	$(RUN) safety_finetune.train --config configs/full.yaml

eval:
	$(RUN) safety_finetune.eval --config configs/smoke.yaml

test:
	$(RUN) pytest tests/ -v

lint:
	$(RUN) ruff check src/ tests/

notebook:
	docker compose up notebook

clean:
	rm -rf outputs/ models/ wandb/
