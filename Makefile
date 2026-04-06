.PHONY: build lock smoke train eval test lint notebook inspect clean

IMAGE := safety-finetune
RUN := docker compose run --rm train

# Regenerate requirements.lock inside the target Python version (3.11)
# Run this when you change pyproject.toml dependencies
lock:
	docker build -f Dockerfile.lock -t $(IMAGE)-lock . && \
	docker run --rm -v $(PWD):/out $(IMAGE)-lock

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

inspect:
	docker compose run --rm --entrypoint python train /app/scripts/inspect_data.py $(ARGS)

notebook:
	docker compose up notebook

clean:
	rm -rf outputs/ models/ wandb/
