.PHONY: install install-dev lint format test train evaluate generate-data clean

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

lint:
	ruff check .
	black --check .

format:
	ruff check --fix .
	black .

test:
	pytest

generate-data:
	python training/processor.py --generate 20000 --output training/data/20k_leak.txt --seed 42

train:
	python training/training.py --dataset_path training/data/dataset_alpaca.json --output_dir ./model

evaluate:
	python training/evaluate.py --model_path ./model --eval_dataset ./model/eval_set.json

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
