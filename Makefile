.DEFAULT_GOAL := help
COMPOSE := docker compose

.PHONY: help up down logs migrate weights test test-all lint format bench

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

.env:
	@cp -n .env.example .env && echo "created .env from .env.example"

up: .env ## Build and start the full stack (api, worker, redis, postgres)
	$(COMPOSE) up --build

down: ## Stop and remove containers
	$(COMPOSE) down

logs: ## Tail service logs
	$(COMPOSE) logs -f

migrate: .env ## Apply database migrations inside the api container
	$(COMPOSE) run --rm api alembic upgrade head

weights: ## Copy trained weights into ./models (make weights WEIGHTS_SRC=/path/to.pt)
	@mkdir -p models
	@test -n "$(WEIGHTS_SRC)" || { echo "Set WEIGHTS_SRC=/path/to/model_multitask_seed47.pt"; exit 1; }
	cp "$(WEIGHTS_SRC)" models/model_multitask_seed47.pt && echo "weights copied to ./models"

test: ## Run the fast test suite (no services required)
	pytest -m "not slow" -q

test-all: ## Run every test including the slow real-model one
	pytest -q

lint: ## Run ruff and mypy
	ruff check app tests
	mypy app

format: ## Auto-fix imports/lint
	ruff check --fix app tests

bench: ## Benchmark a running API (make bench BASE_URL=http://localhost:8000)
	python scripts/bench.py
