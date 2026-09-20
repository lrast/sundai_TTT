.PHONY: install lint test smoke smoke-qa smoke-txlog sweep-small report

install:
	uv sync --group dev

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

test:
	uv run pytest

smoke: smoke-qa smoke-txlog

smoke-qa:
	uv run ctxlab run -c configs/experiments/smoke.yaml
	uv run ctxlab report runs/smoke

smoke-txlog:
	uv run ctxlab run -c configs/experiments/txlog_smoke.yaml
	uv run ctxlab report runs/txlog_smoke

sweep-small:
	uv run ctxlab run -c configs/experiments/position_sweep.yaml
	uv run ctxlab report runs/position_sweep

report:
	uv run ctxlab report $(RUN)
