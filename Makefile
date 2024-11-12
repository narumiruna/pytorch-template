lint:
	uv run ruff check .

test:
	uv run pytest -v -s --cov=src tests

type:
	uv run mypy --install-types --non-interactive .

publish:
	uv build --wheel
	uv publish

.PHONY: lint test publish
