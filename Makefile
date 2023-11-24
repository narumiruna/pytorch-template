install:
	poetry install

lint:
	poetry run flake8 -v

test:
	poetry run pytest -v -s --cov=. tests

publish:
	poetry build -f wheel
	poetry publish

.PHONY: lint test publish
