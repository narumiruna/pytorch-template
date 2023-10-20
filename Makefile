install:
	poetry install

lint: install
	poetry run flake8 -v

test: install
	poetry run pytest -v -s tests

cover: install
	poetry run coverage run -m pytest -v -s tests
	poetry run coverage report -m

publish:
	poetry build -f wheel
	poetry publish -u __token__ -p $(PYPI_TOKEN)

.PHONY: lint test cover
