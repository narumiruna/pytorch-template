install:
	poetry install

lint:
	poetry run flake8 -v .

test: install
	poetry run pytest -v -s --disable-warnings tests

cover: install
	poetry run coverage run -m pytest -v -s --disable-warnings tests
	poetry run coverage report -m

publish:
	poetry build -f wheel
	poetry publish -u __token__ -p $(PYPI_TOKEN)

.PHONY: lint test cover
