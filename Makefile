.PHONY: format tag publish test

format:
	@echo "Formatting with isort and black..."
	isort .
	black .

format-check:
	@echo "Checking formatting with isort and black..."
	isort --check .
	black --check .

lint-check:
	@echo "Running lint checks..."
	flake8 src/ tests/
	mypy src/ tests/

code-check: format-check lint-check

tag:
	@echo "Tagging version..."
	@bash ./tag.sh

# Upload package to PyPI
publish:
	@echo "Uploading package to PyPI..."
	@bash scripts/publish.sh

# Update HF dataset features
update-schema:
	@echo "Updating schema..."
	python scripts/update_schema.py

test:
	@echo "Running tests..."
	pytest
