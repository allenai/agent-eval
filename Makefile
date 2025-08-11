.PHONY: format tag publish test test-core test-leaderboard test-all

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

test-core:
	@echo "Running core tests (excluding leaderboard)..."
	pytest -m "not leaderboard"

test-leaderboard:
	@echo "Running leaderboard tests..."
	pytest -m leaderboard

test: test-core  # Default to core tests

test-all:
	@echo "Running all tests..."
	pytest
