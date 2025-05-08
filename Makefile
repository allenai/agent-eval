.PHONY: format tag publish test

format:
	@echo "Formatting with isort and black..."
	isort .
	black .

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
