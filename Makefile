.PHONY: format tag publish

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
	@bash ./publish.sh
