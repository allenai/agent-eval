.PHONY: format publish

format:
	@echo "Formatting with isort and black..."
	isort .
	black .

publish:
	@echo "Publishing package using publish.sh..."
	@bash ./publish.sh
