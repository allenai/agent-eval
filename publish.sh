#!/usr/bin/env bash

set -euo pipefail

rm -rf dist

export TWINE_NON_INTERACTIVE=1
export TWINE_USERNAME='__token__'
export TWINE_PASSWORD=$PYPI_TOKEN

pip install --upgrade pip setuptools wheel build twine
python -m build
twine upload dist/*

version=$(grep version pyproject.toml | cut -d '"' -f2)
git tag "$version"
git push origin "$version"
