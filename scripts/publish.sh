#!/usr/bin/env bash
set -euo pipefail

# 🔢 Extract version from pyproject.toml
version=$(grep -m 1 'version' pyproject.toml | cut -d '"' -f2)

# ✔ Ensure Git tag for this version exists
if ! git rev-parse "$version" >/dev/null 2>&1; then
  echo "Git tag '$version' not found. Please run './tag.sh' to create it before publishing."
  exit 1
fi


# 🧹 Clean build artifacts
rm -rf dist

# 🔒 Set up PyPI credentials
export TWINE_NON_INTERACTIVE=1
export TWINE_USERNAME='__token__'
export TWINE_PASSWORD="${PYPI_TOKEN:?PYPI_TOKEN must be set}"

# 🛠 Build and upload to PyPI
pip install --upgrade pip setuptools wheel build twine
python -m build
twine upload dist/*

# Report version
version=$(grep -m 1 'version' pyproject.toml | cut -d '"' -f2)
echo "Successfully published version $version"
