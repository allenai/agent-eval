#!/usr/bin/env bash
set -euo pipefail

# 🧠 Ensure we're on the main branch
current_branch=$(git rev-parse --abbrev-ref HEAD)
if [[ "$current_branch" != "main" ]]; then
  echo "You're on branch '$current_branch'. Please switch to 'main' before publishing."
  exit 1
fi

# 🔄 Ensure local main matches remote main
git fetch origin main
if ! git diff --quiet HEAD origin/main; then
  echo "Local 'main' is not in sync with 'origin/main'."
  echo "Please pull the latest changes and push any local commits before publishing."
  exit 1
fi

# 🧼 Ensure working tree is clean
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "You have uncommitted changes. Commit or stash them before publishing."
  exit 1
fi

# 🔢 Extract version from pyproject.toml
version=$(grep version pyproject.toml | cut -d '"' -f2)

# ✅ Ensure the version follows semantic versioning
if ! [[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Version '$version' is not semantic (e.g., 1.2.3)."
  exit 1
fi

# 🏷 Check that tag doesn't already exist
if git rev-parse "$version" >/dev/null 2>&1; then
  echo "Git tag '$version' already exists. Bump the version in pyproject.toml."
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

# 🏷 Create git tag and push it
git tag "$version"
git push origin "$version"

echo "Successfully published version $version"
