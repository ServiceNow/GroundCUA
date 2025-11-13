# Publishing GroundCUA to PyPI

This document explains how the automated publishing workflow works and how to create releases.

## Overview

The GroundCUA package is automatically published to PyPI whenever a new GitHub release is created. The workflow uses GitHub Actions with trusted publishing for secure, tokenless deployment.

## Prerequisites

Before you can publish to PyPI, you need to:

1. **Set up PyPI Trusted Publishing** (one-time setup):
   - Go to https://pypi.org/manage/account/publishing/
   - Add a new pending publisher with:
     - PyPI Project Name: `groundcua`
     - Owner: `xhluca` (or your GitHub username/org)
     - Repository name: `GroundCUA`
     - Workflow name: `publish-python.yaml`
     - Environment name: `pypi`

2. **Configure GitHub Environment** (optional, for additional protection):
   - Go to your repository Settings → Environments
   - Create an environment named `pypi`
   - Add protection rules if desired (e.g., required reviewers)

## Creating a Release

To publish a new version to PyPI:

1. **Update the version** (optional - the workflow will do this automatically):
   ```bash
   # The workflow will update groundcua/version.py based on the release tag
   ```

2. **Create a new release on GitHub**:
   - Go to https://github.com/xhluca/GroundCUA/releases/new
   - Choose a tag name (e.g., `v0.0.2`, `v0.1.0`, etc.)
   - Tag should follow semantic versioning: `vMAJOR.MINOR.PATCH`
   - Fill in the release title and description
   - Click "Publish release"

3. **Automatic publishing**:
   - The GitHub Actions workflow will automatically:
     - Update `groundcua/version.py` with the release tag
     - Build the package
     - Publish to PyPI using trusted publishing

4. **Verify the release**:
   - Check the Actions tab for workflow status
   - Visit https://pypi.org/project/groundcua/ to see the new version

## Local Testing

To test the package locally before publishing:

```bash
# Install in development mode
pip install -e .

# Or build the package locally
python -m build

# Install from the built package
pip install dist/groundcua-0.0.1-py3-none-any.whl
```

## Using the Package

After publishing, users can install the package with:

```bash
# Basic installation
pip install groundcua

# With training dependencies
pip install groundcua[training]

# With all optional dependencies
pip install groundcua[all]
```

## Migration from groundcua_utils.py

If you have existing code using `groundcua_utils.py`, update your imports:

```python
# Old way
from groundcua_utils import prepare_image, create_messages, GROUNDNEXT_SYSTEM_PROMPT

# New way
from groundcua import prepare_image, create_messages, GROUNDNEXT_SYSTEM_PROMPT
```

## Version Numbering

- Follow semantic versioning: `MAJOR.MINOR.PATCH`
- `MAJOR`: Breaking changes
- `MINOR`: New features (backward compatible)
- `PATCH`: Bug fixes (backward compatible)

Current version: **0.0.1**

