#!/usr/bin/env python3
"""
Update version.py with the release tag version.

This script is used during the GitHub Actions release workflow to update
the package version with the release tag.
"""

import argparse
import re
from pathlib import Path


def update_version(version: str, path: str) -> None:
    """
    Update the version in the specified file.
    
    Args:
        version: Version string (e.g., 'v0.0.1' or '0.0.1')
        path: Path to the version.py file
    """
    # Remove 'v' prefix if present
    if version.startswith('v'):
        version = version[1:]
    
    # Validate version format (basic semver check)
    if not re.match(r'^\d+\.\d+\.\d+', version):
        raise ValueError(f"Invalid version format: {version}. Expected semver format (e.g., 0.0.1)")
    
    version_file = Path(path)
    
    if not version_file.exists():
        raise FileNotFoundError(f"Version file not found: {path}")
    
    # Read the current content
    content = version_file.read_text()
    
    # Update the version line
    new_content = re.sub(
        r'__version__\s*=\s*["\'].*["\']',
        f'__version__ = "{version}"',
        content
    )
    
    # Write back
    version_file.write_text(new_content)
    
    print(f"✓ Updated {path} to version {version}")


def main():
    parser = argparse.ArgumentParser(
        description="Update version.py with release tag"
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Version string (e.g., 'v0.0.1' or '0.0.1')"
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Path to version.py file"
    )
    
    args = parser.parse_args()
    update_version(args.version, args.path)


if __name__ == "__main__":
    main()

