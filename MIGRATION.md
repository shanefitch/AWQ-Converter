# Migration Guide: setup.py to pyproject.toml

This guide explains the changes made to the AWQ Quantizer project to address the installation error related to the deprecation of the legacy editable install method in pip.

## What Changed?

1. **Replaced `setup.py` with `pyproject.toml`**:
   - We've migrated from the older `setup.py` approach to the modern Python packaging structure using `pyproject.toml`.
   - This follows the latest Python packaging standards (PEP 517/518).

2. **Updated Installation Instructions**:
   - The installation process now uses the modern pip installation method.
   - We've added troubleshooting steps for common installation issues.

3. **Command-line Tool**:
   - The package now properly registers the `awq_quantizer` command-line tool.
   - You can run `awq_quantizer --help` directly after installation.

## How to Migrate

### Automatic Migration

We've provided a migration script to help you transition smoothly:

```bash
# Run the migration script
./migrate_to_pyproject.py
```

This script will:
1. Uninstall any existing installation of awq_quantizer
2. Install the package using the new pyproject.toml

### Manual Migration

If you prefer to migrate manually:

1. Uninstall any existing installation:
   ```bash
   pip uninstall -y awq_quantizer
   ```

2. Install using the new pyproject.toml:
   ```bash
   # Modern installation method (recommended)
   pip install --use-pep517 -e .
   
   # If you encounter issues, try without the --use-pep517 flag
   pip install -e .
   ```

## Troubleshooting

If you encounter any issues during migration:

1. **Installation errors**:
   ```bash
   # Update pip to the latest version
   pip install --upgrade pip
   
   # Try installing with verbose output
   pip install -v -e .
   
   # Try with PEP 517 explicitly enabled
   pip install --use-pep517 -e .
   ```

2. **Command-line tool not found**:
   - Make sure your Python scripts directory is in your PATH
   - Try reinstalling the package

3. **Other issues**:
   - Check the [INSTALL.md](INSTALL.md) file for more troubleshooting tips
   - Open an issue on the GitHub repository

## Why This Change?

The error you encountered is related to the deprecation of the legacy editable install method in pip. The message indicates that:

1. The old way of doing editable installs (`setup.py develop`) is deprecated
2. Pip 25.0 will enforce this behavior change
3. The recommended replacement is to use a `pyproject.toml` file with setuptools >= 64

This change ensures compatibility with current and future versions of pip and follows modern Python packaging best practices. 