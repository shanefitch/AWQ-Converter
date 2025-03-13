#!/usr/bin/env python3
"""
Migration script to help users migrate from setup.py to pyproject.toml.
This script will:
1. Uninstall any existing installation of awq_quantizer
2. Install the package using the new pyproject.toml
"""

import os
import subprocess
import sys


def run_command(command, description):
    """Run a command and print its output."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"Command output: {e.stdout}")
        print(f"Command error: {e.stderr}")
        return False


def main():
    """Main function."""
    print("AWQ Quantizer Migration Script")
    print("==============================")
    print("This script will help you migrate from setup.py to pyproject.toml.")
    print("It will uninstall any existing installation of awq_quantizer and")
    print("install the package using the new pyproject.toml.")
    
    # Check if pyproject.toml exists
    if not os.path.exists("pyproject.toml"):
        print("Error: pyproject.toml not found. Please make sure you're running this script from the root of the repository.")
        sys.exit(1)
    
    # Uninstall any existing installation
    print("\nStep 1: Uninstalling any existing installation of awq_quantizer...")
    run_command("pip uninstall -y awq_quantizer", "Uninstalling awq_quantizer")
    
    # Install the package using the new pyproject.toml
    print("\nStep 2: Installing the package using the new pyproject.toml...")
    success = run_command("pip install --use-pep517 -e .", "Installing awq_quantizer")
    
    if not success:
        print("\nTrying alternative installation method...")
        success = run_command("pip install -e .", "Installing awq_quantizer (alternative method)")
    
    if success:
        print("\nMigration completed successfully!")
        print("You can now use the awq_quantizer command-line tool:")
        print("  awq_quantizer --help")
    else:
        print("\nMigration failed. Please try manually:")
        print("  pip uninstall -y awq_quantizer")
        print("  pip install --use-pep517 -e .")


if __name__ == "__main__":
    main() 