#!/usr/bin/env python3
"""
Test script to verify the AWQ Quantizer installation.
"""

import importlib.util
import subprocess
import sys


def check_module(module_name):
    """Check if a module can be imported."""
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        print(f"❌ Module {module_name} is not installed")
        return False
    print(f"✅ Module {module_name} is installed")
    return True


def check_command(command):
    """Check if a command is available."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print(f"✅ Command '{command}' is available")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Command '{command}' is not available")
        return False


def main():
    """Main function."""
    print("AWQ Quantizer Installation Test")
    print("==============================")
    
    # Check if the main package is installed
    main_module = check_module("awq_quantizer")
    
    # Check if submodules are installed
    submodules = [
        "awq_quantizer.model_loading",
        "awq_quantizer.quantization",
        "awq_quantizer.utils",
    ]
    
    submodules_ok = all(check_module(module) for module in submodules)
    
    # Check if the command-line tool is available
    command_ok = check_command("awq_quantizer --help")
    
    # Print summary
    print("\nSummary:")
    if main_module and submodules_ok and command_ok:
        print("✅ AWQ Quantizer is installed correctly")
    else:
        print("❌ AWQ Quantizer installation has issues")
        print("\nTroubleshooting:")
        print("1. Try running the migration script: ./migrate_to_pyproject.py")
        print("2. Check the MIGRATION.md and INSTALL.md files for more information")
        print("3. Make sure your Python scripts directory is in your PATH")


if __name__ == "__main__":
    main() 