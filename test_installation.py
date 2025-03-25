#!/usr/bin/env python3
"""
Test script for verifying the installation of the AWQ Quantizer.

This script checks if:
1. The main package can be imported
2. The submodules can be imported
3. The command-line tool is available
4. GPU acceleration is available and lists all GPUs

Usage:
    python test_installation.py
    ./test_installation.py
"""

import importlib
import subprocess
import sys
from typing import Dict, List, Optional, Tuple


def check_module(module_name: str) -> bool:
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"✅ Module '{module_name}' is installed correctly.")
        return True
    except ImportError as e:
        print(f"❌ Module '{module_name}' could not be imported: {e}")
        return False


def check_command(command: str) -> bool:
    """Check if a command is available."""
    try:
        result = subprocess.run(
            command.split(), 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            check=False  # Don't raise an exception on non-zero exit
        )
        if result.returncode == 0:
            print(f"✅ Command '{command}' is available.")
            return True
        else:
            print(f"❌ Command '{command}' failed with error: {result.stderr.strip()}")
            return False
    except FileNotFoundError:
        print(f"❌ Command '{command}' not found.")
        return False


def check_gpu_support() -> Tuple[bool, List[Dict[str, str]]]:
    """Check if CUDA is available through PyTorch and list all GPUs."""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"✅ CUDA is available with {device_count} device(s)")
            
            # Get info for each GPU
            gpu_info = []
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                device_props = torch.cuda.get_device_properties(i)
                
                # Calculate memory info
                total_memory_gb = device_props.total_memory / (1024 ** 3)
                
                # Get compute capability
                compute_capability = f"{device_props.major}.{device_props.minor}"
                
                # Store GPU info
                gpu_info.append({
                    "index": i,
                    "name": device_name,
                    "memory": f"{total_memory_gb:.2f} GB",
                    "compute_capability": compute_capability
                })
                
                # Print GPU info
                print(f"  GPU {i}: {device_name}")
                print(f"    - Memory: {total_memory_gb:.2f} GB")
                print(f"    - Compute Capability: {compute_capability}")
                
            return True, gpu_info
        else:
            print("⚠️ CUDA is not available. GPU acceleration will not be used.")
            return False, []
    except ImportError:
        print("⚠️ Failed to import torch. Cannot check CUDA availability.")
        return False, []


def check_multi_gpu_support() -> bool:
    """Check if the AWQ Quantizer supports multi-GPU processing."""
    try:
        # Check if the multi_gpu argument exists in the CLI
        result = subprocess.run(
            ["awq_quantizer", "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        if "--multi_gpu" in result.stdout:
            print("✅ Multi-GPU support is available")
            return True
        else:
            print("⚠️ Multi-GPU support is not detected in the CLI")
            return False
    except Exception as e:
        print(f"⚠️ Could not determine multi-GPU support: {e}")
        return False


def main():
    """Main function to run all checks."""
    print("Testing AWQ Quantizer Installation\n" + "=" * 30)
    
    # Check main module
    main_module_ok = check_module("awq_quantizer")
    
    # Check submodules if main module is OK
    submodule_checks = []
    if main_module_ok:
        submodules = [
            "awq_quantizer.model_loading",
            "awq_quantizer.quantization",
            "awq_quantizer.utils"
        ]
        for submodule in submodules:
            submodule_checks.append(check_module(submodule))
    
    # Check if the command-line tool is available
    command_ok = check_command("awq_quantizer --help")
    
    # Check GPU support
    gpu_available, gpu_info = check_gpu_support()
    
    # Check multi-GPU support
    multi_gpu_ok = check_multi_gpu_support() if command_ok else False
    
    # Print summary
    print("\nInstallation Summary\n" + "=" * 30)
    if main_module_ok and all(submodule_checks) and command_ok:
        print("✅ AWQ Quantizer is installed correctly!")
        if gpu_available:
            gpu_count = len(gpu_info)
            if gpu_count == 1:
                print(f"✅ GPU acceleration is available: {gpu_info[0]['name']}")
            else:
                print(f"✅ Multi-GPU acceleration is available with {gpu_count} GPUs")
                
            # Print multi-GPU support status
            if multi_gpu_ok and gpu_count > 1:
                print("✅ Multi-GPU processing is supported and can be enabled with --multi_gpu flag")
            elif gpu_count > 1:
                print("⚠️ Multiple GPUs detected but multi-GPU support may not be available")
        else:
            print("⚠️ GPU acceleration is not available. Quantization will be slower.")
    else:
        print("❌ AWQ Quantizer installation has issues.")
        print("\nTroubleshooting:")
        print("1. Make sure you've installed with 'pip install -e .'")
        print("2. Check that your Python environment is activated")
        print("3. If installed in development mode, make sure the source directory is accessible")
        print("4. For GPU support, ensure you have CUDA and the correct PyTorch version installed")
        
        # Print specific help for missing command
        if not command_ok:
            print("\nTo fix missing command-line tool:")
            print("- Check if 'awq_quantizer' is in your PATH")
            print("- Reinstall with 'pip install -e .'")
    
    return 0 if (main_module_ok and all(submodule_checks) and command_ok) else 1


if __name__ == "__main__":
    sys.exit(main()) 