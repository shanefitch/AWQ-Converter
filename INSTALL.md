# Installation Guide

This guide provides instructions for installing and setting up the AWQ Quantizer.

## Prerequisites

- Python 3.12 or later
- CUDA-compatible GPU (recommended for faster quantization)
- Internet connection (if using Hugging Face Hub integration)

## Installation

### Option 1: Install from source

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/awq_quantizer.git
   cd awq_quantizer
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   # Using venv
   python -m venv awq_env
   source awq_env/bin/activate  # On Windows: awq_env\Scripts\activate
   
   # Or using conda
   conda create -n awq_env python=3.12
   conda activate awq_env
   ```

3. Install the package in development mode:
   ```bash
   # Modern installation method (recommended)
   pip install -e .
   
   # If you encounter any issues, try with PEP 517 explicitly enabled
   pip install --use-pep517 -e .
   ```

### Option 2: Install using pip

```bash
# Install directly from GitHub
pip install git+https://github.com/yourusername/awq_quantizer.git
```

## Verifying Installation

To verify that the installation was successful, run the test script:

```bash
python test_quantization.py
```

This will create a small test model, quantize it, and save the results to the `test_output` directory.

You can also verify that the command-line tool is installed correctly:

```bash
# This should display the help message
awq_quantizer --help
```

## Hugging Face Hub Integration

AWQ Quantizer provides seamless integration with Hugging Face Hub. To use this feature, you need to have the `huggingface_hub` package installed, which is included in the requirements.

To authenticate with Hugging Face Hub for accessing private models, you can:

1. Use the `--token` parameter:
   ```bash
   awq_quantizer --hub_model_id your-org/private-model --token "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" --output_dir /path/to/output
   ```

2. Or login using the Hugging Face CLI:
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   ```
   Then you can use AWQ Quantizer without providing a token:
   ```bash
   awq_quantizer --hub_model_id your-org/private-model --output_dir /path/to/output
   ```

## Troubleshooting

### Common Issues

1. **Installation errors**:
   If you encounter errors during installation, try the following:
   ```bash
   # Install with verbose output to see what's happening
   pip install -v -e .
   
   # Or try with PEP 517 explicitly enabled
   pip install --use-pep517 -e .
   
   # If you're using an older version of pip, update it first
   pip install --upgrade pip
   ```

2. **Missing dependencies**:
   If you encounter errors about missing dependencies, try installing them manually:
   ```bash
   pip install safetensors huggingface_hub torch numpy transformers tqdm bitsandbytes accelerate pyyaml
   ```

3. **CUDA issues**:
   If you encounter CUDA-related errors, make sure you have a compatible version of PyTorch installed:
   ```bash
   # For CUDA 12.1
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

4. **BF16 support**:
   BF16 support requires a compatible GPU. If your GPU doesn't support BF16, the tensors will be automatically converted to FP32 before quantization.

5. **Hugging Face Hub connectivity**:
   If you're having trouble connecting to Hugging Face Hub, check your internet connection and firewall settings. You can also try setting the `HF_HUB_OFFLINE=1` environment variable to use cached models.

### Getting Help

If you encounter any issues not covered here, please open an issue on the GitHub repository. 