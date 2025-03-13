# AWQ Quantizer

A tool for converting multi-file BF16 Safetensors models to I32 FP16 AWQ quantized models.

> **Note**: We've recently updated the project to use modern Python packaging with `pyproject.toml` instead of `setup.py`. If you're upgrading from a previous version, please see [MIGRATION.md](MIGRATION.md) for details.

## Overview

This project provides a pipeline for loading, processing, and quantizing large language models stored in the Safetensors format. It specifically targets models with BF16 precision and converts them to AWQ quantized models with I32 FP16 precision.

## Features

- Load multi-file Safetensors models from local storage or Hugging Face Hub
- Direct integration with Hugging Face Hub for seamless model downloading
- Convert BF16 tensors to appropriate formats for quantization
- Apply AWQ (Activation-aware Weight Quantization) to reduce model size
- Save quantized models in Safetensors format with I32 FP16 precision
- Maintain model accuracy while reducing memory footprint

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/awq_quantizer.git
cd awq_quantizer

# Create a virtual environment (optional but recommended)
python -m venv awq_env
source awq_env/bin/activate  # On Windows: awq_env\Scripts\activate

# Install in development mode
pip install -e .

# If you encounter any issues, try with PEP 517 explicitly enabled
# pip install --use-pep517 -e .
```

For more detailed installation instructions, see [INSTALL.md](INSTALL.md).

## Verify Installation

To verify that the installation was successful:

```bash
# Run the test installation script
./test_installation.py

# Or test the command-line tool directly
awq_quantizer --help
```

## Usage

```bash
# Basic usage with local model
awq_quantizer --model_path "path/to/model" --output_dir "path/to/output"

# Using a model directly from Hugging Face Hub
awq_quantizer --hub_model_id "facebook/opt-350m" --output_dir "path/to/output"

# Advanced usage with custom configuration
awq_quantizer --config "path/to/config.yaml"
```

## Hugging Face Hub Integration

AWQ Quantizer provides seamless integration with Hugging Face Hub, allowing you to quantize models directly from the Hub without downloading them manually:

```bash
# Using the hub_model_id parameter (recommended)
awq_quantizer --hub_model_id "facebook/opt-350m" --output_dir "path/to/output"

# For private models, you can provide an authentication token
awq_quantizer --hub_model_id "your-org/private-model" --token "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" --output_dir "path/to/output"

# You can also specify a particular revision
awq_quantizer --hub_model_id "facebook/opt-350m" --revision "v1.0" --output_dir "path/to/output"
```

## Configuration

The quantization process can be customized through YAML configuration files. See `src/awq_quantizer/config/default_config.yaml` for an example.

You can also create a configuration file with the Hugging Face Hub model ID:

```yaml
model:
  hub_model_id: "facebook/opt-350m"
output:
  dir: "quantized_model"
```

And then run:

```bash
awq_quantizer --config "path/to/config.yaml"
```

## Testing

To run the test script:

```bash
# Test with locally created tensors
python test_quantization.py

# Test with a small model from Hugging Face Hub
python test_quantization.py --use_hub
```

## License

[MIT License](LICENSE)

## Acknowledgements

This project implements the AWQ quantization method as described in the paper [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978). 