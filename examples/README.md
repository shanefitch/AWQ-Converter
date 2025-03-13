# AWQ Quantizer Examples

This directory contains example scripts demonstrating how to use the AWQ Quantizer library.

## Basic Usage Example

The `basic_usage.py` script demonstrates the simplest way to use AWQ Quantizer to load a model from Hugging Face Hub and quantize it.

```bash
# Run with default parameters (quantizes facebook/opt-125m)
./basic_usage.py

# Quantize a different model
./basic_usage.py --hub_model_id "facebook/opt-350m" --output_dir "./my_quantized_model"

# Change quantization parameters
./basic_usage.py --bits 8 --group_size 64

# Enable verbose logging
./basic_usage.py --verbose
```

## Advanced Usage Example

The `advanced_usage.py` script demonstrates how to use AWQ Quantizer with a configuration file for more advanced customization options.

```bash
# Run with the provided example configuration
./advanced_usage.py --config config_example.yaml

# Enable verbose logging
./advanced_usage.py --config config_example.yaml --verbose
```

## Benchmark Script

The `benchmark.py` script measures the performance of original and quantized models, comparing inference speed, memory usage, and other metrics.

```bash
# Run with default parameters (benchmarks facebook/opt-125m)
./benchmark.py

# Benchmark a different model
./benchmark.py --hub_model_id "facebook/opt-350m" --output_dir "./my_benchmark_results"

# Change quantization parameters
./benchmark.py --bits 8 --group_size 64

# Adjust benchmark settings
./benchmark.py --num_iterations 20 --input_size 1024 --batch_size 2

# Enable verbose logging
./benchmark.py --verbose
```

## Configuration File

The `config_example.yaml` file provides a template for configuring the AWQ Quantizer. You can modify this file or create your own configuration files based on it.

Key sections in the configuration file:

1. **Model Configuration**: Specifies how to load the model (local path or Hugging Face Hub).
2. **Output Configuration**: Defines where and how to save the quantized model.
3. **Quantization Parameters**: Controls the quantization process with various options.

## Creating Your Own Examples

Feel free to modify these examples or create your own based on them. The AWQ Quantizer library provides a flexible API that can be adapted to various use cases.

For more information, refer to the main [README.md](../README.md) and [INSTALL.md](../INSTALL.md) files. 