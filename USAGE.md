# Usage Guide

This guide provides instructions for using the AWQ Quantizer to convert BF16 Safetensors models to I32 FP16 AWQ quantized models.

## Basic Usage

### Command Line Interface

The AWQ Quantizer provides a command-line interface for easy use:

```bash
# Basic usage with local model
awq_quantizer --model_path /path/to/model --output_dir /path/to/output

# Using a model from Hugging Face Hub (Option 1)
awq_quantizer --model_path facebook/opt-350m --from_hub --output_dir /path/to/output

# Using a model from Hugging Face Hub (Option 2 - Recommended)
awq_quantizer --hub_model_id facebook/opt-350m --output_dir /path/to/output

# Using a custom configuration file
awq_quantizer --config /path/to/config.yaml

# Display help message
awq_quantizer --help
```

### Python API

You can also use the AWQ Quantizer as a Python library:

```python
import torch
from awq_quantizer.model_loading.safetensors_loader import SafetensorsLoader
from awq_quantizer.quantization.awq import AWQQuantizer

# Initialize model loader for local model
loader = SafetensorsLoader(
    model_path="/path/to/model",
    from_hub=False,
)

# Or initialize model loader for Hugging Face Hub model
# loader = SafetensorsLoader(
#     model_path="facebook/opt-350m",
#     from_hub=True,
#     revision="main",  # Optional: specify model revision
#     token=None,       # Optional: provide token for private models
# )

# Load model
tensors = loader.load_all_tensors()

# Convert BF16 tensors to FP16
tensors = loader.convert_tensors_bf16_to_fp16(tensors)

# Initialize quantizer
quantizer = AWQQuantizer(
    bits=4,
    group_size=128,
    symmetric=True,
)

# Quantize tensors
quantized_tensors = quantizer.quantize_tensors(tensors)

# Save quantized tensors
for name, result in quantized_tensors.items():
    tensors_to_save = {
        f"{name}.q": result["tensor_q"].to(torch.int32),
        f"{name}.scales": result["scales"].to(torch.float16),
        f"{name}.zero_points": result["zero_points"].to(torch.int32),
        f"{name}.bits": result["bits"],
        f"{name}.group_size": result["group_size"],
        f"{name}.symmetric": result["symmetric"],
    }
    
    filename = f"{name.replace('/', '_')}.safetensors"
    loader.save_tensors(
        tensors_to_save,
        "/path/to/output",
        filename,
    )
```

## Configuration

### Configuration File

The AWQ Quantizer can be configured using a YAML configuration file. Here's an example:

```yaml
# Model settings
model:
  # Path to the local model
  path: "/path/to/model"
  # Hugging Face Hub model ID (e.g., 'facebook/opt-350m')
  hub_model_id: ""
  # Whether the model is on Hugging Face Hub
  from_hub: false
  # Model revision to use (if from_hub is true)
  revision: "main"
  # Token to use for private models (if from_hub is true)
  token: null

# Quantization settings
quantization:
  method: "awq"
  bits: 4
  group_size: 128
  symmetric: true
  zero_point: "minmax"
  percentile: 0.99
  scale_method: "mse"
  per_channel: true
  skip_layers: []

# Output settings
output:
  dir: "/path/to/output"
  safetensors: true
  fp16: true
  int32: true

# Logging settings
logging:
  level: "INFO"
  to_file: true
  file_path: "quantization.log"

# Hardware settings
hardware:
  device: "cuda"
  num_threads: 4
  mixed_precision: true
```

### Command Line Options

The AWQ Quantizer provides the following command-line options:

#### Model Source Options
- `--model_path`: Path to the local model directory or file
- `--hub_model_id`: Hugging Face Hub model ID (e.g., 'facebook/opt-350m'). When provided, `--from_hub` is automatically set to True.
- `--from_hub`: Load model from Hugging Face Hub. Use with `--model_path` to specify the model ID or with `--hub_model_id`.
- `--revision`: Model revision to use when loading from Hugging Face Hub
- `--token`: Authentication token for private models on Hugging Face Hub

#### Output Options
- `--output_dir`: Directory to save the quantized model
- `--config`: Path to the configuration file

#### Quantization Options
- `--bits`: Bit width for quantization (4 or 8)
- `--group_size`: Group size for quantization
- `--symmetric`: Whether to use symmetric quantization
- `--zero_point`: Zero-point calibration method (none, minmax, percentile)
- `--percentile`: Percentile value if zero_point is "percentile"
- `--scale_method`: Scale calibration method (minmax, mse)
- `--per_channel`: Whether to use per-channel quantization
- `--skip_layers`: List of layer names to skip

#### Logging Options
- `--log_level`: Logging level
- `--log_to_file`: Whether to log to file
- `--log_file`: Log file path

## Examples

### Example 1: Quantizing a Local Model

```bash
awq_quantizer \
  --model_path /path/to/model \
  --output_dir /path/to/output \
  --bits 4 \
  --group_size 128 \
  --symmetric \
  --zero_point minmax \
  --scale_method mse \
  --per_channel \
  --log_level INFO
```

### Example 2: Quantizing a Model from Hugging Face Hub

```bash
# Option 1: Using --model_path and --from_hub
awq_quantizer \
  --model_path facebook/opt-350m \
  --from_hub \
  --output_dir /path/to/output \
  --bits 4 \
  --group_size 128 \
  --symmetric \
  --zero_point minmax \
  --scale_method mse \
  --per_channel \
  --log_level INFO

# Option 2: Using --hub_model_id (Recommended)
awq_quantizer \
  --hub_model_id facebook/opt-350m \
  --output_dir /path/to/output \
  --bits 4 \
  --group_size 128 \
  --symmetric \
  --zero_point minmax \
  --scale_method mse \
  --per_channel \
  --log_level INFO
```

### Example 3: Using a Configuration File

```bash
awq_quantizer --config /path/to/config.yaml
```

You can also create a configuration file with the Hugging Face Hub model ID:

```yaml
model:
  hub_model_id: "facebook/opt-350m"
output:
  dir: "/path/to/output"
```

And then run:

```bash
awq_quantizer --config /path/to/config.yaml
```

## Advanced Usage

### Skipping Layers

You can skip certain layers from quantization using the `--skip_layers` option:

```bash
awq_quantizer \
  --hub_model_id facebook/opt-350m \
  --output_dir /path/to/output \
  --skip_layers "model.embed_tokens" "model.norm" "lm_head"
```

### Custom Quantization Parameters

You can customize the quantization parameters:

```bash
awq_quantizer \
  --hub_model_id facebook/opt-350m \
  --output_dir /path/to/output \
  --bits 8 \
  --group_size 64 \
  --zero_point percentile \
  --percentile 0.95 \
  --scale_method minmax
```

### Using a Specific Model Revision

You can specify a particular revision of a Hugging Face Hub model:

```bash
awq_quantizer \
  --hub_model_id facebook/opt-350m \
  --revision "v1.0" \
  --output_dir /path/to/output
```

### Using Authentication for Private Models

For private models on Hugging Face Hub, you can provide an authentication token:

```bash
awq_quantizer \
  --hub_model_id your-org/private-model \
  --token "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" \
  --output_dir /path/to/output
```

## Performance Considerations

- **Memory Usage**: The quantization process requires loading the entire model into memory. Make sure you have enough RAM or GPU memory.
- **Speed**: Quantization can be slow for large models. Using a GPU can significantly speed up the process.
- **Accuracy**: The quantization parameters can affect the accuracy of the quantized model. Experiment with different parameters to find the best trade-off between model size and accuracy. 