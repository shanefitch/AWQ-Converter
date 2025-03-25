# AWQ Quantizer Migration Guide

This guide will help you migrate from older versions of the AWQ Quantizer to the current version with GPU optimization.

## Major Changes

- **Modern Packaging**: Moved to PyPI-compatible packaging with `pyproject.toml`
- **GPU Acceleration**: Added explicit CUDA device management
- **Memory Efficiency**: Improved memory management for large models
- **Chunked Saving**: Added support for saving large models in chunks
- **Safetensors Format**: Added option to save in the more portable safetensors format
- **Improved Error Handling**: Better recovery from partial downloads and errors

## Migrating Your Code

### Installation

If you were using the old version:

```bash
# Old way
git clone https://github.com/original-repo/awq-quantizer.git
cd awq-quantizer
pip install -e .
```

Update to the new version:

```bash
# New way
pip install awq-quantizer
# Or for latest development version
pip install git+https://github.com/shanefitch/AWQ-Converter.git
```

### API Changes

#### Command Line Interface

The command line interface has new options for GPU and memory optimization:

```bash
# Old way
python -m awq_quantizer.main --model_id MODEL_ID --output_dir OUTPUT_DIR --bits 4 --group_size 128 --symmetric

# New way with GPU options
python -m awq_quantizer.main --model_id MODEL_ID --output_dir OUTPUT_DIR --bits 4 --group_size 128 --symmetric --device cuda --num_workers 2 --save_safetensors
```

New parameters:
- `--device`: Specify device for quantization (cuda, cuda:0, cpu)
- `--num_workers`: Number of worker threads for parallel quantization
- `--max_memory`: Maximum fraction of GPU memory to use
- `--save_safetensors`: Save in safetensors format instead of PyTorch
- `--chunk_size`: Save large models in chunks (number of tensors per chunk)

#### Python API

If you were using the Python API:

```python
# Old way
from awq_quantizer import AWQQuantizer

quantizer = AWQQuantizer(bits=4, group_size=128, symmetric=True)
result = quantizer.quantize(tensor)
```

Update to the new API:

```python
# New way with GPU options
from awq_quantizer.quantization import AWQQuantizer

quantizer = AWQQuantizer(
    bits=4, 
    group_size=128, 
    symmetric=True,
    device="cuda",  # Use GPU if available
    scale_method="mse",
    per_channel=True
)
result = quantizer.quantize(tensor)
```

### Loading Models from Hugging Face

```python
# New way
from awq_quantizer.model_loading import load_model_from_hub

# With better error handling and resume capability
tensors = load_model_from_hub(
    model_id="mistralai/Mistral-7B-v0.1",
    resume_download=True,
    force_download=False
)
```

## Testing Your Installation

We provide a test script to verify your installation:

```bash
# Run the test script
python test_installation.py
```

This will check:
1. If the main package is installed
2. If all required submodules are available
3. If the command-line interface works
4. If GPU acceleration is available

## Output File Format

When using the chunked saving feature, the output directory will contain:

- `metadata.json`: Contains mapping of tensor names to chunk files
- Multiple chunk files (either `.pt` or `.safetensors` depending on the format)

## Troubleshooting

If you encounter issues after migration:

1. Make sure you have the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Check your CUDA version is compatible with the installed PyTorch version:
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"
   ```

3. Run the test script to verify your installation:
   ```bash
   python test_installation.py
   ```

4. Check the logs for any specific error messages.

## Full Example

Here's a complete example of quantizing a model with all the new features:

```bash
python -m awq_quantizer.main \
  --model_id mistralai/Mistral-7B-v0.1 \
  --output_dir ./quantized_model \
  --bits 4 \
  --group_size 128 \
  --symmetric \
  --device cuda \
  --num_workers 2 \
  --save_safetensors \
  --chunk_size 50
```

This will:
1. Download the Mistral 7B model from Hugging Face
2. Quantize it using AWQ with 4-bit precision
3. Use CUDA acceleration
4. Save the quantized model in safetensors format in chunks of 50 tensors each 