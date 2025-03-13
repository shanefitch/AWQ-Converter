#!/usr/bin/env python3
"""
Advanced usage example for AWQ Quantizer.

This script demonstrates how to use the AWQ Quantizer with a configuration file
for more advanced customization options.
"""

import os
import argparse
import yaml
from awq_quantizer import Quantizer
from awq_quantizer.model_loading import load_model
from awq_quantizer.utils import setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description="AWQ Quantizer Advanced Example")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to the configuration YAML file"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    args = parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    
    # Load configuration
    print(f"Loading configuration from {args.config}...")
    config = load_config(args.config)
    
    # Extract configuration values
    model_config = config.get('model', {})
    output_config = config.get('output', {})
    quant_config = config.get('quantization', {})
    
    # Create output directory if it doesn't exist
    output_dir = output_config.get('dir', './quantized_model')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model based on configuration
    print("Loading model...")
    model_data = load_model(
        local_path=model_config.get('local_path'),
        hub_model_id=model_config.get('hub_model_id'),
        revision=model_config.get('revision'),
        token=model_config.get('token'),
        cache_dir=model_config.get('cache_dir')
    )
    
    # Configure quantizer
    print("Configuring quantizer...")
    quantizer = Quantizer(
        bits=quant_config.get('bits', 4),
        group_size=quant_config.get('group_size', 128),
        zero_point=quant_config.get('zero_point', True),
        use_fp16_scale=quant_config.get('use_fp16_scale', True),
        scale_dtype=quant_config.get('scale_dtype', 'fp16'),
        symmetric=quant_config.get('symmetric', False),
        calibration_method=quant_config.get('calibration_method', 'minmax'),
        calibration_percdamp=quant_config.get('calibration_percdamp', 0.01),
        calibration_percentile=quant_config.get('calibration_percentile', 0.999),
    )
    
    # Perform quantization
    print("Quantizing model...")
    quantized_model = quantizer.quantize(model_data)
    
    # Save quantized model
    print(f"Saving quantized model to {output_dir}...")
    save_format = output_config.get('format', 'safetensors')
    quantized_model.save(
        output_dir, 
        format=save_format,
        metadata=output_config.get('metadata', {})
    )
    
    print("Quantization complete!")
    print(f"Original model size: {model_data.get_size_in_gb():.2f} GB")
    print(f"Quantized model size: {quantized_model.get_size_in_gb():.2f} GB")
    print(f"Compression ratio: {model_data.get_size_in_gb() / quantized_model.get_size_in_gb():.2f}x")

if __name__ == "__main__":
    main() 