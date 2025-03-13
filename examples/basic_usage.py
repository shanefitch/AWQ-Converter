#!/usr/bin/env python3
"""
Basic usage example for AWQ Quantizer.

This script demonstrates how to use the AWQ Quantizer to load a model from
Hugging Face Hub and quantize it.
"""

import os
import argparse
from awq_quantizer import Quantizer
from awq_quantizer.model_loading import load_model_from_hub
from awq_quantizer.utils import setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description="AWQ Quantizer Basic Example")
    parser.add_argument(
        "--hub_model_id", 
        type=str, 
        default="facebook/opt-125m",
        help="Hugging Face Hub model ID to quantize (default: facebook/opt-125m)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./quantized_model",
        help="Directory to save the quantized model (default: ./quantized_model)"
    )
    parser.add_argument(
        "--bits", 
        type=int, 
        default=4,
        help="Quantization bit-width (default: 4)"
    )
    parser.add_argument(
        "--group_size", 
        type=int, 
        default=128,
        help="Group size for quantization (default: 128)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading model {args.hub_model_id} from Hugging Face Hub...")
    model_data = load_model_from_hub(args.hub_model_id)
    
    print(f"Quantizing model to {args.bits} bits with group size {args.group_size}...")
    quantizer = Quantizer(
        bits=args.bits,
        group_size=args.group_size,
    )
    
    quantized_model = quantizer.quantize(model_data)
    
    print(f"Saving quantized model to {args.output_dir}...")
    quantized_model.save(args.output_dir)
    
    print("Quantization complete!")
    print(f"Original model size: {model_data.get_size_in_gb():.2f} GB")
    print(f"Quantized model size: {quantized_model.get_size_in_gb():.2f} GB")
    print(f"Compression ratio: {model_data.get_size_in_gb() / quantized_model.get_size_in_gb():.2f}x")

if __name__ == "__main__":
    main() 