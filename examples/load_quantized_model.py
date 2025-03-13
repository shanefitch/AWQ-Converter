#!/usr/bin/env python3
"""
Load Quantized Model Example for AWQ Quantizer.

This script demonstrates how to load a previously quantized model and use it for inference.
"""

import os
import argparse
import torch
from awq_quantizer.model_loading import load_quantized_model
from awq_quantizer.utils import setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description="Load Quantized Model Example")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to the quantized model directory"
    )
    parser.add_argument(
        "--input_text", 
        type=str, 
        default="Hello, world!",
        help="Input text for inference (default: 'Hello, world!')"
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=50,
        help="Maximum length for generated text (default: 50)"
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
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load quantized model
    print(f"Loading quantized model from {args.model_path}...")
    model_data = load_quantized_model(args.model_path)
    model = model_data.get_model().to(device)
    tokenizer = model_data.get_tokenizer()
    
    # Print model information
    print("\nModel Information:")
    print(f"  - Model type: {model.__class__.__name__}")
    print(f"  - Quantization bits: {model_data.get_metadata().get('quantization_bits', 'Unknown')}")
    print(f"  - Original model: {model_data.get_metadata().get('original_model', 'Unknown')}")
    
    # Tokenize input
    print(f"\nInput text: '{args.input_text}'")
    inputs = tokenizer(args.input_text, return_tensors="pt").to(device)
    
    # Generate text
    print("\nGenerating text...")
    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            max_length=args.max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    # Decode and print output
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\nGenerated text:")
    print(output_text)
    
    # Print memory usage
    if torch.cuda.is_available():
        memory_usage = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
        print(f"\nGPU memory usage: {memory_usage:.2f} GB")

if __name__ == "__main__":
    main() 