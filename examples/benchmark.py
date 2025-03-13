#!/usr/bin/env python3
"""
Benchmark script for AWQ Quantizer.

This script measures the performance of original and quantized models,
comparing inference speed, memory usage, and accuracy.
"""

import os
import time
import argparse
import torch
import numpy as np
from awq_quantizer import Quantizer
from awq_quantizer.model_loading import load_model, load_model_from_hub
from awq_quantizer.utils import setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description="AWQ Quantizer Benchmark")
    parser.add_argument(
        "--hub_model_id", 
        type=str, 
        default="facebook/opt-125m",
        help="Hugging Face Hub model ID to benchmark (default: facebook/opt-125m)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./benchmark_results",
        help="Directory to save benchmark results (default: ./benchmark_results)"
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
        "--num_iterations", 
        type=int, 
        default=10,
        help="Number of inference iterations for benchmarking (default: 10)"
    )
    parser.add_argument(
        "--input_size", 
        type=int, 
        default=512,
        help="Input sequence length for benchmarking (default: 512)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1,
        help="Batch size for benchmarking (default: 1)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    return parser.parse_args()

def measure_memory_usage():
    """Measure current GPU memory usage."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
    return 0

def benchmark_inference(model, input_data, num_iterations):
    """Benchmark inference speed."""
    # Warmup
    for _ in range(3):
        _ = model(input_data)
    
    # Measure inference time
    start_time = time.time()
    for _ in range(num_iterations):
        _ = model(input_data)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations
    return avg_time

def main():
    args = parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model {args.hub_model_id} from Hugging Face Hub...")
    model_data = load_model_from_hub(args.hub_model_id)
    original_model = model_data.get_model().to(device)
    
    # Generate random input data
    input_ids = torch.randint(
        0, 1000, (args.batch_size, args.input_size), 
        dtype=torch.long, device=device
    )
    attention_mask = torch.ones(
        (args.batch_size, args.input_size), 
        dtype=torch.long, device=device
    )
    input_data = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
    
    # Benchmark original model
    print("Benchmarking original model...")
    torch.cuda.empty_cache()
    original_memory = measure_memory_usage()
    original_time = benchmark_inference(original_model, input_data, args.num_iterations)
    
    # Quantize model
    print(f"Quantizing model to {args.bits} bits with group size {args.group_size}...")
    quantizer = Quantizer(
        bits=args.bits,
        group_size=args.group_size,
    )
    
    quantized_model_data = quantizer.quantize(model_data)
    quantized_model = quantized_model_data.get_model().to(device)
    
    # Benchmark quantized model
    print("Benchmarking quantized model...")
    torch.cuda.empty_cache()
    quantized_memory = measure_memory_usage()
    quantized_time = benchmark_inference(quantized_model, input_data, args.num_iterations)
    
    # Print results
    print("\n===== Benchmark Results =====")
    print(f"Model: {args.hub_model_id}")
    print(f"Quantization: {args.bits}-bit, group size {args.group_size}")
    print(f"Batch size: {args.batch_size}, Sequence length: {args.input_size}")
    print(f"Number of iterations: {args.num_iterations}")
    print("\nOriginal Model:")
    print(f"  - Memory usage: {original_memory:.2f} GB")
    print(f"  - Average inference time: {original_time * 1000:.2f} ms")
    print("\nQuantized Model:")
    print(f"  - Memory usage: {quantized_memory:.2f} GB")
    print(f"  - Average inference time: {quantized_time * 1000:.2f} ms")
    print("\nComparison:")
    print(f"  - Memory reduction: {(1 - quantized_memory / original_memory) * 100:.2f}%")
    print(f"  - Speedup: {original_time / quantized_time:.2f}x")
    
    # Save results to file
    results_file = os.path.join(args.output_dir, f"benchmark_{args.hub_model_id.replace('/', '_')}_{args.bits}bit.txt")
    with open(results_file, 'w') as f:
        f.write("===== AWQ Quantizer Benchmark Results =====\n")
        f.write(f"Model: {args.hub_model_id}\n")
        f.write(f"Quantization: {args.bits}-bit, group size {args.group_size}\n")
        f.write(f"Batch size: {args.batch_size}, Sequence length: {args.input_size}\n")
        f.write(f"Number of iterations: {args.num_iterations}\n\n")
        f.write("Original Model:\n")
        f.write(f"  - Memory usage: {original_memory:.2f} GB\n")
        f.write(f"  - Average inference time: {original_time * 1000:.2f} ms\n\n")
        f.write("Quantized Model:\n")
        f.write(f"  - Memory usage: {quantized_memory:.2f} GB\n")
        f.write(f"  - Average inference time: {quantized_time * 1000:.2f} ms\n\n")
        f.write("Comparison:\n")
        f.write(f"  - Memory reduction: {(1 - quantized_memory / original_memory) * 100:.2f}%\n")
        f.write(f"  - Speedup: {original_time / quantized_time:.2f}x\n")
    
    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    main() 