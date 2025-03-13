"""
Test script for AWQ Quantizer.
"""

import argparse
import os
import torch
from safetensors.torch import save_file

from src.awq_quantizer.model_loading.safetensors_loader import SafetensorsLoader
from src.awq_quantizer.quantization.awq import AWQQuantizer


def parse_args():
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Test AWQ Quantizer")
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test_output",
        help="Directory to save the test output",
    )
    
    parser.add_argument(
        "--use_hub",
        action="store_true",
        help="Test with a small model from Hugging Face Hub instead of creating test tensors",
    )
    
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default="hf-internal-testing/tiny-random-gptj",
        help="Hugging Face Hub model ID to use for testing (if --use_hub is set)",
    )
    
    return parser.parse_args()


def create_test_tensors():
    """
    Create test tensors.

    Returns:
        Dictionary of test tensors
    """
    # Create a linear weight tensor
    linear_weight = torch.randn(768, 3072, dtype=torch.bfloat16)
    
    # Create an attention weight tensor
    attention_weight = torch.randn(768, 3, 768, dtype=torch.bfloat16)
    
    # Create a small tensor
    small_tensor = torch.randn(10, 10, dtype=torch.bfloat16)
    
    # Create a non-floating point tensor
    int_tensor = torch.randint(0, 100, (100, 100), dtype=torch.int32)
    
    return {
        "model.layers.0.mlp.fc1.weight": linear_weight,
        "model.layers.0.self_attn.qkv_proj.weight": attention_weight,
        "model.layers.0.small_tensor": small_tensor,
        "model.layers.0.int_tensor": int_tensor,
    }


def save_test_tensors(tensors, output_dir):
    """
    Save test tensors.

    Args:
        tensors: Dictionary of test tensors
        output_dir: Output directory
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save tensors
    save_file(tensors, os.path.join(output_dir, "model.safetensors"))


def main():
    """
    Main function.
    """
    # Parse arguments
    args = parse_args()
    
    if args.use_hub:
        print(f"Testing with model from Hugging Face Hub: {args.hub_model_id}")
        
        # Initialize model loader for Hugging Face Hub
        loader = SafetensorsLoader(
            model_path=args.hub_model_id,
            from_hub=True,
            logger_level="INFO",
        )
        
        # Load model
        print("Loading model from Hugging Face Hub")
        loaded_tensors = loader.load_all_tensors()
        
    else:
        # Create test tensors
        print("Creating test tensors")
        tensors = create_test_tensors()
        
        # Save test tensors
        print("Saving test tensors")
        save_test_tensors(tensors, args.output_dir)
        
        # Initialize model loader
        print("Initializing model loader")
        loader = SafetensorsLoader(
            model_path=args.output_dir,
            from_hub=False,
            logger_level="INFO",
        )
        
        # Load model
        print("Loading model")
        loaded_tensors = loader.load_all_tensors()
    
    # Convert BF16 tensors to FP16
    print("Converting BF16 tensors to FP16")
    converted_tensors = loader.convert_tensors_bf16_to_fp16(loaded_tensors)
    
    # Initialize quantizer
    print("Initializing quantizer")
    quantizer = AWQQuantizer(
        bits=4,
        group_size=128,
        symmetric=True,
        zero_point="minmax",
        percentile=0.99,
        scale_method="mse",
        per_channel=True,
        logger_level="INFO",
    )
    
    # Quantize tensors
    print("Quantizing tensors")
    quantized_tensors = quantizer.quantize_tensors(converted_tensors)
    
    # Dequantize tensors
    print("Dequantizing tensors")
    dequantized_tensors = quantizer.dequantize_tensors(quantized_tensors)
    
    # Calculate error
    print("Calculating error")
    for name, tensor in converted_tensors.items():
        if name in dequantized_tensors:
            error = torch.abs(tensor - dequantized_tensors[name]).mean().item()
            print(f"Error for {name}: {error:.6f}")
    
    # Save quantized tensors
    print("Saving quantized tensors")
    quantized_output_dir = os.path.join(args.output_dir, "quantized")
    os.makedirs(quantized_output_dir, exist_ok=True)
    
    # Save metadata
    metadata = {
        "quantization_method": "awq",
        "bits": "4",
        "group_size": "128",
        "symmetric": "True",
        "zero_point": "minmax",
        "percentile": "0.99",
        "scale_method": "mse",
        "per_channel": "True",
    }
    
    # Save each tensor to a separate file
    for name, result in quantized_tensors.items():
        # Create a dictionary with all tensors for this layer
        tensors_to_save = {
            f"{name}.q": result["tensor_q"].to(torch.int32),
            f"{name}.scales": result["scales"].to(torch.float16),
            f"{name}.zero_points": result["zero_points"].to(torch.int32),
            f"{name}.bits": result["bits"],
            f"{name}.group_size": result["group_size"],
            f"{name}.symmetric": result["symmetric"],
        }
        
        # Save to file
        filename = f"{name.replace('/', '_')}.safetensors"
        loader.save_tensors(
            tensors_to_save,
            quantized_output_dir,
            filename,
            metadata,
        )
    
    print(f"Quantized model saved to {quantized_output_dir}")
    
    if args.use_hub:
        print("\nSuccessfully tested AWQ Quantizer with a model from Hugging Face Hub!")
    else:
        print("\nSuccessfully tested AWQ Quantizer with locally created test tensors!")


if __name__ == "__main__":
    main() 