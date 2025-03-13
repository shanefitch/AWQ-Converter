"""
Main module for AWQ Quantizer.
"""

import argparse
import os
import time
from typing import Dict, Optional

import torch

from .model_loading.safetensors_loader import SafetensorsLoader
from .quantization.awq import AWQQuantizer
from .utils.config import load_config
from .utils.logger import get_logger
from .utils.tensor_utils import get_device_from_config


def parse_args():
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="AWQ Quantizer")
    
    model_group = parser.add_argument_group("Model Source")
    model_group.add_argument(
        "--model_path",
        type=str,
        help="Path to the local model directory or file",
    )
    
    model_group.add_argument(
        "--hub_model_id",
        type=str,
        help="Hugging Face Hub model ID (e.g., 'facebook/opt-350m'). When provided, --from_hub is automatically set to True.",
    )
    
    model_group.add_argument(
        "--from_hub",
        action="store_true",
        help="Load model from Hugging Face Hub. Use with --model_path to specify the model ID or with --hub_model_id.",
    )
    
    model_group.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Model revision to use when loading from Hugging Face Hub",
    )
    
    model_group.add_argument(
        "--token",
        type=str,
        help="Authentication token for private models on Hugging Face Hub",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save the quantized model",
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the configuration file",
    )
    
    parser.add_argument(
        "--bits",
        type=int,
        choices=[4, 8],
        default=4,
        help="Bit width for quantization (4 or 8)",
    )
    
    parser.add_argument(
        "--group_size",
        type=int,
        default=128,
        help="Group size for quantization",
    )
    
    parser.add_argument(
        "--symmetric",
        action="store_true",
        help="Whether to use symmetric quantization",
    )
    
    parser.add_argument(
        "--zero_point",
        type=str,
        choices=["none", "minmax", "percentile"],
        default="minmax",
        help="Zero-point calibration method (none, minmax, percentile)",
    )
    
    parser.add_argument(
        "--percentile",
        type=float,
        default=0.99,
        help="Percentile value if zero_point is 'percentile'",
    )
    
    parser.add_argument(
        "--scale_method",
        type=str,
        choices=["minmax", "mse"],
        default="mse",
        help="Scale calibration method (minmax, mse)",
    )
    
    parser.add_argument(
        "--per_channel",
        action="store_true",
        help="Whether to use per-channel quantization",
    )
    
    parser.add_argument(
        "--skip_layers",
        type=str,
        nargs="+",
        default=[],
        help="List of layer names to skip",
    )
    
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level",
    )
    
    parser.add_argument(
        "--log_to_file",
        action="store_true",
        help="Whether to log to file",
    )
    
    parser.add_argument(
        "--log_file",
        type=str,
        default="quantization.log",
        help="Log file path",
    )
    
    return parser.parse_args()


def main():
    """
    Main function.
    """
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Handle hub_model_id parameter
    if args.hub_model_id:
        args.model_path = args.hub_model_id
        args.from_hub = True
    
    # Update configuration with command line arguments
    if args.model_path:
        config["model"]["path"] = args.model_path
    
    if args.from_hub:
        config["model"]["from_hub"] = args.from_hub
    
    if args.revision:
        config["model"]["revision"] = args.revision
    
    if args.token:
        config["model"]["token"] = args.token
    
    if args.output_dir:
        config["output"]["dir"] = args.output_dir
    
    if args.bits:
        config["quantization"]["bits"] = args.bits
    
    if args.group_size:
        config["quantization"]["group_size"] = args.group_size
    
    if args.symmetric:
        config["quantization"]["symmetric"] = args.symmetric
    
    if args.zero_point:
        config["quantization"]["zero_point"] = args.zero_point
    
    if args.percentile:
        config["quantization"]["percentile"] = args.percentile
    
    if args.scale_method:
        config["quantization"]["scale_method"] = args.scale_method
    
    if args.per_channel:
        config["quantization"]["per_channel"] = args.per_channel
    
    if args.skip_layers:
        config["quantization"]["skip_layers"] = args.skip_layers
    
    if args.log_level:
        config["logging"]["level"] = args.log_level
    
    if args.log_to_file:
        config["logging"]["to_file"] = args.log_to_file
    
    if args.log_file:
        config["logging"]["file_path"] = args.log_file
    
    # Initialize logger
    logger = get_logger(
        name="awq_quantizer",
        level=config["logging"]["level"],
        to_file=config["logging"]["to_file"],
        file_path=config["logging"]["file_path"],
    )
    
    # Log configuration
    logger.info("Configuration:")
    for section, params in config.config.items():
        logger.info(f"  {section}:")
        for key, value in params.items():
            logger.info(f"    {key}: {value}")
    
    # Check if model path is provided
    if not config["model"]["path"]:
        logger.error("Model path is not provided")
        raise ValueError("Model path is not provided. Use --model_path or --hub_model_id to specify the model.")
    
    # Check if output directory is provided
    if not config["output"]["dir"]:
        logger.error("Output directory is not provided")
        raise ValueError("Output directory is not provided")
    
    # Get device
    device = get_device_from_config(config.config)
    logger.info(f"Using device: {device}")
    
    # Initialize model loader
    logger.info("Initializing model loader")
    loader = SafetensorsLoader(
        model_path=config["model"]["path"],
        from_hub=config["model"]["from_hub"],
        revision=config["model"]["revision"],
        token=config["model"]["token"],
        logger_name="safetensors_loader",
        logger_level=config["logging"]["level"],
        logger_to_file=config["logging"]["to_file"],
        logger_file_path=config["logging"]["file_path"],
    )
    
    # Load model
    logger.info("Loading model")
    start_time = time.time()
    tensors = loader.load_all_tensors()
    logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
    
    # Convert BF16 tensors to FP16
    logger.info("Converting BF16 tensors to FP16")
    start_time = time.time()
    tensors = loader.convert_tensors_bf16_to_fp16(tensors)
    logger.info(f"Conversion completed in {time.time() - start_time:.2f} seconds")
    
    # Move tensors to device
    logger.info(f"Moving tensors to {device}")
    start_time = time.time()
    for name, tensor in tensors.items():
        tensors[name] = tensor.to(device)
    logger.info(f"Tensors moved in {time.time() - start_time:.2f} seconds")
    
    # Initialize quantizer
    logger.info("Initializing quantizer")
    quantizer = AWQQuantizer(
        bits=config["quantization"]["bits"],
        group_size=config["quantization"]["group_size"],
        symmetric=config["quantization"]["symmetric"],
        zero_point=config["quantization"]["zero_point"],
        percentile=config["quantization"]["percentile"],
        scale_method=config["quantization"]["scale_method"],
        per_channel=config["quantization"]["per_channel"],
        logger_name="awq_quantizer",
        logger_level=config["logging"]["level"],
        logger_to_file=config["logging"]["to_file"],
        logger_file_path=config["logging"]["file_path"],
    )
    
    # Quantize tensors
    logger.info("Quantizing tensors")
    start_time = time.time()
    quantized_tensors = quantizer.quantize_tensors(
        tensors, skip_layers=config["quantization"]["skip_layers"]
    )
    logger.info(f"Quantization completed in {time.time() - start_time:.2f} seconds")
    
    # Save quantized tensors
    logger.info("Saving quantized tensors")
    start_time = time.time()
    
    # Create output directory
    os.makedirs(config["output"]["dir"], exist_ok=True)
    
    # Save metadata
    metadata = {
        "quantization_method": "awq",
        "bits": str(config["quantization"]["bits"]),
        "group_size": str(config["quantization"]["group_size"]),
        "symmetric": str(config["quantization"]["symmetric"]),
        "zero_point": config["quantization"]["zero_point"],
        "percentile": str(config["quantization"]["percentile"]),
        "scale_method": config["quantization"]["scale_method"],
        "per_channel": str(config["quantization"]["per_channel"]),
    }
    
    # Save each tensor to a separate file
    for name, result in quantized_tensors.items():
        # Create a dictionary with all tensors for this layer
        tensors_to_save = {
            f"{name}.q": result["tensor_q"].to(torch.int32).cpu(),
            f"{name}.scales": result["scales"].to(torch.float16).cpu(),
            f"{name}.zero_points": result["zero_points"].to(torch.int32).cpu(),
            f"{name}.bits": result["bits"].cpu(),
            f"{name}.group_size": result["group_size"].cpu(),
            f"{name}.symmetric": result["symmetric"].cpu(),
        }
        
        # Save to file
        filename = f"{name.replace('/', '_')}.safetensors"
        loader.save_tensors(
            tensors_to_save,
            config["output"]["dir"],
            filename,
            metadata,
        )
    
    logger.info(f"Saving completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Quantized model saved to {config['output']['dir']}")


if __name__ == "__main__":
    main() 