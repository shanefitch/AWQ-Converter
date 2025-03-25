"""
Main module for AWQ Quantizer.
"""

import os
import sys
import time
from typing import Dict, List, Optional, Union

import torch

from .utils.config import load_config
from .model_loading import load_model_from_hub, load_model_from_path
from .quantization.awq import AWQQuantizer
from .utils.logger import get_logger


def main() -> int:
    """
    Main entry point for AWQ Quantizer.
    """
    try:
        # Load configuration
        config = load_config()
        
        # Initialize logger
        logger = get_logger(
            name="awq_quantizer",
            level=config["logging"]["level"],
            to_file=config["logging"]["to_file"],
            file_path=config["logging"]["file_path"],
        )
        
        # Log configuration
        logger.info("Configuration:")
        for section, values in config.config.items():
            logger.info(f"  {section}:")
            for key, value in values.items():
                logger.info(f"    {key}: {value}")
        
        # Set device
        device = torch.device(config["hardware"]["device"])
        logger.info(f"Using device: {device}")
        
        # Initialize model loader
        logger.info("Initializing model loader")
        
        if config["model"]["from_hub"]:
            loader = load_model_from_hub(
                model_id=config["model"]["path"],  # Already set to hub_model_id if needed
                revision=config["model"]["revision"],
                token=config["model"]["token"],
                logger_name="awq_quantizer",
                logger_level=config["logging"]["level"],
                logger_to_file=config["logging"]["to_file"],
                logger_file_path=config["logging"]["file_path"],
                resume_download=True,  # Always try to resume downloads
                force_download=False,  # Don't force re-download by default
            )
        else:
            loader = load_model_from_path(
                model_path=config["model"]["path"],
                logger_name="awq_quantizer",
                logger_level=config["logging"]["level"],
                logger_to_file=config["logging"]["to_file"],
                logger_file_path=config["logging"]["file_path"],
            )
        
        # Initialize quantizer
        logger.info("Initializing quantizer")
        quantizer = AWQQuantizer(
            bits=config["quantization"]["bits"],
            group_size=config["quantization"]["group_size"],
            zero_point=config["quantization"]["zero_point"],
            percentile=config["quantization"]["percentile"],
            symmetric=config["quantization"]["symmetric"],
            num_threads=config["hardware"]["num_threads"],
        )
        
        # Load model tensors
        logger.info("Loading model tensors")
        tensors = loader.load_tensors()
        
        # Quantize tensors
        logger.info("Quantizing tensors")
        start_time = time.time()
        quantized_tensors = quantizer.quantize(tensors)
        
        # Save quantized tensors
        logger.info("Saving quantized tensors")
        os.makedirs(config["output"]["dir"], exist_ok=True)
        
        output_path = os.path.join(
            config["output"]["dir"],
            "model.safetensors",
        )
        
        loader.save_tensors(
            tensors=quantized_tensors,
            output_dir=config["output"]["dir"],
            filename="model.safetensors",
        )
        
        logger.info(f"Quantization completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"Quantized model saved to {output_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during quantization: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 