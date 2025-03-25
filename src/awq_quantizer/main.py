"""
Main module for AWQ Quantizer.
"""

import os
import sys
import argparse
import time
import logging
import json
from typing import Dict, List, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

from .utils.config import load_config
from .model_loading import load_model_from_hub, load_model_from_path
from .quantization.awq import AWQQuantizer
from .utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="AWQ Quantizer CLI")
    
    # Model loading arguments
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model ID on HuggingFace Hub or path to local model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save quantized model",
    )
    
    # Quantization arguments
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="Number of bits for quantization",
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
        help="Use symmetric quantization",
    )
    parser.add_argument(
        "--zero_point",
        type=str,
        default="minmax",
        choices=["none", "minmax", "percentile"],
        help="Zero point calibration method",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=0.99,
        help="Percentile for zero point calibration",
    )
    parser.add_argument(
        "--scale_method",
        type=str,
        default="mse",
        choices=["minmax", "mse"],
        help="Scale calibration method",
    )
    parser.add_argument(
        "--per_channel",
        action="store_true",
        help="Use per-channel quantization",
    )
    
    # Hardware arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for quantization (cuda, cuda:0, cuda:1, cpu, or 'all' for all GPUs)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker threads for parallel processing per GPU",
    )
    parser.add_argument(
        "--max_memory",
        type=float,
        default=0.8,
        help="Maximum fraction of GPU memory to use (0.0-1.0)",
    )
    parser.add_argument(
        "--multi_gpu",
        action="store_true",
        help="Use all available GPUs for processing (overrides --device)",
    )
    
    # Logging arguments
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        help="Log file path",
    )
    
    # Saving arguments
    parser.add_argument(
        "--save_safetensors",
        action="store_true",
        help="Save in safetensors format instead of pytorch format",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10,
        help="Number of tensors to save in each chunk (for large models)",
    )
    
    return parser.parse_args()


def get_available_gpus(logger=None) -> List[str]:
    """
    Get a list of available CUDA devices.
    
    Args:
        logger: Optional logger instance
        
    Returns:
        List of device strings (e.g., ["cuda:0", "cuda:1"])
    """
    if not torch.cuda.is_available():
        if logger:
            logger.warning("No CUDA devices available")
        return []
    
    devices = []
    device_count = torch.cuda.device_count()
    
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        if logger:
            logger.info(f"Found CUDA device {i}: {device_name}")
        devices.append(f"cuda:{i}")
    
    return devices


def get_device_memory_info(device_idx: int) -> Tuple[float, float]:
    """
    Get memory information for a CUDA device.
    
    Args:
        device_idx: Device index
        
    Returns:
        Tuple of (total_memory_gb, free_memory_gb)
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0
    
    try:
        # Get total memory
        total_memory = torch.cuda.get_device_properties(device_idx).total_memory
        total_memory_gb = total_memory / 1024**3
        
        # Get free memory
        free_memory = torch.cuda.memory_reserved(device_idx) - torch.cuda.memory_allocated(device_idx)
        free_memory_gb = free_memory / 1024**3
        
        return total_memory_gb, free_memory_gb
    except Exception:
        return 0.0, 0.0


def prepare_tensors_for_quantization(
    tensors: Dict[str, torch.Tensor], 
    device: str,
    max_memory_fraction: float = 0.8,
    logger = None
) -> Dict[str, torch.Tensor]:
    """
    Prepare tensors for quantization by filtering, validating, and moving them to the specified device.
    
    Args:
        tensors: Dictionary of tensors from the model
        device: Device to move tensors to (cuda, cuda:0, etc.)
        max_memory_fraction: Maximum fraction of GPU memory to use (0.0-1.0)
        logger: Logger instance
        
    Returns:
        Dictionary of tensors ready for quantization
    """
    prepared_tensors = {}
    
    # Check if using CUDA
    is_cuda = device.startswith("cuda")
    
    # Get available memory if using CUDA
    if is_cuda and logger:
        device_idx = 0
        if ":" in device:
            device_idx = int(device.split(":")[1])
        
        total_memory = torch.cuda.get_device_properties(device_idx).total_memory
        max_allowed_memory = int(total_memory * max_memory_fraction)
        
        logger.info(f"GPU {device_idx} has {total_memory / 1024**3:.2f} GB total memory")
        logger.info(f"Using up to {max_allowed_memory / 1024**3:.2f} GB for tensors")
        
        # Reserve memory for quantization operations
        current_memory = torch.cuda.memory_allocated(device_idx)
        available_memory = max_allowed_memory - current_memory
        
        logger.info(f"Current memory usage: {current_memory / 1024**3:.2f} GB")
        logger.info(f"Available memory for tensors: {available_memory / 1024**3:.2f} GB")
    
    # Process each tensor
    for name, tensor in tensors.items():
        # Skip non-tensor values
        if not isinstance(tensor, torch.Tensor):
            if logger:
                logger.warning(f"Skipping non-tensor value: {name}")
            continue
            
        # Skip non-floating point tensors
        if not tensor.is_floating_point():
            if logger:
                logger.warning(f"Skipping non-floating point tensor: {name}")
            continue
            
        # Skip tensors with no elements
        if tensor.numel() == 0:
            if logger:
                logger.warning(f"Skipping empty tensor: {name}")
            continue
            
        # Skip tensors that are too small for grouping
        if tensor.numel() < 128:  # Minimum size for group quantization
            if logger:
                logger.warning(f"Skipping tensor too small for grouping: {name}")
            continue
            
        # Move tensor to device (for CUDA, check memory availability first)
        if is_cuda:
            tensor_size = tensor.numel() * tensor.element_size()
            
            if 'available_memory' in locals() and tensor_size > available_memory:
                if logger:
                    logger.warning(
                        f"Tensor {name} size ({tensor_size / 1024**3:.2f} GB) exceeds available GPU memory. "
                        f"Processing on CPU instead."
                    )
                # Keep on CPU if too large
                prepared_tensors[name] = tensor
            else:
                # Track memory usage
                if 'available_memory' in locals():
                    available_memory -= tensor_size
                
                try:
                    # Move to GPU
                    prepared_tensors[name] = tensor.to(device)
                    
                    if logger and logger.level <= logging.DEBUG:
                        logger.debug(
                            f"Moved tensor {name} (shape: {tensor.shape}, " 
                            f"size: {tensor_size / 1024**2:.2f} MB) to {device}"
                        )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        if logger:
                            logger.warning(f"GPU OOM when loading tensor {name}, processing on CPU instead: {e}")
                        # Keep on CPU if CUDA OOM
                        prepared_tensors[name] = tensor
                    else:
                        raise
        else:
            # CPU processing, just add the tensor
            prepared_tensors[name] = tensor
            
    if logger:
        logger.info(f"Prepared {len(prepared_tensors)} tensors for quantization")
        
        if is_cuda:
            current_memory = torch.cuda.memory_allocated(device_idx)
            logger.info(f"Current GPU memory usage after preparation: {current_memory / 1024**3:.2f} GB")
    
    return prepared_tensors


def quantize_tensor_batch(
    tensor_items: List[tuple], 
    quantizer: AWQQuantizer, 
    device: str,
    logger = None
) -> Dict[str, torch.Tensor]:
    """
    Quantize a batch of tensors.
    
    Args:
        tensor_items: List of (name, tensor) tuples
        quantizer: AWQQuantizer instance
        device: Device to quantize on
        logger: Logger instance
        
    Returns:
        Dictionary of quantized tensors
    """
    quantized_tensors = {}
    
    for name, tensor in tensor_items:
        if logger:
            logger.info(f"Quantizing tensor: {name} on {device}")
            
        try:
            # Move tensor to target device if needed
            if tensor.device.type != device.split(':')[0] or (
                ':' in device and str(tensor.device).split(':')[1] != device.split(':')[1]
            ):
                try:
                    tensor = tensor.to(device)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and device != "cpu":
                        if logger:
                            logger.warning(f"GPU OOM when moving tensor {name} to {device}, falling back to CPU: {e}")
                        device = "cpu"
                        tensor = tensor.to("cpu")
                    else:
                        raise
                
            # Quantize tensor
            quantized_tensor = quantizer.quantize(tensor)
            
            # Move result back to CPU to save memory
            if device.startswith('cuda'):
                for key, value in quantized_tensor.items():
                    if isinstance(value, torch.Tensor):
                        quantized_tensor[key] = value.cpu()
            
            quantized_tensors[name] = quantized_tensor
            
            if logger:
                logger.info(f"Successfully quantized tensor: {name} on {device}")
                
        except Exception as e:
            if logger:
                logger.error(f"Failed to quantize tensor {name} on {device}: {e}")
            continue
            
    return quantized_tensors


def partition_tensors(tensors: Dict[str, torch.Tensor], num_partitions: int) -> List[Dict[str, torch.Tensor]]:
    """
    Partition tensors across multiple devices by balancing tensor sizes.
    
    Args:
        tensors: Dictionary of tensors
        num_partitions: Number of partitions to create
        
    Returns:
        List of tensor dictionaries, one for each partition
    """
    if num_partitions <= 1:
        return [tensors]
    
    # Calculate tensor sizes
    tensor_sizes = [(name, tensor.numel() * tensor.element_size()) for name, tensor in tensors.items()]
    # Sort by size (largest first)
    tensor_sizes.sort(key=lambda x: x[1], reverse=True)
    
    # Create partitions with approximately equal total size
    partitions = [dict() for _ in range(num_partitions)]
    partition_sizes = [0] * num_partitions
    
    # Distribute tensors using a greedy approach
    for name, size in tensor_sizes:
        # Find partition with smallest total size
        min_idx = partition_sizes.index(min(partition_sizes))
        # Add tensor to this partition
        partitions[min_idx][name] = tensors[name]
        # Update partition size
        partition_sizes[min_idx] += size
    
    return partitions


def save_model_in_chunks(
    tensors: Dict[str, torch.Tensor],
    output_dir: str,
    chunk_size: int = 10,
    use_safetensors: bool = False,
    logger = None,
) -> None:
    """
    Save model in chunks to handle very large models.
    
    Args:
        tensors: Dictionary of tensors
        output_dir: Directory to save chunks
        chunk_size: Number of tensors per chunk
        use_safetensors: Whether to use safetensors format
        logger: Logger instance
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of tensor names
    tensor_names = list(tensors.keys())
    
    # Calculate number of chunks
    num_chunks = (len(tensor_names) + chunk_size - 1) // chunk_size
    
    if logger:
        logger.info(f"Saving model in {num_chunks} chunks with {chunk_size} tensors per chunk")
    
    # Create metadata file with tensor to chunk mapping
    tensor_to_chunk = {}
    
    # Save quantization parameters in metadata
    example_tensor = next(iter(tensors.values()))
    quantization_params = {
        "bits": example_tensor["bits"].item() if "bits" in example_tensor else None,
        "group_size": example_tensor["group_size"].item() if "group_size" in example_tensor else None,
        "symmetric": example_tensor["symmetric"].item() if "symmetric" in example_tensor else None,
    }
    
    # Save tensors in chunks
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(tensor_names))
        
        # Get tensor names for this chunk
        chunk_tensor_names = tensor_names[start_idx:end_idx]
        
        # Create chunk dictionary
        chunk_tensors = {name: tensors[name] for name in chunk_tensor_names}
        
        # Update tensor to chunk mapping
        for name in chunk_tensor_names:
            tensor_to_chunk[name] = chunk_idx
        
        # Save chunk
        chunk_filename = f"model_chunk_{chunk_idx:04d}"
        chunk_path = os.path.join(output_dir, chunk_filename)
        
        if use_safetensors:
            from safetensors.torch import save_file
            save_file(chunk_tensors, chunk_path + ".safetensors")
            if logger:
                logger.info(f"Saved chunk {chunk_idx+1}/{num_chunks} with {len(chunk_tensors)} tensors in safetensors format")
        else:
            torch.save(chunk_tensors, chunk_path + ".pt")
            if logger:
                logger.info(f"Saved chunk {chunk_idx+1}/{num_chunks} with {len(chunk_tensors)} tensors in PyTorch format")
    
    # Save metadata
    metadata = {
        "num_chunks": num_chunks,
        "chunk_size": chunk_size,
        "tensor_to_chunk": tensor_to_chunk,
        "format": "safetensors" if use_safetensors else "pytorch",
        "num_tensors": len(tensor_names),
        "quantization_params": quantization_params,
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    if logger:
        logger.info(f"Saved metadata file with tensor mapping")


def main() -> int:
    """
    Main entry point for AWQ Quantizer.
    """
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Setup logging
        logger = get_logger(
            name="awq_quantizer",
            level=args.log_level,
            to_file=args.log_file is not None,
            file_path=args.log_file,
        )
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Check if we should use all GPUs
        if args.multi_gpu or args.device.lower() == "all":
            available_devices = get_available_gpus(logger)
            if not available_devices:
                logger.warning("No CUDA devices available, falling back to CPU")
                devices = ["cpu"]
            else:
                devices = available_devices
                logger.info(f"Using {len(devices)} GPU(s) for processing")
        else:
            devices = [args.device]
            
        # Log device information for each device
        for device in devices:
            if device.startswith("cuda"):
                if torch.cuda.is_available():
                    device_idx = int(device.split(':')[1]) if ':' in device else 0
                    device_name = torch.cuda.get_device_name(device_idx)
                    total_memory_gb, free_memory_gb = get_device_memory_info(device_idx)
                    
                    logger.info(f"Using GPU {device}: {device_name}")
                    logger.info(f"  Total memory: {total_memory_gb:.2f} GB")
                    logger.info(f"  Free memory: {free_memory_gb:.2f} GB")
                else:
                    logger.warning(f"CUDA device {device} requested but not available, falling back to CPU")
                    devices = ["cpu"]
            elif device == "cpu":
                logger.info("Using CPU for quantization")
        
        # Load model tensors
        logger.info(f"Loading model from {args.model_id}")
        try:
            model_loader = load_model_from_hub(args.model_id)
            # Call load_tensors to get the dictionary of tensors
            model_tensors = model_loader.load_tensors()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return 1
        
        # Prepare tensors for quantization (keep on CPU at this stage)
        logger.info("Preparing tensors for quantization")
        start_time = time.time()
        # Initially keep tensors on CPU to distribute them later
        prepared_tensors = prepare_tensors_for_quantization(
            model_tensors, 
            device="cpu", 
            max_memory_fraction=args.max_memory,
            logger=logger
        )
        prep_time = time.time() - start_time
        
        if not prepared_tensors:
            logger.error("No valid tensors found for quantization")
            return 1
        
        logger.info(f"Found {len(prepared_tensors)} tensors for quantization (Preparation took {prep_time:.2f}s)")
        
        # If using multiple devices, partition tensors
        if len(devices) > 1:
            logger.info(f"Partitioning tensors across {len(devices)} devices")
            partitioned_tensors = partition_tensors(prepared_tensors, len(devices))
            
            # Log partition distribution
            for i, partition in enumerate(partitioned_tensors):
                logger.info(f"Partition {i} for {devices[i]}: {len(partition)} tensors")
        else:
            # No need to partition if using a single device
            partitioned_tensors = [prepared_tensors]
            
        # Initialize quantizers for each device
        logger.info("Initializing quantizers")
        quantizers = {}
        for device in devices:
            quantizers[device] = AWQQuantizer(
                bits=args.bits,
                group_size=args.group_size,
                symmetric=args.symmetric,
                zero_point=args.zero_point,
                percentile=args.percentile,
                scale_method=args.scale_method,
                per_channel=args.per_channel,
                device=device,  # Set device for each quantizer
                logger_name=f"awq_quantizer_{device}",
                logger_level=args.log_level,
                logger_to_file=args.log_file is not None,
                logger_file_path=args.log_file,
            )
        
        # Quantize tensors with parallel processing
        logger.info(f"Starting quantization with {args.num_workers} worker threads per device")
        start_time = time.time()
        
        # Process each partition on its assigned device
        quantized_tensors = {}
        
        with ThreadPoolExecutor(max_workers=args.num_workers * len(devices)) as executor:
            # Keep track of futures and their metadata
            future_metadata = {}
            
            # Submit tasks for each device partition
            for device_idx, device in enumerate(devices):
                if device_idx >= len(partitioned_tensors):
                    continue  # Skip if no partition for this device
                    
                partition = partitioned_tensors[device_idx]
                if not partition:
                    continue  # Skip empty partitions
                
                # Divide tensors into batches for workers
                tensor_items = list(partition.items())
                workers_per_device = max(1, min(args.num_workers, len(tensor_items)))
                batch_size = max(1, len(tensor_items) // workers_per_device)
                batches = [tensor_items[i:i+batch_size] for i in range(0, len(tensor_items), batch_size)]
                
                # Submit batches to executor
                for batch_idx, batch in enumerate(batches):
                    future = executor.submit(
                        quantize_tensor_batch, 
                        batch, 
                        quantizers[device], 
                        device, 
                        logger
                    )
                    # Store metadata with the future for easy reference
                    future_metadata[future] = {
                        "device": device,
                        "batch_idx": batch_idx,
                        "total_batches": len(batches)
                    }
            
            # Process results as they complete
            for future in as_completed([f for f in future_metadata.keys()]):
                try:
                    batch_result = future.result()
                    metadata = future_metadata[future]
                    device = metadata["device"]
                    batch_idx = metadata["batch_idx"]
                    total_batches = metadata["total_batches"]
                    
                    quantized_tensors.update(batch_result)
                    logger.info(f"Completed batch {batch_idx+1}/{total_batches} on {device} ({len(batch_result)} tensors)")
                except Exception as e:
                    metadata = future_metadata[future]
                    device = metadata["device"]
                    logger.error(f"Error processing batch on {device}: {e}")
        
        quant_time = time.time() - start_time
        
        if not quantized_tensors:
            logger.error("No tensors were successfully quantized")
            return 1
        
        logger.info(f"Quantized {len(quantized_tensors)} tensors in {quant_time:.2f} seconds")
        
        # Clean up memory for all CUDA devices
        if torch.cuda.is_available():
            for device_idx in range(torch.cuda.device_count()):
                with torch.cuda.device(device_idx):
                    torch.cuda.empty_cache()
                if logger.level <= logging.DEBUG:
                    memory_allocated = torch.cuda.memory_allocated(device_idx) / 1024**3
                    logger.debug(f"GPU {device_idx} memory after quantization: {memory_allocated:.2f} GB")
        
        # Save quantized tensors
        logger.info(f"Saving {len(quantized_tensors)} quantized tensors to {args.output_dir}")
        save_start_time = time.time()
        try:
            save_model_in_chunks(
                tensors=quantized_tensors,
                output_dir=args.output_dir,
                chunk_size=args.chunk_size,
                use_safetensors=args.save_safetensors,
                logger=logger,
            )
            save_time = time.time() - save_start_time
            logger.info(f"Successfully saved quantized model in {save_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Failed to save quantized model: {e}")
            return 1
        
        total_time = time.time() - start_time + prep_time
        logger.info(f"Quantization complete in {total_time:.2f} seconds")
        return 0
        
    except Exception as e:
        logger.error(f"Error during quantization: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 