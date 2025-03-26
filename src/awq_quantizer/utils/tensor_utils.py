"""
Tensor utility functions for AWQ Quantizer.
"""

import torch
import os
from typing import Dict, List, Optional, Tuple, Union


def convert_bf16_to_fp16(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert BF16 tensor to FP16.

    Args:
        tensor: Input tensor

    Returns:
        FP16 tensor
    """
    if tensor.dtype == torch.bfloat16:
        return tensor.to(torch.float16)
    return tensor


def convert_fp16_to_bf16(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert FP16 tensor to BF16.

    Args:
        tensor: Input tensor

    Returns:
        BF16 tensor
    """
    if tensor.dtype == torch.float16:
        return tensor.to(torch.bfloat16)
    return tensor


def get_tensor_type(tensor: torch.Tensor) -> str:
    """
    Get tensor type as string.

    Args:
        tensor: Input tensor

    Returns:
        Tensor type as string
    """
    dtype_map = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.int32: "int32",
        torch.int8: "int8",
        torch.uint8: "uint8",
        torch.int16: "int16",
        torch.int64: "int64",
        torch.bool: "bool",
    }
    return dtype_map.get(tensor.dtype, str(tensor.dtype))


def get_tensor_stats(tensor: torch.Tensor) -> Dict[str, float]:
    """
    Get tensor statistics.

    Args:
        tensor: Input tensor

    Returns:
        Dictionary with tensor statistics
    """
    # Convert to float32 for accurate statistics
    tensor_float = tensor.to(torch.float32) if tensor.dtype != torch.float32 else tensor
    
    return {
        "min": float(tensor_float.min().item()),
        "max": float(tensor_float.max().item()),
        "mean": float(tensor_float.mean().item()),
        "std": float(tensor_float.std().item()),
        "abs_mean": float(tensor_float.abs().mean().item()),
        "sparsity": float((tensor_float == 0).float().mean().item()),
    }


def get_percentile_value(tensor: torch.Tensor, percentile: float) -> float:
    """
    Get percentile value from tensor.

    Args:
        tensor: Input tensor
        percentile: Percentile value (0-1)

    Returns:
        Value at the specified percentile
    """
    # Convert to float32 for accurate statistics
    tensor_float = tensor.to(torch.float32) if tensor.dtype != torch.float32 else tensor
    
    # Flatten tensor
    tensor_flat = tensor_float.flatten()
    
    # Sort tensor
    tensor_sorted, _ = torch.sort(tensor_flat)
    
    # Get index
    index = int(percentile * (tensor_sorted.shape[0] - 1))
    
    return float(tensor_sorted[index].item())


def get_optimal_fp16_scale(tensor: torch.Tensor) -> float:
    """
    Get optimal FP16 scale for tensor.

    Args:
        tensor: Input tensor

    Returns:
        Optimal FP16 scale
    """
    # Convert to float32 for accurate statistics
    tensor_float = tensor.to(torch.float32) if tensor.dtype != torch.float32 else tensor
    
    # Get max absolute value
    max_abs = tensor_float.abs().max().item()
    
    # FP16 max value
    fp16_max = 65504.0
    
    # Calculate scale
    scale = fp16_max / max_abs if max_abs > 0 else 1.0
    
    return float(scale)


def apply_dynamic_scale(
    tensor: torch.Tensor, scale: Optional[float] = None
) -> Tuple[torch.Tensor, float]:
    """
    Apply dynamic scale to tensor.

    Args:
        tensor: Input tensor
        scale: Scale factor (if None, calculate optimal scale)

    Returns:
        Scaled tensor and scale factor
    """
    # Convert to float32 for accurate scaling
    tensor_float = tensor.to(torch.float32) if tensor.dtype != torch.float32 else tensor
    
    # Calculate scale if not provided
    if scale is None:
        scale = get_optimal_fp16_scale(tensor_float)
    
    # Apply scale
    tensor_scaled = tensor_float * scale
    
    return tensor_scaled, scale


def revert_dynamic_scale(
    tensor: torch.Tensor, scale: float
) -> torch.Tensor:
    """
    Revert dynamic scale from tensor.

    Args:
        tensor: Input tensor
        scale: Scale factor

    Returns:
        Unscaled tensor
    """
    # Convert to float32 for accurate scaling
    tensor_float = tensor.to(torch.float32) if tensor.dtype != torch.float32 else tensor
    
    # Revert scale
    tensor_unscaled = tensor_float / scale
    
    return tensor_unscaled


def get_device_from_config(config: Dict) -> torch.device:
    """
    Get device from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        PyTorch device
    """
    device_str = config.get("hardware", {}).get("device", "cuda")
    
    if device_str == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    
    if device_str == "mps" and not hasattr(torch.backends, "mps") and not torch.backends.mps.is_available():
        return torch.device("cpu")
    
    return torch.device(device_str)


def filter_safetensor_files(file_paths: List[str]) -> List[str]:
    """
    Filter a list of safetensor file paths to exclude consolidated files when individual files exist.
    
    Args:
        file_paths: List of paths to safetensor files
        
    Returns:
        List of safetensor file paths with consolidated files removed if individual files exist
    """
    # First, separate consolidated and individual files
    consolidated_files = []
    individual_files = []
    
    for path in file_paths:
        if not path.endswith('.safetensors'):
            continue
            
        filename = os.path.basename(path)
        if 'consolidated' in filename.lower():
            consolidated_files.append(path)
        else:
            individual_files.append(path)
    
    # If we have individual files, ignore consolidated ones
    if individual_files:
        return individual_files
    # Otherwise, use consolidated files if they exist
    elif consolidated_files:
        return consolidated_files
    # If neither exists, return empty list
    return []


def is_consolidated_file(file_path: str) -> bool:
    """
    Check if a file path represents a consolidated safetensor file.
    
    Args:
        file_path: Path to the safetensor file
        
    Returns:
        True if the file is a consolidated safetensor file, False otherwise
    """
    if not file_path.endswith('.safetensors'):
        return False
        
    filename = os.path.basename(file_path)
    return 'consolidated' in filename.lower()


def get_model_files(model_path: str) -> List[str]:
    """
    Get all relevant model files from a directory, handling consolidated files appropriately.
    
    Args:
        model_path: Path to the model directory or file
        
    Returns:
        List of paths to model files to load
    """
    if os.path.isfile(model_path):
        return [model_path] if model_path.endswith('.safetensors') else []
        
    # Get all safetensor files in directory
    safetensor_files = []
    for root, _, files in os.walk(model_path):
        for file in files:
            if file.endswith('.safetensors'):
                safetensor_files.append(os.path.join(root, file))
                
    return filter_safetensor_files(safetensor_files)


def filter_consolidated_files(files: List[str]) -> List[str]:
    """
    Filter out consolidated files if individual files exist.
    
    This is useful when loading model files, as consolidated files are redundant
    if the individual files are present. The consolidated files contain the same
    weights as the individual files, just combined into a single file.
    
    Args:
        files: List of file paths
        
    Returns:
        Filtered list of files with consolidated files removed if individual files exist
    """
    # If only one file, return it regardless
    if len(files) <= 1:
        return files
        
    # Split into consolidated and individual files
    consolidated = []
    individual = []
    
    for file in files:
        if is_consolidated_file(file):
            consolidated.append(file)
        else:
            individual.append(file)
            
    # If we have individual files, ignore consolidated ones
    if individual:
        return sorted(individual)
    
    # Otherwise return consolidated files
    return sorted(consolidated) 