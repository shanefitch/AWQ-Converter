"""
Model loading module for AWQ Quantizer.
"""

from typing import Dict, List, Optional, Tuple, Union
import os
import hashlib

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from .safetensors_loader import SafetensorsLoader
from ..utils.logger import get_logger
from ..utils.tensor_utils import get_model_files, is_consolidated_file


def verify_file_hash(file_path: str, expected_hash: Optional[str] = None) -> str:
    """
    Verify a file's SHA256 hash.

    Args:
        file_path: Path to the file
        expected_hash: Expected SHA256 hash (if None, just returns the computed hash)

    Returns:
        Computed SHA256 hash

    Raises:
        ValueError: If the computed hash doesn't match the expected hash
    """
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        # Read the file in chunks to handle large files
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    
    computed_hash = sha256_hash.hexdigest()
    
    if expected_hash and computed_hash != expected_hash:
        raise ValueError(
            f"Hash verification failed for {file_path}. "
            f"Expected: {expected_hash}, Got: {computed_hash}"
        )
    
    return computed_hash


def load_model_from_hub(
    model_id: str,
    revision: str = "main",
    token: Optional[str] = None,
    logger_name: str = "model_loading",
    logger_level: str = "INFO",
    logger_to_file: bool = False,
    logger_file_path: Optional[str] = None,
    resume_download: bool = True,
    force_download: bool = False,
    verify_downloads: bool = True,
) -> SafetensorsLoader:
    """
    Load a model from the Hugging Face Hub.

    Args:
        model_id: Model ID on Hugging Face Hub
        revision: Model revision to use
        token: Token to use for private models
        logger_name: Logger name
        logger_level: Logging level
        logger_to_file: Whether to log to file
        logger_file_path: Log file path
        resume_download: Whether to resume partial downloads
        force_download: Whether to force re-download of files
        verify_downloads: Whether to verify downloaded files

    Returns:
        SafetensorsLoader instance
    """
    logger = get_logger(
        name=logger_name,
        level=logger_level,
        to_file=logger_to_file,
        file_path=logger_file_path,
    )

    try:
        # Try to download the entire snapshot first
        local_dir = snapshot_download(
            repo_id=model_id,
            revision=revision,
            token=token,
            resume_download=resume_download,
            force_download=force_download,
        )
        logger.info(f"Successfully downloaded model snapshot to {local_dir}")
        
    except Exception as e:
        logger.warning(f"Failed to download model snapshot: {e}")
        logger.info("Falling back to individual file downloads")
        local_dir = None

    # Create the loader instance
    loader = SafetensorsLoader(
        model_path=model_id if local_dir is None else local_dir,
        from_hub=True if local_dir is None else False,
        revision=revision,
        token=token,
        logger_name=logger_name,
        logger_level=logger_level,
        logger_to_file=logger_to_file,
        logger_file_path=logger_file_path,
        resume_download=resume_download,
        force_download=force_download,
    )

    return loader


def load_model_from_path(
    model_path: str,
    logger_name: str = "model_loading",
    logger_level: str = "INFO",
    logger_to_file: bool = False,
    logger_file_path: Optional[str] = None,
    verify_files: bool = True,
) -> SafetensorsLoader:
    """
    Load a model from a local path.

    Args:
        model_path: Path to the model
        logger_name: Logger name
        logger_level: Logging level
        logger_to_file: Whether to log to file
        logger_file_path: Log file path
        verify_files: Whether to verify local files

    Returns:
        SafetensorsLoader instance
    """
    return SafetensorsLoader(
        model_path=model_path,
        from_hub=False,
        logger_name=logger_name,
        logger_level=logger_level,
        logger_to_file=logger_to_file,
        logger_file_path=logger_file_path,
        resume_download=False,  # Not used for local files
        force_download=False,  # Not used for local files
    ) 