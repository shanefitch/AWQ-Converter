"""
Safetensors loader module for AWQ Quantizer.
"""

import os
from typing import Dict, Optional

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file

from ..utils.logger import get_logger
from ..utils.tensor_utils import convert_bf16_to_fp16, get_tensor_type, get_model_files, is_consolidated_file, filter_consolidated_files


class SafetensorsLoader:
    """
    Safetensors loader class for AWQ Quantizer.
    """

    def __init__(
        self,
        model_path: str,
        from_hub: bool = False,
        revision: str = "main",
        token: Optional[str] = None,
        logger_name: str = "safetensors_loader",
        logger_level: str = "INFO",
        logger_to_file: bool = False,
        logger_file_path: Optional[str] = None,
        resume_download: bool = True,
        force_download: bool = False,
    ):
        """
        Initialize the Safetensors loader.

        Args:
            model_path: Path to the model or model ID on Hugging Face Hub
            from_hub: Whether the model is on Hugging Face Hub
            revision: Model revision to use (if from_hub is true)
            token: Token to use for private models (if from_hub is true)
            logger_name: Logger name
            logger_level: Logging level
            logger_to_file: Whether to log to file
            logger_file_path: Log file path
            resume_download: Whether to resume partial downloads
            force_download: Whether to force re-download of files
        """
        self.model_path = model_path
        self.from_hub = from_hub
        self.revision = revision
        self.token = token
        self.resume_download = resume_download
        self.force_download = force_download
        
        self.logger = get_logger(
            name=logger_name,
            level=logger_level,
            to_file=logger_to_file,
            file_path=logger_file_path,
        )
        
        self.model_files = get_model_files(model_path)
        if not self.model_files:
            raise ValueError(f"No safetensor files found in {model_path}")
            
        # Filter out consolidated files if individual files exist
        self.model_files = filter_consolidated_files(self.model_files)
        self.logger.info(f"Loading {len(self.model_files)} safetensors files:")
        for file in self.model_files:
            self.logger.info(f"  - {os.path.basename(file)}")
            
        # Initialize tensors dictionary
        self.tensors: Dict[str, torch.Tensor] = {}

    def verify_file(self, file_path: str) -> bool:
        """
        Verify that a safetensor file is valid and complete.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file is valid, False otherwise
        """
        try:
            with safe_open(file_path, framework="pt") as f:
                # Try to read the metadata and first tensor
                _ = f.metadata()
                keys = f.keys()
                if keys:
                    _ = f.get_tensor(keys[0])
            return True
        except Exception as e:
            self.logger.warning(f"File verification failed for {file_path}: {e}")
            return False

    def _get_file_path(self, file: str) -> str:
        """
        Get file path with robust error handling.

        Args:
            file: File name

        Returns:
            File path
        """
        if self.from_hub:
            # Download file from Hugging Face Hub with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    file_path = hf_hub_download(
                        repo_id=self.model_path,
                        filename=file,
                        revision=self.revision,
                        token=self.token,
                        resume_download=self.resume_download,
                        force_download=self.force_download,
                    )
                    
                    # Verify the downloaded file
                    if not self.verify_file(file_path):
                        if attempt < max_retries - 1:
                            self.logger.warning(f"Retrying download of {file} (attempt {attempt + 2}/{max_retries})")
                            self.force_download = True  # Force re-download on retry
                            continue
                        raise ValueError(f"File verification failed after {max_retries} attempts")
                    
                    return file_path
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        self.logger.warning(f"Error downloading {file}, retrying: {e}")
                        continue
                    raise ValueError(f"Failed to download {file} after {max_retries} attempts: {e}")
        else:
            # Get file from local path
            file_path = os.path.join(self.model_path, file)
            if not self.verify_file(file_path):
                raise ValueError(f"Local file verification failed: {file_path}")
            return file_path

    def load_tensors(self) -> Dict[str, torch.Tensor]:
        """
        Load tensors from safetensors files.

        Returns:
            Dictionary mapping tensor names to tensors
        """
        self.tensors = {}
        
        # Load each file
        for file in self.model_files:
            try:
                # Load tensors from file
                file_tensors = load_file(file)
                
                # Add tensors to dictionary
                for name, tensor in file_tensors.items():
                    if name in self.tensors:
                        self.logger.warning(f"Duplicate tensor name: {name}")
                    self.tensors[name] = tensor
                    
                self.logger.debug(f"Loaded {len(file_tensors)} tensors from {os.path.basename(file)}")
                
            except Exception as e:
                self.logger.error(f"Error loading {os.path.basename(file)}: {e}")
                raise
                
        self.logger.info(f"Loaded {len(self.tensors)} total tensors")
        return self.tensors

    def save_tensors(
        self,
        tensors: Dict[str, torch.Tensor],
        output_dir: str,
        filename: str = "model.safetensors",
    ) -> None:
        """
        Save tensors to safetensors file.

        Args:
            tensors: Dictionary mapping tensor names to tensors
            output_dir: Output directory
            filename: Output filename
        """
        from safetensors.torch import save_file
        
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Save tensors
            output_path = os.path.join(output_dir, filename)
            save_file(tensors, output_path)
            
            self.logger.info(f"Saved {len(tensors)} tensors to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving tensors: {e}")
            raise

    def convert_tensors_bf16_to_fp16(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Convert BF16 tensors to FP16.

        Args:
            tensors: Dictionary mapping tensor names to tensors

        Returns:
            Dictionary mapping tensor names to tensors with BF16 converted to FP16
        """
        converted_tensors = {}
        
        for name, tensor in tensors.items():
            converted_tensor = convert_bf16_to_fp16(tensor)
            
            if converted_tensor.dtype != tensor.dtype:
                self.logger.debug(f"Converted tensor {name} from {get_tensor_type(tensor)} to {get_tensor_type(converted_tensor)}")
            
            converted_tensors[name] = converted_tensor
        
        return converted_tensors 