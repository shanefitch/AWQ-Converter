"""
Safetensors loader module for AWQ Quantizer.
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

from ..utils.logger import get_logger
from ..utils.tensor_utils import convert_bf16_to_fp16, get_tensor_type


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
        """
        self.model_path = model_path
        self.from_hub = from_hub
        self.revision = revision
        self.token = token
        
        self.logger = get_logger(
            name=logger_name,
            level=logger_level,
            to_file=logger_to_file,
            file_path=logger_file_path,
        )
        
        self.model_files = []
        self.tensors_metadata = {}
        
        # Find model files
        self._find_model_files()

    def _find_model_files(self) -> None:
        """
        Find model files.
        """
        if self.from_hub:
            # Get model files from Hugging Face Hub
            self.logger.info(f"Finding model files from Hugging Face Hub: {self.model_path}")
            
            try:
                # Get model info
                from huggingface_hub import model_info
                
                info = model_info(
                    self.model_path,
                    revision=self.revision,
                    token=self.token,
                )
                
                # Filter safetensors files
                self.model_files = [
                    file.rfilename
                    for file in info.siblings
                    if file.rfilename.endswith(".safetensors")
                ]
                
                self.logger.info(f"Found {len(self.model_files)} safetensors files")
            
            except Exception as e:
                self.logger.error(f"Error finding model files from Hugging Face Hub: {e}")
                raise
        
        else:
            # Get model files from local path
            self.logger.info(f"Finding model files from local path: {self.model_path}")
            
            if os.path.isfile(self.model_path) and self.model_path.endswith(".safetensors"):
                # Single file
                self.model_files = [os.path.basename(self.model_path)]
                self.model_path = os.path.dirname(self.model_path)
            
            elif os.path.isdir(self.model_path):
                # Directory
                self.model_files = [
                    file
                    for file in os.listdir(self.model_path)
                    if file.endswith(".safetensors")
                ]
            
            else:
                self.logger.error(f"Invalid model path: {self.model_path}")
                raise ValueError(f"Invalid model path: {self.model_path}")
            
            self.logger.info(f"Found {len(self.model_files)} safetensors files")

    def get_tensor_names(self) -> List[str]:
        """
        Get all tensor names from all model files.

        Returns:
            List of tensor names
        """
        tensor_names = []
        
        for file in self.model_files:
            file_path = self._get_file_path(file)
            
            with safe_open(file_path, framework="pt") as f:
                tensor_names.extend(f.keys())
        
        return tensor_names

    def get_tensor_metadata(self) -> Dict[str, Dict]:
        """
        Get metadata for all tensors.

        Returns:
            Dictionary mapping tensor names to metadata
        """
        if not self.tensors_metadata:
            for file in self.model_files:
                file_path = self._get_file_path(file)
                
                with safe_open(file_path, framework="pt") as f:
                    for key in f.keys():
                        tensor_info = f.get_tensor_info(key)
                        self.tensors_metadata[key] = {
                            "dtype": tensor_info.dtype,
                            "shape": tensor_info.shape,
                            "file": file,
                        }
        
        return self.tensors_metadata

    def load_tensor(self, name: str) -> torch.Tensor:
        """
        Load a specific tensor by name.

        Args:
            name: Tensor name

        Returns:
            Tensor
        """
        # Get tensor metadata
        metadata = self.get_tensor_metadata()
        
        if name not in metadata:
            self.logger.error(f"Tensor not found: {name}")
            raise ValueError(f"Tensor not found: {name}")
        
        # Get file path
        file_path = self._get_file_path(metadata[name]["file"])
        
        # Load tensor
        with safe_open(file_path, framework="pt") as f:
            tensor = f.get_tensor(name)
        
        self.logger.debug(f"Loaded tensor: {name}, shape: {tensor.shape}, dtype: {get_tensor_type(tensor)}")
        
        return tensor

    def load_all_tensors(self) -> Dict[str, torch.Tensor]:
        """
        Load all tensors from all model files.

        Returns:
            Dictionary mapping tensor names to tensors
        """
        tensors = {}
        
        # Get tensor names
        tensor_names = self.get_tensor_names()
        
        # Load tensors
        for name in tensor_names:
            tensors[name] = self.load_tensor(name)
        
        return tensors

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

    def save_tensors(
        self,
        tensors: Dict[str, torch.Tensor],
        output_dir: str,
        filename: str = "model.safetensors",
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Save tensors to a safetensors file.

        Args:
            tensors: Dictionary mapping tensor names to tensors
            output_dir: Output directory
            filename: Output filename
            metadata: Metadata to save with the tensors

        Returns:
            Path to the saved file
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save tensors
        output_path = os.path.join(output_dir, filename)
        
        save_file(tensors, output_path, metadata=metadata)
        
        self.logger.info(f"Saved tensors to {output_path}")
        
        return output_path

    def _get_file_path(self, file: str) -> str:
        """
        Get file path.

        Args:
            file: File name

        Returns:
            File path
        """
        if self.from_hub:
            # Download file from Hugging Face Hub
            return hf_hub_download(
                repo_id=self.model_path,
                filename=file,
                revision=self.revision,
                token=self.token,
            )
        
        else:
            # Get file from local path
            return os.path.join(self.model_path, file) 