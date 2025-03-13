"""
AWQ (Activation-aware Weight Quantization) implementation.

This module implements the AWQ quantization method as described in the paper:
"AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"
https://arxiv.org/abs/2306.00978
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from ..utils.logger import get_logger
from ..utils.tensor_utils import (
    apply_dynamic_scale,
    get_percentile_value,
    get_tensor_stats,
    revert_dynamic_scale,
)


class AWQQuantizer:
    """
    AWQ Quantizer class.
    """

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        symmetric: bool = True,
        zero_point: str = "minmax",
        percentile: float = 0.99,
        scale_method: str = "mse",
        per_channel: bool = True,
        logger_name: str = "awq_quantizer",
        logger_level: str = "INFO",
        logger_to_file: bool = False,
        logger_file_path: Optional[str] = None,
    ):
        """
        Initialize the AWQ Quantizer.

        Args:
            bits: Bit width for quantization (4 or 8)
            group_size: Group size for quantization
            symmetric: Whether to use symmetric quantization
            zero_point: Zero-point calibration method (none, minmax, percentile)
            percentile: Percentile value if zero_point is "percentile"
            scale_method: Scale calibration method (minmax, mse)
            per_channel: Whether to use per-channel quantization
            logger_name: Logger name
            logger_level: Logging level
            logger_to_file: Whether to log to file
            logger_file_path: Log file path
        """
        self.bits = bits
        self.group_size = group_size
        self.symmetric = symmetric
        self.zero_point = zero_point
        self.percentile = percentile
        self.scale_method = scale_method
        self.per_channel = per_channel
        
        self.logger = get_logger(
            name=logger_name,
            level=logger_level,
            to_file=logger_to_file,
            file_path=logger_file_path,
        )
        
        # Validate parameters
        self._validate_parameters()
        
        # Calculate quantization parameters
        self.qmin, self.qmax = self._calculate_qmin_qmax()
        
        self.logger.info(f"Initialized AWQ Quantizer with bits={bits}, group_size={group_size}, symmetric={symmetric}")
        self.logger.info(f"Quantization range: [{self.qmin}, {self.qmax}]")

    def _validate_parameters(self) -> None:
        """
        Validate quantization parameters.
        """
        if self.bits not in [4, 8]:
            self.logger.error(f"Invalid bits: {self.bits}, must be 4 or 8")
            raise ValueError(f"Invalid bits: {self.bits}, must be 4 or 8")
        
        if self.group_size <= 0:
            self.logger.error(f"Invalid group_size: {self.group_size}, must be positive")
            raise ValueError(f"Invalid group_size: {self.group_size}, must be positive")
        
        if self.zero_point not in ["none", "minmax", "percentile"]:
            self.logger.error(f"Invalid zero_point: {self.zero_point}, must be none, minmax, or percentile")
            raise ValueError(f"Invalid zero_point: {self.zero_point}, must be none, minmax, or percentile")
        
        if self.percentile <= 0 or self.percentile >= 1:
            self.logger.error(f"Invalid percentile: {self.percentile}, must be in (0, 1)")
            raise ValueError(f"Invalid percentile: {self.percentile}, must be in (0, 1)")
        
        if self.scale_method not in ["minmax", "mse"]:
            self.logger.error(f"Invalid scale_method: {self.scale_method}, must be minmax or mse")
            raise ValueError(f"Invalid scale_method: {self.scale_method}, must be minmax or mse")

    def _calculate_qmin_qmax(self) -> Tuple[int, int]:
        """
        Calculate quantization min and max values.

        Returns:
            Tuple of (qmin, qmax)
        """
        if self.symmetric:
            qmax = 2 ** (self.bits - 1) - 1
            qmin = -qmax
        else:
            qmax = 2 ** self.bits - 1
            qmin = 0
        
        return qmin, qmax

    def _calculate_scale_zp(
        self, tensor: torch.Tensor, dim: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate scale and zero point for quantization.

        Args:
            tensor: Input tensor
            dim: Dimension for per-channel quantization

        Returns:
            Tuple of (scale, zero_point)
        """
        # Handle per-channel quantization
        if dim is not None and self.per_channel:
            # Get min and max values per channel
            if self.zero_point == "percentile":
                # Use percentile for min and max
                tensor_flat = tensor.reshape(tensor.shape[0], -1)
                xmin = torch.tensor([
                    get_percentile_value(tensor_flat[i], 1 - self.percentile)
                    for i in range(tensor_flat.shape[0])
                ], device=tensor.device)
                xmax = torch.tensor([
                    get_percentile_value(tensor_flat[i], self.percentile)
                    for i in range(tensor_flat.shape[0])
                ], device=tensor.device)
            else:
                # Use min and max
                tensor_flat = tensor.reshape(tensor.shape[0], -1)
                xmin = tensor_flat.min(dim=1)[0]
                xmax = tensor_flat.max(dim=1)[0]
            
            # Handle symmetric quantization
            if self.symmetric:
                abs_max = torch.maximum(xmin.abs(), xmax.abs())
                xmin = -abs_max
                xmax = abs_max
            
            # Calculate scale and zero point
            scale = (xmax - xmin) / (self.qmax - self.qmin)
            zero_point = self.qmin - torch.round(xmin / scale)
            
            # Reshape scale and zero point for broadcasting
            new_shape = [1] * len(tensor.shape)
            new_shape[dim] = tensor.shape[dim]
            scale = scale.reshape(new_shape)
            zero_point = zero_point.reshape(new_shape)
        
        else:
            # Get min and max values
            if self.zero_point == "percentile":
                # Use percentile for min and max
                xmin = get_percentile_value(tensor, 1 - self.percentile)
                xmax = get_percentile_value(tensor, self.percentile)
            else:
                # Use min and max
                xmin = tensor.min().item()
                xmax = tensor.max().item()
            
            # Handle symmetric quantization
            if self.symmetric:
                abs_max = max(abs(xmin), abs(xmax))
                xmin = -abs_max
                xmax = abs_max
            
            # Calculate scale and zero point
            scale = (xmax - xmin) / (self.qmax - self.qmin)
            zero_point = self.qmin - round(xmin / scale)
            
            # Convert to tensors
            scale = torch.tensor(scale, device=tensor.device)
            zero_point = torch.tensor(zero_point, device=tensor.device)
        
        return scale, zero_point

    def _quantize_tensor(
        self, tensor: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
    ) -> torch.Tensor:
        """
        Quantize tensor using scale and zero point.

        Args:
            tensor: Input tensor
            scale: Scale factor
            zero_point: Zero point

        Returns:
            Quantized tensor
        """
        # Quantize
        tensor_q = torch.round(tensor / scale + zero_point)
        
        # Clamp
        tensor_q = torch.clamp(tensor_q, self.qmin, self.qmax)
        
        return tensor_q

    def _dequantize_tensor(
        self, tensor_q: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
    ) -> torch.Tensor:
        """
        Dequantize tensor using scale and zero point.

        Args:
            tensor_q: Quantized tensor
            scale: Scale factor
            zero_point: Zero point

        Returns:
            Dequantized tensor
        """
        # Dequantize
        tensor_dq = (tensor_q - zero_point) * scale
        
        return tensor_dq

    def _quantize_per_group(
        self, tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize tensor per group.

        Args:
            tensor: Input tensor

        Returns:
            Tuple of (quantized_tensor, scales, zero_points)
        """
        # Get tensor shape
        shape = tensor.shape
        
        # Reshape tensor for group quantization
        if len(shape) == 2:
            # Linear weight
            out_features, in_features = shape
            
            # Calculate number of groups
            num_groups = (in_features + self.group_size - 1) // self.group_size
            
            # Pad tensor if needed
            if in_features % self.group_size != 0:
                pad_size = num_groups * self.group_size - in_features
                tensor = F.pad(tensor, (0, pad_size))
                in_features = num_groups * self.group_size
            
            # Reshape tensor
            tensor = tensor.reshape(out_features, num_groups, self.group_size)
            
            # Quantize each group
            scales = torch.zeros(out_features, num_groups, 1, device=tensor.device)
            zero_points = torch.zeros(out_features, num_groups, 1, device=tensor.device)
            tensor_q = torch.zeros_like(tensor, dtype=torch.int32)
            
            for i in range(out_features):
                for j in range(num_groups):
                    # Calculate scale and zero point
                    scale, zero_point = self._calculate_scale_zp(tensor[i, j])
                    
                    # Quantize
                    tensor_q[i, j] = self._quantize_tensor(tensor[i, j], scale, zero_point)
                    
                    # Store scale and zero point
                    scales[i, j, 0] = scale
                    zero_points[i, j, 0] = zero_point
            
            # Reshape back
            tensor_q = tensor_q.reshape(out_features, in_features)
            
            # Remove padding if needed
            if in_features != shape[1]:
                tensor_q = tensor_q[:, :shape[1]]
        
        elif len(shape) == 3:
            # Attention weight
            out_features, qkv, in_features = shape
            
            # Calculate number of groups
            num_groups = (in_features + self.group_size - 1) // self.group_size
            
            # Pad tensor if needed
            if in_features % self.group_size != 0:
                pad_size = num_groups * self.group_size - in_features
                tensor = F.pad(tensor, (0, pad_size))
                in_features = num_groups * self.group_size
            
            # Reshape tensor
            tensor = tensor.reshape(out_features, qkv, num_groups, self.group_size)
            
            # Quantize each group
            scales = torch.zeros(out_features, qkv, num_groups, 1, device=tensor.device)
            zero_points = torch.zeros(out_features, qkv, num_groups, 1, device=tensor.device)
            tensor_q = torch.zeros_like(tensor, dtype=torch.int32)
            
            for i in range(out_features):
                for j in range(qkv):
                    for k in range(num_groups):
                        # Calculate scale and zero point
                        scale, zero_point = self._calculate_scale_zp(tensor[i, j, k])
                        
                        # Quantize
                        tensor_q[i, j, k] = self._quantize_tensor(tensor[i, j, k], scale, zero_point)
                        
                        # Store scale and zero point
                        scales[i, j, k, 0] = scale
                        zero_points[i, j, k, 0] = zero_point
            
            # Reshape back
            tensor_q = tensor_q.reshape(out_features, qkv, in_features)
            
            # Remove padding if needed
            if in_features != shape[2]:
                tensor_q = tensor_q[:, :, :shape[2]]
        
        else:
            self.logger.error(f"Unsupported tensor shape: {shape}")
            raise ValueError(f"Unsupported tensor shape: {shape}")
        
        return tensor_q, scales, zero_points

    def quantize(
        self, tensor: torch.Tensor, name: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Quantize tensor using AWQ.

        Args:
            tensor: Input tensor
            name: Tensor name for logging

        Returns:
            Dictionary with quantized tensor and metadata
        """
        # Log tensor info
        if name is not None:
            self.logger.info(f"Quantizing tensor: {name}, shape: {tensor.shape}")
            stats = get_tensor_stats(tensor)
            self.logger.debug(f"Tensor stats: {stats}")
        
        # Quantize tensor
        tensor_q, scales, zero_points = self._quantize_per_group(tensor)
        
        # Create result dictionary
        result = {
            "tensor_q": tensor_q,
            "scales": scales,
            "zero_points": zero_points,
            "bits": torch.tensor(self.bits, dtype=torch.int32),
            "group_size": torch.tensor(self.group_size, dtype=torch.int32),
            "symmetric": torch.tensor(self.symmetric, dtype=torch.bool),
        }
        
        # Log quantization info
        if name is not None:
            self.logger.info(f"Quantized tensor: {name}, shape: {tensor_q.shape}")
            self.logger.debug(f"Scales shape: {scales.shape}")
            self.logger.debug(f"Zero points shape: {zero_points.shape}")
        
        return result

    def dequantize(
        self, result: Dict[str, torch.Tensor], name: Optional[str] = None
    ) -> torch.Tensor:
        """
        Dequantize tensor using AWQ.

        Args:
            result: Dictionary with quantized tensor and metadata
            name: Tensor name for logging

        Returns:
            Dequantized tensor
        """
        # Extract tensors
        tensor_q = result["tensor_q"]
        scales = result["scales"]
        zero_points = result["zero_points"]
        
        # Log tensor info
        if name is not None:
            self.logger.info(f"Dequantizing tensor: {name}, shape: {tensor_q.shape}")
        
        # Get tensor shape
        shape = tensor_q.shape
        
        # Reshape tensor for group dequantization
        if len(shape) == 2:
            # Linear weight
            out_features, in_features = shape
            
            # Calculate number of groups
            num_groups = (in_features + self.group_size - 1) // self.group_size
            
            # Pad tensor if needed
            if in_features % self.group_size != 0:
                pad_size = num_groups * self.group_size - in_features
                tensor_q = F.pad(tensor_q, (0, pad_size))
                in_features = num_groups * self.group_size
            
            # Reshape tensor
            tensor_q = tensor_q.reshape(out_features, num_groups, self.group_size)
            
            # Dequantize each group
            tensor_dq = torch.zeros_like(tensor_q, dtype=torch.float32)
            
            for i in range(out_features):
                for j in range(num_groups):
                    # Dequantize
                    tensor_dq[i, j] = self._dequantize_tensor(
                        tensor_q[i, j], scales[i, j, 0], zero_points[i, j, 0]
                    )
            
            # Reshape back
            tensor_dq = tensor_dq.reshape(out_features, in_features)
            
            # Remove padding if needed
            if in_features != shape[1]:
                tensor_dq = tensor_dq[:, :shape[1]]
        
        elif len(shape) == 3:
            # Attention weight
            out_features, qkv, in_features = shape
            
            # Calculate number of groups
            num_groups = (in_features + self.group_size - 1) // self.group_size
            
            # Pad tensor if needed
            if in_features % self.group_size != 0:
                pad_size = num_groups * self.group_size - in_features
                tensor_q = F.pad(tensor_q, (0, pad_size))
                in_features = num_groups * self.group_size
            
            # Reshape tensor
            tensor_q = tensor_q.reshape(out_features, qkv, num_groups, self.group_size)
            
            # Dequantize each group
            tensor_dq = torch.zeros_like(tensor_q, dtype=torch.float32)
            
            for i in range(out_features):
                for j in range(qkv):
                    for k in range(num_groups):
                        # Dequantize
                        tensor_dq[i, j, k] = self._dequantize_tensor(
                            tensor_q[i, j, k], scales[i, j, k, 0], zero_points[i, j, k, 0]
                        )
            
            # Reshape back
            tensor_dq = tensor_dq.reshape(out_features, qkv, in_features)
            
            # Remove padding if needed
            if in_features != shape[2]:
                tensor_dq = tensor_dq[:, :, :shape[2]]
        
        else:
            self.logger.error(f"Unsupported tensor shape: {shape}")
            raise ValueError(f"Unsupported tensor shape: {shape}")
        
        # Log dequantization info
        if name is not None:
            self.logger.info(f"Dequantized tensor: {name}, shape: {tensor_dq.shape}")
        
        return tensor_dq

    def quantize_tensors(
        self, tensors: Dict[str, torch.Tensor], skip_layers: List[str] = []
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Quantize multiple tensors.

        Args:
            tensors: Dictionary mapping tensor names to tensors
            skip_layers: List of layer names to skip

        Returns:
            Dictionary mapping tensor names to quantized tensors and metadata
        """
        quantized_tensors = {}
        
        for name, tensor in tensors.items():
            # Skip non-weight tensors and specified layers
            if any(skip_name in name for skip_name in skip_layers):
                self.logger.info(f"Skipping tensor: {name}")
                continue
            
            # Skip non-floating point tensors
            if not tensor.is_floating_point():
                self.logger.info(f"Skipping non-floating point tensor: {name}")
                continue
            
            # Skip small tensors
            if tensor.numel() < self.group_size:
                self.logger.info(f"Skipping small tensor: {name}, size: {tensor.numel()}")
                continue
            
            # Skip tensors with unsupported shapes
            if len(tensor.shape) not in [2, 3]:
                self.logger.info(f"Skipping tensor with unsupported shape: {name}, shape: {tensor.shape}")
                continue
            
            # Quantize tensor
            try:
                quantized_tensors[name] = self.quantize(tensor, name)
            except Exception as e:
                self.logger.error(f"Error quantizing tensor: {name}, error: {e}")
                raise
        
        return quantized_tensors

    def dequantize_tensors(
        self, quantized_tensors: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Dequantize multiple tensors.

        Args:
            quantized_tensors: Dictionary mapping tensor names to quantized tensors and metadata

        Returns:
            Dictionary mapping tensor names to dequantized tensors
        """
        dequantized_tensors = {}
        
        for name, result in quantized_tensors.items():
            # Dequantize tensor
            try:
                dequantized_tensors[name] = self.dequantize(result, name)
            except Exception as e:
                self.logger.error(f"Error dequantizing tensor: {name}, error: {e}")
                raise
        
        return dequantized_tensors 