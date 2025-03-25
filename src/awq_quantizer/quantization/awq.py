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
        device: Optional[str] = None,
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
            device: Device to use for quantization (cuda, cuda:0, etc.)
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
        
        # Set device (default to CUDA if available)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Validate device
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"
        
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
        Validate initialization parameters.
        """
        if self.bits not in [4, 8]:
            raise ValueError(f"Unsupported bit width: {self.bits}. Supported: 4, 8.")
            
        if self.group_size <= 0 or not isinstance(self.group_size, int):
            raise ValueError(f"Group size must be a positive integer: {self.group_size}")
            
        if self.zero_point not in ["none", "minmax", "percentile"]:
            raise ValueError(f"Unsupported zero point calibration method: {self.zero_point}")
            
        if self.zero_point == "percentile" and (self.percentile <= 0 or self.percentile >= 1):
            raise ValueError(f"Percentile must be in range (0, 1): {self.percentile}")
            
        if self.scale_method not in ["minmax", "mse"]:
            raise ValueError(f"Unsupported scale calibration method: {self.scale_method}")

    def _calculate_qmin_qmax(self) -> Tuple[int, int]:
        """
        Calculate the minimum and maximum quantized values.

        Returns:
            Tuple of (qmin, qmax)
        """
        if self.symmetric:
            qmin = -(2 ** (self.bits - 1))
            qmax = 2 ** (self.bits - 1) - 1
        else:
            qmin = 0
            qmax = 2 ** self.bits - 1
            
        return qmin, qmax

    def _calculate_scale_zp(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate scale and zero point for quantization.

        Args:
            tensor: Input tensor

        Returns:
            Tuple of (scale, zero_point)
        """
        # Make sure tensor is on the right device
        tensor = tensor.to(self.device)
        
        # Detect tensor rank and decide how to handle channels
        tensor_dim = tensor.dim()
        
        # Default: apply per-tensor quantization
        if tensor_dim <= 1 or not self.per_channel:
            # Use entire tensor as a single group
            return self._compute_scale_zp_for_group(tensor)
            
        # Handle per-channel quantization for 2+D tensors
        num_channels = tensor.size(0)
        scales = []
        zero_points = []
        
        for c in range(num_channels):
            # Get tensor slice for this channel
            channel_tensor = tensor[c]
            
            # Calculate scale and zero point for this channel
            scale, zp = self._compute_scale_zp_for_group(channel_tensor)
            
            scales.append(scale)
            zero_points.append(zp)
            
        # Stack results
        scale = torch.stack(scales).to(self.device)
        zero_point = torch.stack(zero_points).to(self.device)
        
        # Return per-channel scale and zero point
        return scale, zero_point

    def _compute_scale_zp_for_group(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scale and zero point for a single tensor or group.

        Args:
            tensor: Input tensor or tensor group

        Returns:
            Tuple of (scale, zero_point)
        """
        # Ensure the tensor is contiguous
        tensor = tensor.contiguous()
        
        # Get tensor min and max
        if self.zero_point == "percentile":
            tensor_stats = get_tensor_stats(tensor)
            t_min = get_percentile_value(tensor, 1 - self.percentile, tensor_stats)
            t_max = get_percentile_value(tensor, self.percentile, tensor_stats)
        else:
            t_min = tensor.min()
            t_max = tensor.max()
            
        # Handle symmetric quantization
        if self.symmetric:
            abs_max = max(abs(t_min), abs(t_max))
            t_min = -abs_max
            t_max = abs_max
            
        # Compute scale and zero point
        scale = (t_max - t_min) / (self.qmax - self.qmin)
        
        # Avoid division by zero
        scale = torch.clamp(scale, min=1e-10)
        
        if self.symmetric:
            zero_point = torch.zeros_like(scale)
        else:
            zero_point = self.qmin - t_min / scale
            zero_point = zero_point.round().clamp(self.qmin, self.qmax)
            
        return scale, zero_point

    def _quantize_tensor(
        self, 
        tensor: torch.Tensor, 
        scale: torch.Tensor, 
        zero_point: torch.Tensor
    ) -> torch.Tensor:
        """
        Quantize a tensor using the given scale and zero point.

        Args:
            tensor: Input tensor
            scale: Scale factor
            zero_point: Zero point

        Returns:
            Quantized tensor
        """
        # Move tensors to the same device if needed
        tensor = tensor.to(self.device)
        scale = scale.to(self.device)
        zero_point = zero_point.to(self.device)
        
        # Prepare for per-channel quantization if needed
        if self.per_channel and tensor.dim() > 1 and scale.dim() == 1:
            # Reshape scale and zero_point for broadcasting
            new_shape = [scale.size(0)] + [1] * (tensor.dim() - 1)
            scale = scale.reshape(new_shape)
            zero_point = zero_point.reshape(new_shape)
            
        # Apply quantization formula: q = round(x / scale + zero_point)
        tensor_q = torch.round(tensor / scale + zero_point)
        
        # Clamp to quantization range
        tensor_q = torch.clamp(tensor_q, self.qmin, self.qmax)
        
        return tensor_q

    def _dequantize_tensor(
        self, 
        tensor_q: torch.Tensor, 
        scale: torch.Tensor, 
        zero_point: torch.Tensor
    ) -> torch.Tensor:
        """
        Dequantize a tensor using the given scale and zero point.

        Args:
            tensor_q: Quantized tensor
            scale: Scale factor
            zero_point: Zero point

        Returns:
            Dequantized tensor
        """
        # Move tensors to the same device if needed
        tensor_q = tensor_q.to(self.device)
        scale = scale.to(self.device)
        zero_point = zero_point.to(self.device)
        
        # Prepare for per-channel dequantization if needed
        if self.per_channel and tensor_q.dim() > 1 and scale.dim() == 1:
            # Reshape scale and zero_point for broadcasting
            new_shape = [scale.size(0)] + [1] * (tensor_q.dim() - 1)
            scale = scale.reshape(new_shape)
            zero_point = zero_point.reshape(new_shape)
            
        # Apply dequantization formula: x = (q - zero_point) * scale
        tensor_dq = (tensor_q - zero_point) * scale
        
        return tensor_dq

    def _quantize_per_group(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply group-wise quantization to a tensor.

        Args:
            tensor: Input tensor

        Returns:
            Tuple of (quantized tensor, scales, zero points)
        """
        # Skip small tensors
        if tensor.numel() < self.group_size:
            scale, zero_point = self._calculate_scale_zp(tensor)
            tensor_q = self._quantize_tensor(tensor, scale, zero_point)
            return tensor_q, scale, zero_point
            
        # Move tensor to device
        tensor = tensor.to(self.device)
        
        # Handle tensors based on rank
        if tensor.dim() <= 1:
            # Reshape 1D tensor to 2D for group processing
            original_shape = tensor.shape
            tensor = tensor.reshape(1, -1)
            is_1d = True
        else:
            # For multi-dimensional tensors, treat first dim as channels
            is_1d = False
            
        # Get tensor shape
        if is_1d:
            num_channels, num_elements = 1, tensor.numel()
        else:
            num_channels = tensor.size(0)
            num_elements = tensor[0].numel()
            
        # Calculate number of groups per channel
        num_groups = math.ceil(num_elements / self.group_size)
        
        # Prepare output tensors
        result_shape = list(tensor.shape)
        scales = torch.zeros((num_channels, num_groups), device=self.device)
        zero_points = torch.zeros((num_channels, num_groups), device=self.device)
        tensor_q = torch.zeros_like(tensor, dtype=torch.int32, device=self.device)
        
        # Process each channel
        for c in range(num_channels):
            channel_tensor = tensor[c] if not is_1d else tensor[0]
            channel_flat = channel_tensor.reshape(-1)
            
            # Pad to ensure divisibility by group_size
            pad_len = num_groups * self.group_size - channel_flat.size(0)
            if pad_len > 0:
                channel_flat = F.pad(channel_flat, (0, pad_len))
                
            # Reshape to (num_groups, group_size)
            channel_grouped = channel_flat.reshape(num_groups, self.group_size)
            
            # Process each group
            for g in range(num_groups):
                group_tensor = channel_grouped[g]
                
                # Calculate scale and zero point for this group
                scale, zero_point = self._compute_scale_zp_for_group(group_tensor)
                
                # Store scale and zero point
                scales[c, g] = scale
                zero_points[c, g] = zero_point
                
                # Quantize group
                group_q = self._quantize_tensor(group_tensor, scale, zero_point)
                
                # Store quantized values back to the tensor
                start_idx = g * self.group_size
                end_idx = min((g + 1) * self.group_size, num_elements)
                
                if is_1d:
                    tensor_q[0, start_idx:end_idx] = group_q[:end_idx-start_idx]
                else:
                    # Get the correct indices within the flattened tensor
                    tensor_q_flat = tensor_q[c].reshape(-1)
                    tensor_q_flat[start_idx:end_idx] = group_q[:end_idx-start_idx]
                    tensor_q[c] = tensor_q_flat.reshape(channel_tensor.shape)
                    
        # Restore original shape for 1D tensors
        if is_1d:
            tensor_q = tensor_q.reshape(original_shape)
            
        return tensor_q, scales, zero_points

    def quantize(self, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Quantize a single tensor.

        Args:
            tensor: Input tensor to quantize

        Returns:
            Dictionary containing the quantized tensor and metadata:
            {
                'tensor_q': Quantized tensor (int32)
                'scales': Scale factors (float16)
                'zero_points': Zero points (int32)
                'bits': Number of bits (int32)
                'group_size': Group size (int32)
                'symmetric': Whether symmetric quantization was used (bool)
            }
        """
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor, got {type(tensor)}")
            
        if not tensor.is_floating_point():
            raise ValueError(f"Expected floating point tensor, got {tensor.dtype}")
            
        # Move tensor to device for quantization
        original_device = tensor.device
        tensor = tensor.to(self.device)
        
        try:
            # Apply group-wise quantization
            tensor_q, scales, zero_points = self._quantize_per_group(tensor)
            
            # Create metadata
            result = {
                'tensor_q': tensor_q.cpu().to(torch.int32),
                'scales': scales.cpu().to(torch.float16),
                'zero_points': zero_points.cpu().to(torch.int32),
                'bits': torch.tensor(self.bits, dtype=torch.int32),
                'group_size': torch.tensor(self.group_size, dtype=torch.int32),
                'symmetric': torch.tensor(self.symmetric, dtype=torch.bool),
            }
            
            return result
            
        finally:
            # Clean up GPU memory if using CUDA
            if self.device.startswith('cuda'):
                # Be explicit about removing references to free memory
                if 'tensor_q' in locals():
                    del tensor_q
                if 'scales' in locals():
                    del scales
                if 'zero_points' in locals():
                    del zero_points
                torch.cuda.empty_cache()
                
        # Should not reach here due to finally block returning
        return {}

    def quantize_model(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Quantize all tensors in a model.

        Args:
            tensors: Dictionary mapping tensor names to tensors

        Returns:
            Dictionary mapping tensor names to quantized tensor dictionaries
        """
        quantized_tensors = {}
        
        # Iterate through tensors
        for name, tensor in tensors.items():
            try:
                self.logger.info(f"Quantizing tensor: {name}")
                quantized_tensors[name] = self.quantize(tensor)
                self.logger.info(f"Successfully quantized tensor: {name}")
            except Exception as e:
                self.logger.error(f"Error quantizing tensor: {name}, error: {e}")
                continue
                
        return quantized_tensors

    def dequantize(self, quantized_tensor: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Dequantize a tensor from its quantized representation.

        Args:
            quantized_tensor: Dictionary containing the quantized tensor and metadata

        Returns:
            Dequantized tensor
        """
        # Extract components
        tensor_q = quantized_tensor['tensor_q'].to(self.device)
        scales = quantized_tensor['scales'].to(self.device)
        zero_points = quantized_tensor['zero_points'].to(self.device)
        group_size = quantized_tensor['group_size'].item()
        
        # Handle tensors based on rank
        if tensor_q.dim() <= 1:
            # Reshape 1D tensor to 2D for group processing
            original_shape = tensor_q.shape
            tensor_q = tensor_q.reshape(1, -1)
            is_1d = True
        else:
            # For multi-dimensional tensors, treat first dim as channels
            is_1d = False
            
        # Get tensor shape
        if is_1d:
            num_channels, num_elements = 1, tensor_q.numel()
        else:
            num_channels = tensor_q.size(0)
            num_elements = tensor_q[0].numel()
            
        # Calculate number of groups per channel
        num_groups = math.ceil(num_elements / group_size)
        
        # Prepare output tensor
        tensor_dq = torch.zeros_like(tensor_q, dtype=torch.float32)
        
        # Process each channel
        for c in range(num_channels):
            channel_q = tensor_q[c] if not is_1d else tensor_q[0]
            channel_flat = channel_q.reshape(-1)
            
            # Pad to ensure divisibility by group_size
            pad_len = num_groups * group_size - channel_flat.size(0)
            if pad_len > 0:
                channel_flat = F.pad(channel_flat, (0, pad_len))
                
            # Reshape to (num_groups, group_size)
            channel_grouped = channel_flat.reshape(num_groups, group_size)
            
            # Process each group
            for g in range(num_groups):
                group_q = channel_grouped[g]
                
                # Get scale and zero point for this group
                scale = scales[c, g]
                zero_point = zero_points[c, g]
                
                # Dequantize group
                group_dq = self._dequantize_tensor(group_q, scale, zero_point)
                
                # Store dequantized values back to the tensor
                start_idx = g * group_size
                end_idx = min((g + 1) * group_size, num_elements)
                
                if is_1d:
                    tensor_dq[0, start_idx:end_idx] = group_dq[:end_idx-start_idx]
                else:
                    # Get the correct indices within the flattened tensor
                    tensor_dq_flat = tensor_dq[c].reshape(-1)
                    tensor_dq_flat[start_idx:end_idx] = group_dq[:end_idx-start_idx]
                    tensor_dq[c] = tensor_dq_flat.reshape(channel_q.shape)
                    
        # Restore original shape for 1D tensors
        if is_1d:
            tensor_dq = tensor_dq.reshape(original_shape)
            
        # Move result back to CPU
        return tensor_dq.cpu() 