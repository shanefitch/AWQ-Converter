"""
Model loading package.
"""

from typing import Optional, List
import os
from huggingface_hub import snapshot_download, hf_hub_download
from ..utils.tensor_utils import filter_safetensor_files, is_consolidated_file

def load_model_from_hub(
    model_id: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    allow_patterns: Optional[List[str]] = None,
) -> str:
    """
    Load a model from the Hugging Face Hub.
    
    Args:
        model_id: The model ID on Hugging Face Hub
        revision: Optional git revision to use
        token: Optional authentication token
        cache_dir: Optional directory to cache files
        allow_patterns: Optional list of file patterns to download
        
    Returns:
        Path to the downloaded model files
    """
    # Default to downloading only safetensors files if no patterns specified
    if allow_patterns is None:
        allow_patterns = ["*.safetensors"]
    
    try:
        # First try to find a consolidated file
        consolidated_file = None
        try:
            # List files in the repo
            from huggingface_hub import list_repo_files
            files = list_repo_files(model_id, revision=revision, token=token)
            
            # Look for consolidated file
            safetensor_files = [f for f in files if f.endswith('.safetensors')]
            consolidated_files = [f for f in safetensor_files if 'consolidated' in f.lower()]
            
            if consolidated_files:
                # Download just the consolidated file
                consolidated_file = hf_hub_download(
                    repo_id=model_id,
                    filename=consolidated_files[0],
                    revision=revision,
                    token=token,
                    cache_dir=cache_dir,
                )
        except Exception:
            # If listing files fails, fall back to downloading all files
            pass
            
        if consolidated_file:
            print(f"Using consolidated model file: {os.path.basename(consolidated_file)}")
            return os.path.dirname(consolidated_file)
            
        # If no consolidated file or error, download all safetensor files
        print("Downloading all model files...")
        local_dir = snapshot_download(
            repo_id=model_id,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            allow_patterns=allow_patterns,
        )
        return local_dir
        
    except Exception as e:
        raise ValueError(f"Error loading model from Hugging Face Hub: {e}")

def load_model(
    local_path: Optional[str] = None,
    hub_model_id: Optional[str] = None,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> str:
    """
    Load a model from either local path or Hugging Face Hub.
    
    Args:
        local_path: Path to local model files
        hub_model_id: Model ID on Hugging Face Hub
        revision: Optional git revision to use
        token: Optional authentication token
        cache_dir: Optional directory to cache files
        
    Returns:
        Path to the model files
    """
    if local_path and hub_model_id:
        raise ValueError("Cannot specify both local_path and hub_model_id")
    elif not local_path and not hub_model_id:
        raise ValueError("Must specify either local_path or hub_model_id")
        
    if hub_model_id:
        return load_model_from_hub(
            model_id=hub_model_id,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
        )
    else:
        if not os.path.exists(local_path):
            raise ValueError(f"Local path does not exist: {local_path}")
        return local_path 