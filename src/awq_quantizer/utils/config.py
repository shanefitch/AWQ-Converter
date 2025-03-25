"""
Configuration utility for AWQ Quantizer.
"""

import os
from typing import Any, Dict, Optional, Union

import yaml


class Config:
    """
    Configuration class for AWQ Quantizer.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration.

        Args:
            config_path: Path to the configuration file
        """
        self.config = {}
        
        # Load default configuration
        default_config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config",
            "default_config.yaml",
        )
        
        if os.path.exists(default_config_path):
            self.config = self._load_yaml(default_config_path)
        
        # Load user configuration if provided
        if config_path is not None and os.path.exists(config_path):
            user_config = self._load_yaml(config_path)
            self._update_nested_dict(self.config, user_config)
            
        # Initialize model section if not present
        if "model" not in self.config:
            self.config["model"] = {}
            
        # Set defaults for model configuration
        model_config = self.config["model"]
        model_config.setdefault("path", "")
        model_config.setdefault("hub_model_id", "")
        model_config.setdefault("from_hub", False)
        model_config.setdefault("revision", "main")
        model_config.setdefault("token", None)
            
        # Handle hub_model_id from command line or config
        if model_config["hub_model_id"]:
            model_config["path"] = model_config["hub_model_id"]
            model_config["from_hub"] = True
        elif model_config["path"] and "/" in model_config["path"] and not os.path.exists(model_config["path"]):
            # If path looks like a hub ID (contains '/') and doesn't exist locally, treat it as a hub model
            model_config["hub_model_id"] = model_config["path"]
            model_config["from_hub"] = True

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        """
        Load YAML file.

        Args:
            path: Path to the YAML file

        Returns:
            Dictionary containing the YAML content
        """
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _update_nested_dict(self, d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update nested dictionary.

        Args:
            d: Dictionary to update
            u: Dictionary with updates

        Returns:
            Updated dictionary
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.

        Args:
            key: Configuration key (dot-separated for nested keys)
            default: Default value if key is not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.

        Args:
            key: Configuration key (dot-separated for nested keys)
            value: Configuration value
        """
        keys = key.split(".")
        config = self.config
        
        for i, k in enumerate(keys[:-1]):
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value

    def save(self, path: str) -> None:
        """
        Save configuration to file.

        Args:
            path: Path to save the configuration
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def __getitem__(self, key: str) -> Any:
        """
        Get configuration value using dictionary-like access.

        Args:
            key: Configuration key

        Returns:
            Configuration value
        """
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Set configuration value using dictionary-like access.

        Args:
            key: Configuration key
            value: Configuration value
        """
        self.set(key, value)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration.

    Args:
        config_path: Path to the configuration file

    Returns:
        Configuration instance
    """
    return Config(config_path) 