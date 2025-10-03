import os
from box.exceptions import BoxValueError
import yaml
from cnnClassifier import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a YAML file and returns a ConfigBox object.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError as e:
        raise ValueError("yaml file is empty") from e
    except Exception as e:
        raise e
    
@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """
    Create a list of directories.

    Args:
        path_to_directories (list): List of path of directories
        ignore_log (bool, optional): Ignore logging if multiple directories are created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Directory created: {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    """
    Save a dictionary to a JSON file.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON file saved successfully to: {path}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Load a JSON file and return a ConfigBox object.
    """
    with open(path) as f:
        content = json.load(f)
    
    logger.info(f"JSON file loaded successfully from: {path}")
    return ConfigBox(content)
    
@ensure_annotations
def load_bin(path: Path) -> Any:
    """
    Load a binary file and return the data.
    """
  
    data = joblib.load(path)
    logger.info(f"Binary file loaded successfully from: {path}")
    return data

@ensure_annotations
def save_bin(data: Any, path: Path):
    """
    Save a binary file.
    """
    joblib.dump(data, path)
    logger.info(f"Binary file saved successfully to: {path}")

@ensure_annotations
def get_size(path: Path) -> str:
    """
    Get the size of a file in bytes.
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"

@ensure_annotations
def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()

@ensure_annotations
def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())
