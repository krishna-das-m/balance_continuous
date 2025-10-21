import os
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations
from pathlib import Path
import yaml


@ensure_annotations
def read_yaml(path_to_yaml:Path)-> ConfigBox:
    """reads yaml file

    Args:
        path_to_yaml (Path): input path 

    Returns:
        ConfigBox: ConfigBox
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    
@ensure_annotations
def create_directories(path_to_dir:list, verbose=True):
    """create list of directories

    Args:
        path_to_dir (list): list of path of directories
        verbose (bool, optional): _description_. Defaults to True.
    """
    for path in path_to_dir:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")