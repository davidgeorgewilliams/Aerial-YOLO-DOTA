import yaml


def read_yaml_config(yaml_path):
    """
    Read and parse a YAML configuration file.

    This function opens a YAML file at the specified path, reads its contents,
    and returns the parsed data as a Python object (typically a dictionary).

    Args:
        yaml_path (str): The file path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the parsed YAML data.

    Raises:
        FileNotFoundError: If the specified YAML file does not exist.
        yaml.YAMLError: If there's an error parsing the YAML file.

    Example:
        >>> config = read_yaml_config('config.yaml')
        >>> print(config['dataset_path'])

    Note:
        This function uses yaml.safe_load() for secure parsing, which avoids
        arbitrary code execution vulnerabilities that can occur with yaml.load().
    """
    with open(yaml_path, "r") as file:
        return yaml.safe_load(file)
