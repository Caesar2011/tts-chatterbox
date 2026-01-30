import tomllib
from pathlib import Path

def load_project_config():
    """Loads the main project configuration from pyproject.toml."""
    try:
        pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
        return data['tool']['audiobook_producer']['settings']
    except (FileNotFoundError, KeyError) as e:
        raise RuntimeError(
            f"Could not load or parse configuration from pyproject.toml. "
            f"Ensure '[tool.audiobook_producer.settings]' section exists. Error: {e}"
        )

PROJECT_CONFIG = load_project_config()
