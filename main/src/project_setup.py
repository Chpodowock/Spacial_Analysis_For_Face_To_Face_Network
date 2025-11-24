import sys
from pathlib import Path
import importlib
import json

# === Setup paths ===
def setup_paths():
    base_dir = Path.cwd().parent
    src_path = base_dir / "src"

    include_dir_paths = [
        src_path,
        src_path / "display",
        src_path / "entities"
    ]

    for path in include_dir_paths:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.append(path_str)

    return base_dir, src_path


# === Load experiment configuration ===
def load_experiments_dict(filepath):
    def convert(obj):
        if isinstance(obj, list) and len(obj) == 2 and all(isinstance(x, (int, float)) for x in obj):
            return tuple(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        print(f"✅ Loaded experiment definition from {filepath}")
        return convert(data)


# === Import and reload all project modules ===
def import_and_reload_modules():
    import world_model
    import display_manager
    import matrix_simil
    import network_manager

    importlib.reload(world_model)
    importlib.reload(display_manager)
    importlib.reload(matrix_simil)
    importlib.reload(network_manager)

    from world_model import WorldModel
    from display_manager import DisplayManager
    from matrix_simil import MatrixSimilarity
    from network_manager import NetworkManager

    return WorldModel, DisplayManager, MatrixSimilarity, NetworkManager
