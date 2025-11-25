# Spatial Analysis for Face-to-Face Networks

Toolkit for exploring face-to-face contact data: loading raw TIJ events, synchronising with study periods, mapping reader signatures to spatial plans, clustering agents/areas by activity, and visualising results with Plotly/Dash.

## Project layout
- `main/src/`: core logic (world model, period manager, network/matrix utilities, visualisation helpers).
- `main/notebooks/`: Jupyter workflows; `pipeline.ipynb` shows the end-to-end flow and `experiments_config.json` stores plan/reader metadata.
- `main/data/`: expected data folders (`TIJ_with_reader_list/`, `periodes/`, `plans/`, etc.). Data files are not included here.

## Setup
1) Use Python 3.10+ and create/activate a virtual environment.
2) Install dependencies:
```
pip install -r requirements.txt
```

## Running the notebook pipeline
1) Open `main/notebooks/pipeline.ipynb` in Jupyter Lab/Notebook.
2) Early in the notebook, make sure the paths and configs load correctly (already present in the notebook):
```python
from project_setup import setup_paths, load_experiments_dict, import_and_reload_modules
base_dir, src_path = setup_paths()
experiments_definition = load_experiments_dict("experiments_config.json")
WorldModel, DisplayManager, MatrixSimilarity, NetworkManager = import_and_reload_modules()

experiment_id = "WS16"  # choose one from experiments_config.json
world = WorldModel(experiment_id, experiments_definition, base_dir)
world.initialize()
```
3) Continue with the notebook cells to compute signatures, assign agents to areas, build transition graphs, and visualise results. The visualisers rely on Plotly; static exports use Kaleido.

## Notes
- Data files referenced in `experiments_config.json` should live under `main/data/plans/<experiment_id>/` and `main/data/TIJ_with_reader_list/`.
- If you add new experiments, extend `main/notebooks/experiments_config.json` with the offset and plan/reader definitions.
- Visual dashboards use Dash; for non-notebook use you can call the visualiser classes from `main/src/display/` directly.
