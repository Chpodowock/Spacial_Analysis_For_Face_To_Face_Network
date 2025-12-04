# Spatial Analysis for Face-to-Face Networks

Toolkit for exploring face-to-face contact data: loading raw TIJ events, synchronising with study periods, mapping reader signatures to spatial plans, clustering agents/areas by activity, and visualising results with Plotly/Dash.

## Project layout
- `main/src/`: core logic (world model, period manager, network/matrix utilities, visualisation helpers).
- `main/notebooks/`: Jupyter workflows; `pipeline.ipynb` shows the end-to-end flow and `experiments_config.json` stores plan/reader metadata.
- `main/data/`: expected data folders (`TIJ_with_reader_list/`, `periodes/`, `plans/`, etc.). Only sample Data files are included here.

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

experiment_id = "WS16"  # For the sample data
world = WorldModel(experiment_id, experiments_definition, base_dir)
world.initialize()
```
3) Continue with the notebook cells to compute signatures, assign agents to areas, build transition graphs, and visualise results. The visualisers rely on Plotly; static exports use Kaleido.
- Visual dashboards are implemented with Dash. For non-notebook usage, you can instantiate and run the visualiser classes directly from `main/src/display/`.

## Custom data files

- Custom datasets must follow the same folder layout as the provided sample data:
  - `periodes/`: contains the initial time-segmentation files (e.g. high/low/transition periods).
  - `plans/<experiment_name>/`: contains the venue floor plan(s) and reader layout for each experiment.
  - `<raw_contacts_folder>/`: store the raw contact data in the dedicated folder used in the sample data.
  - `tijs/`: contains the preprocessed *tij* files with enriched signature information.

- For any new experiment, extend `main/data/experiments_config.json` with:
  - the time offset, and  
  - the plan/reader definitions used to map RFID readers to the floor plan.
