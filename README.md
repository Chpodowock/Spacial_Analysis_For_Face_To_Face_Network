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
3) (Optional) Add `main/src` to `PYTHONPATH` if you are running code outside the notebook:
   ```
   export PYTHONPATH="$PWD/main/src:$PYTHONPATH"
   ```

## Running the notebook pipeline
1) Launch Jupyter from the repo root so relative paths resolve:
   ```
   jupyter lab
   ```
2) Open `main/notebooks/pipeline.ipynb`.
3) Run the configuration cell (already included):
   ```python
   from project_setup import setup_paths, load_experiments_dict, import_and_reload_modules
   base_dir, src_path = setup_paths()
   experiments_definition = load_experiments_dict("experiments_config.json")
   WorldModel, DisplayManager, MatrixSimilarity, NetworkManager = import_and_reload_modules()

   experiment_id = "WS16"  # choose one from experiments_config.json
   world = WorldModel(experiment_id, experiments_definition, base_dir)
   world.initialize()
   ```
4) Continue through the notebook to compute signatures, assign agents to areas, build transition graphs, and visualise results. Plotly drives the interactive views; static exports use Kaleido.

## Notes
- Data expected paths:
  - Plans: `main/data/plans/<experiment_id>/`
  - Reader-enriched TIJ: `main/data/TIJ_with_reader_list/`
  - Period definitions: `main/data/periodes/`
- Add new experiments by updating `main/data/experiments_config.json` with offsets and plan/reader definitions.
- Visual dashboards use Dash; outside notebooks you can import the visualiser classes from `main/src/display/` directly.
