# ace_evaluation (Standalone)

Standalone evaluation and visualization tools for table tennis physics models — aerodynamics, table contacts, and racket contact models (RCM).

## Structure

```
ace_evaluation/          # Core library: data classes, physics utilities, contact models, ONNX models
data_plotter/            # PySide6/pyqtgraph desktop app for interactive data exploration
publication_plots/       # Publication-ready figure generation (Plotly + matplotlib)
pyproject.toml           # Package metadata and dependencies
```

See per-directory READMEs for details:
- [ace_evaluation/README.md](ace_evaluation/README.md) — library modules and data model
- [data_plotter/README.md](data_plotter/README.md) — desktop plotter app usage
- [DEPENDENCY_GRAPH.md](DEPENDENCY_GRAPH.md) — module dependency graph

## Quick start

```bash
pip install -e .
# Run the desktop plotter
python data_plotter/app.py /path/to/hdf5/folder
# Generate publication figures
python -m publication_plots.publication_plots /path/to/hdf5/folder
```

## HDF5 data pipeline

Process raw recordings into enriched HDF5 files with aerodynamics fitting and contact evaluation:

```bash
python -m ace_evaluation.update_hdf5.update_hdf5 --help
```
