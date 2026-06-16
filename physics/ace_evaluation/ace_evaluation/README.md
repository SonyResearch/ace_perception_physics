# ace_evaluation

Core physics evaluation library for table tennis ball flight analysis.

## Structure

```
ace_evaluation/
├── __init__.py
├── racket_contacts.py          # Racket contact model (Nakashima-based)
├── models/                     # Pre-trained ONNX models for residual contact models
│   ├── RCM-175_wo1203_tangential_COR.onnx
│   ├── residual_model_compact.exponential.dim8.pre1002_wo0428.RCM-118.onnx
│   └── residual_model_compact.exponential.onnx
├── update_hdf5/                # HDF5 data processing pipeline
│   ├── column_mapping.py       # Column name normalisation
│   ├── racket_contacts.py      # Racket contact extraction for HDF5
│   ├── table_contacts.py       # Table contact extraction for HDF5
│   └── update_hdf5.py          # Main HDF5 update entry point
└── utilities/                  # Shared helpers
    ├── aerodynamics_integrator.py
    ├── aerodynamics_utilities.py
    ├── data_classes.py          # MatchCollection / Match / Rally / Shot / FlightSegment
    ├── math_utilities.py
    └── physics_params.py
```

## Key modules

- **`racket_contacts.py`** — Nakashima racket contact model used across the evaluation pipeline.
- **`utilities/data_classes.py`** — Hierarchical data model (`MatchCollection > Match > Game > Rally > Shot > FlightSegment`) loaded from HDF5 files.
- **`update_hdf5/update_hdf5.py`** — Reads raw HDF5 recordings, runs aerodynamics fitting, table/racket contact evaluation, and writes enriched HDF5 output.

## Dependencies

- `numpy`, `scipy`, `h5py`, `onnxruntime`
- `ace_evaluation.utilities` (internal)