# Residual NN Racket Contact Model

This folder contains the scripts, data, and configurations necessary to train the proposed Residual NN Racket Contact Model.

The configuration parameters for training the model are defined in a `yaml` file e.g: `proposed_model.config.yaml`

The data used to train the model is provided as a `CSV` file in `/data/refined_contact_FK_train_processed.csv`

## Quick start

```bash
cd physics/
python residual_rcm/scripts/train_model.py
```

The resulting ONNX will be exported to `./outputs/YYYY-MM-DD/HH-MM-SS`
