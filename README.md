# Model
- Synthetic data generation
- Train ML model
- Export ONNX network to input to EZKL

# Workflow

## Generate synthetic data

```bash
$ cargo run
```

## Train Model and Save to ONNX Format

```bash
$ python src/model.py synthetic_data/credit_data.json
```

This generates the `credit_model.onnx` file.

## Development Setup

Make sure that all the tools defined in `.tool-versions` are installed before you start.

Then install python dependencies:

```bash
$ pip install -r requirements.txt
```

Then build the project.

```bash
$ cargo build
```

## Generate New Synthetic Data

If you want to re-train the model in new synthetic data:

```bash
$ cargo run
```

Then train the model and generate the new ONNX format:

```bash
$ python src/model.py synthetic_data/credit_data.json
```

Commit `synthetic_data/credit_data.json` and `credit_model.onnx` to git
to make sure that we know which data have been used to train the particular
model.

Note: The model is not evaluated for its correctness. This is something that
we could have done with _test_ data, but we haven't for now.
