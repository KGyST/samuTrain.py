# Calamari Operational Rules and Best Practices

## Overview
This document outlines operational rules, troubleshooting tips, and best practices for managing Calamari OCR within the samuTrain framework.

## Terminal Commands

### Run Server
To run the samuTrain server with Calamari OCR, use the following command:

```bash
python scripts/run_server.py --data-folder "data/single_case" --model-folder "models/new_model"
```

### Reset Database
To reset the database, use the following command:

```bash
python scripts/run_server.py --reset-db --data-folder "data/single_case"
```

## Troubleshooting Knowledge

### Calamari Load Error
If loading fails with "checkpoint version 2", it is a known TF 2.4 compatibility issue. Proceed by performing one training cycle to force a version 5 export.

### Static Files
Always serve static files using absolute paths via `os.path.dirname(os.path.abspath(__file__))`.

## Code Style

### General Guidelines
- Follow PEP 8 guidelines for Python code.
- Use descriptive variable and function names.
- Include docstrings for all functions and classes.

### Static File Handling
- Use absolute paths for static files to ensure consistency across different environments.
- Example:
  ```python
  static_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
  ```

## Database Management

### SQLite Backend
- samuTrain uses a built-in SQLite backend (`stockDB` style) to manage data samples virtually.
- Ensure data integrity by using transactions for critical operations.

### Virtual Set Logic
- **Trainset**: Samples tagged with `status='TRAIN'`. These form the core pool for training.
- **Failset**: Samples that the model fails to predict correctly are flagged as `status='FAIL'` in the database.
  - During training, the orchestrator pulls a weighted mix of 'TRAIN' and 'FAIL' samples.
  - When the model successfully predicts a 'FAIL' sample, its status is automatically reverted to 'TRAIN'.
- **Testset**: A protected set of samples (flagged as `status='TEST'`). It is never used for training and runs exclusively at the end of an epoch to report objective efficiency.

## Model Management

### Loading Models
- Use `SavedCalamariModel` to load checkpoints.
- Ensure that the model is built before loading weights to avoid compatibility issues.

### Saving Models
- Save model weights using `graph.save_weights("path/to/best.ckpt")`.
- Omit the `.json` suffix when saving weights.

## Best Practices

### Training
- Always perform a forward pass to build the graph before loading weights.
- Use the Adam optimizer for training and ensure it is created separately if `graph.model.optimizer` does not exist.

### Data Handling
- Ensure that input images are in `uint8` format and not pre-normalized to float [0,1].
- Use `img_len` shape `(B,)` instead of `(B,1)` to avoid incorrect sequence lengths.

### Error Handling
- Implement robust error handling for model loading and training processes.
- Use `expect_partial()` when loading weights to silence warnings about optimizer slots not restored.

## Compatibility Notes

### TensorFlow Version
- Calamari OCR is compatible with TensorFlow 2.x.
- Be aware of TF 2.4 compatibility issues, especially with checkpoint versions.

### Python Version
- Use Python 3.10 for compatibility with the samuTrain framework.

## Summary
This document provides operational rules and best practices for managing Calamari OCR within the samuTrain framework. It covers terminal commands, troubleshooting tips, code style, database management, model management, and best practices for training and data handling.