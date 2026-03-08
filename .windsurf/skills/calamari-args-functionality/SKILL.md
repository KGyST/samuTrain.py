---
name: calamari-args-functionality
description: Detailed functionalities of Calamari command-line arguments for various scripts
---

# Calamari Argument Functionalities

This skill documents the detailed functionalities of command-line arguments for Calamari OCR scripts, including predict.py, train.py, eval.py, and others.

## predict.py Arguments

### --checkpoint
- **Type**: List[str]
- **Description**: Paths to model checkpoints (without .json extension). Supports multiple checkpoints for ensemble prediction. Automatically appends .json and resolves wildcards.
- **Example**: `--checkpoint model1 model2` for ensemble of two models.

### --data
- **Type**: CalamariDataGeneratorParams
- **Description**: Data source configuration. Uses FileDataParams by default, with choices for different data formats (File, PageXML, Abbyy, Hdf5).
- **Sub-arguments**: 
  - --data.files: List of input image files or directories.
  - --data.preload: Whether to preload data into memory.
  - --data.num_processes: Number of processes for data loading.
- **Example**: `--data.files "data/*.png"`

### --verbose
- **Type**: bool
- **Default**: True
- **Description**: Prints prediction results to the log for each processed line.

### --extended_prediction_data
- **Type**: bool
- **Default**: False
- **Description**: Writes extended prediction data including predicted strings, labels, positions, probabilities, and alternatives to a .pred or .json file.

### --extended_prediction_data_format
- **Type**: str
- **Default**: "json"
- **Options**: "pred", "json"
- **Description**: Format for extended data. "pred" includes logits, "json" excludes them for smaller files.

### --output_dir
- **Type**: Optional[str]
- **Description**: Directory to write prediction output files. Defaults to the directory of each input file.

### --predictor
- **Type**: PredictorParams
- **Description**: Prediction parameters including beam search settings.
- **Sub-arguments**: 
  - --predictor.beam_width: Beam width for decoding.
  - --predictor.non_normalized: Use non-normalized probabilities.

### --voter
- **Type**: VoterParams
- **Description**: Voting parameters for ensemble prediction.
- **Sub-arguments**: 
  - --voter.voter: Voting method (e.g., confidence_voter).

## train.py Arguments

Training uses nested arguments under --trainer.

### --trainer.output_dir
- **Type**: str
- **Description**: Directory to save the trained model, checkpoints, and logs.

### --trainer.scenario.data.files
- **Type**: List[str]
- **Description**: Training data files (images and GT text pairs).

### --trainer.gen.train_gen.epochs
- **Type**: int
- **Description**: Number of training epochs.

### --trainer.early_stopping.n_to_go
- **Type**: int
- **Description**: Patience for early stopping (number of epochs without improvement).

### --trainer.network
- **Type**: Optional[str]
- **Description**: Network architecture definition string or path to JSON file. Defaults to CNN-LSTM architecture.
- **Example**: `--trainer.network="cnn=40:3x3,pool=2x2,cnn=60:3x3,pool=2x2,lstm=200,dropout=0.5"`

### --trainer.scenario.data.line_height
- **Type**: int
- **Default**: 48
- **Description**: Target line height for image preprocessing.

### --trainer.auto_upgrade_checkpoints
- **Type**: bool
- **Default**: True
- **Description**: Automatically upgrade older checkpoints for warm start.

## eval.py Arguments

### --gt
- **Type**: CalamariDataGeneratorParams
- **Description**: Ground truth data parameters.
- **Sub-arguments**: Similar to --data in predict.py.

### --pred
- **Type**: Optional[CalamariDataGeneratorParams]
- **Description**: Prediction data parameters. If not provided, uses GT converted to prediction format.

### --n_confusions
- **Type**: int
- **Default**: 10
- **Description**: Number of most common confusions to print. Use -1 for all.

### --n_worst_lines
- **Type**: int
- **Default**: 0
- **Description**: Number of worst recognized lines to print with errors.

### --xlsx_output
- **Type**: Optional[str]
- **Description**: Path to write detailed evaluation results to an XLSX file.

### --skip_empty_gt
- **Type**: bool
- **Default**: False
- **Description**: Ignore empty ground truth lines.

### --checkpoint
- **Type**: Optional[str]
- **Description**: Checkpoint to load text preprocessor for GT text processing.

### --evaluator
- **Type**: EvaluatorParams
- **Description**: Evaluation parameters including CER/SER computation settings.

## Additional Scripts

### cross_fold_train.py
Similar to train.py but for cross-fold validation. Uses CrossFoldTrainerParams with additional fold management arguments.

### Other Utility Scripts
- **dataset_statistics.py**: Analyzes dataset properties like character frequencies and image sizes.
- **ensemble.py**: Combines predictions from multiple models using various voting strategies.
- **experiment.py**: Runs multiple training experiments with different hyperparameters.
- **img_gt_pred_to_html.py**: Converts evaluation results to HTML visualization.
- **split_dirs_to_train_eval.py**: Splits data directories into training and evaluation sets.

For complete parameter lists and advanced options, refer to the source code parameter classes or use `--help` with each script.
