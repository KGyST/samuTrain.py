---
name: calamari-expert
description: Comprehensive Calamari CLI parameters, PAIArgumentParser documentation, flat/nested modes, and detailed argument functionalities for all scripts
---

# Calamari CLI Arguments Documentation

## Overview

Calamari OCR uses `paiargparse.PAIArgumentParser` for hierarchical parameter management. This system supports both **flattened** and **nested** parameter structures, which is crucial for avoiding `UnknownArgumentError` when calling Calamari scripts programmatically.

## PAIArgumentParser Implementation Pattern

### Basic Structure

All Calamari scripts follow this pattern:

```python
from paiargparse import PAIArgumentParser

def parse_args(args=None):
    parser = PAIArgumentParser()
    parser.add_argument("--version", action="version", version="%(prog)s v" + __version__)
    parser.add_root_argument("root", ParamsClass, flat=True/False)
    return parser.parse_args(args).root
```

### Key Scripts and Their Root Arguments

#### 1. train.py
- **Root**: `TrainerParams` (from `CalamariScenario.default_trainer_params()`)
- **Flat Mode**: `False` (nested structure)
- **Key Classes**: `TrainerParams`, `CalamariScenarioParams`

#### 2. predict.py  
- **Root**: `PredictArgs`
- **Flat Mode**: `True` (flattened structure)
- **Key Classes**: `PredictArgs`, `CalamariDataGeneratorParams`, `CTCDecoderParams`, `PredictorParams`

#### 3. eval.py
- **Root**: `EvalArgs`
- **Flat Mode**: `False` (nested structure with ignore list)
- **Ignore Fields**: `["gt.images", "pred.images"]`
- **Key Classes**: `EvalArgs`, `CalamariDataGeneratorParams`, `EvaluatorParams`

#### 4. cross_fold_train.py
- **Root**: `CrossFoldTrainerParams`
- **Flat Mode**: `False` (nested structure)

## Parameter Modes

### `pai_meta` Mode Types

1. **`mode="flat"`**: Parameter appears at top level (e.g., `--checkpoint`)
2. **`mode="ignore"`**: Parameter is hidden from CLI
3. **Default (nested)****: Parameter appears as nested (e.g., `--data.checkpoint`)

### Examples from predict.py

```python
@pai_dataclass
@dataclass
class PredictArgs:
    checkpoint: List[str] = field(
        default_factory=list, 
        metadata=pai_meta(mode="flat", help="Path to the checkpoint without file extension")
    )
    data: CalamariDataGeneratorParams = field(
        default_factory=FileDataParams,
        metadata=pai_meta(mode="flat", choices=DATA_GENERATOR_CHOICES),
    )
    ctc_decoder: CTCDecoderParams = field(
        default_factory=CTCDecoderParams, 
        metadata=pai_meta(mode="ignore")
    )
```

## Flattened vs Nested Structure

### Flattened Structure (`flat=True`)
- All parameters with `mode="flat"` appear at top level
- Example: `--checkpoint model.ckpt` instead of `--root.checkpoint model.ckpt`
- Used in: `predict.py`

### Nested Structure (`flat=False`)  
- Parameters maintain their hierarchical structure
- Example: `--trainer.data.files train.txt`
- Used in: `train.py`, `eval.py`, `cross_fold_train.py`

## Parameter Hierarchy

### Training Parameters (train.py)
```
trainer/
├── scenario/
│   ├── data/
│   │   ├── train/
│   │   │   └── files
│   │   └── validation/
│   │       └── files
│   └── model/
│       ├── network
│       └── layers[]
├── gen/
│   ├── train/
│   └── validation/
├── codec/
└── network (flat)
```

### Prediction Parameters (predict.py)
```
root (flat=True)
├── checkpoint (flat)
├── data (flat)
├── verbose (flat)
├── extended_prediction_data (flat)
├── output_dir (flat)
├── predictor (flat)
└── ctc_decoder (ignore)
```

## Common Parameter Patterns

### Data Generator Parameters
- Always use `CalamariDataGeneratorParams` or subclasses
- Support multiple formats via `DATA_GENERATOR_CHOICES`
- Common fields: `files`, `preload`, `num_processes`

### Model Parameters
- Network definition via `--network` (flat) or nested model params
- Layer configurations: Conv2D, LSTM, Dropout, etc.
- Downscale factor automatically computed

### Codec Parameters
- Character set management
- External dictionary support
- Case sensitivity options

## Avoiding UnknownArgumentError

### 1. Check Parameter Mode
```python
# For flattened scripts (predict.py)
args = ["--checkpoint", "model.ckpt", "--data.files", "train.txt"]

# For nested scripts (train.py)  
args = ["--trainer.checkpoint", "model.ckpt", "--trainer.gen.train.files", "train.txt"]
```

### 2. Use Ignore Lists When Needed
```python
parser.add_root_argument("root", EvalArgs, ignore=["gt.images", "pred.images"])
```

### 3. Respect Choice Constraints
```python
metadata=pai_meta(choices=DATA_GENERATOR_CHOICES)
```

## Key Parameter Files

- `calamari_ocr/ocr/training/params.py`: `TrainerParams`
- `calamari_ocr/ocr/scenario_params.py`: `CalamariScenarioParams`
- `calamari_ocr/ocr/dataset/params.py`: `DataParams`
- `calamari_ocr/ocr/model/params.py`: `ModelParams`
- `calamari_ocr/ocr/predict/params.py`: `PredictorParams`

## Script-Specific Notes

### train.py
- Uses `CalamariScenario.default_trainer_params()` for defaults
- Supports network definition via `--network` flat parameter
- Auto-upgrades checkpoints when `auto_upgrade_checkpoints=True`

### predict.py
- Fully flattened structure for easier CLI usage
- Supports multiple checkpoints for ensemble prediction
- Extended prediction data available in `.pred` or `.json` format

### eval.py
- Ignores image paths in GT/PRED to focus on text comparison
- Supports XLSX output for detailed analysis
- Can load text preprocessor from checkpoint

### cross_fold_train.py
- Manages multiple training folds automatically
- Inherits most parameters from `TrainerParams`

## Best Practices

1. **Always check `flat` mode** before constructing argument lists
2. **Use `pai_meta` mode="flat"** for frequently used top-level parameters
3. **Use `mode="ignore"`** for complex nested objects not needed in CLI
4. **Provide meaningful help text** in `pai_meta` for all CLI-exposed parameters
5. **Use `choices`** for enumerated parameters to enable validation
6. **Test with `--help`** to verify parameter structure before programmatic use

# Calamari Argument Functionalities

This section documents the detailed functionalities of command-line arguments for Calamari OCR scripts, including predict.py, train.py, eval.py, and others.

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
