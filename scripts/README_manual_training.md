# Manual Training Script for samuTrain

## Overview

The `try_manual_training.py` script implements manual training with immediate weight updates and PRED/CERT feedback for Calamari OCR models. Unlike traditional training that runs on entire datasets, this script allows you to train on individual cases or small batches with granular control.

## Key Features

- **Single Case Training**: Train on one image+ground truth pair at a time
- **Immediate Weight Updates**: Model weights are updated after each training step
- **PRED/CERT Feedback**: Get prediction and confidence values before and after training
- **CER Calculation**: Character Error Rate computed automatically
- **Database Integration**: Updates predictions in samuTrain database
- **Temporary Training**: Uses temporary datasets to avoid interfering with main training
- **Model Saving**: Option to save updated model permanently after training
- **Automatic Naming**: Timestamped model names for easy version tracking

## Usage

### Command Line Usage

```bash
# Train on all cases in a dataset
python try_manual_training.py data/64_case models/generic_ocr_model

# Train on specific case ID
python try_manual_training.py data/single_case models/generic_ocr_model --case-id 1

# Train and save updated model
python try_manual_training.py data/single_case models/generic_ocr_model --save-model --output-model models/manual_trained

# Train with automatic model naming (timestamped)
python try_manual_training.py data/single_case models/generic_ocr_model --save-model
```

### Programmatic Usage

```python
from try_manual_training import CalamariManualTrainer

# Initialize trainer
trainer = CalamariManualTrainer("models/generic_ocr_model/best.ckpt.json")

# Train on single case
result = trainer.train_single_case("path/to/image.png", "ground truth text")

print(f"Before: '{result['pred_before']}'")
print(f"After:  '{result['pred']}'")
print(f"CER:    {result['cer']:.3f}")
print(f"Loss:   {result['loss']:.4f}")
print(f"Time:   {result['training_time']:.2f}s")
```

## Output Format

The script returns a dictionary with the following fields:

```python
{
    "loss": float,           # Training loss value
    "pred": str,             # Prediction after training
    "cer": float,            # Character Error Rate
    "pred_before": str,      # Prediction before training
    "training_time": float,  # Time taken for training step
    "status": str            # "ok", "error", or "stopped"
}
```

## Technical Implementation

### How It Works

1. **Temporary Dataset Creation**: Creates a temporary directory with the single case data
2. **Continue Learning**: Uses Calamari's continue learning with warmstart from existing model
3. **Single Epoch Training**: Runs exactly one epoch on the single case
4. **Weight Update**: Model weights are updated during training
5. **Prediction Comparison**: Gets predictions before and after training
6. **Database Update**: Updates the samuTrain database with new predictions

### Key Components

- **CalamariManualTrainer**: Main class for manual training operations
- **Temporary Training**: Uses `calamari_train` with temporary datasets
- **Weight Updates**: Leverages Calamari's warmstart mechanism
- **Prediction Pipeline**: Uses MultiPredictor for inference

## Requirements

- Python 3.10+ (in venv_310)
- Calamari OCR library
- TensorFlow 2.15.0+
- samuTrain database setup

## Examples

### Example 1: Single Case Training
```bash
cd "e:\Git\samuTrain.py"
.\venv_310\Scripts\python.exe .\scripts\try_manual_training.py data/single_case models/generic_ocr_model --case-id 1
```

### Example 2: Batch Training
```bash
cd "e:\Git\samuTrain.py"
.\venv_310\Scripts\python.exe .\scripts\try_manual_training.py data/64_case models/generic_ocr_model
```

### Example 3: Model Saving
```bash
# Save with custom name
python try_manual_training.py data/single_case models/generic_ocr_model --save-model --output-model models/my_trained_model

# Save with automatic timestamp (e.g., models/generic_ocr_model_manual_trained_20260427_052400)
python try_manual_training.py data/single_case models/generic_ocr_model --save-model
```

### Example 4: Programmatic Usage
```python
# See example_manual_training.py for complete example
from try_manual_training import CalamariManualTrainer

trainer = CalamariManualTrainer("models/generic_ocr_model/best.ckpt.json")
result = trainer.train_single_case("data/single_case/01002c.bin.png", "1962")
```

## Model Saving Feature

The script now supports permanent model saving after manual training:

- **Automatic Naming**: Uses timestamp format `model_manual_trained_YYYYMMDD_HHMMSS`
- **Custom Names**: Specify your own output path with `--output-model`
- **Complete Models**: Saves full model with checkpoints, logs, and metadata
- **Validation**: Only saves if at least one training case was successful
- **Integration**: Saved models can be used immediately for further training or prediction

## Notes

- Training time per case is typically 60-120 seconds
- Model weights are permanently updated during training
- Database is updated with new predictions automatically
- Temporary files are cleaned up automatically
- Script handles both .bin.png and regular image formats

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure the model folder contains `best.ckpt.json` or `best.ckpt`
2. **Image not found**: Check that image paths are correct and files exist
3. **Database errors**: Ensure samuTrain database is properly initialized
4. **Memory issues**: Reduce batch size or close other applications

### Performance Tips

- Use SSD storage for faster I/O
- Ensure sufficient RAM (8GB+ recommended)
- Close unnecessary applications during training
- Consider GPU acceleration if available

## Integration with samuTrain

This script integrates seamlessly with the samuTrain ecosystem:

- Uses the same database schema
- Follows the same file path conventions
- Compatible with existing model formats
- Updates the same prediction tables
- Works with the same data preprocessing pipeline
