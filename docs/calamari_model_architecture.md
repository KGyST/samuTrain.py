# Calamari Model Architecture

## Overview
This document provides an in-depth look at the architecture of the Calamari OCR model, focusing on its key components and their interactions.

## Key Components

### 1. Graph (TensorFlow Model)
The `graph` is the core TensorFlow model responsible for OCR predictions. It processes input images and produces predictions (`y_pred`).

### 2. TrainerParams
`TrainerParams` is an object that contains training hyperparameters and configuration settings. It manages the training process and ensures consistency across sessions.

### 3. Codec
The `Codec` object handles encoding and decoding between text and numerical labels. It ensures that text data is properly formatted for the model and vice versa.

### 4. SavedCalamariModel
This utility class is responsible for loading and saving Calamari models. It ensures that models can be checkpointed and resumed efficiently.

## Data Flow

### Entry Point
The script starts in the `main()` function, which orchestrates the entire training process.

### Loading Model and Data
- `load_checkpoint()` or `SavedCalamariModel` loads the model (`graph`), training parameters (`params`), and character codec (`codec`).
- `_collect_chars_from_gt()` reads ground truth files to extract the character set.
- `_load_pairs()` loads image and ground truth pairs from the data directory.

### Data Structures
- `graph`: The TensorFlow model that performs OCR predictions.
- `params`: `TrainerParams` object containing training hyperparameters and configuration.
- `codec`: `Codec` object for encoding/decoding between text and numerical labels.
- `charset_from_gt`: Set of characters extracted from ground truth files.
- `all_pairs`: List of tuples containing `(image, gt_text, name)` for training.

### Training Loop
- `train_batch()` is the core function that performs learning/weight updates.
- It processes batches of images and labels, applies preprocessing, and performs forward and backward passes.

### Preprocessing
- `_make_input_dict()` converts raw images into the format expected by the model.
  - Input: `image: np.ndarray`, `target_height: int`, `max_width: int`
  - Output: `dict` with keys `'img'` and `'img_len'`
- `_encode_gt()` encodes ground truth text into numerical labels using the `codec`.
  - Input: `codec: Codec`, `text: str`
  - Output: `encoded_label: np.ndarray`

### Forward Pass
- The `graph` (TensorFlow model) processes the input batch and produces predictions (`y_pred`).
- CTC (Connectionist Temporal Classification) loss is computed using `tf.keras.backend.ctc_batch_cost`.

### Backward Pass
- Gradients are computed using `tf.GradientTape`.
- The `optimizer` (Adam optimizer) updates the model weights using `apply_gradients()`.

### Post-processing
- `_decode()` converts model predictions back into text using the `codec`.
- Metrics like Character Error Rate (CER) are computed using `_cer()`.

### Saving Checkpoints
- `_save_checkpoint()` saves the model weights and training state to disk.
- This ensures that training can be resumed later if interrupted.

## Classes and Data Structures

- **`graph`**: The TensorFlow model that performs OCR predictions.
- **`optimizer`**: The Adam optimizer responsible for updating model weights.
- **`codec`**: Handles encoding/decoding between text and numerical labels.
- **`TrainerParams`**: Manages training hyperparameters and configuration.
- **`SavedCalamariModel`**: Utility for loading and saving Calamari models.
- **`cv2`**: OpenCV library used for image preprocessing (e.g., resizing, grayscale conversion).

## Summary
The Calamari OCR model is structured around a TensorFlow-based graph that processes images and produces text predictions. The training process involves loading data, preprocessing, forward and backward passes, and saving checkpoints for resuming training. Key components include the `graph`, `codec`, `TrainerParams`, and `SavedCalamariModel`.