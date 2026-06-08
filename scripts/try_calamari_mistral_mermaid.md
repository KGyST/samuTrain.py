# Mermaid Diagram for `scripts/try_calamari_mistral.py`

```mermaid
graph TD
    A[scripts/try_calamari_mistral.py] --"num_epochs: int, checkpoint_path: str"--> B[main]
    B --"checkpoint_path: str"--> C[load_checkpoint]
    B --"checkpoint_json: str"--> C2[SavedCalamariModel]
    B --"data_dir: str"--> D[_collect_chars_from_gt]
    B --"data_dir: str, charset: set[str]"--> E[_load_pairs]

    C --"checkpoint_dir: str"--> F[graph: TensorFlow Model]
    C --"checkpoint_dir: str"--> G[params: TrainerParams]
    C --"checkpoint_dir: str"--> H[("codec: Codec")]
    C2 --"checkpoint_json: str"--> F
    C2 --"checkpoint_json: str"--> G
    C2 --"checkpoint_json: str"--> H

    D --"data_dir: str"--> I["charset_from_gt: set[str]"]
    E --"data_dir: str, charset: set[str]"--> J["all_pairs: List[Tuple[np.ndarray, str, str]]"]

    B --"graph: TensorFlow Model"--> K[train_batch]
    B --"optimizer: tf.keras.optimizers.Adam"--> K
    H[("codec: Codec")] ----> K
    B --"batch_images: List[np.ndarray], batch_labels: List[str], target_height: int, max_width: int"--> K
    K --"batch_images: List[np.ndarray], batch_labels: List[str], target_height: int, max_width: int"--> L[preprocess_batch]
    L --"img: np.ndarray, target_height: int, max_width: int"--> M[_make_input_dict]
    M --"image: np.ndarray, target_height: int, max_width: int"--> O["dict: {'img': np.ndarray, 'img_len': np.ndarray}"]

    L --"codec: Codec, label: str"--> P[_encode_gt]
    P --"codec: Codec, text: str"--> R[("encoded_label: np.ndarray")]

    K --"x_batch: dict"--> S[forward_pass]
    S --"x_batch: dict"--> T[graph: TensorFlow Model]
    T --"x_batch: dict"--> U["outputs: {'blank_last_softmax': np.ndarray, 'out_len': np.ndarray}"]
    S --"labels_t: tf.Tensor, y_pred: tf.Tensor, out_len: tf.Tensor, gt_len_t: tf.Tensor"--> V[tf.keras.backend.ctc_batch_cost]
    V --"labels_t: tf.Tensor, y_pred: tf.Tensor, out_len: tf.Tensor, gt_len_t: tf.Tensor"--> W[loss: float]

    K --"loss: float, graph.trainable_variables: List[tf.Variable]"--> X[backward_pass]
    X --"loss: float"--> Y[tf.GradientTape]
    Y --"grads: List[tf.Tensor]"--> Z[optimizer.apply_gradients]

    K --"y_pred: np.ndarray"--> AB[_decode]
    H[("codec: Codec")] ----> AB
    AB --"y_pred: np.ndarray, codec: Codec"--> AC[("pred: str, conf: float")]

    B --"graph: TensorFlow Model"--> AD[save_final_checkpoint]
    G ----> AD
    H[("codec: Codec")] ----> AD
    B --"output_dir: str, step: int"--> AD
    AD --"graph: TensorFlow Model, params: TrainerParams, codec: Codec, output_dir: str, step: int"--> AE[_save_checkpoint]
    AE --"graph: TensorFlow Model, weights_path: str"--> AF[graph.save_weights]

    style A fill:#f9f,stroke:#333
    style K fill:#bbf,stroke:#333
    style C fill:#f96,stroke:#333
    style C2 fill:#f96,stroke:#333
```

## Explanation

### Key Functions and Data Flow

1. **Entry Point**: The script starts in `main()`, which orchestrates the entire training process.

2. **Loading Model and Data**:
   - `load_checkpoint()` or `SavedCalamariModel` loads the model (`graph`), training parameters (`params`), and character codec (`codec`).
   - `_collect_chars_from_gt()` reads ground truth files to extract the character set.
   - `_load_pairs()` loads image and ground truth pairs from the data directory.

3. **Data Structures**:
    - `graph`: The TensorFlow model that performs OCR predictions.
    - `params`: `TrainerParams` object containing training hyperparameters and configuration.
    - `codec`: `Codec` object for encoding/decoding between text and numerical labels, created via `SavedCalamariModel` or `Codec(raw_charset)`.
    - `charset_from_gt`: Set of characters extracted from ground truth files.
    - `all_pairs`: List of tuples containing `(image, gt_text, name)` for training.

4. **Training Loop**:
   - `train_batch()` is the core function that performs the learning/weight updating.
   - It processes batches of images and labels, applies preprocessing, and performs forward and backward passes.

5. **Preprocessing**:
   - `_make_input_dict()` converts raw images into the format expected by the model.
     - Input: `image: np.ndarray`, `target_height: int`, `max_width: int`
     - Output: `dict` with keys `'img'` and `'img_len'`
   - `_encode_gt()` encodes ground truth text into numerical labels using the `codec`.
     - Input: `codec: Codec`, `text: str`
     - Output: `encoded_label: np.ndarray`

6. **Forward Pass**:
   - The `graph` (TensorFlow model) processes the input batch and produces predictions (`y_pred`).
   - CTC (Connectionist Temporal Classification) loss is computed using `tf.keras.backend.ctc_batch_cost`.

7. **Backward Pass**:
   - Gradients are computed using `tf.GradientTape`.
   - The `optimizer` (Adam optimizer) updates the model weights using `apply_gradients()`.

8. **Post-processing**:
   - `_decode()` converts model predictions back into text using the `codec`.
   - Metrics like Character Error Rate (CER) are computed using `_cer()`.

9. **Saving Checkpoints**:
   - `_save_checkpoint()` saves the model weights and training state to disk.
   - This ensures that training can be resumed later if interrupted.

### Classes and Data Structures

- **`graph`**: The TensorFlow model that performs OCR predictions.
- **`optimizer`**: The Adam optimizer responsible for updating model weights.
- **`codec`**: Handles encoding/decoding between text and numerical labels.
- **`TrainerParams`**: Manages training hyperparameters and configuration.
- **`SavedCalamariModel`**: Utility for loading and saving Calamari models.
- **`cv2`**: OpenCV library used for image preprocessing (e.g., resizing, grayscale conversion).

### Summary

The script `scripts/try_calamari_mistral.py` orchestrates the training process by:
1. Loading the model and data.
2. Preprocessing images and labels.
3. Performing forward and backward passes to update weights.
4. Saving checkpoints for resuming training.

The core learning happens in `train_batch()`, where the model processes the data, computes loss, and updates weights using gradient descent.