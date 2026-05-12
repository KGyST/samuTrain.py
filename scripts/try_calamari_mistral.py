"""
train_single_cases.py
---------------------
Loads a Calamari model from checkpoint, iterates over image/gt pairs in a folder,
does one gradient step per image, prints PRED + CER, and saves a checkpoint after
each image update.

Folder structure expected:
    data/single_case/
        01002c.bin.png
        01002c.gt.txt
        010071.bin.png
        010071.gt.txt
        ...

Checkpoint layout written:
    <output_dir>/
        checkpoint_0001/
            saved_model/           ← Keras SavedModel (loadable by Calamari)
            trainer_params.json    ← updated params JSON
        checkpoint_0002/
            ...
"""

import os
import glob
import json
import tempfile
import shutil

import cv2
import numpy as np
import tensorflow as tf

from tensorflow import keras

from calamari_ocr.ocr.training.params import TrainerParams
from calamari_ocr.ocr.scenario import CalamariScenario
from calamari_ocr.ocr.dataset.codec import Codec
from calamari_ocr.ocr import SavedCalamariModel

# ---------------------------------------------------------------------------
# CONFIG — edit these paths
# ---------------------------------------------------------------------------
CHECKPOINT_JSON: str = r"E:\Git\samuTrain.py\models\generic_ocr_model\best.ckpt.json"
DATA_DIR: str = r"E:\Git\samuTrain.py\data\10_case"
OUTPUT_DIR: str = r"E:\Git\samuTrain.py\models\fine_tuned"
LEARNING_RATE: float = 1e-4  # fallback LR if not readable from checkpoint

# TARGET_HEIGHT and downscale factor are derived from model params at load time.
# You can override TARGET_HEIGHT here (set to None = auto from model line_height).
TARGET_HEIGHT_OVERRIDE = None  # e.g. 48 — or None for auto
MODEL_EXPECTED_FIXED_WIDTH = 48 # From error message: expected shape=(None, None, 48, 1)
# ---------------------------------------------------------------------------


def assert_valid_path(path: str) -> None:
    """Helper function to validate file or directory paths."""
    assert os.path.exists(path), f"Path does not exist: {path}"
    assert os.path.isabs(path), f"Path must be absolute: {path}"


def assert_valid_image(image: np.ndarray) -> None:
    """Helper function to validate image data."""
    assert image is not None, "Image data is None"
    assert image.ndim in (2, 3), f"Invalid image dimensions: {image.ndim}"


# ---------------------------------------------------------------------------
# CER helper
# ---------------------------------------------------------------------------
def _cer(gt: str, pred: str) -> float:
    """Character Error Rate via Levenshtein distance."""
    assert isinstance(gt, str), "Ground truth text must be a string"
    assert isinstance(pred, str), "Predicted text must be a string"

    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    # dynamic programming
    m, n = len(gt), len(pred)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            if gt[i - 1] == pred[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n] / len(gt)


# ---------------------------------------------------------------------------
# Image → model input dict
# ---------------------------------------------------------------------------
def _make_input_dict(image: np.ndarray, target_height: int, max_width: int) -> dict:
    """Convert image to model input dictionary."""
    assert_valid_image(image)
    assert isinstance(target_height, int) and target_height > 0, "Invalid target height"
    assert isinstance(max_width, int) and max_width > 0, "Invalid max width"

    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    h, w = image.shape
    scale = target_height / h
    new_w = max(1, int(w * scale))
    # Cap width so CTC output sequence length stays within model's trained range.
    if new_w > max_width:
        new_w = max_width
    resized = cv2.resize(image, (new_w, target_height))

    # Normalize and reshape to match the expected input shape (1, None, 48, 1)
    # Transpose from (height, width) to (width, height) to match model's expected input
    img = resized.transpose((1, 0)).astype(np.float32) / 255.0  # Transpose and Normalize to [0, 1]
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Ensure the shape matches the model's expected input shape
    # The assertions here should reflect the transposed shape (batch, width, height, channels)
    assert img.shape[2] == target_height, f"Image height mismatch after transpose: {img.shape[2]} != {target_height}"
    assert img.shape[3] == 1, f"Image channel mismatch: {img.shape[3]} != 1"

    print(f"Final image shape: {img.shape}")  # Debug print

    return {
        "img": img,
        "img_len": np.array([[img.shape[1]]], dtype=np.int32), # img_len should be the width here (img.shape[1])
    }


# ---------------------------------------------------------------------------
# GT text → label indices
# ---------------------------------------------------------------------------
def _encode_gt(codec: Codec, text: str) -> np.ndarray:
    """Encode ground truth text into label indices."""
    assert isinstance(text, str), "Ground truth text must be a string"
    # charset[0] == "" (blank), so real chars start at index 1
    indices = [codec.charset.index(c) for c in text if c in codec.charset]
    return np.array(indices, dtype=np.int32)


# ---------------------------------------------------------------------------
# Greedy CTC decode of softmax output  (blank = last column)
# ---------------------------------------------------------------------------
def _decode(codec: Codec, softmax_np: np.ndarray) -> tuple:
    """Returns (predicted_text, avg_confidence)."""
    assert isinstance(softmax_np, np.ndarray), "Softmax output must be a numpy array"
    best = np.argmax(softmax_np, axis=-1)  # [T]
    conf = float(np.mean(np.max(softmax_np, axis=-1)))
    blank_idx = softmax_np.shape[-1] - 1
    chars, prev = [], None
    for idx in best:
        if idx != blank_idx and idx != prev:
            if 0 < idx < len(codec.charset):  # 0 is blank in charset too
                chars.append(codec.charset[idx])
        prev = idx
    return "".join(chars), conf


# ---------------------------------------------------------------------------
# Single gradient step — one image, one GT string
# ---------------------------------------------------------------------------
def _gradient_step(graph, optimizer, codec: Codec, image: np.ndarray, gt_text: str,
                   target_height: int, max_width: int) -> tuple:
    """
    Forward pass → CTC loss → backprop → apply gradients.
    Returns (pred_text, confidence, loss_value).
    """
    assert_valid_image(image)
    assert isinstance(gt_text, str) and len(gt_text) > 0, "Ground truth text must be a non-empty string"

    inp = _make_input_dict(image, target_height, max_width)
    labels = _encode_gt(codec, gt_text)

    if len(labels) == 0:
        raise ValueError(f"GT text '{gt_text}' produced zero labels — chars not in codec?")

    # batch of 1
    x_batch = {
        "img": tf.constant(inp["img"], dtype=tf.float32),
        "img_len": tf.constant(inp["img_len"], dtype=tf.int32),
    }
    labels_t = tf.constant(labels[np.newaxis, :], dtype=tf.int32)  # [1, L]
    gt_len_t = tf.constant([[len(labels)]], dtype=tf.int32)  # [1, 1]

    print(f"Input shape to model: {x_batch['img'].shape}")  # Debug print

    with tf.GradientTape() as tape:
        # The loaded model is the prediction (serve) graph — a plain Functional keras
        # model that takes only inputs (no targets). Call it directly.
        outputs = graph(x_batch, training=True)
        y_pred = outputs["blank_last_softmax"]  # [1, T, C], blank is last
        out_len = outputs["out_len"]  # [1]

        seq_len = int(out_len.numpy()[0])
        if seq_len < len(labels):
            raise ValueError(
                f"CTC output seq_len={seq_len} < gt_len={len(labels)} for '{gt_text}'. "
                f"Image width {inp['img'].shape[2]}px is too narrow after downscaling. "
                f"Increase MAX_INPUT_WIDTH or use shorter GT text."
            )

        # ctc_batch_cost expects labels 0-based (no blank), blank-last softmax
        # our labels are already 1-based (blank=0 in charset), subtract 1 here:
        loss = tf.reduce_mean(
            tf.keras.backend.ctc_batch_cost(
                labels_t - 1,  # 0-based GT
                y_pred,
                tf.cast(tf.expand_dims(out_len, -1), tf.int32),  # output lengths
                gt_len_t,  # GT lengths
            )
        )

    grads = tape.gradient(loss, graph.trainable_variables)
    optimizer.apply_gradients(
        [(g, v) for g, v in zip(grads, graph.trainable_variables) if g is not None]
    )

    pred_text, conf = _decode(codec, y_pred.numpy()[0])
    return pred_text, conf, float(loss.numpy())


# ---------------------------------------------------------------------------
# Save checkpoint (Keras SavedModel + trainer_params.json)
# ---------------------------------------------------------------------------
def _save_checkpoint(graph, params: TrainerParams, codec: Codec,
                     output_dir: str, step: int) -> str:
    """Save checkpoint with weights and parameters."""
    assert_valid_path(output_dir)
    assert isinstance(step, int) and step > 0, "Step must be a positive integer"

    ckpt_name = f"checkpoint_{step:04d}"
    ckpt_dir = os.path.join(output_dir, ckpt_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save weights as .h5 — this is the format Calamari's SavedCalamariModel expects.
    # The file is named without extension here; Calamari appends .h5 itself when loading.
    weights_path = os.path.join(ckpt_dir, "checkpoint")
    graph.save_weights(weights_path + ".h5")

    # Write trainer_params.json so this dir is a valid Calamari checkpoint.
    params_copy = TrainerParams.from_dict(params.to_dict())
    params_copy.output_dir = output_dir
    params_copy.current_epoch = step
    params_copy.scenario.data.codec.charset = codec.charset

    params_path = os.path.join(ckpt_dir, "trainer_params.json")
    with open(params_path, "w", encoding="utf-8") as f:
        f.write(params_copy.to_json(indent=2))

    return ckpt_dir


# ---------------------------------------------------------------------------
# Load image/gt pairs from folder
# ---------------------------------------------------------------------------
def _load_pairs(data_dir: str):
    """
    Yields (image_np, gt_text, base_name) for every .bin.png / .gt.txt pair.
    """
    assert_valid_path(data_dir)

    png_files = sorted(glob.glob(os.path.join(data_dir, "*.bin.png")))
    if not png_files:
        # also accept plain .png
        png_files = sorted(glob.glob(os.path.join(data_dir, "*.png")))

    for png_path in png_files:
        base = png_path.replace(".bin.png", "").replace(".png", "")
        gt_path = base + ".gt.txt"
        if not os.path.exists(gt_path):
            print(f"  [skip] no GT file for {os.path.basename(png_path)}")
            continue

        img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  [skip] could not read image {png_path}")
            continue

        with open(gt_path, "r", encoding="utf-8") as f:
            gt_text = f.read().strip()

        if not gt_text:
            print(f"  [skip] empty GT for {os.path.basename(png_path)}")
            continue

        yield img, gt_text, os.path.basename(base)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Main function to execute the training process."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load model ---
    print(f"Loading model from: {CHECKPOINT_JSON}")
    with open(CHECKPOINT_JSON, "r") as f:
        ckpt_dict = json.load(f)

    params = TrainerParams.from_dict(ckpt_dict)
    scenario = CalamariScenario(params.scenario)

    temp_dir = tempfile.mkdtemp()
    params.output_dir = temp_dir

    # --- Load the actual checkpoint weights via Keras ---
    # SavedCalamariModel resolves the .json → .h5 path and handles version upgrades.
    # keras.models.load_model with custom_objects loads the RootGraph with real weights.
    # (scenario.graph only creates a fresh uninitialized graph — weights would be random.)
    ckpt = SavedCalamariModel(CHECKPOINT_JSON, auto_update=True)
    graph = keras.models.load_model(
        ckpt.ckpt_path,
        custom_objects=CalamariScenario.model_cls().all_custom_objects(),
    )
    print(f"Weights loaded from: {ckpt.ckpt_path}")

    # --- Derive geometry from model params (do NOT hardcode these) ---
    model_params = params.scenario.model
    ds = model_params.compute_downscale_factor()  # IntVec2D(x=?, y=?)
    downscale_x = ds.x  # width  downscale factor (e.g. 4 for two pool2x2)
    downscale_y = ds.y  # height downscale factor

    # Line height: use the value stored in the checkpoint data params
    line_height = params.scenario.data.line_height
    target_height = TARGET_HEIGHT_OVERRIDE if TARGET_HEIGHT_OVERRIDE else line_height

    # Maximum safe input width: enough output time-steps for reasonable words.
    # We allow up to 80 output time-steps (covers ~40-char words with CTC gaps).
    max_input_width = 80 * downscale_x

    print(f"Model geometry: line_height={line_height}px, "
          f"downscale x={downscale_x} y={downscale_y}, "
          f"max_input_width={max_input_width}px")
    print(f"Using target_height={target_height}px")

    # The authoritative charset lives in scenario.data.codec (set during training).
    # params.codec.resolved_include_chars is a whitelist subset — do NOT use it.
    raw_charset = params.scenario.data.codec.charset
    if not raw_charset:
        raise RuntimeError(
            "Codec charset is empty in params.scenario.data.codec.charset.\n"
            "Open best.ckpt.json and check the path: scenario -> data -> codec -> charset"
        )
    codec = Codec(raw_charset)
    print(f"Codec charset ({codec.size()} chars): {codec.charset}")

    # Reuse Adam from checkpoint params (same hyper-parameters as training)
    lr_value = LEARNING_RATE
    try:
        lr_value = params.learning_rate.lr
        print(f"  Using LR from checkpoint: {lr_value}")
    except Exception:
        print(f"  Could not read LR from params, using default: {lr_value}")

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_value)

    print(f"Codec size: {codec.size()} chars")
    print(f"Output dir: {OUTPUT_DIR}")
    print()

    # --- Iterate over image/gt pairs ---
    step = 0
    pairs = list(_load_pairs(DATA_DIR))
    if not pairs:
        print(f"No image/gt pairs found in {DATA_DIR}")
        return

    print(f"Found {len(pairs)} image/gt pair(s).\n")
    print(f"{'Step':>4}  {'File':<20}  {'GT':<12}  {'PRED':<12}  {'CER':>6}  {'Conf':>6}  {'Loss':>8}")
    print("-" * 76)

    for img, gt_text, name in pairs:
        step += 1

        # Check all GT chars are in codec
        missing = [c for c in gt_text if c not in codec.charset]
        if missing:
            print(f"  [skip] '{name}' — chars not in codec: {missing}")
            continue

        try:
            pred, conf, loss = _gradient_step(graph, optimizer, codec, img, gt_text,
                                              target_height, max_input_width)
        except Exception as e:
            print(f"  [error] '{name}': {e}")
            continue

        cer = _cer(gt_text, pred)

        print(f"{step:>4}  {name:<20}  {gt_text:<12}  {pred:<12}  {cer:>6.3f}  {conf:>6.3f}  {loss:>8.4f}")

        # Save checkpoint after every weight update
        ckpt_dir = _save_checkpoint(graph, params, codec, OUTPUT_DIR, step)
        print(f"       → checkpoint: {ckpt_dir}")
        print()

    print("Done.")
    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()