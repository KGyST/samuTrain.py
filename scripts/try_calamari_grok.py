import os
import glob
import json
import random
import argparse
import tempfile
import shutil
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from calamari_ocr.ocr.training.params import TrainerParams
from calamari_ocr.ocr.scenario import CalamariScenario
from calamari_ocr.ocr import SavedCalamariModel
from calamari_ocr.ocr.dataset.codec import Codec

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
DATA_DIR = r"E:\Git\samuTrain.py\data\10_case"
OUTPUT_DIR = r"E:\Git\samuTrain.py\models\fine_tuned_lowlevel"
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 4          # effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS
TARGET_HEIGHT = None          # None = use model default

def _cer(gt: str, pred: str) -> float:
    if not gt: return 0.0 if not pred else 1.0
    # simple Levenshtein CER
    m, n = len(gt), len(pred)
    dp = [list(range(n+1))]
    for i in range(1, m+1):
        prev = dp[0][:]
        dp.append([i] + [0]*n)
        for j in range(1, n+1):
            cost = 0 if gt[i-1] == pred[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[m][n] / len(gt)

def _make_input(img: np.ndarray, target_height: int, max_width: int):
    if img.ndim == 3 and img.shape[2] in (3,4):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY if img.shape[2]==3 else cv2.COLOR_BGRA2GRAY)
    h, w = img.shape
    scale = target_height / h
    new_w = min(max(1, int(w * scale)), max_width)
    resized = cv2.resize(img, (new_w, target_height))
    img_tensor = resized.transpose((1, 0)).astype(np.uint8)  # (W, H) -> model expects this internally
    img_tensor = np.expand_dims(img_tensor, axis=-1)         # (W, H, 1)
    img_tensor = np.expand_dims(img_tensor, axis=0)          # (1, W, H, 1)
    return {
        "img": img_tensor,
        "img_len": np.array([new_w], dtype=np.int32)   # IMPORTANT: raw width!
    }

def _encode_gt(codec: Codec, text: str):
    return np.array([codec.char2code[c] + 1 for c in text if c in codec.char2code], dtype=np.int32)  # 1-based

def main(epochs: int = 1, checkpoint: str = None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load checkpoint
    ckpt_json = checkpoint or r"E:\Git\samuTrain.py\models\generic_ocr_model\best.ckpt.json"
    print(f"Loading from {ckpt_json}")

    saved_model = SavedCalamariModel(ckpt_json, auto_update=True)
    params: TrainerParams = saved_model.trainer_params
    scenario = CalamariScenario(params.scenario)

    # Get the real graph
    model_and_graph = scenario.create_model_and_graph()
    root_graph = model_and_graph[1] if len(model_and_graph) > 1 else model_and_graph[0]
    graph = root_graph.graph if hasattr(root_graph, 'graph') else root_graph.submodules[0]  # CalamariGraph

    # Build the model
    dummy = {"img": np.zeros((1, 48, 256, 1), dtype=np.uint8),
             "img_len": np.array([256], dtype=np.int32)}
    graph.predict(dummy)  # force build

    codec: Codec = params.scenario.data.codec
    print(f"Codec size: {len(codec.charset)} chars")

    # Geometry
    line_height = params.scenario.data.line_height
    target_h = TARGET_HEIGHT or line_height
    ds = params.scenario.model.compute_downscale_factor()
    max_w = 80 * ds.x
    print(f"Target height: {target_h}, max width: {max_w}")

    # Load data
    pngs = sorted(glob.glob(os.path.join(DATA_DIR, "*.png")))
    pairs = []
    for p in pngs:
        base = os.path.splitext(p)[0].replace(".bin", "")
        gt_path = base + ".gt.txt"
        if not os.path.exists(gt_path): continue
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        with open(gt_path, encoding="utf-8") as f:
            gt = f.read().strip()
        if img is not None and gt:
            pairs.append((img, gt, os.path.basename(base)))

    print(f"Loaded {len(pairs)} samples")

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    global_step = 0

    for epoch in range(epochs):
        random.shuffle(pairs)
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")

        for i in range(0, len(pairs), BATCH_SIZE):
            batch = pairs[i:i+BATCH_SIZE]
            batch_imgs = [x[0] for x in batch]
            batch_gts = [x[1] for x in batch]
            batch_names = [x[2] for x in batch]

            # Prepare batch
            inputs_list = [_make_input(img, target_h, max_w) for img in batch_imgs]
            gt_list = [_encode_gt(codec, gt) for gt in batch_gts]

            # Stack
            imgs = tf.concat([inp["img"] for inp in inputs_list], axis=0)
            img_lens = tf.concat([inp["img_len"] for inp in inputs_list], axis=0)

            max_gt_len = max(len(g) for g in gt_list)
            gts_padded = tf.pad(tf.stack(gt_list), [[0,0],[0, max_gt_len]], constant_values=0)
            gt_lens = tf.constant([[len(g)] for g in gt_list], dtype=tf.int32)

            x_batch = {"img": imgs, "img_len": img_lens}
            y_batch = {"gt": gts_padded, "gt_len": gt_lens}

            # Gradient accumulation
            accum_grads = None
            for accum_step in range(GRAD_ACCUM_STEPS):
                with tf.GradientTape() as tape:
                    outputs = graph.train(x_batch, y_batch)   # important: use .train()
                    loss_dict = scenario.model.loss(x_batch, y_batch, outputs)
                    loss = tf.reduce_mean(list(loss_dict.values())[0])  # ctc-loss

                grads = tape.gradient(loss, graph.trainable_variables)
                if accum_grads is None:
                    accum_grads = grads
                else:
                    accum_grads = [a + b for a, b in zip(accum_grads, grads)]

            # Apply gradients
            optimizer.apply_gradients(zip(accum_grads, graph.trainable_variables))

            # Predict after update (for logging)
            for j in range(len(batch)):
                single_x = {"img": tf.expand_dims(imgs[j], 0), "img_len": tf.expand_dims(img_lens[j], 0)}
                pred_out = graph.predict(single_x)
                softmax = pred_out["blank_last_softmax"][0].numpy()

                pred_str, conf = _decode(codec, softmax)
                cer = _cer(batch_gts[j], pred_str)

                global_step += 1
                print(f"{global_step:5d} | {batch_names[j]:20s} | GT: {batch_gts[j]:20s} | "
                      f"PRED: {pred_str:20s} | CER: {cer:.4f} | Conf: {conf:.4f} | Loss: {loss.numpy():.5f}")

    # Save final checkpoint
    ckpt_dir = os.path.join(OUTPUT_DIR, f"checkpoint_{global_step:05d}")
    os.makedirs(ckpt_dir, exist_ok=True)
    graph.save_weights(os.path.join(ckpt_dir, "checkpoint"))
    params.output_dir = OUTPUT_DIR
    with open(os.path.join(ckpt_dir, "trainer_params.json"), "w", encoding="utf-8") as f:
        f.write(params.to_json(indent=2))

    print(f"\nTraining finished. Checkpoint saved to {ckpt_dir}")


def _decode(codec: Codec, softmax: np.ndarray):
    best = np.argmax(softmax, axis=-1)
    conf = float(np.mean(np.max(softmax, axis=-1)))
    blank_idx = len(codec.charset) - 1  # blank is last
    chars, prev = [], None
    for idx in best:
        if idx != blank_idx and idx != prev:
            if 0 <= idx < len(codec.charset):
                chars.append(codec.charset[idx])
        prev = idx
    return "".join(chars), conf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    main(args.epochs, args.checkpoint)