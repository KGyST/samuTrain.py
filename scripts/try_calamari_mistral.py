"""
Improved Calamari OCR training script:
- Loads checkpoints to continue training (default), or start from scratch (--from-scratch).
- Uses batching for efficiency.
- Collects characters from GT files to validate samples.
- Pads images/labels to handle variable sizes in batches.
- Returns pred/cert after every weight update.
- Saves checkpoints only at the end or on KeyboardInterrupt.
"""

import os
import glob
import json
import random
import argparse
import h5py
import numpy as np
import tensorflow as tf
import cv2

from calamari_ocr.ocr.training.params import TrainerParams
from calamari_ocr.ocr.scenario import CalamariScenario
from calamari_ocr.ocr.dataset.codec import Codec

UTF_8 = 'utf-8'
BIN_PNG = ".bin.png"
DEFAULT_CHECKPOINT_JSON = r"E:\Git\samuTrain.py\models\generic_ocr_model\best.ckpt.json"

def get_cer(sGt: str, sPred: str) -> float:
    if len(sGt) == 0:
        return 0.0 if len(sPred) == 0 else 1.0
    iLenGt: int = len(sGt)
    iLenPred: int = len(sPred)
    dpList: list[int] = list(range(iLenPred + 1))
    for iIdx in range(1, iLenGt + 1):
        iPrev, dpList[0] = dpList[0], iIdx
        for jIdx in range(1, iLenPred + 1):
            iTemp = dpList[jIdx]
            if sGt[iIdx - 1] == sPred[jIdx - 1]:
                dpList[jIdx] = iPrev
            else:
                dpList[jIdx] = 1 + min(iPrev, dpList[jIdx], dpList[jIdx - 1])
            iPrev = iTemp
    return dpList[iLenPred] / iLenGt

def make_input_dict(image: np.ndarray, iTargetHeight: int, iMaxWidth: int) -> dict:
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    h, w = image.shape
    fScale = iTargetHeight / h
    iNewW = max(1, int(w * fScale))
    if iNewW > iMaxWidth:
        iNewW = iMaxWidth
    resized = cv2.resize(image, (iNewW, iTargetHeight))
    img = resized.transpose((1, 0))
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return {"img": img.astype(np.uint8), "img_len": np.array([iNewW], dtype=np.int32)}

def encode_gt(codec, sText: str) -> np.ndarray:
    indices = [codec.charset.index(c) + 1 for c in sText if c in codec.charset]
    return np.array(indices, dtype=np.int32)

def decode_prediction(codec, softmax_np: np.ndarray) -> tuple[str, float]:
    if softmax_np.size == 0:
        return "", 0.0
    best = np.argmax(softmax_np, axis=-1)
    fConf = float(np.mean(np.max(softmax_np, axis=-1)))
    iClasses = softmax_np.shape[-1]
    iBlankIdx = iClasses - 1
    chars, iPrev = [], None
    for iIdx in best:
        if iIdx != iBlankIdx and iIdx != iPrev:
            if 0 <= iIdx < len(codec.charset):
                chars.append(codec.charset[iIdx])
        iPrev = iIdx
    return "".join(chars), fConf

def load_pairs(sDataDir: str, charSet: set[str]) -> list:
    pairs = []
    pngFiles = sorted(glob.glob(os.path.join(sDataDir, "*" + BIN_PNG)))
    if not pngFiles:
        pngFiles = sorted(glob.glob(os.path.join(sDataDir, "*.png")))
    for sPngPath in pngFiles:
        sBase = sPngPath.replace(BIN_PNG, "").replace(".png", "")
        sGtPath = sBase + ".gt.txt"
        if not os.path.exists(sGtPath):
            continue
        img = cv2.imread(sPngPath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        with open(sGtPath, "r", encoding="utf-8") as f:
            sGtText = f.read().strip()
        if not sGtText:
            continue
        if not set(sGtText).issubset(charSet):
            continue
        pairs.append((img, sGtText, os.path.basename(sBase)))
    return pairs

def save_checkpoint(graph, params, codec, sOutputDir: str, iStep: int) -> str:
    sCkptDir = os.path.join(sOutputDir, f"checkpoint_{iStep:04d}")
    os.makedirs(sCkptDir, exist_ok=True)
    sWeightsPath = os.path.join(sCkptDir, "checkpoint.h5")
    with h5py.File(sWeightsPath, 'w') as f:
        weights = graph.get_weights()
        for i, weight in enumerate(weights):
            f.create_dataset(f'weight_{i}', data=weight)
    checkpointData = {
        "model_params": params.scenario.model.to_dict(),
        "trainer_params": params.to_dict(),
        "codec": {"charset": codec.charset},
        "checkpoint_path": "checkpoint.h5",
    }
    with open(os.path.join(sCkptDir, "checkpoint.json"), "w", encoding=UTF_8) as f:
        json.dump(checkpointData, f, indent=2)
    return sCkptDir

def train_batch(graph, optimizer, codec, batchImages: list, batchLabels: list,
                iTargetHeight: int, iMaxWidth: int) -> list:
    xBatchImgs = []
    xBatchLens = []
    labelsList = []
    iMaxBatchWidth = 0
    iMaxLabelLen = 0
    for img, sLabel in zip(batchImages, batchLabels):
        inp = make_input_dict(img, iTargetHeight, iMaxWidth)
        xBatchImgs.append(inp["img"])
        xBatchLens.append(inp["img_len"])
        encodedLabel = encode_gt(codec, sLabel)
        labelsList.append(encodedLabel)
        iMaxBatchWidth = max(iMaxBatchWidth, inp["img"].shape[1])
        iMaxLabelLen = max(iMaxLabelLen, len(encodedLabel))
    paddedImgs = [tf.pad(img, [[0, 0], [0, iMaxBatchWidth - img.shape[1]], [0, 0], [0, 0]]) for img in xBatchImgs]
    paddedLabels = [np.pad(label, (0, iMaxLabelLen - len(label)), constant_values=0) for label in labelsList]
    labelsT = tf.stack(paddedLabels)
    gtLenT = tf.constant([[len(l)] for l in labelsList], dtype=tf.int32)
    xBatch = {"img": tf.concat(paddedImgs, axis=0), "img_len": tf.concat(xBatchLens, axis=0)}
    with tf.GradientTape() as tape:
        outputs = graph.train(xBatch, {})
        yPred = outputs["blank_last_softmax"]
        outLen = outputs["out_len"]
        loss = tf.reduce_mean(tf.keras.backend.ctc_batch_cost(labelsT - 1, yPred, tf.expand_dims(outLen, -1), gtLenT))
    grads = tape.gradient(loss, graph.trainable_variables)
    optimizer.apply_gradients(zip(grads, graph.trainable_variables))
    yPred_np = yPred.numpy()
    results = []
    for i, sLabel in enumerate(batchLabels):
        sPred, fConf = decode_prediction(codec, yPred_np[i])
        results.append((sPred, get_cer(sLabel, sPred), fConf, float(loss.numpy())))
    return results

def collect_chars_from_gt(sDataDir: str) -> set[str]:
    chars = set()
    for f in glob.glob(os.path.join(sDataDir, "*.gt.txt")):
        with open(f, "r", encoding="utf-8") as fh:
            chars.update(fh.read().strip())
    return chars

def run_main(iEpochs: int, sCheckpointPath: str, bFromScratch: bool,
             sDataDir: str, sOutputDir: str, fLr: float | None, iBatchSize: int) -> None:
    if fLr is None:
        fLr = 5e-4 if bFromScratch else 1e-5
    os.makedirs(sOutputDir, exist_ok=True)
    sCheckpoint = sCheckpointPath if sCheckpointPath else DEFAULT_CHECKPOINT_JSON

    with open(sCheckpoint, "r", encoding=UTF_8) as f:
        params_dict = json.load(f)
    params = TrainerParams.from_dict(params_dict)

    if bFromScratch:
        gtChars = sorted(collect_chars_from_gt(sDataDir))
        if not gtChars:
            raise ValueError("No characters found in GT files")
        codec = Codec(gtChars)
        params.scenario.data.codec = codec
        params.scenario.model.classes = len(codec) + 1

    scenario = CalamariScenario(params.scenario)
    things = scenario.create_model_and_graph()
    graph = things[1].submodules[0]

    iTargetHeight = params.scenario.data.line_height
    iMaxInputWidth = 80 * params.scenario.model.compute_downscale_factor().x

    codec = params.scenario.data.codec
    allPairs = load_pairs(sDataDir, set(codec.charset))
    if not allPairs:
        raise ValueError("No training pairs found")
    print(f"Loaded {len(allPairs)} training pairs, charset size: {len(codec.charset)}, classes: {params.scenario.model.classes}")

    dummy_img = np.zeros((1, iMaxInputWidth, iTargetHeight, 1), dtype=np.uint8)
    dummy_len = np.array([iMaxInputWidth], dtype=np.int32)
    graph.predict({"img": dummy_img, "img_len": dummy_len})

    if not bFromScratch:
        sCkptDir = sCheckpoint.replace(".json", "")
        loadedModel = tf.keras.models.load_model(sCkptDir)
        graph.set_weights(loadedModel.get_layer('root').get_weights())
        print("Loaded pre-trained weights")

    optimizer = tf.keras.optimizers.Adam(learning_rate=fLr, clipnorm=1.0)
    for iEpoch in range(iEpochs):
        random.shuffle(allPairs)
        epochLoss = []
        epochCer = []
        for i in range(0, len(allPairs), iBatchSize):
            batchPairs = allPairs[i:i + iBatchSize]
            batchImages = [img for img, _, _ in batchPairs]
            batchLabels = [label for _, label, _ in batchPairs]
            results = train_batch(graph, optimizer, codec, batchImages, batchLabels, iTargetHeight, iMaxInputWidth)
            for (sPred, fCer, fConf, fLoss), (_, sLabel, sName) in zip(results, batchPairs):
                print(f"{sName}: {sPred} - {sLabel} (CER: {fCer:.2f} Conf: {fConf:.2f})")
                epochLoss.append(fLoss)
                epochCer.append(fCer)
        avgLoss = sum(epochLoss) / len(epochLoss)
        avgCer = sum(epochCer) / len(epochCer)
        print(f"[Epoch {iEpoch+1}/{iEpochs}] avg_loss: {avgLoss:.4f} avg_CER: {avgCer:.4f}")
    save_checkpoint(graph, params, codec, sOutputDir, iEpochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--from-scratch", action="store_true", help="Start training from random weights instead of loading a checkpoint")
    parser.add_argument("--data-dir", type=str, default=r"E:\Git\samuTrain.py\data\10_case", help="Directory containing .bin.png and .gt.txt files")
    parser.add_argument("--output-dir", type=str, default=r"E:\Git\samuTrain.py\models\fine_tuned_fixed", help="Directory to save checkpoints")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: 5e-4 for --from-scratch, 1e-5 otherwise)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    args = parser.parse_args()
    run_main(args.epochs, args.checkpoint, args.from_scratch,
             args.data_dir, args.output_dir, args.lr, args.batch_size)
