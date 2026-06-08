---
name: calamari-finetuning
description: Calamari 2.3.1 Fine-Tuning
modeSlugs:
  - architect
  - code
  - ask
  - debug
  - orchestrator
  - refactor
---

# SKILL: 

Knowledge extracted from live debugging of `calamari_ocr 2.3.1` + `tfaip`.
Use this before writing any code that loads, runs, or trains a Calamari model.

## Package versions this applies to

| Package | Version |
|---|---|
| calamari_ocr | 2.3.1 (editable install) |
| tfaip | whatever ships with calamari 2.3.1 |
| tensorflow / keras | 2.x (TF1-compat warnings present) |
| Python | 3.10 |

---

## Object hierarchy (confirmed by runtime probe)

```
CalamariScenario.create_model_and_graph()
    -> list of length 2

    [0]  calamari_ocr.ocr.model.model.Model
         tfaip ModelBase subclass
         NOT a Keras model — has no load_weights, no trainable_variables
         has: .cer_metric, .wrap_model_with_loss_and_metric, .root_graph_cls, etc.

    [1]  tfaip.model.graphbase.RootGraph
         IS a Keras layer (subclasses tf.keras.Model)
         but trainable_variables = [] until after first forward pass
         has: .model  -> points back to [0] (the tfaip wrapper, NOT useful)
              .submodules[0]  -> CalamariGraph  ✓ THIS IS THE REAL MODEL

    [1].submodules[0]  calamari_ocr.ocr.model.graph.CalamariGraph
         tfaip GraphBase subclass, itself a Keras Model
         has load_weights, trainable_variables (after build)
         THIS is what you train
```

**Common mistake:** using `things[0]`, `things[1]`, or `things[1].model` as the Keras model.
All three give `AttributeError: 'Model' object has no attribute 'load_weights'`.

---

## Loading a checkpoint

```python
from calamari_ocr.ocr.training.params import TrainerParams
from calamari_ocr.ocr.scenario import CalamariScenario
import json, numpy as np

with open("path/to/best.ckpt.json") as f:
    params = TrainerParams.from_dict(json.load(f))

scenario = CalamariScenario(params.scenario)
things   = scenario.create_model_and_graph()
graph    = things[1].submodules[0]          # CalamariGraph — the real Keras model
codec    = params.scenario.data.codec       # Codec with .charset list

# MUST build before load_weights — variables don't exist until a forward pass
dummy_img = np.zeros((1, 48, 256, 1), dtype=np.uint8)
dummy_len = np.array([256], dtype=np.int32)             # shape (B,) NOT (B,1)
graph.predict({"img": dummy_img, "img_len": dummy_len}) # builds the graph

# Now load weights
graph.load_weights("path/to/best.ckpt").expect_partial()
# .expect_partial() silences warnings about optimizer slots not restored
```

---

## Calling the graph — tfaip rules

`CalamariGraph` is a `GraphBase` subclass. tfaip **intercepts and blocks** direct calls:

```python
graph(inputs)          # BLOCKED — raises ValueError
graph(inputs, targets) # BLOCKED
```

Allowed entry points:

```python
graph.predict(inputs)          # inference — wraps inputs in {"predict": inputs}
graph.train(inputs, targets)   # training forward pass
```

Both ultimately call `build_graph(inputs, training)` internally.

---

## Input dict format

```python
inputs = {
    "img":     np.ndarray  shape (B, H, W, 1)  dtype uint8
    "img_len": np.ndarray  shape (B,)           dtype int32   # RAW pixel width
}
```

**Critical details:**

- `dtype` must be `uint8` — `graph.py` does `tf.cast(inputs["img"], float32) / 255.0` internally. Do NOT pre-normalise to float [0,1] — this would double-normalise.
- `img_len` shape must be `(B,)` not `(B,1)` — `graph.py` calls `K.flatten(inputs["img_len"])`. A 2D shape causes incorrect sequence lengths.
- `img_len` is the **raw pixel width**. `CalamariGraph.build_graph()` downscales it internally via `params.compute_downscaled(shape)`. Do not pre-downscale.
- Images are grayscale single-channel. Convert BGR→gray with cv2 before adding dims.

```python
# Correct image prep
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)        # (H, W)
img = img[np.newaxis, :, :, np.newaxis]             # (1, H, W, 1)  uint8
img_len = np.array([img.shape[2]], dtype=np.int32)  # (1,)  raw width
inputs = {"img": img, "img_len": img_len}
```

---

## Target dict format (for training)

```python
targets = {
    "gt":     int32 tensor  shape (B, max_label_len)   # 1-based label indices
    "gt_len": int32 tensor  shape (B, 1)               # actual label length per sample
}
```

**Label encoding:** Calamari uses **1-based** codec indices. Blank = last class index (0-based from the softmax end). To encode:

```python
labels = [codec.charset.index(c) + 1 for c in text if c in codec.charset]
# charset.index() is 0-based; +1 makes it 1-based as Calamari expects
```

**CTC loss** subtracts 1 back to 0-based internally:
```python
# from model.py _loss():
K.ctc_batch_cost(targets["gt"] - 1, outputs["blank_last_softmax"], ...)
```

---

## Graph outputs

`build_graph()` returns a dict with these keys (from `graph.py`):

```python
{
    "blank_last_softmax":  (B, T, C)   softmax with blank as LAST class
    "out_len":             (B,)        sequence length after downscaling
    "blank_last_logits":   (B, T, C)   logits version of above
    "logits":              (B, T, C)   blank rolled to FIRST position (for ctc_ops)
    "softmax":             (B, T, C)   softmax of rolled logits
    "decoded":             (B, T)      greedy decoded, 1-based, -1 padded → +1 applied
}
```

For training use `"blank_last_softmax"` and `"out_len"`.

---

## Manual CTC loss (for GradientTape training)

```python
with tf.GradientTape() as tape:
    outputs = graph.train(x_batch, targets)
    y_pred  = outputs["blank_last_softmax"]   # (B, T, C)
    out_len = outputs["out_len"]              # (B,)

    loss = tf.reduce_mean(
        tf.keras.backend.ctc_batch_cost(
            labels_padded - 1,                          # 0-based for ctc
            y_pred,
            tf.cast(tf.expand_dims(out_len, -1), tf.int32),  # (B,1)
            tf.constant(gt_len, dtype=tf.int32),        # (B,1)
        )
    )

grads = tape.gradient(loss, graph.trainable_variables)
optimizer.apply_gradients(zip(grads, graph.trainable_variables))
```

**Do not use `graph.model.optimizer`** — it may not exist on `CalamariGraph`.
Create your own: `optimizer = tf.keras.optimizers.Adam(lr=1e-4)`

---

## Greedy decode

```python
def decode(codec, softmax_np):          # softmax_np: (T, C) for one sample
    best_path  = np.argmax(softmax_np, axis=-1)
    confidence = float(np.mean(np.max(softmax_np, axis=-1)))
    blank_idx  = softmax_np.shape[-1] - 1   # blank is LAST
    chars, prev = [], None
    for idx in best_path:
        if idx != blank_idx and idx != prev:
            if 0 <= idx < len(codec.charset):
                chars.append(codec.charset[idx])
        prev = idx
    return "".join(chars), confidence
# codec.charset is 0-based; output indices from softmax argmax are also 0-based
```

---

## Codec

```python
codec = params.scenario.data.codec
codec.charset          # list of characters, 0-indexed
len(codec.charset)     # number of classes excluding blank
                       # total softmax classes = len(charset) + 1 (blank last)

# Encode text to 1-based labels
[codec.charset.index(c) + 1 for c in text if c in codec.charset]

# Chars not in charset are silently skipped — check for empty result
```

---

## Batching variable-size inputs

Images in a batch may have different H and W — pad to max:

```python
max_h = max(d["img"].shape[1] for d in batch)
max_w = max(d["img"].shape[2] for d in batch)
padded = np.pad(img, ((0,0),(0,max_h-h),(0,max_w-w),(0,0)), mode="constant")
# img_len holds ORIGINAL widths — do not update after padding
```

Labels may have different lengths — pad to max_len, track actual lengths:

```python
padded  = np.zeros((B, max_len), dtype=np.int32)
gt_len  = np.zeros((B, 1),       dtype=np.int32)
for i, lbl in enumerate(labels):
    padded[i, :len(lbl)] = lbl
    gt_len[i, 0] = len(lbl)
```

---

## Saving weights

```python
graph.save_weights("path/to/best.ckpt")   # omit .json suffix
# To overwrite the original checkpoint:
weights_path = checkpoint_path.replace(".json", "")
graph.save_weights(weights_path)
```

---

## Architecture params

```python
params.scenario.model          # ModelParams
params.scenario.model.layers   # list of LayerParams (Conv2D, MaxPool2D, BiLSTM, Dropout)
params.scenario.model.classes  # int — codec size + 1 (including blank)
params.scenario.model.compute_downscaled(width)   # int — width after all pooling
params.scenario.model.compute_downscale_factor()  # IntVec2D(x, y) — total stride
```

Default architecture (from `params.py`):
```
Conv2D(filters=40) → MaxPool2D → Conv2D(filters=60) → MaxPool2D → BiLSTM → Dropout(0.5)
```
Two MaxPool2D layers = width downscaled by 4x, height downscaled by 4x.

---

## What does NOT work

| Attempt | Error | Reason |
|---|---|---|
| `things[0].load_weights(...)` | `AttributeError: no load_weights` | things[0] is tfaip Model wrapper |
| `things[1].load_weights(...)` | Same | things[1].trainable_variables is [] before build |
| `things[1].model.load_weights(...)` | Same | .model points back to things[0] |
| `graph(inputs)` | `ValueError: A Graph must not be called directly` | tfaip blocks __call__ |
| `img_len shape (B,1)` | Wrong sequence length / CTC error | Must be (B,) — graph calls K.flatten |
| `img` as float32 [0,1] | Silent double-normalisation | Graph does /255 internally |
| Load weights before build | Weights load into nothing | Variables don't exist pre-build |
| Use `graph.model.optimizer` | May not exist | Create Adam separately |
