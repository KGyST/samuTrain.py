# samuTrain.py — Agent Guide

## Environment

- **Python**: 3.10.8 in `venv_310/` — always use `.\venv_310\Scripts\python.exe` (not `python`)
- **No project-level build config**: no pyproject.toml, no Makefile, no pytest.ini at root.
- **No CI**: `.github/` does not exist. No pre-commit hooks.
- **Key packages**: tensorflow 2.15.0, calamari_ocr 2.3.1 (editable from `lib/calamari/`), fastapi, opencv, h5py, numpy 1.26.4
- **Database**: SQLite at `samu.db` (36 KB). Run `python scripts/run_server.py --reset-db` to reset.
- **`PYTHONPATH` for server**: `src/` and `lib/calamari/` added via `sys.path.insert` in `run_server.py`

## Architecture

```
samuTrain.py/
├── src/            — FastAPI server, DB, bridge to Calamari
│   ├── main.py     — FastAPI app, routes, training orchestration
│   ├── bridge.py   — OCRBridge: all Calamari calls isolated here
│   ├── db.py       — SQLite (table `cases`: img_path, ocr_text, status, etc.)
│   ├── run_server.py— entrypoint: `.\venv_310\Scripts\python src/run_server.py --data-folder "data/..." --model-folder "models/..."`
│   └── engines/    — LearnerInterface base, CalamariLearner, ContinueLearningEngine
├── scripts/        — standalone training and exploration scripts
├── data/           — .bin.png + .gt.txt pairs (UTF-8). Subdirs: 10_case, 64_case, 1962, single_case, etc.
├── models/         — checkpoints (SavedModel dirs). Default output: models/fine_tuned_fixed/
├── lib/            — vendored submodules: calamari_ocr (editable), samuLearnUI.ts, samuTeszt-python
└── docs/           — Calamari integration, architecture, training guides
```

## Training Script (`scripts/try_calamari_mistral.py`)

The primary fine-tuning/from-scratch training script.

```
.\venv_310\Scripts\python scripts/try_calamari_mistral.py [--from-scratch] [--epochs N] [--lr LR] [--output-dir ...]
```

Defaults:
| Arg | Default | Notes |
|---|---|---|
| `--epochs` | 1 | |
| `--lr` | `5e-4` (from-scratch), `1e-5` (fine-tune) | |
| `--data-dir` | `data/10_case/` | |
| `--output-dir` | `models/fine_tuned_fixed/` | |
| `--checkpoint` | `models/generic_ocr_model/best.ckpt.json` | Use `--from-scratch` for random init |
| `--batch-size` | 8 | |

**Checkpoint load cycle**: saves `checkpoint_NNNN/checkpoint.h5` + parent `checkpoint_NNNN.json`. Reload with `--checkpoint models/.../checkpoint_NNNN.json`.

## Calamari Critical Knowledge (verified by runtime probe)

### Graph access
```python
scenario = CalamariScenario(params.scenario)
things   = scenario.create_model_and_graph()  # returns [tfaip.Model, RootGraph]
graph    = things[1].submodules[0]             # CalamariGraph — the real Keras model
```
`things[0]` (tfaip Model) and `things[1]` (RootGraph) have `trainable_variables=[]` until a forward pass. Only `things[1].submodules[0]` is trainable.

### Calling the graph
```python
graph.predict({"img": ..., "img_len": ...})    # inference
graph.train({"img": ..., "img_len": ...}, {})  # training forward pass
```
Direct `graph(inputs)` raises `ValueError` (tfaip blocks it).

### Input format
```python
{
    "img":     np.ndarray  shape (B, W, H, 1)  dtype np.uint8   # transposed (W = time axis)
    "img_len": np.ndarray  shape (B,)           dtype np.int32   # raw pixel width (not downscaled)
}
```
- **`dtype` must be `uint8`** — the graph does `/ 255.0` internally. Pre-normalizing to float gives double-normalization.
- **`img_len` shape must be `(B,)`**, not `(B,1)`. The graph calls `K.flatten()`.
- **Time axis is W** (width after transpose), not H. LSTM iterates over columns.

### Label encoding
```python
labels = [codec.charset.index(c) + 1 for c in text if c in codec.charset]  # 1-based
```
Calamari uses 1-based indices with blank at the last class (`classes - 1`).

### Greedy decode
```python
iBlankIdx = softmax_np.shape[-1] - 1   # always classes-1
for idx in best:
    if idx != iBlankIdx and idx != prev:
        if 0 <= idx < len(codec.charset):
            chars.append(codec.charset[idx])
    prev = idx
```

### Classes convention
- **Pre-trained checkpoint** (best.ckpt): `classes = len(codec.charset)` (blank = charset[-1])
- **From-scratch**: `classes = len(codec) + 1` (blank is separate class at `classes - 1`)
- Decode always uses `iBlankIdx = softmax_np.shape[-1] - 1` which handles both.

### CTC loss
```python
loss = tf.reduce_mean(
    K.ctc_batch_cost(
        labelsT - 1,                              # 0-based labels
        outputs["blank_last_softmax"],            # softmax probs, NOT logits
        tf.expand_dims(out_len, -1),              # (B, 1)
        tf.constant(gt_len, dtype=tf.int32),      # (B, 1)
    )
)
```
Keras `ctc_batch_cost` applies `log(y + epsilon)` internally. Feed **softmax probabilities**, not logits. Blank is always the last class.

## Server

```bash
.\venv_310\Scripts\python .\scripts\run_server.py --data-folder "data/single_case" --model-folder "models/new_model"
```

Options: `--reset-db`, `--new`, `--port`. The server auto-detects the last training.

## Data Format

- Images: `.bin.png` (grayscale, binary thresholded)
- GT text: `.gt.txt` (UTF-8 encoded)
- Pairs are matched by basename: `foo.bin.png` + `foo.gt.txt`
- DB status lifecycle: `TRAIN` → prediction fail → `FAIL` → prediction pass → back to `TRAIN`

## Data Directories

| Dir | Samples | Notes |
|---|---|---|
| `data/single_case/` | 3 | Minimal test |
| `data/10_case/` | 10 | Quick fine-tuning |
| `data/64_case/` | 37 | Better diversity |
| `data/1962/` | 10 | Historical text |
| `data/new_case/` | 15 | New additions |

## Keras / TensorFlow quirks

- **SavedModel checkpoint**: `models/generic_ocr_model/best.ckpt` is a SavedModel dir, NOT a single `.ckpt` file. Load via `tf.keras.models.load_model(path)`. The `root` layer inside is a `RootGraph` (subclasses `keras.Model`). Its `submodules[0]` is the `CalamariGraph`.
- **Weight loading**: `graph.load_weights()` does not exist on `CalamariGraph` (it's a `Layer`, not a `Model`). Use `graph.set_weights(loaded.get_layer('root').get_weights())` instead, or use our custom `load_checkpoint_weights(graph, json_path)` for h5-format checkpoints.
- **Custom checkpoint format** (written by `try_calamari_mistral.py`): h5py archive with numbered datasets (`weight_0`..`weight_N`) + sibling `<name>.json` metadata. Load path: `models/xxx/checkpoint_NNNN.json`.
- **double-normalization trap**: do NOT pass `float32/255.0` — the graph casts to float32 and divides by 255 internally.

## Existing instruction files to preserve

- `.windsurfrules` — points to `venv_310`, README, ARCHITECTURE
- `.continue/rules/calamari-finetuning.md` — authoritative Calamari 2.3.1 API reference (graph access, input format, CTC loss, label encoding)
- `.continue/rules/calamari-expert.md` — server run/reset commands, troubleshooting

## No tests

No test framework is configured. The project has `lib/calamari/` tests (pytest-based) but no project-level tests. Validation is done manually via `try_` scripts.
