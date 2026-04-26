# samuTrain

## What is this?
**samuTrain** is a high-level **Training Orchestrator** designed to manage and optimize machine learning workflows. 
While it currently utilizes **Calamari OCR** as its primary "dummy project," the framework is architected to be engine-agnostic. 
It provides a structured layer for process management, hardware resource allocation, and dynamic dataset optimization via database-driven logic.

## How it works: Virtual Set Logic
Unlike legacy systems that physically move files between directories, **samuTrain** uses a built-in SQLite backend (`stockDB` style) to manage data samples virtually.
This ensures data integrity and high-speed batching.

1. **Trainset (The Base)**: Samples tagged with `status='TRAIN'`. These form the core pool for training.
2. **Failset (The Virtual Buffer)**: Samples that the model fails to predict correctly are flagged as `status='FAIL'` in the database.
   - **Logic**: During training, the orchestrator pulls a weighted mix of 'TRAIN' and 'FAIL' samples.
   - **Recovery**: When the model successfully predicts a 'FAIL' sample, its status is automatically reverted to 'TRAIN'.
3. **Testset (The Static Control)**: A protected set of samples (flagged as `status='TEST'`). 
   - **Role**: It is never used for training. It runs exclusively at the end of an epoch to report objective efficiency (e.g., "Accuracy: 82%").

## Strategic Decoupling
A core goal of **samuTrain** is the strict separation of management logic from the ML engine:
- **Engine Agnosticism**: All engine-specific calls (currently Calamari) are isolated in `src/bridge.py`. The orchestrator treats the engine as a replaceable component.
- **Environment Isolation**: Engine dependencies (e.g., TensorFlow 2.15.0) are confined to the `venv_310`, keeping the orchestrator's environment clean.
- **Example Python Runner**: `cd "e:\Git\samuTrain.py"; .\venv_310\Scripts\python.exe .\scripts\try_start_training.py  data/64_case models/generic_ocr_model_3`

## Other docs
- For technical details, see ARCHITECTURE.md
- For roadmap, see ROADMAP.md

