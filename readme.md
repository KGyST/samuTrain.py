# samuTrain

## What is this?
**samuTrain** is a high-level **Training Orchestrator** designed to manage and optimize machine learning workflows. 
While it currently utilizes **Calamari OCR** as its primary "dummy project," the framework is architected to be engine-agnostic. 
It provides a structured layer for process management, hardware resource allocation, and dynamic dataset optimization via database-driven logic.

## Strategic Decoupling
A core goal of **samuTrain** is the strict separation of management logic from the ML engine:
- **Engine Agnosticism**: All engine-specific calls (currently Calamari) are isolated in `src/bridge.py`. The orchestrator treats the engine as a replaceable component.
- **Environment Isolation**: Engine dependencies (e.g., TensorFlow 2.15.0) are confined to the `venv_310`, keeping the orchestrator's environment clean.
- **Example Python Runner**: `cd "e:\Git\samuTrain.py"; .\venv_310\Scripts\python.exe .\scripts\try_start_training.py  data/64_case models/generic_ocr_model_3`

## ARCHITECTURE.md
- For technical details, see ARCHITECTURE.md


