# Calamari OCR Integration with samuTrain

## Overview
This document describes how Calamari OCR is integrated into the samuTrain framework, highlighting the engine-agnostic design and strategic decoupling principles.

## samuTrain Architecture

### What is samuTrain?
**samuTrain** is a high-level **Training Orchestrator** designed to manage and optimize machine learning workflows. While it currently utilizes **Calamari OCR** as its primary "dummy project," the framework is architected to be engine-agnostic.

### Virtual Set Logic
Unlike legacy systems that physically move files between directories, **samuTrain** uses a built-in SQLite backend (`stockDB` style) to manage data samples virtually. This ensures data integrity and high-speed batching.

#### Key Components
1. **Trainset (The Base)**: Samples tagged with `status='TRAIN'`. These form the core pool for training.
2. **Failset (The Virtual Buffer)**: Samples that the model fails to predict correctly are flagged as `status='FAIL'` in the database.
   - **Logic**: During training, the orchestrator pulls a weighted mix of 'TRAIN' and 'FAIL' samples.
   - **Recovery**: When the model successfully predicts a 'FAIL' sample, its status is automatically reverted to 'TRAIN'.
3. **Testset (The Static Control)**: A protected set of samples (flagged as `status='TEST'`).
   - **Role**: It is never used for training. It runs exclusively at the end of an epoch to report objective efficiency (e.g., "Accuracy: 82%").

## Strategic Decoupling

### Engine Agnosticism
A core goal of **samuTrain** is the strict separation of management logic from the ML engine:
- **Engine Agnosticism**: All engine-specific calls (currently Calamari) are isolated in `src/bridge.py`. The orchestrator treats the engine as a replaceable component.
- **Environment Isolation**: Engine dependencies (e.g., TensorFlow 2.15.0) are confined to the `venv_310`, keeping the orchestrator's environment clean.

### Benefits
- **Flexibility**: The framework can easily switch to other ML engines without significant changes to the core logic.
- **Modularity**: Engine-specific code is isolated, making it easier to maintain and update.
- **Scalability**: The framework can scale to accommodate multiple engines or projects simultaneously.

## Integration with Calamari OCR

### How Calamari OCR Fits In
Calamari OCR is integrated into samuTrain as the primary engine for OCR tasks. The integration is designed to be seamless and efficient, leveraging Calamari's capabilities for optical character recognition.

### Key Integration Points
1. **Bridge Module**: The `src/bridge.py` module acts as the interface between samuTrain and Calamari OCR. It handles all engine-specific calls and ensures compatibility.
2. **Training Orchestration**: samuTrain manages the training process, including data loading, preprocessing, and model evaluation. Calamari OCR is responsible for the actual OCR predictions and training.
3. **Data Management**: samuTrain's SQLite backend manages the data samples, ensuring that the correct samples are fed into Calamari OCR for training and evaluation.

### Example Python Runner
To run the training process, use the following command:

```bash
cd "e:\Git\samuTrain.py"
.\[venv_310](venv_310)\Scripts\python.exe .\scripts\try_start_training.py  data/64_case models/generic_ocr_model_3
```

## Engine-Agnostic Principles

### Design Philosophy
- **Modularity**: The framework is designed to be modular, allowing for easy replacement of the ML engine.
- **Abstraction**: Engine-specific details are abstracted away from the core logic, making the framework more flexible and maintainable.
- **Extensibility**: The framework is built to be extensible, allowing for the integration of new engines and features as needed.

### Benefits of Engine Agnosticism
- **Future-Proofing**: The framework can adapt to new ML engines and technologies as they emerge.
- **Maintainability**: Isolating engine-specific code makes it easier to maintain and update the framework.
- **Scalability**: The framework can scale to accommodate multiple engines or projects simultaneously.

## Summary
Calamari OCR is integrated into samuTrain as the primary engine for OCR tasks. The integration leverages samuTrain's engine-agnostic design principles, ensuring flexibility, modularity, and scalability. The `src/bridge.py` module acts as the interface between samuTrain and Calamari OCR, handling all engine-specific calls and ensuring compatibility. This design allows samuTrain to manage the training process efficiently while leveraging Calamari's capabilities for optical character recognition.