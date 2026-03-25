# What is this file for: Constants for samuTrain scripts
# When it is created: 2026-03-25

# Training defaults
DEFAULT_EPOCHS = 5
DEFAULT_NETWORK = "cnn=8:3x3,pool=2x2,lstm=32"
DEFAULT_BATCH_SIZE = 1
DEFAULT_PROCESSES = 1

# File names
TRAINER_PARAMS = "trainer_params.json"
BEST_CKPT = "best.ckpt.json"
BEST_CKPT_DIR = "best.ckpt"
CHECKPOINT_DIR = "checkpoint"

# File extensions
IMAGE_EXT = ".bin.png"
GROUND_TRUTH_EXT = ".gt.txt"

# Encoding
UTF_8 = 'utf-8'

# Environment settings
ENV_TF_LOG_LEVEL = "3"
ENV_PYTHON_WARNINGS = "ignore"
ENV_CALAMARI_LOG_LEVEL = "ERROR"
ENV_PROTOBUF_IMPL = "python"

# Backup settings
BACKUP_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
BACKUP_SUFFIX = ".old"

# Model extension suffix
MODEL_EXT_SUFFIX = "_extended"

# Progress tracking
PROGRESS_STEPS = [
  "params_created",
  "training_started", 
  "training_completed",
  "training_failed",
  "error"
]

# Interruption and shutdown
SHUTDOWN_TIMEOUT = 30.0  # seconds
FORCE_TERMINATE_TIMEOUT = 5.0  # seconds
CHECK_INTERRUPT_INTERVAL = 1.0  # seconds

# Error messages
ERROR_MESSAGES = {
  "data_not_found": "Data folder not found",
  "model_not_found": "Model folder not found", 
  "invalid_dataset": "Invalid dataset structure",
  "missing_checkpoint": "Missing checkpoint files",
  "training_failed": "Training operation failed",
  "training_interrupted": "Training was interrupted",
  "force_terminate_failed": "Failed to force terminate training"
}

# Success messages
SUCCESS_MESSAGES = {
  "training_started": "Training started successfully",
  "training_completed": "Training completed successfully",
  "model_created": "New model created successfully",
  "model_continued": "Model continuation completed successfully",
  "training_interrupted": "Training interrupted gracefully"
}
