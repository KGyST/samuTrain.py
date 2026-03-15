import os
import sys
import glob
import json
import shutil
import signal
import argparse
import subprocess

# Reduce Tensorflow noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["CALAMARI_LOG_LEVEL"] = "ERROR"

training_process = None


# ------------------------------------------------------------
# DATASET
# ------------------------------------------------------------

def normalize_data_path(data_path: str) -> str:
  if "*" in data_path or data_path.endswith(".bin.png"):
    return data_path

  if os.path.isdir(data_path):
    return os.path.join(data_path, "*.bin.png")

  return data_path


def verify_dataset(data_pattern: str) -> bool:
  images = glob.glob(data_pattern)

  if not images:
    print(f"❌ No images found: {data_pattern}")
    return False

  for img in images:
    gt = img.replace(".bin.png", ".gt.txt")
    if not os.path.exists(gt):
      print(f"❌ Missing GT file: {gt}")
      return False

  return True


# ------------------------------------------------------------
# CHECKPOINT DISCOVERY
# ------------------------------------------------------------

def is_valid_checkpoint_folder(path: str) -> bool:
  return os.path.exists(os.path.join(path, "trainer_params.json"))


def get_last_checkpoint_folder(model_folder: str) -> str:

  checkpoint_root = os.path.join(model_folder, "checkpoint")

  if not os.path.exists(checkpoint_root):
    raise RuntimeError("No checkpoint folder found")

  folders = sorted(glob.glob(os.path.join(checkpoint_root, "checkpoint_*")))

  for f in reversed(folders):
    if is_valid_checkpoint_folder(f):
      return f

  raise RuntimeError("No valid checkpoint found")


# ------------------------------------------------------------
# SIGNAL HANDLING
# ------------------------------------------------------------

def signal_handler(signum, frame):
  global training_process

  print("\n🛑 Stopping training...")

  if training_process:
    training_process.terminate()
    try:
      training_process.wait(timeout=30)
    except subprocess.TimeoutExpired:
      training_process.kill()

  sys.exit(0)


# ------------------------------------------------------------
# INITIAL TRAINING
# ------------------------------------------------------------

def start_initial_training(data_pattern, epochs, output_dir, network):

  global training_process

  os.makedirs(output_dir, exist_ok=True)

  cmd = [
    sys.executable,
    "-m",
    "calamari_ocr.scripts.train",
    "--train.images",
    data_pattern,
    "--trainer.epochs",
    str(epochs),
    "--trainer.output_dir",
    output_dir,
    "--trainer.best_model_prefix",
    "best",
    "--trainer.gen",
    "TrainOnly",
    "--train.batch_size",
    "1",
    "--val.batch_size",
    "1",
    "--codec.auto_compute",
    "True",
  ]

  if network:
    cmd += ["--network", network]

  print("\n🚀 Initial training\n")
  print(" ".join(cmd))
  print()

  training_process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
  )

  for line in training_process.stdout:
    print(line.strip())

  training_process.wait()

  success = training_process.returncode == 0
  training_process = None

  return success


# ------------------------------------------------------------
# CONTINUE TRAINING
# ------------------------------------------------------------

def collect_chars(data_folder):

  chars = set()

  for root, _, files in os.walk(data_folder):
    for f in files:
      if f.endswith(".gt.txt"):
        p = os.path.join(root, f)
        try:
          with open(p, encoding="utf-8") as fh:
            chars.update(fh.read())
        except:
          pass

  return sorted(chars)


def continue_learning(data_folder, checkpoint_folder, network=None):

  global training_process

  model_dir = os.path.abspath(os.path.join(checkpoint_folder, "..", ".."))

  checkpoint = os.path.join(checkpoint_folder, "best.ckpt")

  if not os.path.exists(checkpoint):
    checkpoint = os.path.join(checkpoint_folder, "best.ckpt.json")

  if not os.path.exists(checkpoint):
    raise RuntimeError("Checkpoint not found")

  chars = collect_chars(data_folder)

  charset_file = None

  if chars:
    charset_file = os.path.join(checkpoint_folder, "extended_charset.txt")

    with open(charset_file, "w", encoding="utf-8") as f:
      f.write("".join(chars))

  cmd = [
    sys.executable,
    "-m",
    "calamari_ocr.scripts.train",
    "--warmstart.model",
    checkpoint,
    "--trainer.output_dir",
    model_dir,
    "--train.images",
    os.path.join(data_folder, "*.bin.png"),
    "--train.skip_invalid",
    "True",
    "--train.batch_size",
    "1",
    "--trainer.gen",
    "TrainOnly",
  ]

  if network:
    cmd += ["--network", network]

  if charset_file:
    cmd += [
      "--codec.include_files",
      charset_file,
      "--codec.auto_compute",
      "True",
      "--codec.keep_loaded",
      "True",
    ]

  print("\n🔄 Continue training\n")
  print(" ".join(cmd))
  print()

  training_process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
  )

  for line in training_process.stdout:
    print(line.strip())

  training_process.wait()

  success = training_process.returncode == 0
  training_process = None

  return success


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

  signal.signal(signal.SIGINT, signal_handler)
  signal.signal(signal.SIGTERM, signal_handler)

  parser = argparse.ArgumentParser()

  parser.add_argument("data_folder")
  parser.add_argument("model_folder")

  parser.add_argument("--new", action="store_true")
  parser.add_argument("--epochs", type=int, default=5)
  parser.add_argument("--network")
  parser.add_argument("--force", action="store_true")

  args = parser.parse_args()

  data_pattern = normalize_data_path(args.data_folder)

  if not verify_dataset(data_pattern):
    sys.exit(1)

  model_exists = os.path.exists(args.model_folder)

  if args.new:

    if model_exists:
      if not args.force:
        print("❌ Model folder exists")
        sys.exit(1)

      shutil.rmtree(args.model_folder)

    success = start_initial_training(
      data_pattern,
      args.epochs,
      args.model_folder,
      args.network
    )

    if not success:
      sys.exit(1)

    print("\n✅ Model created")

  else:

    if not model_exists:
      print("❌ Model folder missing")
      sys.exit(1)

    ckpt = get_last_checkpoint_folder(args.model_folder)

    success = continue_learning(
      args.data_folder,
      ckpt,
      args.network
    )

    if not success:
      sys.exit(1)

    print("\n✅ Training continued")


if __name__ == "__main__":
  main()
