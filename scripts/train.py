import os
import subprocess
import argparse
import sys
import glob

# Protocol Buffers compatibility
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

def verify_dataset(data_pattern):
  images = glob.glob(data_pattern)
  if not images:
    print(f"❌ No images found matching: {data_pattern}")
    return False
  
  for img in images:
    gt_file = img.replace(".bin.png", ".gt.txt")
    if not os.path.exists(gt_file):
      print(f"❌ Missing Ground Truth file: {gt_file}")
      return False
  return True

def main():
  parser = argparse.ArgumentParser(description="samuTrain - Calamari 2.3.1 Clean")
  parser.add_argument("--data", default="data/single_case/*.bin.png", help="Path to images")
  parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
  parser.add_argument("--output", default="models/generic_latin", help="Model output directory")
  args = parser.parse_args()

  if not verify_dataset(args.data):
    sys.exit(1)

  if not os.path.exists(args.output):
    os.makedirs(args.output)

  # Direct Calamari 2.3.1 parameters based on paiargparse suggestions
  cmd = [
    sys.executable, "-m", "calamari_ocr.scripts.train",
    "--train.images", args.data,
    "--trainer.epochs", str(args.epochs),
    "--trainer.output_dir", args.output,
    "--trainer.best_model_prefix", "best",
    "--trainer.gen", "TrainOnly",
    "--train.batch_size", "1",
    "--val.batch_size", "1",
    "--codec.auto_compute", "True",
    "--network", "cnn=40:3x3,pool=2x2,lstm=100,dropout=0.5"
  ]
  
  print(f"🚀 Starting training (Calamari 2.3.1 API)...")
  print(f"Command: {' '.join(cmd)}\n")
  
  # No try-except: let it fail loud and clear if there's an issue
  subprocess.run(cmd, check=True)
  
  print(f"\n✅ Training finished successfully. Model in: {args.output}")

if __name__ == "__main__":
  main()