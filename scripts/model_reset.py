import os
import sys
from calamari_ocr.ocr.training.trainer import Trainer
from calamari_ocr.ocr.scenario import CalamariScenario

# 2-space indentation, no Hungarian comments
def migrate():
  model_dir = "models/new_model"
  checkpoint_path = os.path.join(model_dir, "best.ckpt.json")
  data_dir = "data/single_case"

  if not os.path.exists(checkpoint_path):
    print(f"❌ Model not found at {checkpoint_path}")
    return

  print("🚀 Starting model migration to v5...")
  try:
    scenario = CalamariScenario()
    params = scenario.default_trainer_params()
    
    # Critical: Use the existing model as warmstart
    params.warmstart = checkpoint_path
    params.output_dir = model_dir
    params.epochs = 1
    
    # Point to at least one valid image/gt pair
    params.train.images = [os.path.join(data_dir, "*.bin.png")]
    params.train.gt_extension = ".gt.txt"
    
    trainer = scenario.cls().create_trainer(params)
    
    # This might still trigger the TF 2.4 error during warmstart load.
    # If it does, we must initialize a NEW model and train from scratch,
    # or use the 'calamari-predict' CLI which sometimes bypasses this.
    trainer.train()
    
    # Explicitly save in the new format
    trainer.scenario.save_model(params.output_dir)
    print("✅ Migration successful. Model is now v5.")
    
  except Exception as e:
    print(f"❌ Migration failed: {e}")
    print("💡 Alternative: Delete 'models/new_model' and let the system train a fresh one.")

if __name__ == "__main__":
  migrate()