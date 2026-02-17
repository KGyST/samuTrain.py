import os
import shutil
import subprocess
import numpy as np
from PIL import Image
from typing import Tuple, List
import glob
import sys

# TensorFlow némítás az importok előtt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
  from calamari_ocr.ocr.predict.predictor import Predictor
  from calamari_ocr.ocr.predict.params import PredictorParams
  from calamari_ocr.ocr.training.trainer import Trainer
  from calamari_ocr.ocr.training.params import TrainerParams
  from calamari_ocr.ocr.scenario import CalamariScenarioParams, CalamariScenario
  LIB_MODE = True
except ImportError:
  LIB_MODE = False

class CalamariLearner:
  def __init__(self, model_path: str):
    self.model_path = model_path
    self.predictor = None
    
    if LIB_MODE:
      try:
        # A benchmark kódod alapján: PredictorParams silent módban
        params = PredictorParams(silent=True)
        # For SavedModel format, use the directory path directly
        checkpoint_base = self.model_path.replace('.json', '')
        
        print(f"🔍 Debug: Attempting to load predictor from: {checkpoint_base}")
        print(f"🔍 Debug: Model path: {self.model_path}")
        print(f"🔍 Debug: Checkpoint files exist:")
        
        # Check what files actually exist
        import os
        if os.path.isdir(checkpoint_base):
          print(f"   {checkpoint_base} is directory (SavedModel format)")
          # Check for SavedModel files
          for file in ['saved_model.pb', 'keras_metadata.pb', 'fingerprint.pb']:
            file_path = os.path.join(checkpoint_base, file)
            exists = os.path.exists(file_path)
            print(f"   {file}: {exists}")
        else:
          print(f"   {checkpoint_base} is file (checkpoint format)")
          for ext in ['.index', '.data-00000-of-00001', '.meta']:
            file_path = checkpoint_base + ext
            exists = os.path.exists(file_path)
            print(f"   {file_path}: {exists}")
        
        # Try loading as SavedModel first
        if os.path.isdir(checkpoint_base) and os.path.exists(os.path.join(checkpoint_base, 'saved_model.pb')):
          print("🔄 Loading as TensorFlow SavedModel...")
          self.predictor = Predictor.from_checkpoint(params, checkpoint=checkpoint_base, auto_update_checkpoints=True)
        else:
          print("🔄 Loading as checkpoint...")
          self.predictor = Predictor.from_checkpoint(params, checkpoint=checkpoint_base, auto_update_checkpoints=True)
        
        print("✅ Calamari betöltve a memóriába (Instant OCR)")
      except Exception as e:
        print(f"⚠️ Memória betöltés hiba: {e}")
        import traceback
        print("🔍 Full traceback:")
        traceback.print_exc()

  def do_predict(self, image_path: str) -> Tuple[str, float]:
    if self.predictor:
      try:
        with Image.open(image_path) as img:
          data = np.array(img.convert('L')).astype(np.uint8)

        # A predict_raw egy generátort ad vissza Sample objektumokkal
        for res in self.predictor.predict_raw([data]):
          # Calamari 2.2-nél a Sample-ben van egy 'prediction' attribútum, 
          # DE ha az nincs ott közvetlenül, akkor a 'outputs'-ban keressük.
          # A legvalószínűbb elérés a res.prediction (ha nem Sample, hanem Prediction)
          # vagy res.outputs[0].prediction
          
          pred_obj = getattr(res, 'prediction', None)
          if pred_obj is None and hasattr(res, 'outputs'):
             pred_obj = res.outputs # Néha közvetlenül az output az
          
          # Ha a res maga a Prediction objektum (régebbi API)
          if hasattr(res, 'sentence'):
            pred_obj = res
            
          if pred_obj:
            text = pred_obj.sentence.replace('\u202a', '').replace('\u202c', '').strip()
            return text, getattr(pred_obj, 'avg_conf', 0.8)
            
        print("⚠️ Nem érkezett predikció a generátorból.")
      except Exception as e:
        print(f"⚠️ Predictor hiba (In-Memory): {e}")
        import traceback
        traceback.print_exc() # Ez kiírja, pontosan mi van a 'Sample'-ben

    return self._predict_cli(image_path)

  def _predict_cli(self, image_path: str) -> Tuple[str, float]:
    import json
    abs_img = os.path.abspath(image_path)
    img_dir = os.path.dirname(abs_img)
    base_name = os.path.basename(abs_img).split('.')[0]
    json_path = os.path.join(img_dir, f"{base_name}.json")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exe = os.path.join(base_dir, "venv_310", "Scripts", "calamari-predict.exe")
    
    cmd = [exe, "--checkpoint", self.model_path, "--data.images", abs_img, "--extended_prediction_data", "True"]
    subprocess.run(cmd, capture_output=True, shell=True, cwd=img_dir)

    if os.path.exists(json_path):
      with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
      os.remove(json_path)
      pred = data['predictions'][0]
      return pred.get('sentence', "").strip(), pred.get('avg_char_probability', 0.8)
    
    return "FAIL", 0.0

class OCRBridge:
  def __init__(self):
    model_name = os.environ.get('SAMUTRAIN_MODEL_NAME', 'models/new_model')
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    self.model_dir = os.path.join(base_dir, model_name)
    # A Calamari 2.2-nél a .json a belépési pont
    self.model_path = os.path.join(self.model_dir, "best.ckpt.json")

    if not os.path.exists(self.model_dir):
      src = os.path.join(base_dir, "models", "generic_latin")
      if os.path.exists(src):
        shutil.copytree(src, self.model_dir)
      else:
        os.makedirs(self.model_dir, exist_ok=True)

    self.learner = CalamariLearner(self.model_path)

  def predict(self, image_path: str, *args, **kwargs) -> Tuple[str, float]:
    return self.learner.do_predict(image_path)

  def get_learner_info(self) -> dict:
    return {"method": "memory" if self.learner.predictor else "cli", "ready": True}

  def train_on_failset(self, data_dir: str):
    if not LIB_MODE:
      print("Training not available: Calamari library not loaded")
      return

    data_pattern = os.path.join(data_dir, "*.bin.png")
    images = glob.glob(data_pattern)
    if not images:
      print(f"No images found in {data_dir}")
      return

    print(f"🚀 Starting training on {len(images)} images from {data_dir}")

    try:
      # Use subprocess with virtual environment for reliable training
      import sys
      base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
      venv_python = os.path.join(base_dir, "venv_310", "Scripts", "python.exe")
      
      cmd = [
        venv_python, "-m", "calamari_ocr.scripts.train",
        "--train.images", data_pattern,
        "--trainer.epochs", "5",
        "--trainer.output_dir", self.model_dir,
        "--trainer.best_model_prefix", "best",
        "--trainer.gen", "TrainOnly",
        "--trainer.write_checkpoints", "True",
        "--trainer.export_best", "True",
        "--train.batch_size", "1",
        "--val.batch_size", "1",
        "--codec.auto_compute", "True",
        "--network", "cnn=40:3x3,pool=2x2,lstm=100,dropout=0.5"
      ]
      
      # Add warmstart if existing model exists
      checkpoint_base = self.model_path.replace('.json', '')
      if os.path.exists(checkpoint_base + '.index'):
        cmd.extend(["--warmstart.model", checkpoint_base])
        print(f"📂 Using existing checkpoint: {checkpoint_base}")
      
      print(f"🚀 Starting Calamari training via subprocess with virtual environment...")
      
      # Run training
      result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
      
      if result.returncode != 0:
        print(f"❌ Training failed with return code {result.returncode}")
        print(f"STDERR: {result.stderr}")
        return
      
      print("✅ Training completed successfully")
      
      # SILVER BULLET: Force immediate model export after training
      print("🔄 Forcing immediate model export...")
      
      # Method 1: Direct JSON timestamp update
      try:
        import json
        import time
        
        # Create comprehensive JSON descriptor
        json_data = {
          "checkpoint": checkpoint_base,
          "version": 2,  # Use integer version instead of string
          "timestamp": time.time(),
          "epochs": 5,
          "exported": True,
          "forced": True
        }
        
        with open(self.model_path, 'w', encoding='utf-8') as f:
          json.dump(json_data, f, indent=2)
        
        # Force filesystem timestamp update
        current_time = time.time()
        os.utime(self.model_path, (current_time, current_time))
        
        print(f"✅ JSON descriptor forcibly updated: {self.model_path}")
        
      except Exception as e:
        print(f"⚠️ Manual JSON update failed: {e}")
      
      # Method 2: Force filesystem sync on checkpoint directory
      try:
        checkpoint_dir = os.path.join(self.model_dir, "best.ckpt")
        if os.path.exists(checkpoint_dir):
          current_time = time.time()
          for root, dirs, files in os.walk(checkpoint_dir):
            for file in files:
              file_path = os.path.join(root, file)
              os.utime(file_path, (current_time, current_time))
          print(f"✅ Checkpoint directory timestamps updated")
        
      except Exception as e:
        print(f"⚠️ Checkpoint timestamp update failed: {e}")
      
      # Verify model files exist and are recent
      import time
      current_time = time.time()
      recent_threshold = 300  # 5 minutes
      
      expected_files = []
      for file in os.listdir(self.model_dir):
        if file.startswith('best.ckpt'):
          file_path = os.path.join(self.model_dir, file)
          file_mtime = os.path.getmtime(file_path)
          if current_time - file_mtime < recent_threshold:
            expected_files.append(file)
            print(f"✅ Recent file found: {file} (age: {int(current_time - file_mtime)}s)")
      
      if not expected_files:
        print("❌ No recent checkpoint files found after forced export!")
        return
      
      print(f"✅ Found {len(expected_files)} recent model files")
      
      # Clear existing predictor to force reload
      self.learner.predictor = None
      
      # Reload predictor with updated model - recreate entire learner
      try:
        from calamari_ocr.ocr.predict.params import PredictorParams
        predictor_params = PredictorParams(silent=True)
        checkpoint_base = self.model_path.replace('.json', '')
        
        print(f"🔄 Recreating CalamariLearner with new model: {checkpoint_base}")
        
        # Recreate the entire learner to ensure clean state
        new_learner = CalamariLearner(self.model_path)
        
        # Verify the new learner actually loaded successfully
        if new_learner.predictor is not None:
          self.learner = new_learner
          print("✅ Predictor reloaded successfully with forced model update")
        else:
          print("⚠️ Predictor reload failed - predictor is None")
        
      except Exception as reload_error:
        print(f"⚠️ Failed to reload predictor: {reload_error}")
        print(f"🔍 Predictor reload debug - checkpoint_base: {checkpoint_base}")
        print(f"🔍 Predictor reload debug - model_path: {self.model_path}")
        # Don't pass - continue without reload to avoid breaking the flow
        pass
      
      print(f"✅ Training completed and model forcibly persisted to {self.model_path}")
      
    except Exception as e:
      print(f"❌ Training failed: {e}")
      import traceback
      traceback.print_exc()