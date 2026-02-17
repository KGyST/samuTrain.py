import os
import shutil
import subprocess
import numpy as np
from PIL import Image
from typing import Tuple, List

# TensorFlow némítás az importok előtt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
  from calamari_ocr.ocr.predict.predictor import Predictor
  from calamari_ocr.ocr.predict.params import PredictorParams
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
        # A checkpoint elérési útja (kiterjesztés nélkül a biztosabb)
        checkpoint_base = self.model_path.replace('.json', '')
        
        self.predictor = Predictor.from_checkpoint(params, checkpoint=checkpoint_base)
        print("✅ Calamari betöltve a memóriába (Instant OCR)")
      except Exception as e:
        print(f"⚠️ Memória betöltés hiba: {e}")

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
    model_name = os.environ.get('SAMUTRAIN_MODEL_NAME', 'new_model')
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    self.model_dir = os.path.join(base_dir, "models", model_name)
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