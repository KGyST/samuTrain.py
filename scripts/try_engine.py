import os
import sys

# Ensure the project root is in the path so we can import our own modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engines.calamari_learner import CalamariLearner

def main():
  # Path to your test image in data/single_case/
  img_path = "data/single_case/test001.bin.png"
  
  # Note: This expects a model to exist. 
  # If you haven't trained one yet, use the 'generic_latin' path we discussed.
  model_path = "models/generic_latin/best.ckpt"
  
  print(f"🚀 Initializing CalamariLearner with model: {model_path}")
  learner = CalamariLearner(model_path=model_path)
  
  if not os.path.exists(img_path):
    print(f"❌ Test image not found: {img_path}")
    return

  if not learner.is_available():
    print(f"⚠️ Model files (.json) not found at {model_path}")
    print("Please run the download script or train a model first.")
    return

  print(f"🔮 Predicting for: {img_path}")
  try:
    text, confidence = learner.predict(img_path)
    
    print("\n" + "="*30)
    print(f"RESULT: '{text}'")
    print(f"CONFIDENCE: {confidence:.4f}")
    print("="*30)
    
    if text:
      print("🎯 SUCCESS: The engine returned a prediction!")
    else:
      print("❓ The engine ran, but returned an empty string.")
      
  except Exception as e:
    print(f"❌ Critical error during prediction: {e}")
    import traceback
    traceback.print_exc()

if __name__ == "__main__":
  main()