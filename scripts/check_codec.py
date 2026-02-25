#!/usr/bin/env python3
import os
import sys

# Add src/ and project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CALAMARI_LOG_LEVEL'] = 'ERROR'

from calamari_ocr.ocr.predict.predictor import Predictor
from calamari_ocr.ocr.predict.params import PredictorParams

def check_codec():
  """Check model codec information"""
  
  model_path = os.path.join(project_root, 'models', 'new_model', 'best.ckpt')
  params = PredictorParams(silent=True)
  predictor = Predictor.from_checkpoint(params, checkpoint=model_path)
  
  print('📋 Model codec information:')
  print(f'Model type: {type(predictor.model)}')
  
  model_attrs = [attr for attr in dir(predictor.model) if not attr.startswith('_')]
  print(f'Model attributes: {model_attrs}')
  
  # Try to find codec in model
  if hasattr(predictor.model, 'codec'):
    codec = predictor.model.codec
    print(f'Codec type: {type(codec)}')
    codec_attrs = [attr for attr in dir(codec) if not attr.startswith('_')]
    print(f'Codec attributes: {codec_attrs}')
    
    if hasattr(codec, 'charset'):
      charset = codec.charset
      print(f'Charset: {charset}')
      print(f'Charset size: {len(charset)}')
      print(f'Charset as list: {list(charset)}')
    
    if hasattr(codec, 'code2char'):
      code2char = codec.code2char
      print(f'Code2Char mapping size: {len(code2char)}')
      print(f'Sample mappings: {list(code2char.items())[:10]}')
      
      # Check for common characters
      common_chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', 'ö', 'ő', 'l']
      print(f'Common characters in codec:')
      for char in common_chars:
        has_char = char in charset if hasattr(codec, 'charset') else char in code2char.values()
        print(f'  {char}: {"✅" if has_char else "❌"}')
  else:
    print('❌ No codec found in model')
    
    # Try other possible locations
    if hasattr(predictor.model, 'scenario'):
      scenario = predictor.model.scenario
      print(f'Scenario type: {type(scenario)}')
      if hasattr(scenario, 'codec'):
        codec = scenario.codec
        print(f'Found codec in scenario: {type(codec)}')
        if hasattr(codec, 'charset'):
          print(f'Charset: {codec.charset}')

if __name__ == "__main__":
  check_codec()
