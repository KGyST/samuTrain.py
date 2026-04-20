import os
import sys

# Add src/ to sys.path
scriptDir = os.path.dirname(os.path.abspath(__file__))
projectRoot = os.path.dirname(scriptDir)
srcDir = os.path.join(projectRoot, 'src')
sys.path.insert(0, srcDir)

def try_ocr(test_image: str) -> None:
  """
  Central script to be called so only this function needs to be imported by the test_ script.
  """
  from bridge import OCRBridge
  
  assert os.path.isabs(test_image), f"Path must be absolute: {test_image}"
  assert os.path.exists(test_image), f"Test image missing: {test_image}"

  try:
    ocrBridge = OCRBridge()
    ocrBridge.predict(test_image)
      
  except Exception:
    pass

def main() -> None:
  """
  Main function to parse arguments and run OCR test.
  """
  sTestImage = os.path.abspath('data/64_case/010001.bin.png')
  try_ocr(sTestImage)

if __name__ == "__main__":
  main()

