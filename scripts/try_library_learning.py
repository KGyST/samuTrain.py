# What is this file for: Test script for library-based continue learning
# When it is created: 2026-03-10

import os
import sys
import tempfile
import shutil

# Add src/ and project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

# Force minimal logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ['CALAMARI_LOG_LEVEL'] = 'ERROR'

def test_continue_learning_engine():
  """Test the new library-based continue learning engine"""
  print("🧪 Testing ContinueLearningEngine...")
  
  try:
    from engines.continue_learning import ContinueLearningEngine
    
    # Test with a dummy model directory
    engine = ContinueLearningEngine("models/new_model")
    print("✅ Engine initialized successfully")
    
    # Test character collection
    test_data_dir = os.path.join(project_root, "data")
    if os.path.exists(test_data_dir):
      chars = engine.collect_chars(test_data_dir)
      print(f"✅ Character collection works: found {len(chars)} characters")
    else:
      print("⚠️ No data directory found for character collection test")
    
    # Test network detection
    model_dir = os.path.join(project_root, "models", "new_model")
    if os.path.exists(model_dir):
      network = engine.get_current_network(model_dir)
      print(f"✅ Network detection works: {network}")
    else:
      print("⚠️ No model directory found for network detection test")
    
    print("✅ ContinueLearningEngine tests passed")
    return True
    
  except Exception as e:
    print(f"❌ ContinueLearningEngine test failed: {e}")
    import traceback
    traceback.print_exc()
    return False

def test_calamari_learner():
  """Test the enhanced CalamariLearner"""
  print("\n🧪 Testing enhanced CalamariLearner...")
  
  try:
    from engines.calamari_learner import CalamariLearner
    
    model_path = os.path.join(project_root, "models", "new_model", "best.ckpt.json")
    learner = CalamariLearner(model_path)
    print("✅ CalamariLearner initialized successfully")
    
    # Test learning engine access
    engine = learner.get_learning_engine()
    print("✅ Learning engine access works")
    
    print("✅ CalamariLearner tests passed")
    return True
    
  except Exception as e:
    print(f"❌ CalamariLearner test failed: {e}")
    import traceback
    traceback.print_exc()
    return False

def test_bridge_integration():
  """Test bridge integration"""
  print("\n🧪 Testing bridge integration...")
  
  try:
    from bridge import OCRBridge
    
    bridge = OCRBridge()
    print("✅ OCRBridge initialized successfully")
    
    # Test that the continue_learning method exists
    if hasattr(bridge, 'continue_learning'):
      print("✅ Bridge has continue_learning method")
    else:
      print("❌ Bridge missing continue_learning method")
      return False
    
    print("✅ Bridge integration tests passed")
    return True
    
  except Exception as e:
    print(f"❌ Bridge integration test failed: {e}")
    import traceback
    traceback.print_exc()
    return False

def test_database_integration():
  """Test database integration"""
  print("\n🧪 Testing database integration...")
  
  try:
    from db import get_training_sessions, insert_training_session, update_training_session
    
    # Test training sessions methods exist
    print("✅ Database training session methods available")
    
    # Get current statistics
    stats = get_training_sessions(limit=1)
    print(f"✅ Can query training sessions: {len(stats)} returned")
    
    print("✅ Database integration tests passed")
    return True
    
  except Exception as e:
    print(f"❌ Database integration test failed: {e}")
    import traceback
    traceback.print_exc()
    return False

if __name__ == "__main__":
  print("🚀 Starting library-based learning tests...\n")
  
  tests = [
    test_continue_learning_engine,
    test_calamari_learner,
    test_bridge_integration,
    test_database_integration
  ]
  
  passed = 0
  total = len(tests)
  
  for test in tests:
    if test():
      passed += 1
  
  print(f"\n📊 Test Results: {passed}/{total} tests passed")
  
  if passed == total:
    print("🎉 All tests passed! Library-based learning is ready.")
  else:
    print("⚠️ Some tests failed. Check the implementation.")
