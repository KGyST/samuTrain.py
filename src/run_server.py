#!/usr/bin/env python3
"""
samuTrain V2 Server Launcher
Run this script to start the samuTrain OCR monitoring server
Enhanced with signal handling, library integration, and validation from try_start_training
"""

import os
import sys
import argparse
import signal
import subprocess
import codecs
import glob
import json
import shutil
from datetime import datetime

# Add src/ and project root to sys.path for proper module imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

# Force minimal logging (from try_start_training)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ['CALAMARI_LOG_LEVEL'] = 'ERROR'

# Protocol Buffers compatibility (from try_start_training)
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Set UTF-8 encoding for stdout to handle emoji characters (from try_start_training)
UTF_8 = 'utf-8'
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter(UTF_8)(sys.stdout.detach())

# Import Calamari library components for library mode (from try_start_training)
try:
    from calamari_ocr.ocr.scenario import CalamariScenario
    from calamari_ocr.scripts.train import main as calamari_train
    LIB_MODE = True
except ImportError:
    LIB_MODE = False

# Global variables for graceful shutdown
server_process = None
shutdown_requested = False

def clean_unicode_text(text: str) -> str:
    """Remove or replace invisible Unicode control characters for better display (from try_start_training)"""
    if not text:
        return text
    
    # Handle both actual Unicode characters and their escaped representations
    replacements = {
        '\u202a': '[LTR]',  # Left-to-Right Embedding
        '\u202b': '[RTL]',  # Right-to-Left Embedding
        '\u202c': '[PDF]',  # Pop Directional Formatting
        '\u202d': '[LRO]',  # Left-to-Right Override
        '\u202e': '[RLO]',  # Right-to-Left Override
    }

    replacements = {
        **replacements,
        **{f'{repr(r)[1:-1]}': s for r, s in replacements.items()}
    }
    
    # Replace control characters with readable alternatives or remove them
    cleaned = text
    for char, replacement in replacements.items():
        cleaned = cleaned.replace(char, replacement)
    
    return cleaned

def normalize_data_path(data_path):
    """Normalize data path with glob pattern (from try_start_training)"""
    # If already contains glob pattern or .bin.png, return as-is
    if "*" in data_path or data_path.endswith(".bin.png"):
        return data_path
    
    # If it's a directory, add glob pattern
    if os.path.isdir(data_path):
        if not data_path.endswith(os.sep):
            data_path += os.sep
        return data_path + "*.bin.png"
    
    # If it looks like a directory path (ends with /), add glob pattern
    if data_path.endswith("/"):
        return data_path + "*.bin.png"
    
    # Otherwise, assume it's a glob pattern
    return data_path

def verify_dataset(data_pattern):
    """Verify that dataset exists (from try_start_training)"""
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

def verify_model_folder(model_folder):
    """Verify that model folder exists and has required files"""
    if not os.path.exists(model_folder):
        print(f"❌ Model folder not found: {model_folder}")
        return False
    
    # Check for model files
    required_files = ["best.ckpt.json", "trainer_params.json"]
    for file in required_files:
        file_path = os.path.join(model_folder, file)
        if not os.path.exists(file_path):
            print(f"⚠️ Missing model file: {file}")
    
    return True

def start_initial_training(data_pattern, epochs, output_dir, network):
    """Start training from scratch using try_start_training logic"""
    if not network:
        network = "cnn=8:3x3,pool=2x2,lstm=32"
    
    print("🚀 Starting initial training from scratch...")
    print(f"📊 Data: {data_pattern}")
    print(f"🎯 Epochs: {epochs}")
    print(f"📁 Output: {output_dir}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert to absolute path
    data_pattern = os.path.abspath(data_pattern)
    output_dir = os.path.abspath(output_dir)
    
    try:
        # Use library API for training
        trainer_params = CalamariScenario.default_trainer_params()
        trainer_params.output_dir = output_dir
        trainer_params.epochs = epochs
        trainer_params.network = network
        trainer_params.gen.train.images = [data_pattern]
        trainer_params.gen.val.images = [data_pattern]  # Use same data for validation
        trainer_params.gen.setup.train.batch_size = 1
        trainer_params.gen.setup.val.batch_size = 1
        trainer_params.codec.auto_compute = True
        trainer_params.progress_bar = False
        
        result = calamari_train(trainer_params)
        print("✅ Initial training completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Initial training failed: {e}")
        return False

def signal_handler(signum, frame=None):
    """Handle Ctrl+C gracefully and save best model (from try_start_training)"""
    global server_process, shutdown_requested
    print(f"\nReceived signal {signum}. Shutting down server gracefully...")
    shutdown_requested = True
    
    if server_process:
        print("Terminating server process...")
        server_process.terminate()
        try:
            server_process.wait(timeout=30)  # Wait up to 30 seconds
            print("Server terminated gracefully")
        except subprocess.TimeoutExpired:
            print("Server didn't terminate gracefully, forcing kill...")
            server_process.kill()
            server_process.wait()
            print("Server force-killed")
    
    print("Exiting...")
    sys.exit(0)

def validate_environment():
    """Validate environment before starting server"""
    print("🔍 Validating environment...")
    
    # Check virtual environment
    venv_python = os.path.join(project_root, 'venv_310', 'Scripts', 'python.exe')
    if not os.path.exists(venv_python):
        print("⚠️ Virtual environment not found, may cause issues")
    
    # Check Calamari library
    if not LIB_MODE:
        print("⚠️ Calamari library not available, some features may be limited")
    else:
        print("✅ Calamari library available")
    
    print("✅ Environment validation completed")

# Force use of virtual environment
venv_python = os.path.join(project_root, 'venv_310', 'Scripts', 'python.exe')
if os.path.exists(venv_python) and sys.executable != venv_python:
    print(f"⚠️ Switching to virtual environment Python: {venv_python}")
    os.execv(venv_python, [venv_python] + sys.argv)

if __name__ == "__main__":
    # Set up signal handlers for graceful shutdown (from try_start_training)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser(description="Start samuTrain V2 Server")
    parser.add_argument("--data-folder", default="data/64_case",
                       help="Path to data folder (default: data/64_case)")
    parser.add_argument("--model-folder", default="models/new_model",
                       help="Path to model folder (default: models/new_model)")
    parser.add_argument("--validate", action="store_true",
                       help="Validate environment before starting")
    
    # Note: --reset-db is deprecated - use --new instead (auto-resets database)
    parser.add_argument("--reset-db", action="store_true",
                       help=argparse.SUPPRESS)  # Hidden for backward compatibility
    
    # New parameters from try_start_training for new model support
    parser.add_argument("--new", action="store_true",
                       help="Create new model on startup")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Training epochs for new model (default: 5)")
    parser.add_argument("--network", default=None,
                       help="Network architecture (default: cnn=8:3x3,pool=2x2,lstm=32)")
    parser.add_argument("--force", action="store_true",
                       help="Force overwrite existing model directory")
    parser.add_argument("--auto-continue", action="store_true",
                       help="Automatically continue with same data after initial training")
    
    args = parser.parse_args()

    # Ensure UTF-8 output on Windows (avoids UnicodeEncodeError for emoji in db.py)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    
    # Set data folder as environment variable for other modules
    os.environ['SAMUTRAIN_DATA_FOLDER'] = args.data_folder
    os.environ['SAMUTRAIN_MODEL_FOLDER'] = args.model_folder
    
    # Validate environment if requested
    if args.validate:
        validate_environment()
    
    # Handle deprecated --reset-db parameter
    if args.reset_db:
        print("⚠️ --reset-db is deprecated, using --new instead")
        args.new = True
    
    # Auto-reset database when creating new model
    if args.new:
        print("🔄 Auto-resetting database for new model...")
        from db import db
        success = db.reset_database()
        if not success:
            print("❌ Database reset failed")
            sys.exit(1)
        print("✅ Database reset completed")
    
    # Handle new model creation before starting server
    if args.new:
        print(f"🆕 Creating new model: {args.model_folder}")
        
        # Normalize data path
        data_path = normalize_data_path(args.data_folder)
        
        # Verify dataset
        if not verify_dataset(data_path):
            print("❌ Dataset validation failed")
            sys.exit(1)
        
        # Check model folder existence
        model_exists = os.path.exists(args.model_folder)
        
        # Validate arguments
        if args.new and model_exists:
            if not args.force:
                print(f"❌ Model folder already exists: {args.model_folder}")
                print(f"   Use --force to overwrite or choose a different folder")
                sys.exit(1)
            shutil.rmtree(args.model_folder)
            print(f"🗑️ Removed existing model folder: {args.model_folder}")
        
        # Start initial training
        if not start_initial_training(data_path, args.epochs, args.model_folder, args.network):
            print("❌ Initial training failed")
            sys.exit(1)
        
        # Handle auto-continuation for new models
        if args.auto_continue:
            print("\n🤖 Auto-continuation enabled, proceeding...")
            # Import here to avoid circular imports
            from bridge import OCRBridge
            from db import db
            ocr_bridge = OCRBridge(db)
            success = ocr_bridge.continue_learning(
                args.data_folder, args.model_folder, args.network, epochs=args.epochs
            )
            if success:
                print(f"\n🎉 Complete training workflow finished! Model: {args.model_folder}")
            else:
                print("❌ Continuation failed")
                sys.exit(1)
        else:
            print(f"\n🎉 New model created! Model saved to: {args.model_folder}")
    
    from main import app
    import uvicorn
    
    print("🔄🚀 Starting samuTrain V2 Server...")
    print(f"📁 Data folder: {args.data_folder}")
    print(f"📁 Model folder: {args.model_folder}")
    if args.new:
        print("🆕 Creating new model (database auto-reset)")
    if LIB_MODE:
        print("🔬 Calamari library mode: ENABLED")
    else:
        print("⚠️ Calamari library mode: DISABLED")
    print("📊 UI will be available at: http://127.0.0.1:8000")
    print("📚 API docs at: http://127.0.0.1:8000/docs")
    print("⏹️  Press Ctrl+C to stop the server")
    
    try:
        uvicorn.run(app, host="127.0.0.1", port=8000, access_log=False)
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT)
    except Exception as e:
        print(f"❌ Server error: {e}")
        signal_handler(signal.SIGTERM)
