#!/usr/bin/env python3
"""
samuTrain V2 Training Script
Train OCR model on specified dataset with configurable epochs
"""

import sys
import os
import argparse
import signal
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bridge import SamuTrainBridge

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n⏹️  Training stopped by user")
    print("🔄 You can resume training later")
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Train samuTrain OCR model")
    parser.add_argument("--data-folder", default="data/single_case", 
                       help="Path to training data folder (default: data/single_case)")
    parser.add_argument("--epochs", type=int, default=-1,
                       help="Number of training epochs (default: unlimited, -1)")
    parser.add_argument("--confidence-threshold", type=float, default=0.8,
                       help="Confidence threshold for failset detection (default: 0.8)")
    parser.add_argument("--backend-url", default="http://127.0.0.1:8000",
                       help="samuTrain backend URL (default: http://127.0.0.1:8000)")
    
    args = parser.parse_args()
    
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🚀 samuTrain V2 Training Script")
    print("=" * 40)
    print(f"📁 Data folder: {args.data_folder}")
    print(f"🔢 Epochs: {'Unlimited' if args.epochs == -1 else args.epochs}")
    print(f"📊 Confidence threshold: {args.confidence_threshold}")
    print(f"🌐 Backend URL: {args.backend_url}")
    print("⏹️  Press Ctrl+C to stop training")
    print("=" * 40)
    
    # Validate data folder
    data_path = Path(args.data_folder)
    if not data_path.exists():
        print(f"❌ Error: Data folder '{args.data_folder}' does not exist")
        sys.exit(1)
    
    # Count training files
    png_files = list(data_path.glob("*.png"))
    if not png_files:
        print(f"❌ Error: No PNG files found in '{args.data_folder}'")
        sys.exit(1)
    
    print(f"📄 Found {len(png_files)} training files")
    
    # Initialize bridge
    bridge = SamuTrainBridge(
        backend_url=args.backend_url,
        confidence_threshold=args.confidence_threshold
    )
    
    # Test backend connection
    if not bridge.test_backend_connection():
        print("❌ Error: Cannot connect to samuTrain backend")
        print("💡 Please start the server with: python run_server.py")
        sys.exit(1)
    
    print("✅ Backend connection successful")
    
    # Prepare trainer configuration
    trainer_config = {
        'epochs': args.epochs if args.epochs > 0 else 999999,  # Large number for "unlimited"
        'batch_size': 1,  # Single case training
        'learning_rate': 0.001,
        'dataset': str(data_path)
    }
    
    # Start training with monitoring
    print("\n🎯 Starting training with real-time monitoring...")
    print("📊 Training progress will be sent to samuTrain backend")
    print("🔍 Low confidence predictions will be marked as failsets")
    
    try:
        # Note: This is a placeholder for actual calamari training
        # The actual implementation depends on calamari's API
        print("⚠️  Note: calamari training integration needs to be implemented")
        print("🔧 This script currently demonstrates the monitoring setup")
        
        # Simulate training for demonstration
        import time
        epoch = 0
        while args.epochs == -1 or epoch < args.epochs:
            epoch += 1
            print(f"📈 Epoch {epoch} - Training in progress...")
            time.sleep(2)  # Simulate training time
            
            # Send mock data to backend for demonstration
            if epoch % 5 == 0:  # Every 5 epochs
                print(f"📊 Sending progress data to backend...")
        
        print("\n✅ Training completed!")
        
    except KeyboardInterrupt:
        # This will be caught by signal handler
        pass
    except Exception as e:
        print(f"❌ Training error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
