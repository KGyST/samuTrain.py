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

import requests
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
    
    print("Starting samuTrain V2 Training Script")
    print("=" * 40)
    print(f"Data folder: {args.data_folder}")
    print(f"Epochs: {'Unlimited' if args.epochs == -1 else args.epochs}")
    print(f"Confidence threshold: {args.confidence_threshold}")
    print(f"Backend URL: {args.backend_url}")
    print("Press Ctrl+C to stop training")
    print("=" * 40)
    
    # Validate data folder
    data_path = Path(args.data_folder)
    if not data_path.exists():
        print(f"Error: Data folder '{args.data_folder}' does not exist")
        sys.exit(1)
    
    # Count training files
    png_files = list(data_path.glob("*.png"))
    if not png_files:
        print(f"Error: No PNG files found in '{args.data_folder}'")
        sys.exit(1)
    
    print(f"Found {len(png_files)} training files")
    
    # Initialize bridge
    bridge = SamuTrainBridge(
        backend_url=args.backend_url,
        confidence_threshold=args.confidence_threshold
    )
    
    # Test backend connection
    if not bridge.test_backend_connection():
        print("Error: Cannot connect to samuTrain backend")
        print("Please start the server with: python run_server.py")
        sys.exit(1)
    
    print("Backend connection successful")
    
    # Prepare trainer configuration
    trainer_config = {
        'epochs': args.epochs if args.epochs > 0 else 999999,  # Large number for "unlimited"
        'batch_size': 1,  # Single case training
        'learning_rate': 0.001,
        'dataset': str(data_path)
    }
    
    # Start training with monitoring
    print("\nStarting training with real-time monitoring...")
    print("Training progress will be sent to samuTrain backend")
    print("Low confidence predictions will be marked as failsets")
    
    try:
        # Note: This is a placeholder for actual calamari training
        # The actual implementation depends on calamari's API
        print("Note: calamari training integration needs to be implemented")
        print("This script currently demonstrates the monitoring setup")
        print("Learning step logging enabled")
        print("=" * 60)
        
        # Simulate training for demonstration
        import time
        import random
        epoch = 0
        
        # Track case states for learning step logging
        case_states = {}  # img_path -> {'is_failset': bool, 'confidence': float}
        
        # Initialize bridge for sending data to backend
        bridge = SamuTrainBridge(
            backend_url=args.backend_url,
            confidence_threshold=args.confidence_threshold
        )
        
        if not bridge.test_backend_connection():
            print("Error: Cannot connect to samuTrain backend")
            print("Please start the server with: python run_server.py")
            sys.exit(1)
        
        print("Backend connection successful - will send OCR cases during training")
        
        while args.epochs == -1 or epoch < args.epochs:
            epoch += 1
            print(f"\nEpoch {epoch} - Training in progress... 🚀")
            
            # Simulate training on each image file
            for png_file in png_files:
                img_path = str(png_file)
                base_name = png_file.stem
                # Remove .bin from basename to match GT file naming
                if base_name.endswith('.bin'):
                    base_name = base_name.replace('.bin', '')
                gt_path = png_file.parent / f"{base_name}.gt.txt"
                
                # Read ground truth if available
                gt_text = ""
                try:
                    if gt_path.exists():
                        with open(gt_path, 'r', encoding='utf-8') as f:
                            gt_text = f.read().strip()
                        print(f"  Read GT text from {gt_path}: '{gt_text}'")
                    else:
                        print(f"  GT file not found: {gt_path}")
                except Exception as e:
                    print(f"Warning: Could not read GT file {gt_path}: {e}")
                
                # Simulate OCR prediction with varying confidence
                confidence = random.uniform(0.1, 1.0)
                was_failset = case_states.get(img_path, {}).get('is_failset', False)
                
                # Determine if this prediction passes confidence threshold
                is_good_prediction = confidence >= args.confidence_threshold
                is_failset = not is_good_prediction
                
                # Log learning step with emojis
                if was_failset:
                    if is_failset:
                        print(f"LEARN: {base_name} | was in failset, failed, stays in failset | conf: {confidence:.3f} 😞")
                    else:
                        print(f"LEARN: {base_name} | was in failset, ok, moved to trainset | conf: {confidence:.3f} 🎉")
                else:
                    if is_failset:
                        print(f"LEARN: {base_name} | was in trainset, failed, moved to failset | conf: {confidence:.3f} 😢")
                    else:
                        print(f"LEARN: {base_name} | was in trainset, ok, stays in trainset | conf: {confidence:.3f} 😊")
                
                # Send OCR case to backend
                try:
                    # Create image path for frontend static serving
                    # Use the actual filename as it appears in the data folder
                    static_url = f"/static/{png_file.name}"
                    
                    case_data = {
                        'image_path': static_url,
                        'ocr_text': gt_text,  # Send actual GT text (empty string if file is empty)
                        'confidence': confidence,
                        'gt_text': gt_text,
                        'is_failset': is_failset
                    }
                    
                    response = requests.post(f"{args.backend_url}/api/cases", json=case_data, timeout=5)
                    if response.status_code == 200:
                        result = response.json()
                        print(f"✅ Sent case {base_name} to backend (ID: {result.get('case_id', 'N/A')}) 📤")
                        print(f"  Image URL: {static_url} 🖼️")
                        print(f"  Full URL: {args.backend_url}{static_url} 🔗")
                    else:
                        print(f"✗ Failed to send case {base_name}: {response.status_code} - {response.text}")
                        
                except Exception as e:
                    print(f"✗ Error sending case {base_name}: {e}")
                
                # Update case state
                case_states[img_path] = {'is_failset': is_failset, 'confidence': confidence}
            
            time.sleep(2)  # Simulate training time
            
            # Send mock data to backend for demonstration
            if epoch % 5 == 0:  # Every 5 epochs
                print(f"Sending progress data to backend... 📊")
        
        print("\n🎉 Training completed! ✨")
        
    except KeyboardInterrupt:
        # This will be caught by signal handler
        pass
    except Exception as e:
        print(f"Training error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
